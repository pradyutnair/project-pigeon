#!/usr/bin/env python3
"""
Training script for ConceptGeo model with WandB logging.

Two-stage curriculum training (automatic):
1. Stage 1 (Concept): Train concept embedding head to convergence
2. Stage 2 (Distance): Freeze concept head, train geocell head

The training automatically runs both stages sequentially.

Usage:
    python training/train_concept_geo.py --data-root data --geoguessr-id 6906237dc7731161a37282b2
    python training/train_concept_geo.py --concept-epochs 30 --geocell-epochs 50
    python training/train_concept_geo.py --country-filter Japan  # Train on specific country
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from tqdm import tqdm
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from dataset import (
    PanoramaCBMDataset, 
    create_splits,
    create_splits_stratified,
    SubsetDataset,
    get_statistics,
    print_statistics
)

from models.geoclip_backbone import FrozenGeoCLIP, create_backbone
from models.concept_embedding import ConceptEmbeddingModule, ConceptBank
from models.concept_geo import ConceptGeo, ConceptGeoOutput
from models.geocell_head import (
    GeocelClassificationHead, 
    haversine_distance,
    haversine_matrix,
    smooth_labels,
    geocell_loss
)
from training.losses import (
    ConceptGeoLoss,
    concept_classification_loss,
    image_note_contrastive_loss,
    compute_accuracy,
    compute_class_weights,
    FocalLoss
)


def collate_fn(batch):
    """Custom collate function to handle metadata dict."""
    images = torch.stack([b[0] for b in batch])
    concept_idx = torch.tensor([b[1] for b in batch])
    country_idx = torch.tensor([b[2] for b in batch])
    coords = torch.stack([b[3] for b in batch])
    metadata = [b[4] for b in batch]
    return images, concept_idx, country_idx, coords, metadata


def create_concept_bank_from_samples(
    samples: List[Dict],
    backbone: FrozenGeoCLIP,
    prompt_template: str = "a street view photo showing {}"
) -> Tuple[torch.Tensor, Dict[str, int], Dict[int, str]]:
    """
    Create concept bank from dataset samples.
    
    Args:
        samples: List of sample dicts with 'meta_name' field
        backbone: Backbone model for text encoding
        prompt_template: Template for concept prompts
        
    Returns:
        concept_bank: Tensor of concept embeddings (num_concepts, embed_dim)
        concept_to_idx: Mapping from concept name to index
        idx_to_concept: Mapping from index to concept name
    """
    # Extract unique concepts
    concept_names = sorted(set(s['meta_name'] for s in samples))
    concept_to_idx = {name: i for i, name in enumerate(concept_names)}
    idx_to_concept = {i: name for name, i in concept_to_idx.items()}
    
    print(f"Creating concept bank with {len(concept_names)} concepts...")
    
    # Create prompts
    prompts = [prompt_template.format(name) for name in concept_names]
    
    # Encode in batches
    batch_size = 32
    embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_emb = backbone.encode_text(batch_prompts)
            embeddings.append(batch_emb.cpu())
    
    concept_bank = torch.cat(embeddings, dim=0)
    concept_bank = F.normalize(concept_bank, dim=-1)
    
    print(f"Concept bank shape: {concept_bank.shape}")
    
    return concept_bank, concept_to_idx, idx_to_concept


def create_geocell_coords_from_samples(
    samples: List[Dict],
    num_cells: int = 500
) -> torch.Tensor:
    """
    Create geocell coordinates by clustering sample locations.
    
    Args:
        samples: List of sample dicts with 'lat' and 'lng' fields
        num_cells: Number of geocells to create
        
    Returns:
        geocell_coords: Tensor of shape (num_cells, 2) as (lng, lat)
    """
    from sklearn.cluster import KMeans
    
    # Extract coordinates
    coords = []
    for s in samples:
        if s['lat'] is not None and s['lng'] is not None:
            coords.append([s['lng'], s['lat']])
    
    coords = np.array(coords)
    print(f"Clustering {len(coords)} locations into {num_cells} geocells...")
    
    # Cluster
    kmeans = KMeans(n_clusters=num_cells, random_state=42, n_init=10)
    kmeans.fit(coords)
    
    geocell_coords = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
    
    return geocell_coords


def log_sample_predictions(
    model: ConceptGeo,
    dataloader: DataLoader,
    idx_to_concept: Dict[int, str],
    device: torch.device,
    num_samples: int = 20
) -> List:
    """Log sample predictions to wandb table."""
    model.eval()
    samples_logged = 0
    table_data = []
    
    with torch.no_grad():
        for batch in dataloader:
            if samples_logged >= num_samples:
                break
            images, concept_idx, country_idx, coords, metadata = batch
            images = images.to(device)
            outputs = model(images)
            
            for i in range(min(images.shape[0], num_samples - samples_logged)):
                gt_concept = idx_to_concept[concept_idx[i].item()]
                pred_concept_idx = outputs.concept_logits[i].argmax().item()
                pred_concept = idx_to_concept[pred_concept_idx]
                
                # Top 5 concepts
                top_vals, top_idxs = torch.topk(outputs.concept_activations[i], k=5)
                top_concepts = [idx_to_concept[idx.item()] for idx in top_idxs]
                top_activations = top_vals.tolist()
                
                # Distance
                pred_coord = outputs.pred_coords[i].cpu()
                gt_coord = coords[i][[1, 0]]
                distance = haversine_distance(pred_coord.unsqueeze(0), gt_coord.unsqueeze(0)).item()
                
                is_correct = (gt_concept == pred_concept)
                
                table_data.append([
                    metadata[i]['pano_id'],
                    gt_concept,
                    pred_concept,
                    is_correct,
                    gt_coord,
                    pred_coord,
                    f"{distance:.1f}",
                    metadata[i].get('country', 'Unknown'),
                    ", ".join([f"{c}: {a:.2f}" for c, a in zip(top_concepts[:3], top_activations[:3])])
                ])
                samples_logged += 1
    
    if WANDB_AVAILABLE and wandb.run is not None:
        columns = ["pano_id", "gt_concept", "pred_concept", "correct", "gt_coord", "pred_coord", "distance_km", "country", "top_concepts"]
        table = wandb.Table(columns=columns, data=table_data)
        wandb.log({"predictions": table})
    
    return table_data


def train_epoch(
    model: ConceptGeo,
    train_loader: DataLoader,
    criterion: ConceptGeoLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    geocell_coords: torch.Tensor,
    stage: str = "all",
    log_wandb: bool = True
) -> Dict[str, float]:
    """
    Train for one epoch with optional wandb logging.
    
    Args:
        model: ConceptGeo model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device
        epoch: Current epoch
        geocell_coords: Geocell coordinates
        stage: "concept" - only concept loss
               "geocell" - only geocell loss
               "all" - both losses
        log_wandb: Whether to log to wandb
    """
    model.train()
    
    total_loss = 0
    total_concept_loss = 0
    total_geocell_loss = 0
    total_contrastive_loss = 0
    correct_concepts = 0
    total_samples = 0
    
    stage_desc = {"concept": "[Concept Stage]", "geocell": "[Distance Stage]", "all": "[Joint]"}
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} {stage_desc.get(stage, '')}")
    
    for batch_idx, batch in enumerate(pbar):
        images, concept_idx, country_idx, coords, metadata = batch
        
        images = images.to(device)
        concept_idx = concept_idx.to(device)
        coords = coords.to(device)
        
        # Extract notes (only used in concept stage)
        notes = [m['note'] for m in metadata]
        
        # Forward pass
        outputs = model(images)
        
        # Encode notes for contrastive loss (concept stage only)
        note_features = None
        note_mask = None
        if stage in ["concept", "all"]:
            valid_notes = [n for n in notes if len(n.strip()) > 0]
            if len(valid_notes) > 1:
                note_mask = torch.tensor([len(n.strip()) > 0 for n in notes], device=device)
                note_features = model.encode_notes(valid_notes)
        
        # Convert coords from (lat, lng) to (lng, lat) for geocell loss
        gt_coords_lnglat = coords[:, [1, 0]]
        
        # Compute stage-specific loss
        if stage == "concept":
            # Only concept classification + contrastive alignment
            loss, loss_dict = criterion(
                concept_logits=outputs.concept_logits,
                concept_labels=concept_idx,
                geocell_logits=None,  # Skip geocell loss
                gt_coords=None,
                geocell_coords=None,
                image_features=outputs.image_features,
                note_features=note_features,
                note_mask=note_mask
            )
        elif stage == "geocell":
            # Only geocell classification
            loss, loss_dict = criterion(
                concept_logits=None,  # Skip concept loss
                concept_labels=None,
                geocell_logits=outputs.geocell_logits,
                gt_coords=gt_coords_lnglat,
                geocell_coords=geocell_coords,
                image_features=None,
                note_features=None,
                note_mask=None
            )
        else:  # "all"
            loss, loss_dict = criterion(
                concept_logits=outputs.concept_logits,
                concept_labels=concept_idx,
                geocell_logits=outputs.geocell_logits,
                gt_coords=gt_coords_lnglat,
                geocell_coords=geocell_coords,
                image_features=outputs.image_features,
                note_features=note_features,
                note_mask=note_mask
            )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(stage), max_norm=1.0)
        optimizer.step()
        
        # Track metrics
        total_loss += loss_dict['total']
        total_concept_loss += loss_dict['concept']
        total_geocell_loss += loss_dict['geocell']
        total_contrastive_loss += loss_dict['contrastive']
        
        # Concept accuracy (always track for monitoring)
        pred_concepts = outputs.concept_logits.argmax(dim=-1)
        correct_concepts += (pred_concepts == concept_idx).sum().item()
        total_samples += images.shape[0]
        
        # Update progress bar
        if stage == "concept":
            pbar.set_postfix({
                'concept_loss': f"{loss_dict['concept']:.4f}",
                'concept_acc': f"{correct_concepts/total_samples:.2%}"
            })
        elif stage == "geocell":
            pbar.set_postfix({
                'geocell_loss': f"{loss_dict['geocell']:.4f}",
                'concept_acc': f"{correct_concepts/total_samples:.2%}"
            })
        else:
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'concept_acc': f"{correct_concepts/total_samples:.2%}"
            })
        
        # Log batch metrics to wandb
        if log_wandb and WANDB_AVAILABLE and wandb.run is not None and batch_idx % 10 == 0:
            wandb.log({
                'batch/loss': loss_dict['total'],
                'batch/concept_loss': loss_dict['concept'],
                'batch/geocell_loss': loss_dict['geocell'],
                'batch/contrastive_loss': loss_dict['contrastive'],
            })
    
    num_batches = len(train_loader)
    metrics = {
        'train/loss': total_loss / num_batches,
        'train/concept_loss': total_concept_loss / num_batches,
        'train/geocell_loss': total_geocell_loss / num_batches,
        'train/contrastive_loss': total_contrastive_loss / num_batches,
        'train/concept_acc': correct_concepts / total_samples
    }
    
    return metrics


@torch.no_grad()
def evaluate(
    model: ConceptGeo,
    val_loader: DataLoader,
    criterion: ConceptGeoLoss,
    device: torch.device,
    geocell_coords: torch.Tensor
) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    
    total_loss = 0
    correct_concepts = 0
    correct_top5 = 0
    total_samples = 0
    
    all_pred_coords = []
    all_gt_coords = []
    
    for batch in tqdm(val_loader, desc="Evaluating"):
        images, concept_idx, country_idx, coords, metadata = batch
        
        images = images.to(device)
        concept_idx = concept_idx.to(device)
        coords = coords.to(device)
        
        # Forward pass
        outputs = model(images)
        
        # Convert coords from (lat, lng) to (lng, lat)
        gt_coords_lnglat = coords[:, [1, 0]]
        
        # Compute loss (without contrastive for eval)
        loss, loss_dict = criterion(
            concept_logits=outputs.concept_logits,
            concept_labels=concept_idx,
            geocell_logits=outputs.geocell_logits,
            gt_coords=gt_coords_lnglat,
            geocell_coords=geocell_coords
        )
        
        total_loss += loss_dict['total']
        
        # Top-1 concept accuracy
        pred_concepts = outputs.concept_logits.argmax(dim=-1)
        correct_concepts += (pred_concepts == concept_idx).sum().item()
        
        # Top-5 concept accuracy
        _, top5_preds = torch.topk(outputs.concept_logits, k=min(5, outputs.concept_logits.shape[1]), dim=-1)
        correct_top5 += (top5_preds == concept_idx.unsqueeze(-1)).any(dim=-1).sum().item()
        
        total_samples += images.shape[0]
        
        # Collect coordinates for distance metrics
        all_pred_coords.append(outputs.pred_coords.cpu())
        all_gt_coords.append(gt_coords_lnglat.cpu())
    
    # Compute distance metrics
    all_pred_coords = torch.cat(all_pred_coords, dim=0)
    all_gt_coords = torch.cat(all_gt_coords, dim=0)
    
    distances = haversine_distance(all_pred_coords, all_gt_coords)
    
    # Distance thresholds
    thresholds = [1, 25, 200, 750, 2500]  # km
    
    num_batches = len(val_loader)
    metrics = {
        'val/loss': total_loss / num_batches,
        'val/concept_acc_top1': correct_concepts / total_samples,
        'val/concept_acc_top5': correct_top5 / total_samples,
        'val/median_distance_km': distances.median().item(),
        'val/mean_distance_km': distances.mean().item(),
    }
    
    for t in thresholds:
        acc = (distances <= t).float().mean().item()
        metrics[f'val/acc_{t}km'] = acc
    
    return metrics


def train_stage(
    model: ConceptGeo,
    train_loader: DataLoader,
    val_loader: DataLoader,
    geocell_coords: torch.Tensor,
    idx_to_concept: Dict[int, str],
    device: torch.device,
    output_dir: Path,
    stage: str,
    num_epochs: int,
    learning_rate: float,
    weight_decay: float,
    lambda_contrastive: float,
    log_wandb: bool = True,
    class_weights: Optional[torch.Tensor] = None,
    use_focal_loss: bool = True,
    focal_gamma: float = 2.0
) -> Dict[str, float]:
    """
    Train a single stage (concept or geocell).
    
    Args:
        class_weights: Pre-computed class weights for concept imbalance
        use_focal_loss: Whether to use focal loss for concept stage
        focal_gamma: Gamma parameter for focal loss
    
    Returns:
        best_metrics: Dictionary of best metrics from this stage
    """
    print("\n" + "="*60)
    stage_names = {"concept": "STAGE 1: CONCEPT EMBEDDING", "geocell": "STAGE 2: DISTANCE PREDICTION"}
    print(f"{stage_names.get(stage, stage)}")
    print("="*60)
    
    # Set model training stage
    model.set_training_stage(stage)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.trainable_parameters(stage))
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function with class balancing for concept stage
    if stage == "concept":
        criterion = ConceptGeoLoss(
            lambda_concept=1.0,
            lambda_geocell=0.0,
            lambda_contrastive=lambda_contrastive,
            use_label_smoothing=True,
            smoothing_constant=65.0,
            class_weights=class_weights,
            use_focal_loss=use_focal_loss,
            focal_gamma=focal_gamma
        )
        best_metric_name = "val/concept_acc_top1"
        best_metric_val = 0
        metric_higher_is_better = True
    else:  # geocell
        criterion = ConceptGeoLoss(
            lambda_concept=0.0,
            lambda_geocell=1.0,
            lambda_contrastive=0.0,
            use_label_smoothing=True,
            smoothing_constant=65.0
        )
        best_metric_name = "val/median_distance_km"
        best_metric_val = float('inf')
        metric_higher_is_better = False
    
    # Optimizer
    optimizer = AdamW(
        model.trainable_parameters(stage),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Warmup + Cosine Annealing scheduler
    # Warmup for first 3 epochs (or 10% of epochs, whichever is larger)
    warmup_epochs = max(3, int(num_epochs * 0.1))
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup from 0.1 * lr to lr
            return 0.1 + 0.9 * (epoch / warmup_epochs)
        else:
            # Cosine annealing after warmup
            progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            return 0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    print(f"Using warmup for {warmup_epochs} epochs, then cosine annealing")
    
    best_epoch = 0
    best_metrics = {}
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n--- {stage.upper()} Epoch {epoch}/{num_epochs} ---")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer,
            device, epoch, geocell_coords,
            stage=stage, log_wandb=log_wandb
        )
        
        # Evaluate
        val_metrics = evaluate(
            model, val_loader, criterion, device, geocell_coords
        )
        
        scheduler.step()
        
        # Combine metrics with stage prefix
        metrics = {
            **{f"{stage}/{k}": v for k, v in train_metrics.items()},
            **{f"{stage}/{k}": v for k, v in val_metrics.items()},
            'epoch': epoch,
            'stage': stage,
            'lr': scheduler.get_last_lr()[0]
        }
        
        # Log to wandb
        if log_wandb and WANDB_AVAILABLE and wandb.run is not None:
            wandb.log(metrics)
            if epoch % 10 == 0:
                log_sample_predictions(model, val_loader, idx_to_concept, device, num_samples=20)
        
        # Print metrics
        if stage == "concept":
            print(f"  Loss: {train_metrics['train/concept_loss']:.4f}, "
                  f"Concept Acc: {val_metrics['val/concept_acc_top1']:.2%} "
                  f"(top-5: {val_metrics['val/concept_acc_top5']:.2%})")
        else:
            print(f"  Loss: {train_metrics['train/geocell_loss']:.4f}, "
                  f"Median Dist: {val_metrics['val/median_distance_km']:.1f} km, "
                  f"25km: {val_metrics['val/acc_25km']:.2%}, "
                  f"200km: {val_metrics['val/acc_200km']:.2%}")
        
        # Check if best
        current_metric = val_metrics[best_metric_name]
        is_best = (metric_higher_is_better and current_metric > best_metric_val) or \
                  (not metric_higher_is_better and current_metric < best_metric_val)
        
        if is_best:
            best_metric_val = current_metric
            best_epoch = epoch
            best_metrics = val_metrics.copy()
            
            torch.save({
                'epoch': epoch,
                'stage': stage,
                'model_state_dict': model.state_dict(),
                'metrics': val_metrics
            }, output_dir / f"best_{stage}.pt")
            
            if stage == "concept":
                print(f"  ★ New best! (concept_acc={best_metric_val:.2%})")
            else:
                print(f"  ★ New best! (median_dist={best_metric_val:.1f} km)")
            
            if log_wandb and WANDB_AVAILABLE and wandb.run is not None:
                wandb.run.summary[f'best_{stage}_epoch'] = epoch
                wandb.run.summary[f'best_{stage}_{best_metric_name}'] = best_metric_val
    
    print(f"\n{stage.upper()} complete! Best at epoch {best_epoch}")
    return best_metrics


def main(args):
    """Main training function - runs concept stage then geocell stage automatically."""
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    date_str = datetime.now().strftime("%Y%m%d")
    output_dir = Path(args.output_dir) / f"pigeon-cbm_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Determine country filter for tags
    country_filter = getattr(args, 'country_filter', None)
    country_tag = country_filter if country_filter else "global"
    
    # Initialize wandb with auto-generated name and tags
    log_wandb = WANDB_AVAILABLE and not args.no_wandb
    if log_wandb:
        run_name = f"pigeon-cbm-{timestamp}"
        tags = [
            args.backbone,                          # backbone type
            f"geocells-{args.num_geocells}",       # num geocells
            country_tag,                            # country or global
            "curriculum-training",                  # training type
            f"concept-{args.concept_epochs}ep",     # concept epochs
            f"geocell-{args.geocell_epochs}ep",     # geocell epochs
        ]
        if args.lambda_contrastive > 0:
            tags.append("contrastive-loss")
        if args.use_focal_loss:
            tags.append(f"focal-loss-g{args.focal_gamma}")
        tags.append(f"weights-{args.class_weight_strategy}")
        
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            tags=tags,
            config=vars(args)
        )
        print(f"WandB: {args.wandb_project}/{run_name}")
        print(f"Tags: {tags}")
    else:
        print("WandB logging disabled")
    
    # Save args
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # ============================================
    # 1. LOAD DATASET
    # ============================================
    print("\n" + "="*50)
    print("Loading dataset...")
    print("="*50)
    
    dataset = PanoramaCBMDataset(
        geoguessr_id=args.geoguessr_id,
        data_root=args.data_root,
        require_coordinates=True,
        image_size=(224, 224)  # GeoCLIP uses CLIP ViT-L/14 which expects 224x224
    )
    
    # Print statistics
    stats = get_statistics(dataset.samples)
    print_statistics(stats)
    
    # Split data
    if args.stratified_split:
        train_samples, val_samples, test_samples = create_splits_stratified(
            dataset.samples, 
            train_ratio=0.8, 
            val_ratio=0.1, 
            test_ratio=0.1
        )
    else:
        train_samples, val_samples, test_samples = create_splits(
            dataset.samples,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1
        )
    
    print(f"Split sizes: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")
    
    # Create data loaders
    train_dataset = SubsetDataset(dataset, train_samples)
    val_dataset = SubsetDataset(dataset, val_samples)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # ============================================
    # 2. CREATE BACKBONE AND CONCEPT BANK
    # ============================================
    print("\n" + "="*50)
    print("Loading backbone and creating concept bank...")
    print("="*50)
    
    backbone = FrozenGeoCLIP(model_name=args.backbone, device=device)
    
    # Create concept bank from dataset
    concept_bank, concept_to_idx, idx_to_concept = create_concept_bank_from_samples(
        dataset.samples,
        backbone,
        prompt_template="a street view photo showing {}"
    )
    concept_bank = concept_bank.to(device)
    
    # Save concept mappings
    with open(output_dir / "concept_mappings.json", "w") as f:
        json.dump({
            'concept_to_idx': concept_to_idx,
            'idx_to_concept': {str(k): v for k, v in idx_to_concept.items()}
        }, f, indent=2)
    
    # Create geocell coordinates
    geocell_coords = create_geocell_coords_from_samples(
        train_samples,
        num_cells=args.num_geocells
    ).to(device)
    
    # Save geocell coords
    torch.save(geocell_coords, output_dir / "geocell_coords.pt")
    
    # ============================================
    # 3. CREATE MODEL
    # ============================================
    print("\n" + "="*50)
    print("Creating ConceptGeo model...")
    print("="*50)
    
    model = ConceptGeo(
        backbone=backbone,
        concept_bank=concept_bank,
        num_geocells=args.num_geocells,
        geocell_coords=geocell_coords,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        top_k=5,
        use_concept_for_geocell=True
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # ============================================
    # 3.5 COMPUTE CLASS WEIGHTS FOR IMBALANCE (optional)
    # ============================================
    # Print class distribution info
    from collections import Counter
    concept_counts = Counter(s.get('concept_idx', 0) for s in train_samples)
    most_common = concept_counts.most_common(5)
    least_common = concept_counts.most_common()[-5:]
    print(f"\nClass distribution:")
    print(f"  Most common concepts: {[(idx_to_concept.get(c, c), n) for c, n in most_common]}")
    print(f"  Least common concepts: {[(idx_to_concept.get(c, c), n) for c, n in least_common]}")
    
    # Compute class weights only if strategy is not "none"
    if args.class_weight_strategy != "none":
        print(f"Computing class weights (strategy: {args.class_weight_strategy})...")
        class_weights = compute_class_weights(
            train_samples,
            num_classes=len(concept_to_idx),
            strategy=args.class_weight_strategy,
            smoothing=0.1
        )
        class_weights = class_weights.to(device)
        print(f"  Class weight range: {class_weights.min():.2f} - {class_weights.max():.2f}")
    else:
        class_weights = None
        print("No class weights (using focal loss only for imbalance handling)")
    
    print(f"Using focal loss: {args.use_focal_loss} (gamma={args.focal_gamma})")
    
    # Log additional config to wandb
    if log_wandb:
        wandb.config.update({
            'total_params': total_params,
            'num_concepts': len(concept_to_idx),
            'num_train_samples': len(train_samples),
            'num_val_samples': len(val_samples),
            'use_focal_loss': args.use_focal_loss,
            'focal_gamma': args.focal_gamma,
            'class_weight_strategy': args.class_weight_strategy
        })
    
    # ============================================
    # 4. STAGE 1: CONCEPT TRAINING
    # ============================================
    concept_metrics = train_stage(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        geocell_coords=geocell_coords,
        idx_to_concept=idx_to_concept,
        device=device,
        output_dir=output_dir,
        stage="concept",
        num_epochs=args.concept_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lambda_contrastive=args.lambda_contrastive,
        log_wandb=log_wandb,
        class_weights=class_weights,
        use_focal_loss=args.use_focal_loss,
        focal_gamma=args.focal_gamma
    )
    
    # Load best concept model before geocell training
    print("\nLoading best concept model for geocell stage...")
    checkpoint = torch.load(output_dir / "best_concept.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # ============================================
    # 5. STAGE 2: GEOCELL TRAINING
    # ============================================
    geocell_metrics = train_stage(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        geocell_coords=geocell_coords,
        idx_to_concept=idx_to_concept,
        device=device,
        output_dir=output_dir,
        stage="geocell",
        num_epochs=args.geocell_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lambda_contrastive=0.0,  # Not used in geocell stage
        log_wandb=log_wandb
    )
    
    # ============================================
    # 6. SAVE FINAL RESULTS
    # ============================================
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    
    # Save final model (with both heads trained)
    torch.save({
        'model_state_dict': model.state_dict(),
        'concept_metrics': concept_metrics,
        'geocell_metrics': geocell_metrics,
        'config': vars(args)
    }, output_dir / "final_model.pt")
    
    # Finish wandb run
    if log_wandb:
        wandb.run.summary['final_concept_acc'] = concept_metrics.get('val/concept_acc_top1', 0)
        wandb.run.summary['final_median_distance_km'] = geocell_metrics.get('val/median_distance_km', 0)
        wandb.finish()
    
    # Print summary
    print(f"\nConcept Stage Results:")
    print(f"  Best Concept Acc: {concept_metrics.get('val/concept_acc_top1', 0):.2%}")
    print(f"\nGeocell Stage Results:")
    print(f"  Median Distance: {geocell_metrics.get('val/median_distance_km', 0):.1f} km")
    print(f"  25km Accuracy: {geocell_metrics.get('val/acc_25km', 0):.2%}")
    print(f"  200km Accuracy: {geocell_metrics.get('val/acc_200km', 0):.2%}")
    
    print(f"\nModel checkpoints saved to: {output_dir}")
    print(f"  - best_concept.pt  (Stage 1)")
    print(f"  - best_geocell.pt  (Stage 2)")
    print(f"  - final_model.pt   (Complete model)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ConceptGeo model with two-stage curriculum training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python training/train_concept_geo.py
  python training/train_concept_geo.py --concept-epochs 30 --geocell-epochs 50
  python training/train_concept_geo.py --country-filter Japan --backbone streetclip
        """
    )
    
    # Data arguments
    parser.add_argument("--data-root", type=str, default="project-pigeon/data",
                        help="Root directory for data")
    parser.add_argument("--geoguessr-id", type=str, default="6906237dc7731161a37282b2",
                        help="GeoGuessr map ID")
    parser.add_argument("--country-filter", type=str, default=None,
                        help="Filter dataset to specific country (e.g., 'Japan', 'France')")
    parser.add_argument("--stratified-split", action="store_true",
                        help="Use stratified split to ensure all concepts in train")
    
    # Model arguments
    parser.add_argument("--backbone", type=str, default="geoclip",
                        choices=["geoclip", "streetclip"],
                        help="Backbone model to use")
    parser.add_argument("--num-geocells", type=int, default=500,
                        help="Number of geocells for classification")
    parser.add_argument("--hidden-dim", type=int, default=512,
                        help="Hidden dimension for heads")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout probability")
    
    # Training arguments (separate epochs for each stage)
    parser.add_argument("--concept-epochs", type=int, default=30,
                        help="Number of epochs for concept stage")
    parser.add_argument("--geocell-epochs", type=int, default=50,
                        help="Number of epochs for geocell stage")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loader workers")
    
    # Loss weights
    parser.add_argument("--lambda-contrastive", type=float, default=0.3,
                        help="Weight for image-note contrastive loss (concept stage)")
    
    # Class imbalance handling
    parser.add_argument("--use-focal-loss", action="store_true", default=True,
                        help="Use focal loss for concept classification (default: True)")
    parser.add_argument("--no-focal-loss", dest="use_focal_loss", action="store_false",
                        help="Disable focal loss, use standard cross-entropy")
    parser.add_argument("--focal-gamma", type=float, default=1.0,
                        help="Gamma for focal loss (0=standard CE, 1=gentle, 2=aggressive). Default: 1.0")
    parser.add_argument("--class-weight-strategy", type=str, default="none",
                        choices=["none", "inverse_freq", "inverse_sqrt", "effective_num"],
                        help="Strategy for computing class weights (default: none). "
                             "Options: none (focal loss only), "
                             "inverse_freq (1/freq), "
                             "inverse_sqrt (1/sqrt(freq)), "
                             "effective_num (Class-Balanced Loss paper)")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, default="runs",
                        help="Output directory for checkpoints")
    
    # WandB arguments
    parser.add_argument("--wandb-project", type=str, default="pigeon-cbm",
                        help="WandB project name")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable WandB logging")
    
    args = parser.parse_args()
    main(args)

