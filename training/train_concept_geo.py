#!/usr/bin/env python3
"""
Training script for ConceptGeo model with WandB logging.

Finetunes concept and geocell heads on GeoGuessr meta dataset
using frozen GeoCLIP backbone.

Multi-objective training:
1. Concept classification (metaName supervision)
2. Geocell classification (with label smoothing)
3. Image-Note contrastive alignment

Usage:
    python training/train_concept_geo.py --data-root data --geoguessr-id 6906237dc7731161a37282b2
    python training/train_concept_geo.py --wandb-project conceptgeo --wandb-name exp1
    python training/train_concept_geo.py --no-wandb  # disable logging
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
    compute_accuracy
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
                    f"{distance:.1f}",
                    metadata[i].get('country', 'Unknown'),
                    ", ".join([f"{c}: {a:.2f}" for c, a in zip(top_concepts[:3], top_activations[:3])])
                ])
                samples_logged += 1
    
    if WANDB_AVAILABLE and wandb.run is not None:
        columns = ["pano_id", "gt_concept", "pred_concept", "correct", "distance_km", "country", "top_concepts"]
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
    log_wandb: bool = True
) -> Dict[str, float]:
    """Train for one epoch with optional wandb logging."""
    model.train()
    
    total_loss = 0
    total_concept_loss = 0
    total_geocell_loss = 0
    total_contrastive_loss = 0
    correct_concepts = 0
    total_samples = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        images, concept_idx, country_idx, coords, metadata = batch
        
        images = images.to(device)
        concept_idx = concept_idx.to(device)
        coords = coords.to(device)
        
        # Extract notes
        notes = [m['note'] for m in metadata]
        
        # Forward pass
        outputs = model(images)
        
        # Encode notes
        note_features = None
        note_mask = None
        valid_notes = [n for n in notes if len(n.strip()) > 0]
        
        if len(valid_notes) > 1:
            note_mask = torch.tensor([len(n.strip()) > 0 for n in notes], device=device)
            note_features = model.encode_notes(valid_notes)
        
        # Compute loss
        # Convert coords from (lat, lng) to (lng, lat) for geocell loss
        gt_coords_lnglat = coords[:, [1, 0]]  # Swap columns
        
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
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track metrics
        total_loss += loss_dict['total']
        total_concept_loss += loss_dict['concept']
        total_geocell_loss += loss_dict['geocell']
        total_contrastive_loss += loss_dict['contrastive']
        
        # Concept accuracy
        pred_concepts = outputs.concept_logits.argmax(dim=-1)
        correct_concepts += (pred_concepts == concept_idx).sum().item()
        total_samples += images.shape[0]
        
        # Update progress bar
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


def main(args):
    """Main training function with optional WandB logging."""
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize wandb
    log_wandb = WANDB_AVAILABLE and not args.no_wandb
    if log_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args)
        )
        print(f"WandB logging enabled: {args.wandb_project}/{args.wandb_name or wandb.run.name}")
    else:
        print("WandB logging disabled")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"conceptgeo_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
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
    
    # Count parameters
    trainable_params = sum(p.numel() for p in model.trainable_parameters())
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    
    # ============================================
    # 4. SETUP TRAINING
    # ============================================
    print("\n" + "="*50)
    print("Setting up training...")
    print("="*50)
    
    # Loss function
    criterion = ConceptGeoLoss(
        lambda_concept=args.lambda_concept,
        lambda_geocell=args.lambda_geocell,
        lambda_contrastive=args.lambda_contrastive,
        use_label_smoothing=True,
        smoothing_constant=65.0
    )
    
    # Optimizer (only trainable parameters)
    optimizer = AdamW(
        model.trainable_parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.learning_rate * 0.01
    )
    
    # Log additional config to wandb
    if log_wandb:
        wandb.config.update({
            'trainable_params': trainable_params,
            'num_concepts': len(concept_to_idx),
            'num_train_samples': len(train_samples),
            'num_val_samples': len(val_samples)
        })
    
    # ============================================
    # 5. TRAINING LOOP
    # ============================================
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)
    
    best_val_acc = 0
    best_epoch = 0
    history = []
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, 
            device, epoch, geocell_coords, log_wandb=log_wandb
        )
        
        # Evaluate
        val_metrics = evaluate(
            model, val_loader, criterion, device, geocell_coords
        )
        
        # Update scheduler
        scheduler.step()
        
        # Combine metrics
        metrics = {**train_metrics, **val_metrics, 'epoch': epoch, 'lr': scheduler.get_last_lr()[0]}
        history.append(metrics)
        
        # Log to wandb
        if log_wandb:
            wandb.log(metrics)
            
            # Log sample predictions every 10 epochs
            if epoch % 10 == 0:
                log_sample_predictions(model, val_loader, idx_to_concept, device, num_samples=20)
        
        # Print metrics
        print(f"Train Loss: {train_metrics['train/loss']:.4f}, "
              f"Concept Acc: {train_metrics['train/concept_acc']:.2%}")
        print(f"Val Loss: {val_metrics['val/loss']:.4f}, "
              f"Concept Acc: {val_metrics['val/concept_acc_top1']:.2%} (top-5: {val_metrics['val/concept_acc_top5']:.2%})")
        print(f"Median Distance: {val_metrics['val/median_distance_km']:.1f} km")
        print(f"Distance Accuracies: "
              f"1km={val_metrics['val/acc_1km']:.2%}, "
              f"25km={val_metrics['val/acc_25km']:.2%}, "
              f"200km={val_metrics['val/acc_200km']:.2%}")
        
        # Save best model
        if val_metrics['val/concept_acc_top1'] > best_val_acc:
            best_val_acc = val_metrics['val/concept_acc_top1']
            best_epoch = epoch
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics
            }, output_dir / "best_model.pt")
            
            print(f"  New best model saved! (concept_acc={best_val_acc:.2%})")
            
            if log_wandb:
                wandb.run.summary['best_epoch'] = epoch
                wandb.run.summary['best_concept_acc'] = best_val_acc
                wandb.run.summary['best_median_distance_km'] = val_metrics['val/median_distance_km']
        
        # Save checkpoint every N epochs
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics
            }, output_dir / f"checkpoint_epoch_{epoch}.pt")
    
    # ============================================
    # 6. SAVE FINAL RESULTS
    # ============================================
    print("\n" + "="*50)
    print("Training complete!")
    print("="*50)
    
    # Save training history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': history[-1]
    }, output_dir / "final_model.pt")
    
    # Finish wandb run
    if log_wandb:
        wandb.finish()
    
    print(f"\nBest model at epoch {best_epoch} with concept_acc={best_val_acc:.2%}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ConceptGeo model")
    
    # Data arguments
    parser.add_argument("--data-root", type=str, default="data",
                        help="Root directory for data")
    parser.add_argument("--geoguessr-id", type=str, default="6906237dc7731161a37282b2",
                        help="GeoGuessr map ID")
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
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loader workers")
    
    # Loss weights
    parser.add_argument("--lambda-concept", type=float, default=0.5,
                        help="Weight for concept loss")
    parser.add_argument("--lambda-geocell", type=float, default=1.0,
                        help="Weight for geocell loss")
    parser.add_argument("--lambda-contrastive", type=float, default=0.3,
                        help="Weight for contrastive loss")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, default="runs",
                        help="Output directory for checkpoints")
    parser.add_argument("--save-every", type=int, default=10,
                        help="Save checkpoint every N epochs")
    
    # WandB arguments
    parser.add_argument("--wandb-project", type=str, default="conceptgeo",
                        help="WandB project name")
    parser.add_argument("--wandb-name", type=str, default=None,
                        help="WandB run name (auto-generated if not provided)")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable WandB logging")
    
    args = parser.parse_args()
    main(args)

