#!/usr/bin/env python3
"""
Benchmark evaluation script for ConceptGeo model.

Evaluates on standard geo-localization benchmarks:
- Im2GPS3k
- GWS15k
- YFCC4k
- YFCC26k

Distance threshold metrics:
- 1km (Street-level)
- 25km (City-level)
- 200km (Region-level)
- 750km (Country-level)
- 2500km (Continent-level)
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.geocell_head import haversine_distance


class BenchmarkDataset(Dataset):
    """
    Dataset for loading benchmark images.
    """
    
    def __init__(
        self,
        image_dir: str,
        metadata_path: str,
        transform=None
    ):
        """
        Args:
            image_dir: Directory containing images
            metadata_path: Path to JSON/CSV with image metadata
            transform: Image transform
        """
        self.image_dir = Path(image_dir)
        self.transform = transform
        
        # Load metadata
        self.samples = self._load_metadata(metadata_path)
        print(f"Loaded {len(self.samples)} samples from {metadata_path}")
    
    def _load_metadata(self, path: str) -> List[Dict]:
        """Load metadata from file."""
        path = Path(path)
        
        if path.suffix == '.json':
            with open(path) as f:
                data = json.load(f)
            return data
        elif path.suffix == '.csv':
            import pandas as pd
            df = pd.read_csv(path)
            return df.to_dict('records')
        else:
            raise ValueError(f"Unknown metadata format: {path.suffix}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image_name = sample.get('image', sample.get('filename', sample.get('IMG_ID')))
        image_path = self.image_dir / image_name
        
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Get coordinates
        lat = sample.get('lat', sample.get('LAT'))
        lng = sample.get('lng', sample.get('LON'))
        
        coords = torch.tensor([lng, lat], dtype=torch.float32)  # (lng, lat)
        
        return image, coords, sample


def compute_distance_metrics(
    pred_coords: torch.Tensor,
    gt_coords: torch.Tensor
) -> Dict[str, float]:
    """
    Compute distance-based metrics.
    
    Args:
        pred_coords: Predicted coordinates (N, 2) as (lng, lat)
        gt_coords: Ground truth coordinates (N, 2) as (lng, lat)
        
    Returns:
        Dictionary with metrics
    """
    distances = haversine_distance(pred_coords, gt_coords)
    
    # Distance thresholds
    thresholds = {
        'street_1km': 1,
        'city_25km': 25,
        'region_200km': 200,
        'country_750km': 750,
        'continent_2500km': 2500
    }
    
    metrics = {}
    
    for name, threshold in thresholds.items():
        acc = (distances <= threshold).float().mean().item()
        metrics[name] = acc
    
    # Additional metrics
    metrics['median_km'] = distances.median().item()
    metrics['mean_km'] = distances.mean().item()
    metrics['std_km'] = distances.std().item()
    
    return metrics


@torch.no_grad()
def evaluate_model(
    model,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[Dict[str, float], torch.Tensor, torch.Tensor]:
    """
    Evaluate model on benchmark.
    
    Args:
        model: ConceptGeo model
        dataloader: Benchmark dataloader
        device: Device to run on
        
    Returns:
        metrics: Dictionary with evaluation metrics
        pred_coords: All predicted coordinates
        gt_coords: All ground truth coordinates
    """
    model.eval()
    
    all_pred_coords = []
    all_gt_coords = []
    all_concept_activations = []
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        images, coords, _ = batch
        images = images.to(device)
        
        # Forward pass
        outputs = model(images)
        
        all_pred_coords.append(outputs.pred_coords.cpu())
        all_gt_coords.append(coords)
        all_concept_activations.append(outputs.concept_activations.cpu())
    
    # Concatenate
    pred_coords = torch.cat(all_pred_coords, dim=0)
    gt_coords = torch.cat(all_gt_coords, dim=0)
    concept_activations = torch.cat(all_concept_activations, dim=0)
    
    # Compute metrics
    metrics = compute_distance_metrics(pred_coords, gt_coords)
    
    return metrics, pred_coords, gt_coords, concept_activations


def evaluate_on_benchmark(
    model,
    benchmark_name: str,
    benchmark_config: Dict,
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 4
) -> Dict[str, float]:
    """
    Evaluate model on a specific benchmark.
    
    Args:
        model: ConceptGeo model
        benchmark_name: Name of benchmark
        benchmark_config: Benchmark configuration (paths, etc.)
        device: Device to run on
        batch_size: Batch size
        num_workers: Number of data workers
        
    Returns:
        Dictionary with metrics
    """
    from torchvision import transforms
    
    # Create transform (match CLIP preprocessing)
    transform = transforms.Compose([
        transforms.Resize((336, 336)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])
    
    # Create dataset
    dataset = BenchmarkDataset(
        image_dir=benchmark_config['image_dir'],
        metadata_path=benchmark_config['metadata_path'],
        transform=transform
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Evaluate
    metrics, pred_coords, gt_coords, concept_acts = evaluate_model(
        model, dataloader, device
    )
    
    return metrics


def load_benchmark_configs(benchmarks_path: str) -> Dict[str, Dict]:
    """Load benchmark configurations from JSON file."""
    with open(benchmarks_path) as f:
        configs = json.load(f)
    return configs


def main(args):
    """Main evaluation function."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from {args.model_path}...")
    
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Load model configuration
    model_dir = Path(args.model_path).parent
    
    # Load concept mappings
    with open(model_dir / "concept_mappings.json") as f:
        mappings = json.load(f)
    
    idx_to_concept = {int(k): v for k, v in mappings['idx_to_concept'].items()}
    
    # Load geocell coords
    geocell_coords = torch.load(model_dir / "geocell_coords.pt")
    
    # Create model
    from models.geoclip_backbone import FrozenGeoCLIP
    from models.concept_geo import ConceptGeo
    
    backbone = FrozenGeoCLIP(model_name=args.backbone, device=device)
    
    # Create concept bank
    num_concepts = len(idx_to_concept)
    concept_bank = F.normalize(torch.randn(num_concepts, backbone.embed_dim), dim=-1)
    
    model = ConceptGeo(
        backbone=backbone,
        concept_bank=concept_bank.to(device),
        num_geocells=geocell_coords.shape[0],
        geocell_coords=geocell_coords.to(device)
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully!")
    
    # Load benchmark configs
    benchmark_configs = load_benchmark_configs(args.benchmarks_config)
    
    # Select benchmarks to evaluate
    if args.benchmarks:
        benchmarks = args.benchmarks.split(',')
    else:
        benchmarks = list(benchmark_configs.keys())
    
    # Evaluate on each benchmark
    results = {}
    
    for benchmark in benchmarks:
        if benchmark not in benchmark_configs:
            print(f"Warning: Benchmark {benchmark} not found in config")
            continue
        
        print(f"\n{'='*50}")
        print(f"Evaluating on {benchmark}...")
        print('='*50)
        
        metrics = evaluate_on_benchmark(
            model,
            benchmark,
            benchmark_configs[benchmark],
            device,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        results[benchmark] = metrics
        
        # Print results
        print(f"\nResults for {benchmark}:")
        print(f"  Street (1km):    {metrics['street_1km']:.2%}")
        print(f"  City (25km):     {metrics['city_25km']:.2%}")
        print(f"  Region (200km):  {metrics['region_200km']:.2%}")
        print(f"  Country (750km): {metrics['country_750km']:.2%}")
        print(f"  Continent (2.5k):{metrics['continent_2500km']:.2%}")
        print(f"  Median distance: {metrics['median_km']:.1f} km")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ConceptGeo on benchmarks")
    
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--backbone", type=str, default="geoclip",
                        choices=["geoclip", "streetclip"],
                        help="Backbone model used")
    parser.add_argument("--benchmarks-config", type=str, 
                        default="data/benchmarks/benchmarks.json",
                        help="Path to benchmarks config JSON")
    parser.add_argument("--benchmarks", type=str, default=None,
                        help="Comma-separated list of benchmarks to evaluate")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data workers")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save results JSON")
    
    args = parser.parse_args()
    main(args)




