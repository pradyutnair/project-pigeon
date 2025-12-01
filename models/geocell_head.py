"""
Geocell Classification Head for geo-localization.

This module implements the geocell classification head that predicts
which geographic cell an image belongs to. Uses label smoothing
based on haversine distance to improve training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from typing import Optional, Tuple
from pathlib import Path


class GeocelClassificationHead(nn.Module):
    """
    Geocell classification head.
    
    Takes image features (optionally concatenated with concept activations)
    and predicts which geocell the image belongs to.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_geocells: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        use_concept_features: bool = True,
        num_concepts: Optional[int] = None
    ):
        """
        Initialize geocell head.
        
        Args:
            input_dim: Dimension of projected features
            num_geocells: Number of geocells to classify
            hidden_dim: Hidden layer dimension
            dropout: Dropout probability
            use_concept_features: Whether to also use concept activations
            num_concepts: Number of concepts (required if use_concept_features=True)
        """
        super().__init__()
        
        self.num_geocells = num_geocells
        self.use_concept_features = use_concept_features
        
        # Compute total input dimension
        total_input_dim = input_dim
        if use_concept_features:
            assert num_concepts is not None, "num_concepts required when use_concept_features=True"
            total_input_dim += num_concepts
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_geocells)
        )
        
        print(f"GeocelClassificationHead initialized:")
        print(f"  Input dim: {total_input_dim}")
        print(f"  Num geocells: {num_geocells}")
        print(f"  Use concept features: {use_concept_features}")
    
    def forward(
        self,
        projected_features: torch.Tensor,
        concept_activations: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            projected_features: Features from CEM (B, input_dim)
            concept_activations: Optional concept activations (B, num_concepts)
            
        Returns:
            logits: Geocell logits (B, num_geocells)
        """
        if self.use_concept_features and concept_activations is not None:
            features = torch.cat([projected_features, concept_activations], dim=-1)
        else:
            features = projected_features
        
        logits = self.classifier(features)
        return logits


class GeocellManager:
    """
    Manager for geocell data.
    
    Handles loading geocell coordinates and computing predictions.
    """
    
    def __init__(
        self,
        geocell_path: Optional[str] = None,
        geocell_coords: Optional[torch.Tensor] = None
    ):
        """
        Initialize geocell manager.
        
        Args:
            geocell_path: Path to geocell CSV file
            geocell_coords: Pre-loaded geocell coordinates (num_geocells, 2) as (lng, lat)
        """
        if geocell_coords is not None:
            self.geocell_coords = geocell_coords
        elif geocell_path is not None:
            self.geocell_coords = self._load_geocells(geocell_path)
        else:
            raise ValueError("Either geocell_path or geocell_coords must be provided")
        
        self.num_geocells = self.geocell_coords.shape[0]
        print(f"GeocellManager initialized with {self.num_geocells} geocells")
    
    def _load_geocells(self, path: str) -> torch.Tensor:
        """Load geocell centroids from CSV."""
        df = pd.read_csv(path)
        coords = torch.tensor(df[['lng', 'lat']].values, dtype=torch.float32)
        return coords
    
    def get_geocell_coords(self, device: str = 'cpu') -> torch.Tensor:
        """Get geocell coordinates tensor."""
        return self.geocell_coords.to(device)
    
    def predict_coords(
        self,
        geocell_probs: torch.Tensor,
        top_k: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict coordinates from geocell probabilities.
        
        Args:
            geocell_probs: Softmax probabilities (B, num_geocells)
            top_k: Number of top predictions to return
            
        Returns:
            pred_coords: Predicted coordinates (B, 2) as (lng, lat)
            pred_cells: Predicted geocell indices (B,)
        """
        # Get top-k predictions
        topk_probs, topk_cells = torch.topk(geocell_probs, k=top_k, dim=-1)
        
        # Get coordinates of top prediction
        pred_cells = topk_cells[:, 0]  # (B,)
        geocells = self.geocell_coords.to(geocell_probs.device)
        pred_coords = geocells[pred_cells]  # (B, 2)
        
        return pred_coords, pred_cells


def haversine_distance(
    coords1: torch.Tensor,
    coords2: torch.Tensor
) -> torch.Tensor:
    """
    Compute haversine distance between two sets of coordinates.
    
    Args:
        coords1: (B, 2) as (lng, lat) in degrees
        coords2: (B, 2) as (lng, lat) in degrees
        
    Returns:
        distances: (B,) distances in km
    """
    R = 6371.0  # Earth's radius in km
    
    lng1, lat1 = coords1[:, 0], coords1[:, 1]
    lng2, lat2 = coords2[:, 0], coords2[:, 1]
    
    # Convert to radians
    lng1 = torch.deg2rad(lng1)
    lat1 = torch.deg2rad(lat1)
    lng2 = torch.deg2rad(lng2)
    lat2 = torch.deg2rad(lat2)
    
    dlng = lng2 - lng1
    dlat = lat2 - lat1
    
    a = torch.sin(dlat / 2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlng / 2)**2
    c = 2 * torch.arcsin(torch.sqrt(a))
    
    return R * c


def haversine_matrix(
    coords: torch.Tensor,
    geocell_coords: torch.Tensor
) -> torch.Tensor:
    """
    Compute haversine distances from coords to all geocells.
    
    Args:
        coords: (B, 2) as (lng, lat) in degrees
        geocell_coords: (N, 2) geocell centroids as (lng, lat) in degrees
        
    Returns:
        distances: (B, N) distances in km
    """
    B = coords.shape[0]
    N = geocell_coords.shape[0]
    
    R = 6371.0  # Earth's radius in km
    
    # Expand for broadcasting
    lng1 = coords[:, 0:1]  # (B, 1)
    lat1 = coords[:, 1:2]  # (B, 1)
    lng2 = geocell_coords[:, 0:1].T  # (1, N)
    lat2 = geocell_coords[:, 1:2].T  # (1, N)
    
    # Convert to radians
    lng1 = torch.deg2rad(lng1)
    lat1 = torch.deg2rad(lat1)
    lng2 = torch.deg2rad(lng2)
    lat2 = torch.deg2rad(lat2)
    
    dlng = lng2 - lng1
    dlat = lat2 - lat1
    
    a = torch.sin(dlat / 2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlng / 2)**2
    c = 2 * torch.arcsin(torch.sqrt(torch.clamp(a, min=0, max=1)))
    
    return R * c  # (B, N)


def smooth_labels(
    distances: torch.Tensor,
    smoothing_constant: float = 65.0
) -> torch.Tensor:
    """
    Compute soft labels based on haversine distances.
    
    Label smoothing penalizes predictions based on actual distance
    rather than treating all incorrect cells equally.
    
    Args:
        distances: (B, N) distances in km to each geocell
        smoothing_constant: Controls smoothing strength (from PIGEON config)
        
    Returns:
        soft_labels: (B, N) soft label probabilities
    """
    # Convert distances to soft labels using exponential decay
    # Closer cells get higher weight
    weights = torch.exp(-distances / smoothing_constant)
    
    # Normalize to sum to 1
    soft_labels = weights / weights.sum(dim=-1, keepdim=True)
    
    return soft_labels


def geocell_loss(
    logits: torch.Tensor,
    gt_coords: torch.Tensor,
    geocell_coords: torch.Tensor,
    use_label_smoothing: bool = True,
    smoothing_constant: float = 65.0
) -> torch.Tensor:
    """
    Compute geocell classification loss.
    
    Args:
        logits: Geocell logits (B, N)
        gt_coords: Ground truth coordinates (B, 2) as (lng, lat)
        geocell_coords: Geocell centroids (N, 2) as (lng, lat)
        use_label_smoothing: Whether to use distance-based label smoothing
        smoothing_constant: Smoothing constant for label smoothing
        
    Returns:
        loss: Scalar loss
    """
    if use_label_smoothing:
        # Compute distances to all geocells
        distances = haversine_matrix(gt_coords, geocell_coords)
        
        # Create soft labels
        soft_labels = smooth_labels(distances, smoothing_constant)
        
        # Cross-entropy with soft labels
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -torch.sum(soft_labels * log_probs, dim=-1).mean()
    else:
        # Find nearest geocell (hard labels)
        distances = haversine_matrix(gt_coords, geocell_coords)
        hard_labels = distances.argmin(dim=-1)
        
        loss = F.cross_entropy(logits, hard_labels)
    
    return loss


if __name__ == "__main__":
    # Test the module
    print("Testing GeocelClassificationHead...")
    
    # Create dummy data
    batch_size = 8
    input_dim = 512
    num_concepts = 100
    num_geocells = 2203
    
    # Create head
    head = GeocelClassificationHead(
        input_dim=input_dim,
        num_geocells=num_geocells,
        use_concept_features=True,
        num_concepts=num_concepts
    )
    
    # Test forward pass
    projected = torch.randn(batch_size, input_dim)
    concepts = torch.sigmoid(torch.randn(batch_size, num_concepts))
    
    logits = head(projected, concepts)
    print(f"Logits shape: {logits.shape}")  # (8, 2203)
    
    # Test loss computation
    geocell_coords = torch.randn(num_geocells, 2) * 180  # Dummy coords
    geocell_coords[:, 0] = torch.clamp(geocell_coords[:, 0], -180, 180)  # lng
    geocell_coords[:, 1] = torch.clamp(geocell_coords[:, 1], -90, 90)   # lat
    
    gt_coords = torch.randn(batch_size, 2) * 180
    gt_coords[:, 0] = torch.clamp(gt_coords[:, 0], -180, 180)
    gt_coords[:, 1] = torch.clamp(gt_coords[:, 1], -90, 90)
    
    loss = geocell_loss(
        logits, gt_coords, geocell_coords,
        use_label_smoothing=True,
        smoothing_constant=65.0
    )
    print(f"Loss: {loss.item():.4f}")
    
    # Test haversine distance
    dist = haversine_distance(gt_coords[:2], geocell_coords[:2])
    print(f"Sample distances: {dist.tolist()}")
    
    print("\nTest passed!")




