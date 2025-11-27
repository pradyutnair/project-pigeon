"""
Loss functions for ConceptGeo training.

Includes:
- Concept classification loss (metaName supervision)
- Class-weighted loss for handling class imbalance
- Focal loss for hard example mining
- Geocell classification loss (with label smoothing)
- Image-Note contrastive loss (InfoNCE)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from collections import Counter
import numpy as np


def compute_class_weights(
    samples: List[Dict],
    num_classes: int,
    strategy: str = "inverse_freq",
    smoothing: float = 0.1
) -> torch.Tensor:
    """
    Compute class weights for handling class imbalance.
    
    Args:
        samples: List of sample dicts with 'concept_idx' field
        num_classes: Total number of concept classes
        strategy: Weighting strategy
            - "inverse_freq": 1 / frequency (standard)
            - "inverse_sqrt": 1 / sqrt(frequency) (less aggressive)
            - "effective_num": (1-β) / (1-β^n) where β=0.9999 (from Class-Balanced Loss paper)
        smoothing: Smoothing factor to avoid extreme weights
        
    Returns:
        weights: Tensor of shape (num_classes,)
    """
    # Count class frequencies
    concept_counts = Counter(s.get('concept_idx', 0) for s in samples)
    
    # Initialize all classes with smoothing count
    counts = torch.ones(num_classes) * smoothing
    for idx, count in concept_counts.items():
        if idx < num_classes:
            counts[idx] = count + smoothing
    
    total = counts.sum()
    
    if strategy == "inverse_freq":
        # Standard inverse frequency weighting
        weights = total / (num_classes * counts)
    elif strategy == "inverse_sqrt":
        # Less aggressive - good for moderate imbalance
        weights = torch.sqrt(total / (num_classes * counts))
    elif strategy == "effective_num":
        # From "Class-Balanced Loss Based on Effective Number of Samples"
        beta = 0.9999
        effective_num = 1.0 - torch.pow(beta, counts)
        weights = (1.0 - beta) / effective_num
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Normalize weights to have mean 1
    weights = weights / weights.mean()
    
    return weights


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    From "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    - Down-weights easy examples (high p_t)
    - Focuses on hard examples (low p_t)
    - α provides class balancing
    - γ controls focusing strength (γ=0 is standard CE, γ=2 is common)
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean"
    ):
        """
        Args:
            gamma: Focusing parameter. Higher = more focus on hard examples
            alpha: Class weights (num_classes,). If None, no class weighting
            reduction: "mean", "sum", or "none"
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, C) raw logits
            targets: (B,) class indices
        """
        # Compute softmax probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Get probability of true class
        batch_size = logits.shape[0]
        p_t = probs[torch.arange(batch_size, device=logits.device), targets]
        
        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Compute cross-entropy
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # Apply focal weight
        loss = focal_weight * ce_loss
        
        # Apply class weights if provided
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_t = alpha[targets]
            loss = alpha_t * loss
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class ClassBalancedConceptLoss(nn.Module):
    """
    Concept classification loss with class balancing options.
    
    Supports:
    - Standard weighted cross-entropy
    - Focal loss
    - Combination of both
    """
    
    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        use_focal: bool = True,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0
    ):
        """
        Args:
            class_weights: Pre-computed class weights (num_classes,)
            use_focal: Whether to use focal loss
            focal_gamma: Gamma parameter for focal loss
            label_smoothing: Label smoothing factor
        """
        super().__init__()
        self.class_weights = class_weights
        self.use_focal = use_focal
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        
        if use_focal:
            self.focal_loss = FocalLoss(
                gamma=focal_gamma,
                alpha=class_weights,
                reduction="mean"
            )
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, C) raw logits
            targets: (B,) class indices
        """
        if self.use_focal:
            return self.focal_loss(logits, targets)
        else:
            # Standard weighted cross-entropy
            weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
            return F.cross_entropy(
                logits, targets,
                weight=weight,
                label_smoothing=self.label_smoothing
            )


def concept_classification_loss(
    concept_logits: torch.Tensor,
    concept_labels: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
    use_focal: bool = False,
    focal_gamma: float = 2.0,
    multi_label: bool = False
) -> torch.Tensor:
    """
    Compute concept classification loss with optional class balancing.
    
    Args:
        concept_logits: Raw logits from concept head (B, num_concepts)
        concept_labels: Ground truth labels
            - If multi_label=False: (B,) indices
            - If multi_label=True: (B, num_concepts) multi-hot
        class_weights: Optional class weights for imbalance (num_concepts,)
        use_focal: Whether to use focal loss
        focal_gamma: Gamma for focal loss
        multi_label: Whether this is multi-label classification
        
    Returns:
        loss: Scalar loss
    """
    if multi_label:
        # Binary cross-entropy for multi-label (with optional pos_weight)
        probs = torch.sigmoid(concept_logits)
        if class_weights is not None:
            pos_weight = class_weights.to(concept_logits.device)
            return F.binary_cross_entropy_with_logits(
                concept_logits, concept_labels.float(),
                pos_weight=pos_weight
            )
        return F.binary_cross_entropy(probs, concept_labels.float())
    else:
        # Single-label classification
        if use_focal:
            focal = FocalLoss(gamma=focal_gamma, alpha=class_weights)
            return focal(concept_logits, concept_labels)
        else:
            weight = class_weights.to(concept_logits.device) if class_weights is not None else None
            return F.cross_entropy(concept_logits, concept_labels, weight=weight)


def image_note_contrastive_loss(
    image_features: torch.Tensor,
    note_features: torch.Tensor,
    temperature: float = 0.07
) -> torch.Tensor:
    """
    Compute InfoNCE contrastive loss between images and notes.
    
    Matches each image to its corresponding note in the batch.
    
    Args:
        image_features: Normalized image features (B, D)
        note_features: Normalized note text features (B, D)
        temperature: Temperature for scaling logits
        
    Returns:
        loss: Scalar loss
    """
    # Ensure normalized
    image_features = F.normalize(image_features, dim=-1)
    note_features = F.normalize(note_features, dim=-1)
    
    # Compute similarity matrix
    logits = image_features @ note_features.T / temperature  # (B, B)
    
    # Labels: diagonal is positive
    labels = torch.arange(logits.shape[0], device=logits.device)
    
    # Bidirectional loss
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    
    return (loss_i2t + loss_t2i) / 2


def concept_contrastive_loss(
    projected_features: torch.Tensor,
    concept_bank: torch.Tensor,
    concept_labels: torch.Tensor,
    temperature: float = 0.07
) -> torch.Tensor:
    """
    Contrastive loss to align projected features with concept embeddings.
    
    Encourages the projected features to be similar to the embedding
    of their ground truth concept.
    
    Args:
        projected_features: Features from CEM (B, D)
        concept_bank: Concept embeddings (num_concepts, D)
        concept_labels: Ground truth concept indices (B,)
        temperature: Temperature for scaling
        
    Returns:
        loss: Scalar loss
    """
    # Normalize
    projected_features = F.normalize(projected_features, dim=-1)
    concept_bank = F.normalize(concept_bank, dim=-1)
    
    # Compute similarities to all concepts
    logits = projected_features @ concept_bank.T / temperature  # (B, num_concepts)
    
    # Cross-entropy to match correct concept
    return F.cross_entropy(logits, concept_labels)


class ConceptGeoLoss(nn.Module):
    """
    Combined loss for ConceptGeo training.
    
    Combines:
    - Concept classification loss (with class balancing options)
    - Geocell classification loss  
    - Image-note contrastive loss (optional)
    """
    
    def __init__(
        self,
        lambda_concept: float = 0.5,
        lambda_geocell: float = 1.0,
        lambda_contrastive: float = 0.3,
        use_label_smoothing: bool = True,
        smoothing_constant: float = 65.0,
        temperature: float = 0.07,
        # Class balancing options
        class_weights: Optional[torch.Tensor] = None,
        use_focal_loss: bool = True,
        focal_gamma: float = 2.0
    ):
        """
        Initialize combined loss.
        
        Args:
            lambda_concept: Weight for concept loss
            lambda_geocell: Weight for geocell loss
            lambda_contrastive: Weight for contrastive loss
            use_label_smoothing: Whether to use label smoothing for geocell
            smoothing_constant: Smoothing constant for geocell loss
            temperature: Temperature for contrastive loss
            class_weights: Pre-computed class weights for concept imbalance (num_concepts,)
            use_focal_loss: Whether to use focal loss for concepts
            focal_gamma: Gamma parameter for focal loss (higher = more focus on hard examples)
        """
        super().__init__()
        
        self.lambda_concept = lambda_concept
        self.lambda_geocell = lambda_geocell
        self.lambda_contrastive = lambda_contrastive
        self.use_label_smoothing = use_label_smoothing
        self.smoothing_constant = smoothing_constant
        self.temperature = temperature
        
        # Class balancing
        self.class_weights = class_weights
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        
        print(f"ConceptGeoLoss initialized:")
        print(f"  lambda_concept: {lambda_concept}")
        print(f"  lambda_geocell: {lambda_geocell}")
        print(f"  lambda_contrastive: {lambda_contrastive}")
        if use_focal_loss:
            print(f"  Using Focal Loss (gamma={focal_gamma})")
        if class_weights is not None:
            print(f"  Class weights: min={class_weights.min():.2f}, max={class_weights.max():.2f}")
    
    def forward(
        self,
        concept_logits: Optional[torch.Tensor] = None,
        concept_labels: Optional[torch.Tensor] = None,
        geocell_logits: Optional[torch.Tensor] = None,
        gt_coords: Optional[torch.Tensor] = None,
        geocell_coords: Optional[torch.Tensor] = None,
        image_features: Optional[torch.Tensor] = None,
        note_features: Optional[torch.Tensor] = None,
        note_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss.
        
        Supports stage-specific training by passing None for unused components.
        
        Args:
            concept_logits: Concept classification logits (B, num_concepts) or None
            concept_labels: Ground truth concept indices (B,) or None
            geocell_logits: Geocell classification logits (B, num_geocells) or None
            gt_coords: Ground truth coordinates (B, 2) as (lng, lat) or None
            geocell_coords: Geocell centroids (N, 2) or None
            image_features: Image features for contrastive (B, D) or None
            note_features: Note features for contrastive (B, D) or None
            note_mask: Boolean mask for samples with valid notes (B,) or None
            
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual losses
        """
        loss_dict = {'concept': 0.0, 'geocell': 0.0, 'contrastive': 0.0}
        total_loss = None
        
        # Determine device from available tensors
        device = None
        for t in [concept_logits, geocell_logits, image_features]:
            if t is not None:
                device = t.device
                break
        
        # 1. Concept classification loss (if concept stage) - with class balancing
        loss_concept = torch.tensor(0.0, device=device)
        if concept_logits is not None and concept_labels is not None:
            loss_concept = concept_classification_loss(
                concept_logits, concept_labels,
                class_weights=self.class_weights,
                use_focal=self.use_focal_loss,
                focal_gamma=self.focal_gamma
            )
            loss_dict['concept'] = loss_concept.item()
            if total_loss is None:
                total_loss = self.lambda_concept * loss_concept
            else:
                total_loss = total_loss + self.lambda_concept * loss_concept
        
        # 2. Geocell classification loss (if geocell stage)
        loss_geocell = torch.tensor(0.0, device=device)
        if geocell_logits is not None and gt_coords is not None and geocell_coords is not None:
            from models.geocell_head import geocell_loss
            loss_geocell = geocell_loss(
                geocell_logits, gt_coords, geocell_coords,
                use_label_smoothing=self.use_label_smoothing,
                smoothing_constant=self.smoothing_constant
            )
            loss_dict['geocell'] = loss_geocell.item()
            if total_loss is None:
                total_loss = self.lambda_geocell * loss_geocell
            else:
                total_loss = total_loss + self.lambda_geocell * loss_geocell
        
        # 3. Image-note contrastive loss (if notes available, typically concept stage)
        loss_contrastive = torch.tensor(0.0, device=device)
        if image_features is not None and note_features is not None:
            if note_mask is not None:
                # Only use samples with valid notes
                valid_image = image_features[note_mask]
                valid_note = note_features[note_mask]
                
                if valid_image.shape[0] > 1:  # Need at least 2 for contrastive
                    loss_contrastive = image_note_contrastive_loss(
                        valid_image, valid_note, self.temperature
                    )
            else:
                loss_contrastive = image_note_contrastive_loss(
                    image_features, note_features, self.temperature
                )
            loss_dict['contrastive'] = loss_contrastive.item()
            if total_loss is None:
                total_loss = self.lambda_contrastive * loss_contrastive
            else:
                total_loss = total_loss + self.lambda_contrastive * loss_contrastive
        
        # Ensure we have some loss
        if total_loss is None:
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


def compute_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    top_k: int = 1
) -> float:
    """
    Compute top-k accuracy.
    
    Args:
        logits: Prediction logits (B, C)
        labels: Ground truth indices (B,)
        top_k: k for top-k accuracy
        
    Returns:
        accuracy: Float accuracy value
    """
    with torch.no_grad():
        _, top_indices = torch.topk(logits, k=top_k, dim=-1)
        correct = (top_indices == labels.unsqueeze(-1)).any(dim=-1)
        accuracy = correct.float().mean().item()
    
    return accuracy


if __name__ == "__main__":
    # Test losses
    print("Testing loss functions...")
    
    batch_size = 8
    num_concepts = 100
    num_geocells = 2203
    embed_dim = 512
    
    # Create dummy data
    concept_logits = torch.randn(batch_size, num_concepts)
    concept_labels = torch.randint(0, num_concepts, (batch_size,))
    
    geocell_logits = torch.randn(batch_size, num_geocells)
    gt_coords = torch.randn(batch_size, 2) * 100
    geocell_coords = torch.randn(num_geocells, 2) * 100
    
    image_features = F.normalize(torch.randn(batch_size, embed_dim), dim=-1)
    note_features = F.normalize(torch.randn(batch_size, embed_dim), dim=-1)
    
    # Test individual losses
    loss_c = concept_classification_loss(concept_logits, concept_labels)
    print(f"Concept loss: {loss_c.item():.4f}")
    
    loss_n = image_note_contrastive_loss(image_features, note_features)
    print(f"Contrastive loss: {loss_n.item():.4f}")
    
    # Test combined loss
    criterion = ConceptGeoLoss()
    total_loss, loss_dict = criterion(
        concept_logits, concept_labels,
        geocell_logits, gt_coords, geocell_coords,
        image_features, note_features
    )
    
    print(f"\nCombined loss: {total_loss.item():.4f}")
    print(f"Loss breakdown: {loss_dict}")
    
    # Test accuracy
    acc = compute_accuracy(concept_logits, concept_labels, top_k=5)
    print(f"\nTop-5 accuracy: {acc:.2%}")
    
    print("\nTest passed!")

