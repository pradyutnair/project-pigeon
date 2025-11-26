"""
Loss functions for ConceptGeo training.

Includes:
- Concept classification loss (metaName supervision)
- Geocell classification loss (with label smoothing)
- Image-Note contrastive loss (InfoNCE)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


def concept_classification_loss(
    concept_logits: torch.Tensor,
    concept_labels: torch.Tensor,
    multi_label: bool = False
) -> torch.Tensor:
    """
    Compute concept classification loss.
    
    Args:
        concept_logits: Raw logits from concept head (B, num_concepts)
        concept_labels: Ground truth labels
            - If multi_label=False: (B,) indices
            - If multi_label=True: (B, num_concepts) multi-hot
        multi_label: Whether this is multi-label classification
        
    Returns:
        loss: Scalar loss
    """
    if multi_label:
        # Binary cross-entropy for multi-label
        probs = torch.sigmoid(concept_logits)
        return F.binary_cross_entropy(probs, concept_labels.float())
    else:
        # Cross-entropy for single-label
        return F.cross_entropy(concept_logits, concept_labels)


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
    - Concept classification loss
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
        temperature: float = 0.07
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
        """
        super().__init__()
        
        self.lambda_concept = lambda_concept
        self.lambda_geocell = lambda_geocell
        self.lambda_contrastive = lambda_contrastive
        self.use_label_smoothing = use_label_smoothing
        self.smoothing_constant = smoothing_constant
        self.temperature = temperature
        
        print(f"ConceptGeoLoss initialized:")
        print(f"  lambda_concept: {lambda_concept}")
        print(f"  lambda_geocell: {lambda_geocell}")
        print(f"  lambda_contrastive: {lambda_contrastive}")
    
    def forward(
        self,
        concept_logits: torch.Tensor,
        concept_labels: torch.Tensor,
        geocell_logits: torch.Tensor,
        gt_coords: torch.Tensor,
        geocell_coords: torch.Tensor,
        image_features: Optional[torch.Tensor] = None,
        note_features: Optional[torch.Tensor] = None,
        note_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss.
        
        Args:
            concept_logits: Concept classification logits (B, num_concepts)
            concept_labels: Ground truth concept indices (B,)
            geocell_logits: Geocell classification logits (B, num_geocells)
            gt_coords: Ground truth coordinates (B, 2) as (lng, lat)
            geocell_coords: Geocell centroids (N, 2)
            image_features: Image features for contrastive (B, D)
            note_features: Note features for contrastive (B, D) 
            note_mask: Boolean mask for samples with valid notes (B,)
            
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual losses
        """
        loss_dict = {}
        
        # 1. Concept classification loss
        loss_concept = concept_classification_loss(concept_logits, concept_labels)
        loss_dict['concept'] = loss_concept.item()
        
        # 2. Geocell classification loss
        from models.geocell_head import geocell_loss
        loss_geocell = geocell_loss(
            geocell_logits, gt_coords, geocell_coords,
            use_label_smoothing=self.use_label_smoothing,
            smoothing_constant=self.smoothing_constant
        )
        loss_dict['geocell'] = loss_geocell.item()
        
        # 3. Image-note contrastive loss (if notes available)
        loss_contrastive = torch.tensor(0.0, device=concept_logits.device)
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
        
        # Combined loss
        total_loss = (
            self.lambda_concept * loss_concept +
            self.lambda_geocell * loss_geocell +
            self.lambda_contrastive * loss_contrastive
        )
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

