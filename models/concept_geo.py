"""
ConceptGeo: Main model for interpretable geo-localization.

Combines:
- Frozen GeoCLIP/StreetCLIP backbone
- Concept Embedding Module (CEM) for interpretability
- Geocell classification head
- Image-Note contrastive alignment

Architecture:
    Image -> Frozen Backbone -> Image Features
                                    |
                                    v
                            Concept Embedding Module
                                    |
                    +---------------+---------------+
                    |                               |
                    v                               v
            Concept Activations              Geocell Logits
            (Interpretability)               (Classification)
                                                    |
                                                    v
                                            Predicted Coords
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, NamedTuple
from dataclasses import dataclass


@dataclass
class ConceptGeoOutput:
    """Output container for ConceptGeo model."""
    # Concept outputs
    concept_activations: torch.Tensor  # (B, num_concepts)
    concept_logits: torch.Tensor  # (B, num_concepts)
    projected_features: torch.Tensor  # (B, embed_dim)
    
    # Geocell outputs
    geocell_logits: torch.Tensor  # (B, num_geocells)
    geocell_probs: torch.Tensor  # (B, num_geocells)
    topk_cells: torch.Tensor  # (B, k)
    topk_probs: torch.Tensor  # (B, k)
    
    # Coordinate prediction
    pred_coords: torch.Tensor  # (B, 2) as (lng, lat)
    pred_cells: torch.Tensor  # (B,)
    
    # Image features (for contrastive)
    image_features: torch.Tensor  # (B, embed_dim)


class ConceptGeo(nn.Module):
    """
    ConceptGeo: Interpretable Geo-localization Model.
    
    Uses a frozen backbone with trainable concept and geocell heads.
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        concept_bank: torch.Tensor,
        num_geocells: int,
        geocell_coords: torch.Tensor,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        top_k: int = 5,
        use_concept_for_geocell: bool = True
    ):
        """
        Initialize ConceptGeo.
        
        Args:
            backbone: Frozen backbone model with encode_image and encode_text methods
            concept_bank: Pre-computed concept embeddings (num_concepts, embed_dim)
            num_geocells: Number of geocells for classification
            geocell_coords: Geocell centroids (num_geocells, 2) as (lng, lat)
            hidden_dim: Hidden dimension for heads
            dropout: Dropout probability
            top_k: Number of top geocell candidates
            use_concept_for_geocell: Whether to use concept activations in geocell head
        """
        super().__init__()
        
        self.backbone = backbone
        self.top_k = top_k
        self.use_concept_for_geocell = use_concept_for_geocell
        
        # Get embedding dimension from backbone
        self.embed_dim = backbone.embed_dim
        
        # Store geocell coordinates (frozen)
        self.register_buffer('geocell_coords', geocell_coords)
        self.num_geocells = num_geocells
        
        # Concept Embedding Module
        from models.concept_embedding import ConceptEmbeddingModule
        self.concept_module = ConceptEmbeddingModule(
            input_dim=self.embed_dim,
            concept_bank=concept_bank,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        self.num_concepts = concept_bank.shape[0]
        
        # Geocell Classification Head
        from models.geocell_head import GeocelClassificationHead
        self.geocell_head = GeocelClassificationHead(
            input_dim=concept_bank.shape[1],  # concept_dim
            num_geocells=num_geocells,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_concept_features=use_concept_for_geocell,
            num_concepts=self.num_concepts if use_concept_for_geocell else None
        )
        
        print(f"ConceptGeo initialized:")
        print(f"  Backbone embed_dim: {self.embed_dim}")
        print(f"  Num concepts: {self.num_concepts}")
        print(f"  Num geocells: {num_geocells}")
        print(f"  Top-k: {top_k}")
    
    def trainable_parameters(self):
        """Return only trainable parameters (not frozen backbone)."""
        params = []
        params.extend(self.concept_module.parameters())
        params.extend(self.geocell_head.parameters())
        return params
    
    def freeze_backbone(self):
        """Ensure backbone is frozen."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def forward(
        self,
        images: torch.Tensor,
        return_features: bool = True
    ) -> ConceptGeoOutput:
        """
        Forward pass.
        
        Args:
            images: Input images (B, C, H, W)
            return_features: Whether to include image features in output
            
        Returns:
            ConceptGeoOutput with all predictions
        """
        # 1. Extract frozen image features
        image_features = self.backbone.encode_image(images)  # (B, embed_dim)
        
        # 2. Concept Embedding Module
        projected, concept_activations, concept_logits = self.concept_module(
            image_features, return_logits=True
        )
        
        # 3. Geocell Classification
        if self.use_concept_for_geocell:
            geocell_logits = self.geocell_head(projected, concept_activations)
        else:
            geocell_logits = self.geocell_head(projected)
        
        geocell_probs = F.softmax(geocell_logits, dim=-1)
        
        # 4. Get top-k predictions
        topk_probs, topk_cells = torch.topk(geocell_probs, k=self.top_k, dim=-1)
        
        # 5. Predict coordinates from top cell
        pred_cells = geocell_probs.argmax(dim=-1)
        pred_coords = self.geocell_coords[pred_cells]
        
        return ConceptGeoOutput(
            concept_activations=concept_activations,
            concept_logits=concept_logits,
            projected_features=projected,
            geocell_logits=geocell_logits,
            geocell_probs=geocell_probs,
            topk_cells=topk_cells,
            topk_probs=topk_probs,
            pred_coords=pred_coords,
            pred_cells=pred_cells,
            image_features=image_features if return_features else None
        )
    
    def encode_notes(self, notes: List[str]) -> torch.Tensor:
        """
        Encode note text using backbone.
        
        Args:
            notes: List of note strings
            
        Returns:
            Note features (B, embed_dim)
        """
        return self.backbone.encode_notes(notes)
    
    def get_explanation(
        self,
        outputs: ConceptGeoOutput,
        idx_to_concept: Dict[int, str],
        top_k_concepts: int = 5
    ) -> List[Dict]:
        """
        Generate human-readable explanations for predictions.
        
        Args:
            outputs: Model outputs
            idx_to_concept: Mapping from concept index to name
            top_k_concepts: Number of top concepts to include
            
        Returns:
            List of explanation dicts, one per sample
        """
        explanations = []
        batch_size = outputs.concept_activations.shape[0]
        
        for i in range(batch_size):
            activations = outputs.concept_activations[i]
            top_values, top_indices = torch.topk(activations, k=top_k_concepts)
            
            explanation = {
                'predicted_location': {
                    'lng': outputs.pred_coords[i, 0].item(),
                    'lat': outputs.pred_coords[i, 1].item()
                },
                'geocell': outputs.pred_cells[i].item(),
                'geocell_confidence': outputs.geocell_probs[i].max().item(),
                'top_concepts': [
                    {
                        'name': idx_to_concept[idx.item()],
                        'activation': val.item()
                    }
                    for idx, val in zip(top_indices, top_values)
                ]
            }
            explanations.append(explanation)
        
        return explanations


class ConceptGeoWithNotes(ConceptGeo):
    """
    ConceptGeo with integrated note encoding for contrastive training.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = nn.Parameter(torch.tensor(0.07))
    
    def forward_with_notes(
        self,
        images: torch.Tensor,
        notes: Optional[List[str]] = None
    ) -> Tuple[ConceptGeoOutput, Optional[torch.Tensor]]:
        """
        Forward pass with optional note encoding.
        
        Args:
            images: Input images (B, C, H, W)
            notes: Optional list of note strings
            
        Returns:
            outputs: ConceptGeoOutput
            note_features: Encoded note features if notes provided
        """
        outputs = self.forward(images, return_features=True)
        
        note_features = None
        if notes is not None:
            # Filter out empty notes
            valid_mask = [len(n.strip()) > 0 for n in notes]
            valid_notes = [n for n, v in zip(notes, valid_mask) if v]
            
            if len(valid_notes) > 0:
                note_features = self.encode_notes(valid_notes)
        
        return outputs, note_features


def create_concept_geo(
    backbone_name: str = "geoclip",
    concept_names: List[str] = None,
    num_geocells: int = 2203,
    geocell_coords: torch.Tensor = None,
    device: str = "cuda"
) -> ConceptGeo:
    """
    Factory function to create ConceptGeo model.
    
    Args:
        backbone_name: "geoclip" or "streetclip"
        concept_names: List of concept name strings
        num_geocells: Number of geocells
        geocell_coords: Geocell centroids (N, 2)
        device: Device to load on
        
    Returns:
        ConceptGeo model
    """
    from models.geoclip_backbone import FrozenGeoCLIP
    from models.concept_embedding import ConceptBank
    
    # Create backbone
    backbone = FrozenGeoCLIP(model_name=backbone_name, device=device)
    
    # Create concept bank if concept names provided
    if concept_names is not None:
        concept_bank_obj = ConceptBank(
            concept_names=concept_names,
            clip_model=backbone,
            prompt_template="a street view photo showing {}"
        )
        concept_bank = concept_bank_obj.get_embeddings()
    else:
        # Dummy concept bank for testing
        concept_bank = F.normalize(torch.randn(100, backbone.embed_dim), dim=-1)
    
    # Default geocell coords if not provided
    if geocell_coords is None:
        geocell_coords = torch.randn(num_geocells, 2) * 100
        print("Warning: Using dummy geocell coordinates")
    
    # Create model
    model = ConceptGeo(
        backbone=backbone,
        concept_bank=concept_bank.to(device),
        num_geocells=num_geocells,
        geocell_coords=geocell_coords.to(device)
    )
    
    return model.to(device)


if __name__ == "__main__":
    # Test the model
    print("Testing ConceptGeo model...")
    
    # Create dummy data
    batch_size = 4
    num_concepts = 100
    num_geocells = 500
    embed_dim = 512
    
    # Mock backbone
    class MockBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_dim = embed_dim
            
        def encode_image(self, images):
            B = images.shape[0]
            return F.normalize(torch.randn(B, self.embed_dim), dim=-1)
        
        def encode_notes(self, notes):
            B = len(notes)
            return F.normalize(torch.randn(B, self.embed_dim), dim=-1)
    
    backbone = MockBackbone()
    concept_bank = F.normalize(torch.randn(num_concepts, embed_dim), dim=-1)
    geocell_coords = torch.randn(num_geocells, 2) * 100
    
    # Create model
    model = ConceptGeo(
        backbone=backbone,
        concept_bank=concept_bank,
        num_geocells=num_geocells,
        geocell_coords=geocell_coords
    )
    
    # Test forward pass
    images = torch.randn(batch_size, 3, 224, 224)
    outputs = model(images)
    
    print(f"Concept activations shape: {outputs.concept_activations.shape}")
    print(f"Geocell logits shape: {outputs.geocell_logits.shape}")
    print(f"Pred coords shape: {outputs.pred_coords.shape}")
    print(f"Top-k cells shape: {outputs.topk_cells.shape}")
    
    # Test explanation generation
    idx_to_concept = {i: f"concept_{i}" for i in range(num_concepts)}
    explanations = model.get_explanation(outputs, idx_to_concept, top_k_concepts=3)
    
    print(f"\nSample explanation:")
    print(f"  Location: {explanations[0]['predicted_location']}")
    print(f"  Confidence: {explanations[0]['geocell_confidence']:.2%}")
    print(f"  Top concepts: {explanations[0]['top_concepts']}")
    
    # Test trainable parameters
    trainable_params = list(model.trainable_parameters())
    total_params = sum(p.numel() for p in trainable_params)
    print(f"\nTrainable parameters: {total_params:,}")
    
    print("\nTest passed!")

