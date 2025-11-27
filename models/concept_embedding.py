"""
Concept Embedding Module (CEM) for interpretable geo-localization.

This module projects image features to a concept-aligned space without
forcing through a scalar bottleneck. It computes concept activations
via cosine similarity with a concept bank.

Key difference from vanilla CBM: Concepts are embedded vectors, not bottleneck 
scalars. Information flows through the full embedding space, then we project 
onto concepts for interpretability without forcing all info through a narrow bottleneck.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import numpy as np


class ConceptEmbeddingModule(nn.Module):
    """
    Concept Embedding Module (CEM).
    
    Projects image features to concept-aligned space and computes
    interpretable concept activations via cosine similarity.
    
    Architecture:
        image_features (B, input_dim) 
            -> MLP projector 
            -> projected (B, concept_dim)
            -> cosine_sim(projected, concept_bank.T) 
            -> concept_activations (B, num_concepts)
    """
    
    def __init__(
        self,
        input_dim: int,
        concept_bank: torch.Tensor,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        temperature: float = 1.0,
        learnable_temperature: bool = True,
        temp_min: float = 0.01,
        temp_max: float = 5.0
    ):
        """
        Initialize CEM.
        
        Args:
            input_dim: Dimension of input image features (e.g., 512 for GeoCLIP)
            concept_bank: Pre-computed concept embeddings of shape (num_concepts, concept_dim)
            hidden_dim: Hidden dimension in MLP projector
            dropout: Dropout probability
            temperature: Initial temperature for cosine similarity scaling (default: 1.0)
            learnable_temperature: Whether temperature is learnable
            temp_min: Minimum temperature value (prevents exploding logits)
            temp_max: Maximum temperature value (prevents vanishing gradients)
        """
        super().__init__()
        
        # Store concept bank (frozen)
        self.register_buffer('concept_bank', concept_bank)
        self.num_concepts = concept_bank.shape[0]
        self.concept_dim = concept_bank.shape[1]
        
        # MLP projector: input_dim -> hidden_dim -> concept_dim
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.concept_dim),
        )
        
        # Temperature parameter with clamping bounds
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.learnable_temperature = learnable_temperature
        
        if learnable_temperature:
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.register_buffer('temperature', torch.tensor(temperature))
        
        print(f"ConceptEmbeddingModule initialized:")
        print(f"  Input dim: {input_dim}")
        print(f"  Concept dim: {self.concept_dim}")
        print(f"  Num concepts: {self.num_concepts}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Temperature: {temperature} (learnable={learnable_temperature}, range=[{temp_min}, {temp_max}])")
    
    def forward(
        self, 
        image_features: torch.Tensor,
        return_logits: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            image_features: Image features of shape (B, input_dim)
            return_logits: Whether to also return raw logits before sigmoid
            
        Returns:
            projected: Projected features in concept space (B, concept_dim)
            concept_activations: Concept activation scores (B, num_concepts)
            concept_logits: (optional) Raw logits before sigmoid
        """
        # Project to concept-aligned space
        projected = self.projector(image_features)  # (B, concept_dim)
        projected = F.normalize(projected, dim=-1)
        
        # Clamp temperature to prevent numerical instability
        # Small temp -> huge logits -> unstable gradients
        # Large temp -> flat logits -> slow learning
        temp = self.temperature.clamp(min=self.temp_min, max=self.temp_max)
        
        # Compute concept activations via cosine similarity
        # concept_bank: (num_concepts, concept_dim)
        # projected: (B, concept_dim)
        concept_logits = projected @ self.concept_bank.T / temp  # (B, num_concepts)
        
        # Sigmoid gives interpretable [0, 1] activations
        concept_activations = torch.sigmoid(concept_logits)
        
        if return_logits:
            return projected, concept_activations, concept_logits
        
        return projected, concept_activations
    
    def get_top_concepts(
        self, 
        concept_activations: torch.Tensor,
        k: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get top-k activated concepts.
        
        Args:
            concept_activations: (B, num_concepts)
            k: Number of top concepts to return
            
        Returns:
            top_values: Top activation values (B, k)
            top_indices: Indices of top concepts (B, k)
        """
        return torch.topk(concept_activations, k=k, dim=-1)


class ConceptBank:
    """
    Helper class to create and manage concept banks.
    
    Creates CLIP text embeddings for concept names.
    """
    
    def __init__(
        self,
        concept_names: List[str],
        clip_model: nn.Module,
        prompt_template: str = "a street view photo showing {}"
    ):
        """
        Initialize concept bank.
        
        Args:
            concept_names: List of concept name strings
            clip_model: CLIP model with encode_text method
            prompt_template: Template for creating text prompts
        """
        self.concept_names = concept_names
        self.prompt_template = prompt_template
        self.num_concepts = len(concept_names)
        
        # Create concept embeddings
        self.concept_embeddings = self._create_embeddings(clip_model)
    
    def _create_embeddings(self, clip_model: nn.Module) -> torch.Tensor:
        """Create CLIP text embeddings for all concepts."""
        prompts = [self.prompt_template.format(name) for name in self.concept_names]
        
        with torch.no_grad():
            embeddings = clip_model.encode_text(prompts)
            embeddings = F.normalize(embeddings, dim=-1)
        
        return embeddings
    
    def get_embeddings(self) -> torch.Tensor:
        """Get concept embeddings tensor."""
        return self.concept_embeddings
    
    def get_concept_name(self, idx: int) -> str:
        """Get concept name by index."""
        return self.concept_names[idx]
    
    def get_concept_names(self, indices: List[int]) -> List[str]:
        """Get multiple concept names by indices."""
        return [self.concept_names[i] for i in indices]
    
    def save(self, path: str):
        """Save concept bank to file."""
        torch.save({
            'concept_names': self.concept_names,
            'concept_embeddings': self.concept_embeddings,
            'prompt_template': self.prompt_template
        }, path)
        print(f"Saved concept bank to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'ConceptBank':
        """Load concept bank from file."""
        data = torch.load(path)
        bank = cls.__new__(cls)
        bank.concept_names = data['concept_names']
        bank.concept_embeddings = data['concept_embeddings']
        bank.prompt_template = data['prompt_template']
        bank.num_concepts = len(bank.concept_names)
        print(f"Loaded concept bank from {path} ({bank.num_concepts} concepts)")
        return bank


def create_concept_bank_from_dataset(
    samples: List[dict],
    clip_model: nn.Module,
    meta_name_key: str = 'meta_name',
    prompt_template: str = "a street view photo showing {}"
) -> ConceptBank:
    """
    Create concept bank from dataset samples.
    
    Args:
        samples: List of sample dictionaries with meta_name field
        clip_model: CLIP model for text encoding
        meta_name_key: Key for concept name in sample dict
        prompt_template: Template for creating text prompts
        
    Returns:
        ConceptBank instance
    """
    # Extract unique concept names
    concept_names = sorted(set(s[meta_name_key] for s in samples))
    print(f"Found {len(concept_names)} unique concepts")
    
    return ConceptBank(
        concept_names=concept_names,
        clip_model=clip_model,
        prompt_template=prompt_template
    )


class ConceptClassificationHead(nn.Module):
    """
    Simple classification head for concept prediction.
    
    Used when concept activations should directly predict the concept label
    (single-label classification).
    """
    
    def __init__(self, input_dim: int, num_concepts: int, hidden_dim: int = 256):
        """
        Args:
            input_dim: Input feature dimension
            num_concepts: Number of concepts to predict
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_concepts)
        )
        
        self.num_concepts = num_concepts
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            features: Input features (B, input_dim)
            
        Returns:
            logits: Raw logits (B, num_concepts)
            probs: Softmax probabilities (B, num_concepts)
        """
        logits = self.classifier(features)
        probs = F.softmax(logits, dim=-1)
        return logits, probs


if __name__ == "__main__":
    # Test the module
    print("Testing ConceptEmbeddingModule...")
    
    # Create dummy concept bank
    num_concepts = 100
    concept_dim = 512
    concept_bank = F.normalize(torch.randn(num_concepts, concept_dim), dim=-1)
    
    # Create module
    module = ConceptEmbeddingModule(
        input_dim=512,
        concept_bank=concept_bank,
        hidden_dim=256
    )
    
    # Test forward pass
    batch_size = 8
    image_features = torch.randn(batch_size, 512)
    
    projected, activations = module(image_features)
    
    print(f"Projected shape: {projected.shape}")  # (8, 512)
    print(f"Activations shape: {activations.shape}")  # (8, 100)
    print(f"Activations range: [{activations.min():.3f}, {activations.max():.3f}]")
    
    # Test top concepts
    top_values, top_indices = module.get_top_concepts(activations, k=5)
    print(f"Top 5 concepts for first sample: {top_indices[0].tolist()}")
    print(f"Top 5 values for first sample: {top_values[0].tolist()}")
    
    print("\nTest passed!")


