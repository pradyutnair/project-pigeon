"""
Frozen GeoCLIP/StreetCLIP backbone for feature extraction.
All parameters are frozen - only used as a feature extractor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union
from transformers import CLIPModel, CLIPTokenizer, CLIPProcessor


class FrozenGeoCLIP(nn.Module):
    """
    GeoCLIP as a frozen feature extractor.
    
    Wraps either GeoCLIP or StreetCLIP and provides:
    - Image encoding (frozen)
    - Text encoding for notes (frozen)
    
    All parameters are frozen - this is used purely for feature extraction.
    """
    
    def __init__(self, model_name: str = "geoclip", device: str = "cuda"):
        """
        Initialize frozen backbone.
        
        Args:
            model_name: Either "geoclip" or "streetclip"
            device: Device to load model on
        """
        super().__init__()
        self.model_name = model_name
        self._device = device
        
        if model_name == "geoclip":
            self._init_geoclip()
        elif model_name == "streetclip":
            self._init_streetclip()
        else:
            raise ValueError(f"Unknown model: {model_name}. Use 'geoclip' or 'streetclip'")
        
        # Freeze all parameters
        self._freeze_all()
        
    def _init_geoclip(self):
        """Initialize GeoCLIP model."""
        try:
            from geoclip import GeoCLIP
            geoclip = GeoCLIP()
            
            # Extract components - GeoCLIP wraps CLIP in image_encoder.CLIP
            clip_model = geoclip.image_encoder.CLIP
            self.vision_model = clip_model.vision_model
            self.visual_projection = clip_model.visual_projection
            self.text_model = clip_model.text_model
            self.text_projection = clip_model.text_projection
            
            # GeoCLIP uses CLIP ViT-L/14 (vision_hidden=1024, projection to 768)
            self.embed_dim = 768
            self.vision_hidden_dim = 1024
            
            # Use CLIP tokenizer
            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            
            print(f"Loaded GeoCLIP backbone (embed_dim={self.embed_dim})")
            
        except ImportError:
            raise ImportError("geoclip not installed. Run: pip install geoclip")
    
    def _init_streetclip(self):
        """Initialize StreetCLIP model from HuggingFace."""
        try:
            clip_model = CLIPModel.from_pretrained("geolocal/StreetCLIP")
            
            # Extract components
            self.vision_model = clip_model.vision_model
            self.visual_projection = clip_model.visual_projection
            self.text_model = clip_model.text_model
            self.text_projection = clip_model.text_projection
            
            # StreetCLIP uses CLIP ViT-B/16 (embed_dim=512)
            self.embed_dim = 512
            self.vision_hidden_dim = 768
            
            # Use CLIP tokenizer
            self.tokenizer = CLIPTokenizer.from_pretrained("geolocal/StreetCLIP")
            self.processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")
            
            print(f"Loaded StreetCLIP backbone (embed_dim={self.embed_dim})")
            
        except Exception as e:
            raise ImportError(f"Failed to load StreetCLIP: {e}")
    
    def _freeze_all(self):
        """Freeze all parameters."""
        for param in self.parameters():
            param.requires_grad = False
        print("All backbone parameters frozen")
    
    @property
    def device(self):
        """Get device of model parameters."""
        return next(self.parameters()).device
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract image features (frozen).
        
        Args:
            images: Tensor of shape (B, C, H, W), preprocessed images
            
        Returns:
            Image features of shape (B, embed_dim)
        """
        with torch.no_grad():
            # Get vision model outputs
            vision_outputs = self.vision_model(pixel_values=images)
            
            # Pool and project
            pooled_output = vision_outputs.pooler_output  # (B, vision_hidden_dim)
            image_features = self.visual_projection(pooled_output)  # (B, embed_dim)
            
            # Normalize
            image_features = F.normalize(image_features, dim=-1)
            
        return image_features
    
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        Encode text strings (frozen).
        
        Args:
            texts: List of text strings
            
        Returns:
            Text features of shape (B, embed_dim)
        """
        with torch.no_grad():
            # Tokenize
            tokens = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt"
            ).to(self.device)
            
            # Get text model outputs
            text_outputs = self.text_model(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask
            )
            
            # Pool and project
            pooled_output = text_outputs.pooler_output  # (B, text_hidden_dim)
            text_features = self.text_projection(pooled_output)  # (B, embed_dim)
            
            # Normalize
            text_features = F.normalize(text_features, dim=-1)
            
        return text_features
    
    def encode_notes(self, notes: List[str], strip_html: bool = True) -> torch.Tensor:
        """
        Encode note descriptions (frozen).
        
        Args:
            notes: List of note strings (may contain HTML)
            strip_html: Whether to strip HTML tags from notes
            
        Returns:
            Note features of shape (B, embed_dim)
        """
        if strip_html:
            import re
            notes = [re.sub(r'<[^>]+>', '', note) for note in notes]
        
        return self.encode_text(notes)
    
    def get_image_preprocessing(self):
        """Get the image preprocessing transform for this model."""
        return self.processor.image_processor
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - just encode images.
        
        Args:
            images: Tensor of shape (B, C, H, W)
            
        Returns:
            Image features of shape (B, embed_dim)
        """
        return self.encode_image(images)


class FrozenBackboneWithProjection(nn.Module):
    """
    Frozen backbone with an optional trainable projection layer.
    
    This allows adding a small trainable projection on top of frozen features
    if needed for domain adaptation.
    """
    
    def __init__(
        self, 
        model_name: str = "geoclip",
        projection_dim: Optional[int] = None,
        device: str = "cuda"
    ):
        """
        Args:
            model_name: Either "geoclip" or "streetclip"
            projection_dim: If provided, adds a trainable projection layer
            device: Device to load model on
        """
        super().__init__()
        
        self.backbone = FrozenGeoCLIP(model_name=model_name, device=device)
        self.embed_dim = self.backbone.embed_dim
        
        # Optional trainable projection
        if projection_dim is not None:
            self.projection = nn.Linear(self.embed_dim, projection_dim)
            self.embed_dim = projection_dim
            print(f"Added trainable projection: {self.backbone.embed_dim} -> {projection_dim}")
        else:
            self.projection = None
    
    def trainable_parameters(self):
        """Return only trainable parameters (projection layer if exists)."""
        if self.projection is not None:
            return self.projection.parameters()
        return iter([])  # Empty iterator
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images with optional projection."""
        features = self.backbone.encode_image(images)
        
        if self.projection is not None:
            features = self.projection(features)
            features = F.normalize(features, dim=-1)
        
        return features
    
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode text (always frozen, no projection)."""
        return self.backbone.encode_text(texts)
    
    def encode_notes(self, notes: List[str], strip_html: bool = True) -> torch.Tensor:
        """Encode notes (always frozen, no projection)."""
        return self.backbone.encode_notes(notes, strip_html=strip_html)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode_image(images)


def create_backbone(
    model_name: str = "geoclip",
    projection_dim: Optional[int] = None,
    device: str = "cuda"
) -> Union[FrozenGeoCLIP, FrozenBackboneWithProjection]:
    """
    Factory function to create backbone.
    
    Args:
        model_name: "geoclip" or "streetclip"
        projection_dim: If provided, adds trainable projection layer
        device: Device to load on
        
    Returns:
        Backbone model
    """
    if projection_dim is not None:
        return FrozenBackboneWithProjection(
            model_name=model_name,
            projection_dim=projection_dim,
            device=device
        )
    else:
        return FrozenGeoCLIP(model_name=model_name, device=device)


if __name__ == "__main__":
    # Test the backbone
    print("Testing FrozenGeoCLIP backbone...")
    
    # Test with GeoCLIP
    try:
        backbone = FrozenGeoCLIP(model_name="geoclip", device="cpu")
        print(f"GeoCLIP embed_dim: {backbone.embed_dim}")
        
        # Test image encoding
        dummy_images = torch.randn(2, 3, 224, 224)
        features = backbone.encode_image(dummy_images)
        print(f"Image features shape: {features.shape}")
        
        # Test text encoding
        texts = ["a street view photo", "a mountain landscape"]
        text_features = backbone.encode_text(texts)
        print(f"Text features shape: {text_features.shape}")
        
        # Test note encoding
        notes = ["<p>This is a <strong>test</strong> note.</p>", "Another note"]
        note_features = backbone.encode_notes(notes)
        print(f"Note features shape: {note_features.shape}")
        
        print("GeoCLIP test passed!")
        
    except ImportError as e:
        print(f"GeoCLIP not available: {e}")
    
    # Test with StreetCLIP
    try:
        backbone = FrozenGeoCLIP(model_name="streetclip", device="cpu")
        print(f"\nStreetCLIP embed_dim: {backbone.embed_dim}")
        
        dummy_images = torch.randn(2, 3, 224, 224)
        features = backbone.encode_image(dummy_images)
        print(f"Image features shape: {features.shape}")
        
        print("StreetCLIP test passed!")
        
    except Exception as e:
        print(f"StreetCLIP not available: {e}")

