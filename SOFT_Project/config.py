# config.py
from dataclasses import dataclass

@dataclass
class SIFTConfig:
    """
    Configuration class for SIFT framework parameters.
    Values are based on Section 6.3 of the paper.
    """
    # Model Dimensions
    input_dim: int = 1024       # Dimension of raw features (simulated)
    embed_dim: int = 768        # Dimension d (Backbone output)
    attention_dim: int = 128    # Internal dimension for attention mechanism
    
    # SIFT Specific Hyperparameters
    separation_margin: float = 0.5  # ξ (xi) in Eq. 9
    gamma_scale: float = 1.0        # γ (gamma) in Eq. 12 (Gating scaling)
    lambda_loss: float = 0.1        # λ (lambda) in Eq. 16 (Loss weight)
    
    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 10
    seq_len: int = 20           # N: Number of posts per user timeline
    num_classes: int = 1        # Binary classification
    dropout: float = 0.1
