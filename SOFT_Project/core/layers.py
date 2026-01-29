# core/layers.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentIntentDisentanglement(nn.Module):
    """
    Implements Section 5.2: Latent Intent Disentanglement.
    Maps semantic embeddings to Information Closure (alpha) and Social Interaction (beta).
    """
    def __init__(self, input_dim):
        super().__init__()
        # Equation 6: Projection for Closure Score (alpha)
        self.head_alpha = nn.Linear(input_dim, 1)
        # Equation 7: Projection for Interaction Score (beta)
        self.head_beta = nn.Linear(input_dim, 1)

    def forward(self, h):
        """
        Args:
            h: Semantic embeddings (Batch, Seq_Len, Dim)
        Returns:
            alpha: Information closure scores (Batch, Seq_Len)
            beta: Social interaction scores (Batch, Seq_Len)
        """
        # Sigmoid ensures scores are in [0, 1]
        alpha = torch.sigmoid(self.head_alpha(h)).squeeze(-1)
        beta = torch.sigmoid(self.head_beta(h)).squeeze(-1)
        return alpha, beta


class IntentGatedAttention(nn.Module):
    """
    Implements Section 5.4: Intent-Gated Attention Aggregation.
    Weights posts based on the magnitude of their intent divergence.
    """
    def __init__(self, embed_dim, hidden_dim, gamma_scale=1.0):
        super().__init__()
        self.gamma = gamma_scale
        
        # Parameters for Eq. 11
        self.w_att = nn.Linear(embed_dim, hidden_dim)
        self.query_vector = nn.Parameter(torch.randn(hidden_dim, 1))

    def forward(self, h, alpha, beta):
        """
        Args:
            h: Post embeddings (Batch, Seq_Len, Dim)
            alpha: Closure scores (Batch, Seq_Len)
            beta: Interaction scores (Batch, Seq_Len)
        Returns:
            r_i: Aggregated account representation (Batch, Dim)
        """
        # 1. Calculate Intent Divergence Delta (Eq. 8)
        # Delta \in [-1, 1]
        delta = alpha - beta
        
        # 2. Compute Raw Attention Energy (Eq. 11)
        # u = tanh(W * h)
        u = torch.tanh(self.w_att(h))  # (Batch, Seq, Hidden)
        # e_raw = u * q
        e_raw = torch.matmul(u, self.query_vector).squeeze(-1)  # (Batch, Seq)
        
        # 3. Intent Gating (Eq. 12)
        # Amplifies attention for posts with high intent clarity (|delta| is large)
        gating_factor = 1.0 + self.gamma * torch.abs(delta)
        e_gated = e_raw * gating_factor
        
        # 4. Normalization (Eq. 13)
        att_weights = F.softmax(e_gated, dim=1).unsqueeze(-1)  # (Batch, Seq, 1)
        
        # 5. Weighted Aggregation (Eq. 14)
        r_i = torch.sum(h * att_weights, dim=1)  # (Batch, Dim)
        
        return r_i
