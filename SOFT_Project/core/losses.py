# core/losses.py
import torch
import torch.nn as nn

class SIFTLoss(nn.Module):
    """
    Implements the Unified Optimization Objective (Section 5.5).
    Combines Binary Cross Entropy with Manifold Separation Loss.
    """
    def __init__(self, margin=0.5, lambda_val=0.1):
        super().__init__()
        self.xi = margin        # ξ
        self.lambda_val = lambda_val
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, alpha, beta, targets):
        """
        Args:
            logits: Prediction logits (Batch, 1)
            alpha: Closure scores (Batch, Seq_Len)
            beta: Interaction scores (Batch, Seq_Len)
            targets: Ground truth (Batch, 1). 0=Human, 1=LLM/Bot.
        """
        # --- 1. Classification Loss (Eq. 16 part 1) ---
        loss_cls = self.bce_loss(logits, targets)

        # --- 2. Manifold Separation Loss (Eq. 9 & 10) ---
        # Delta = alpha - beta
        delta = alpha - beta  # (Batch, Seq_Len)
        
        # Expand targets to match sequence length: (Batch, 1) -> (Batch, Seq_Len)
        y_expanded = targets.expand_as(delta)

        # Hinge Loss Calculation:
        # If Bot (y=1): Penalize if delta < xi. (Force delta -> 1)
        loss_llm = torch.relu(self.xi - delta)
        
        # If Human (y=0): Penalize if delta > -xi. (Force delta -> -1)
        loss_human = torch.relu(self.xi + delta)
        
        # Combine using ground truth as mask (Eq. 10)
        # L_g = y * loss_llm + (1-y) * loss_human
        loss_matrix = y_expanded * loss_llm + (1.0 - y_expanded) * loss_human
        
        # Average over batch and sequence
        loss_manifold = loss_matrix.mean()

        # --- 3. Total Loss (Eq. 16) ---
        total_loss = loss_cls + self.lambda_val * loss_manifold

        return total_loss, loss_cls, loss_manifold
