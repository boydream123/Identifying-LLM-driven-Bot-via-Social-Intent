# models/sift_model.py
import torch
import torch.nn as nn
from core.layers import LatentIntentDisentanglement, IntentGatedAttention
from models.backbone import MultimodalBackbone

class SIFT(nn.Module):
    """
    SIFT: Identifying LLM-driven Bot via Social Intent.
    """
    def __init__(self, config):
        super().__init__()
        
        # 1. Feature Adaptation (Phi)
        self.backbone = MultimodalBackbone(config.input_dim, config.embed_dim)
        
        # 2. Latent Intent Disentanglement (Psi)
        self.disentanglement = LatentIntentDisentanglement(config.embed_dim)
        
        # 3. Intent-Gated Attention Aggregation
        self.aggregator = IntentGatedAttention(
            config.embed_dim, 
            config.attention_dim, 
            gamma_scale=config.gamma_scale
        )
        
        # 4. Final Classifier (C_psi)
        self.classifier = nn.Linear(config.embed_dim, config.num_classes)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs):
        # Step 1: Extract Semantic Embeddings
        # h: (Batch, Seq_Len, Embed_Dim)
        h = self.backbone(inputs)
        
        # Step 2: Disentangle Intent
        # alpha, beta: (Batch, Seq_Len)
        alpha, beta = self.disentanglement(h)
        
        # Step 3: Aggregate based on Social Intent
        # r_i: (Batch, Embed_Dim)
        r_i = self.aggregator(h, alpha, beta)
        
        # Step 4: Classification
        r_i = self.dropout(r_i)
        logits = self.classifier(r_i)
        
        return logits, alpha, beta
