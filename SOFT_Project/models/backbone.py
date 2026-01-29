# models/backbone.py
import torch
import torch.nn as nn

class MultimodalBackbone(nn.Module):
    """
    Wrapper for the LLM/VLM Backbone (e.g., Qwen2-VL).
    Section 5.1 describes using LoRA here. 
    
    For this implementation, we simulate the extraction of h_{i,j}.
    """
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        # In a real scenario, this would be the LoRA layers.
        # Here we use a projection to simulate semantic feature extraction.
        self.projection = nn.Linear(input_dim, embed_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        """
        x: Concatenated multimodal features (Batch, Seq_Len, Input_Dim)
        returns: h_{i,j} (Batch, Seq_Len, Embed_Dim)
        """
        return self.activation(self.projection(x))
