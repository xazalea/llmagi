"""
Mixture of Experts (MoE) System
Efficient scaling with sparse expert activation
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import numpy as np


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts System
    
    Features:
    - Sparse expert activation (only activate needed experts)
    - Efficient scaling (more experts, same parameters)
    - Specialized experts for different tasks
    - Dynamic routing
    """
    
    def __init__(
        self,
        hidden_dim: int = 2048,
        num_experts: int = 64,
        num_experts_per_token: int = 4,
        expert_capacity_factor: float = 1.25,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.expert_capacity_factor = expert_capacity_factor
        
        # Expert networks
        self.experts = nn.ModuleList([
            ExpertNetwork(hidden_dim) for _ in range(num_experts)
        ])
        
        # Router
        self.router = Router(hidden_dim, num_experts)
        
        # Gate network
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with sparse expert activation.
        
        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            
        Returns:
            Output tensor [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = x.shape
        
        # Flatten for processing
        x_flat = x.view(-1, hidden_dim)  # [batch*seq_len, hidden_dim]
        
        # Route to experts
        router_logits = self.router(x_flat)  # [batch*seq_len, num_experts]
        expert_weights, expert_indices = self._top_k_routing(
            router_logits, self.num_experts_per_token
        )
        
        # Process with selected experts
        output = torch.zeros_like(x_flat)
        
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (expert_indices == expert_idx)
            
            if expert_mask.any():
                expert_input = x_flat[expert_mask]
                expert_output = self.experts[expert_idx](expert_input)
                expert_weights_selected = expert_weights[expert_mask]
                
                # Weighted combination
                output[expert_mask] += expert_output * expert_weights_selected.unsqueeze(-1)
        
        # Reshape back
        output = output.view(batch_size, seq_len, hidden_dim)
        
        return output
    
    def _top_k_routing(
        self,
        router_logits: torch.Tensor,
        k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Top-k routing to experts"""
        # Get top-k experts
        top_k_weights, top_k_indices = torch.topk(
            router_logits, k, dim=-1
        )
        
        # Softmax over top-k
        top_k_weights = torch.softmax(top_k_weights, dim=-1)
        
        return top_k_weights, top_k_indices


class ExpertNetwork(nn.Module):
    """Individual expert network"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.expert = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Expert forward pass"""
        return self.expert(x)


class Router(nn.Module):
    """Router for expert selection"""
    
    def __init__(self, hidden_dim: int, num_experts: int):
        super().__init__()
        self.router = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route to experts"""
        return self.router(x)


class SparseAttention(nn.Module):
    """Sparse attention for efficiency"""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 16,
        sparsity: float = 0.9,  # 90% sparsity
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.sparsity = sparsity
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sparse attention forward pass"""
        batch_size, seq_len, hidden_dim = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        # Apply sparsity (top-k attention)
        k = int(seq_len * (1 - self.sparsity))
        top_k_scores, top_k_indices = torch.topk(scores, k, dim=-1)
        
        # Create sparse attention mask
        sparse_scores = torch.zeros_like(scores)
        sparse_scores.scatter_(-1, top_k_indices, top_k_scores)
        
        # Apply mask if provided
        if mask is not None:
            sparse_scores = sparse_scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        attn_weights = torch.softmax(sparse_scores, dim=-1)
        
        # Apply attention
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape and project
        attn_output = attn_output.view(batch_size, seq_len, hidden_dim)
        output = self.o_proj(attn_output)
        
        return output

