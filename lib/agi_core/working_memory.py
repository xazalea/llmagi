"""
Working Memory System
Short-term memory for active reasoning and planning
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import deque


class WorkingMemory(nn.Module):
    """
    Working Memory System
    
    Features:
    - Active memory for current task
    - Capacity-limited (7±2 items)
    - Fast access and update
    - Integration with long-term memory
    """
    
    def __init__(
        self,
        hidden_dim: int = 2048,
        capacity: int = 7,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.capacity = capacity
        
        # Memory slots
        self.memory_slots = nn.Parameter(
            torch.randn(capacity, hidden_dim)
        )
        
        # Attention mechanism for memory access
        self.memory_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8, batch_first=True
        )
        
        # Memory update network
        self.update_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Current memory state
        self.current_memory = deque(maxlen=capacity)
        
    def store(
        self,
        information: torch.Tensor,
        importance: float = 1.0,
    ):
        """
        Store information in working memory.
        
        Args:
            information: Information to store [batch, hidden_dim]
            importance: Importance weight
        """
        # Add to current memory
        self.current_memory.append({
            'information': information,
            'importance': importance,
        })
        
        # Update memory slots
        if len(self.current_memory) <= self.capacity:
            idx = len(self.current_memory) - 1
            self.memory_slots.data[idx] = information.squeeze(0)
    
    def retrieve(
        self,
        query: torch.Tensor,
        top_k: int = 3,
    ) -> torch.Tensor:
        """
        Retrieve relevant information from working memory.
        
        Args:
            query: Query vector [batch, hidden_dim]
            top_k: Number of items to retrieve
            
        Returns:
            Retrieved information [batch, top_k, hidden_dim]
        """
        if len(self.current_memory) == 0:
            return torch.zeros(query.shape[0], top_k, self.hidden_dim)
        
        # Get memory items
        memory_items = torch.stack([
            item['information'] for item in self.current_memory
        ])  # [num_items, hidden_dim]
        
        # Compute similarity
        similarities = torch.cosine_similarity(
            query.unsqueeze(1), memory_items.unsqueeze(0), dim=-1
        )  # [batch, num_items]
        
        # Get top-k
        top_k_similarities, top_k_indices = torch.topk(similarities, min(top_k, len(self.current_memory)), dim=-1)
        
        # Retrieve
        retrieved = []
        for b in range(query.shape[0]):
            batch_retrieved = memory_items[top_k_indices[b]]
            retrieved.append(batch_retrieved)
        
        retrieved = torch.stack(retrieved)  # [batch, top_k, hidden_dim]
        
        return retrieved
    
    def update(
        self,
        new_information: torch.Tensor,
        query: torch.Tensor,
    ):
        """
        Update working memory with new information.
        
        Args:
            new_information: New information [batch, hidden_dim]
            query: Query to determine what to update
        """
        # Retrieve relevant memory
        relevant_memory = self.retrieve(query, top_k=1)
        
        # Update
        if relevant_memory.shape[1] > 0:
            combined = torch.cat([
                relevant_memory.squeeze(1),
                new_information
            ], dim=-1)
            
            updated = self.update_network(combined)
            
            # Store updated information
            self.store(updated, importance=1.0)
    
    def clear(self):
        """Clear working memory"""
        self.current_memory.clear()
        self.memory_slots.data.zero_()


class EpisodicBuffer(nn.Module):
    """
    Episodic Buffer
    
    Features:
    - Temporary storage for episodes
    - Compression before moving to long-term memory
    - Fast access during episode
    """
    
    def __init__(
        self,
        hidden_dim: int = 2048,
        buffer_size: int = 100,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.buffer_size = buffer_size
        
        # Buffer
        self.buffer = deque(maxlen=buffer_size)
        
        # Compression network
        self.compressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
        )
        
    def add(
        self,
        episode: Dict,
        timestamp: float,
    ):
        """Add episode to buffer"""
        self.buffer.append({
            'episode': episode,
            'timestamp': timestamp,
            'compressed': self.compressor(episode.get('embedding', torch.randn(1, self.hidden_dim))),
        })
    
    def get_recent(
        self,
        num_episodes: int = 10,
    ) -> List[Dict]:
        """Get recent episodes"""
        return list(self.buffer)[-num_episodes:]
    
    def compress_and_clear(self) -> List[Dict]:
        """Compress and clear buffer"""
        compressed = [
            {
                'compressed_embedding': item['compressed'],
                'timestamp': item['timestamp'],
            }
            for item in self.buffer
        ]
        self.buffer.clear()
        return compressed

