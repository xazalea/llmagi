"""
Phase II Training: Reasoning & Planning Curriculum
Train reasoning, planning, and code execution capabilities
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lib.planner.core_planner import CorePlanner


class ReasoningDataset(Dataset):
    """Dataset for reasoning and planning training"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.samples = self._load_samples()
    
    def _load_samples(self) -> List[Dict]:
        """Load reasoning samples"""
        # Would load actual reasoning tasks, code execution traces, etc.
        return [
            {
                'type': 'reasoning',
                'problem': 'If A implies B and B implies C, what can we conclude?',
                'solution': 'A implies C',
            },
            {
                'type': 'planning',
                'goal': 'Make coffee',
                'steps': ['Get coffee beans', 'Grind beans', 'Brew coffee'],
            },
            {
                'type': 'code',
                'code': 'def add(a, b): return a + b',
                'execution': {'result': 5, 'inputs': [2, 3]},
            },
        ] * 100
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


class Phase2Trainer:
    """Phase II Training: Reasoning & Planning"""
    
    def __init__(
        self,
        hidden_dim: int = 2048,
        learning_rate: float = 1e-4,
        batch_size: int = 16,
        num_epochs: int = 50,
    ):
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        
        # Initialize planner
        self.planner = CorePlanner(hidden_dim=hidden_dim)
        
        # Loss functions
        self.planning_loss = nn.CrossEntropyLoss()
        self.reasoning_loss = nn.MSELoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.planner.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
        )
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.planner.to(self.device)
        
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Train Phase II"""
        print("Starting Phase II Training: Reasoning & Planning")
        
        for epoch in range(self.num_epochs):
            self.planner.train()
            total_loss = 0.0
            
            for batch_idx, batch in enumerate(train_loader):
                loss = self._train_step(batch)
                total_loss += loss.item()
                
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")
            
            # Validation
            if (epoch + 1) % 10 == 0:
                val_loss = self._validate(val_loader)
                print(f"Epoch {epoch} Validation Loss: {val_loss:.4f}")
    
    def _train_step(self, batch: List[Dict]) -> torch.Tensor:
        """Single training step"""
        self.optimizer.zero_grad()
        
        losses = []
        
        for sample in batch:
            # Create input embeddings (simplified)
            input_embeddings = torch.randn(1, 100, self.hidden_dim).to(self.device)
            input_tokens = torch.randint(0, 1000, (1, 100)).to(self.device)
            
            # Forward pass
            output = self.planner(input_tokens, input_embeddings)
            
            # Compute loss based on task type
            if sample['type'] == 'reasoning':
                # Reasoning loss (simplified)
                loss = self.reasoning_loss(
                    output['reasoning_trace'][0] if output['reasoning_trace'] else torch.zeros(1, self.hidden_dim).to(self.device),
                    torch.randn(1, self.hidden_dim).to(self.device)
                )
                losses.append(loss)
            
            elif sample['type'] == 'planning':
                # Planning loss (simplified)
                plan = output.get('plan')
                if plan and plan.tasks:
                    # Check if plan matches expected steps
                    expected_steps = len(sample.get('steps', []))
                    actual_steps = len(plan.tasks)
                    loss = torch.abs(torch.tensor(expected_steps - actual_steps).float()).to(self.device)
                    losses.append(loss)
        
        # Total loss
        total_loss = sum(losses) / len(losses) if losses else torch.tensor(0.0).to(self.device)
        
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss
    
    def _validate(self, val_loader: DataLoader) -> float:
        """Validation"""
        self.planner.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                loss = self._train_step(batch)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def save_checkpoint(self, path: str):
        """Save checkpoint"""
        torch.save({
            'planner_state_dict': self.planner.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load checkpoint"""
        checkpoint = torch.load(path)
        self.planner.load_state_dict(checkpoint['planner_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def main():
    parser = argparse.ArgumentParser(description='Phase II Training')
    parser.add_argument('--data_path', type=str, required=True, help='Path to training data')
    parser.add_argument('--hidden_dim', type=int, default=2048, help='Hidden dimension')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/phase2', help='Checkpoint path')
    
    args = parser.parse_args()
    
    # Create datasets
    train_dataset = ReasoningDataset(args.data_path)
    val_dataset = ReasoningDataset(args.data_path)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create trainer
    trainer = Phase2Trainer(
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
    )
    
    # Train
    trainer.train(train_loader, val_loader)
    
    # Save checkpoint
    Path(args.checkpoint_path).mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(f"{args.checkpoint_path}/phase2_final.pt")


if __name__ == '__main__':
    main()

