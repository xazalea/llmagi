"""
Phase I Training: Representation & Multimodal Alignment
Train unified tokenization and cross-modal alignment
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Tuple
import argparse
from pathlib import Path
import sys

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lib.tokenization.unified_tokenizer import UnifiedTokenizer, ModalityType


class MultimodalDataset(Dataset):
    """Dataset for multimodal training"""
    
    def __init__(self, data_path: str, modality: str = "mixed"):
        self.data_path = Path(data_path)
        self.modality = modality
        # In practice, would load actual data
        self.samples = self._load_samples()
    
    def _load_samples(self) -> List[Dict]:
        """Load samples (placeholder)"""
        # Would load actual video, image, text, audio data
        return [
            {'modality': 'text', 'data': 'sample text'},
            {'modality': 'image', 'data': torch.randn(3, 256, 256)},
            {'modality': 'video', 'data': torch.randn(3, 16, 256, 256)},
            {'modality': 'audio', 'data': torch.randn(1, 24000)},
        ] * 100
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


class Phase1Trainer:
    """Phase I Training: Multimodal Alignment"""
    
    def __init__(
        self,
        hidden_dim: int = 2048,
        vocab_size: int = 65536,
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        num_epochs: int = 100,
    ):
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        
        # Initialize tokenizer
        self.tokenizer = UnifiedTokenizer(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
        )
        
        # Loss functions
        self.reconstruction_loss = nn.MSELoss()
        self.cross_modal_loss = nn.CosineEmbeddingLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.tokenizer.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
        )
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer.to(self.device)
        
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Train Phase I"""
        print("Starting Phase I Training: Multimodal Alignment")
        
        for epoch in range(self.num_epochs):
            self.tokenizer.train()
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
    
    def _train_step(self, batch: Dict) -> torch.Tensor:
        """Single training step"""
        self.optimizer.zero_grad()
        
        # Encode different modalities
        losses = []
        
        for sample in batch:
            modality_str = sample['modality']
            data = sample['data']
            
            # Map to modality type
            modality_map = {
                'text': ModalityType.TEXT,
                'image': ModalityType.IMAGE,
                'video': ModalityType.VIDEO,
                'audio': ModalityType.AUDIO,
            }
            
            modality = modality_map.get(modality_str, ModalityType.TEXT)
            
            # Encode
            if isinstance(data, str):
                encoded = self.tokenizer.encode(modality, data)
            else:
                if not isinstance(data, torch.Tensor):
                    data = torch.tensor(data)
                encoded = self.tokenizer.encode(modality, data.to(self.device))
            
            # Reconstruction loss (simplified)
            if modality == ModalityType.IMAGE or modality == ModalityType.VIDEO:
                decoded = self.tokenizer.decode(
                    encoded['tokens'],
                    modality,
                )
                loss = self.reconstruction_loss(decoded, data.to(self.device))
                losses.append(loss)
        
        # Cross-modal alignment loss (simplified)
        if len(batch) >= 2:
            # Align embeddings from different modalities
            embeddings = []
            for sample in batch[:2]:
                modality_str = sample['modality']
                modality = modality_map.get(modality_str, ModalityType.TEXT)
                data = sample['data']
                
                if isinstance(data, str):
                    encoded = self.tokenizer.encode(modality, data)
                else:
                    if not isinstance(data, torch.Tensor):
                        data = torch.tensor(data)
                    encoded = self.tokenizer.encode(modality, data.to(self.device))
                
                embeddings.append(encoded['embeddings'].mean(dim=1))
            
            if len(embeddings) == 2:
                # Cross-modal alignment
                target = torch.ones(embeddings[0].shape[0]).to(self.device)
                cross_loss = self.cross_modal_loss(embeddings[0], embeddings[1], target)
                losses.append(cross_loss)
        
        # Total loss
        total_loss = sum(losses) / len(losses) if losses else torch.tensor(0.0).to(self.device)
        
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss
    
    def _validate(self, val_loader: DataLoader) -> float:
        """Validation"""
        self.tokenizer.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                loss = self._train_step(batch)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def save_checkpoint(self, path: str):
        """Save checkpoint"""
        torch.save({
            'tokenizer_state_dict': self.tokenizer.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load checkpoint"""
        checkpoint = torch.load(path)
        self.tokenizer.load_state_dict(checkpoint['tokenizer_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def main():
    parser = argparse.ArgumentParser(description='Phase I Training')
    parser.add_argument('--data_path', type=str, required=True, help='Path to training data')
    parser.add_argument('--hidden_dim', type=int, default=2048, help='Hidden dimension')
    parser.add_argument('--vocab_size', type=int, default=65536, help='Vocabulary size')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/phase1', help='Checkpoint path')
    
    args = parser.parse_args()
    
    # Create datasets
    train_dataset = MultimodalDataset(args.data_path, modality='mixed')
    val_dataset = MultimodalDataset(args.data_path, modality='mixed')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create trainer
    trainer = Phase1Trainer(
        hidden_dim=args.hidden_dim,
        vocab_size=args.vocab_size,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
    )
    
    # Train
    trainer.train(train_loader, val_loader)
    
    # Save checkpoint
    Path(args.checkpoint_path).mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(f"{args.checkpoint_path}/phase1_final.pt")


if __name__ == '__main__':
    main()

