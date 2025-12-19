"""
Free Training Script
Optimized for Google Colab, Kaggle, and other free platforms
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import os
from pathlib import Path

# Import system components
from implementations.v1.agi_unified_system import AGIUnifiedSystem
from lib.data_integration.multi_dataset_loader import MultiDatasetLoader
from lib.agi_core.mixture_of_experts import MixtureOfExperts
from lib.agi_core.advanced_reasoning import AdvancedReasoningSystem

# Configuration for free platforms
class FreeTrainingConfig:
    """Configuration optimized for free platforms"""
    # Model config
    hidden_dim = 1024  # Reduced for free tier
    vocab_size = 32768  # Reduced vocabulary
    num_experts = 32  # Reduced experts
    
    # Training config
    batch_size = 4  # Small batch for free GPU
    gradient_accumulation_steps = 4  # Simulate larger batch
    learning_rate = 1e-4
    num_epochs = 10
    save_every = 5  # Save every 5 epochs
    
    # Memory optimization
    use_gradient_checkpointing = True
    use_mixed_precision = True
    use_sparse_attention = True
    
    # Data config
    max_samples_per_dataset = 1000  # Limit for free tier
    compression_ratio = 0.1


def setup_free_environment():
    """Setup environment for free platforms"""
    # Detect platform
    if 'COLAB_GPU' in os.environ:
        platform = 'colab'
        print("Detected Google Colab")
    elif 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        platform = 'kaggle'
        print("Detected Kaggle")
    else:
        platform = 'local'
        print("Detected local environment")
    
    # Setup paths
    if platform == 'colab':
        data_dir = Path('/content/data')
        checkpoint_dir = Path('/content/checkpoints')
        # Mount Google Drive for persistent storage
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            checkpoint_dir = Path('/content/drive/MyDrive/checkpoints')
            print("Mounted Google Drive")
        except:
            print("Google Drive mount failed, using local storage")
    elif platform == 'kaggle':
        data_dir = Path('/kaggle/input')
        checkpoint_dir = Path('/kaggle/working/checkpoints')
    else:
        data_dir = Path('./data')
        checkpoint_dir = Path('./checkpoints')
    
    data_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    return platform, data_dir, checkpoint_dir


def create_optimized_model(config):
    """Create optimized model for free platforms"""
    # Use smaller model with MoE for efficiency
    model = AGIUnifiedSystem(
        hidden_dim=config.hidden_dim,
        vocab_size=config.vocab_size,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        enable_agi=True,
    )
    
    # Add MoE for efficiency
    if hasattr(model, 'planner'):
        # Replace dense layers with MoE
        model.planner.reasoning_transformer = MixtureOfExperts(
            hidden_dim=config.hidden_dim,
            num_experts=config.num_experts,
        )
    
    # Enable gradient checkpointing
    if config.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    return model


def train_free(config, model, train_loader, val_loader, checkpoint_dir):
    """Training loop optimized for free platforms"""
    device = next(model.parameters()).device
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.01,
    )
    
    # Mixed precision scaler
    scaler = GradScaler() if config.use_mixed_precision else None
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            if isinstance(batch, dict):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
            else:
                batch = batch.to(device)
            
            # Forward pass with mixed precision
            if config.use_mixed_precision and scaler:
                with autocast():
                    output = model(batch)
                    loss = criterion(output, batch.get('target', torch.zeros_like(output)))
                    loss = loss / config.gradient_accumulation_steps
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                output = model(batch)
                loss = criterion(output, batch.get('target', torch.zeros_like(output)))
                loss = loss / config.gradient_accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            
            total_loss += loss.item() * config.gradient_accumulation_steps
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item() * config.gradient_accumulation_steps:.4f}")
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")
        
        # Validation
        if (epoch + 1) % 5 == 0:
            val_loss = validate(model, val_loader, device, config)
            print(f"Epoch {epoch} Validation Loss: {val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config.save_every == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")


def validate(model, val_loader, device, config):
    """Validation loop"""
    model.eval()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in val_loader:
            if isinstance(batch, dict):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
            else:
                batch = batch.to(device)
            
            if config.use_mixed_precision:
                with autocast():
                    output = model(batch)
                    loss = criterion(output, batch.get('target', torch.zeros_like(output)))
            else:
                output = model(batch)
                loss = criterion(output, batch.get('target', torch.zeros_like(output)))
            
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def main():
    """Main training function"""
    print("=" * 80)
    print("Free Training Script for AGI System")
    print("=" * 80)
    
    # Setup
    platform, data_dir, checkpoint_dir = setup_free_environment()
    config = FreeTrainingConfig()
    
    # Check GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("Using CPU (training will be slow)")
    
    # Create model
    print("\nCreating optimized model...")
    model = create_optimized_model(config)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    
    # Setup data
    print("\nSetting up data...")
    data_loader = MultiDatasetLoader(cache_dir=str(data_dir / 'cache'))
    data_loader.register_uci_datasets(['Iris', 'Wine Quality'])
    data_loader.register_openimages()
    
    # Create dummy data loaders (replace with actual data)
    from torch.utils.data import TensorDataset
    train_data = TensorDataset(
        torch.randn(1000, 100, config.hidden_dim),
        torch.randint(0, 10, (1000,))
    )
    val_data = TensorDataset(
        torch.randn(200, 100, config.hidden_dim),
        torch.randint(0, 10, (200,))
    )
    
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False)
    
    # Train
    print("\nStarting training...")
    train_free(config, model, train_loader, val_loader, checkpoint_dir)
    
    # Save final model
    final_path = checkpoint_dir / "final_model.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")


if __name__ == '__main__':
    main()

