# Free Deployment & Training Guide

## Overview

This guide provides step-by-step instructions for deploying and training the AGI system for **FREE** using various cloud platforms.

## 🆓 Free Platforms

### 1. Google Colab (Recommended)
- **Free Tier**: 
  - T4 GPU (16GB) - Free
  - A100 GPU (40GB) - Available with Colab Pro ($10/month) or free credits
  - 12.7GB RAM
  - 107GB disk space
- **Limits**: 
  - Free: ~12 hours runtime
  - Pro: 24 hours runtime
- **Best For**: Training, experimentation, demos

### 2. Kaggle Notebooks
- **Free Tier**:
  - P100 GPU (16GB) - 30 hours/week
  - 13GB RAM
  - 20GB disk space
- **Limits**: 30 GPU hours per week
- **Best For**: Training, competitions, sharing

### 3. Hugging Face Spaces
- **Free Tier**:
  - CPU - Free
  - T4 GPU - Free (limited hours)
  - 16GB RAM
  - 50GB disk space
- **Limits**: Limited GPU hours
- **Best For**: Deployment, demos, inference

### 4. Replicate
- **Free Tier**: Limited credits
- **Best For**: API deployment

### 5. Modal
- **Free Tier**: Limited credits
- **Best For**: Serverless deployment

## 🚀 Quick Start: Google Colab

### Step 1: Create Colab Notebook

1. Go to [Google Colab](https://colab.research.google.com/)
2. Create a new notebook
3. Set runtime to GPU: `Runtime > Change runtime type > GPU (T4)`

### Step 2: Setup Environment

```python
# Install dependencies
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install transformers accelerate datasets
!pip install sentencepiece tokenizers
!pip install pillow opencv-python librosa soundfile
!pip install wandb tensorboard

# Clone repository
!git clone https://github.com/yourusername/newllm.git
%cd newllm

# Install project
!pip install -e .
```

### Step 3: Download Datasets

```python
# Download UCI datasets
import urllib.request
import os

# Create data directory
os.makedirs('data/uci', exist_ok=True)

# Download sample UCI datasets
uci_datasets = {
    'iris': 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
    'wine': 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',
}

for name, url in uci_datasets.items():
    urllib.request.urlretrieve(url, f'data/uci/{name}.data')
    print(f"Downloaded {name}")

# Download Open Images (sample)
# Note: Full dataset is large, use sample for free tier
!wget -q https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv -O data/openimages/annotations.csv
```

### Step 4: Training Script

Create `train_colab.py`:

```python
import torch
from implementations.v1.agi_unified_system import AGIUnifiedSystem
from implementations.training.train_agi import AGITrainer
from torch.utils.data import DataLoader

# Initialize system
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Create trainer
trainer = AGITrainer(
    hidden_dim=2048,
    learning_rate=1e-4,
    batch_size=8,  # Smaller for Colab
    num_epochs=10,  # Start small
)

# Load data
from lib.data_integration.multi_dataset_loader import MultiDatasetLoader
data_loader = MultiDatasetLoader()
data_loader.register_uci_datasets(['Iris', 'Wine Quality'])
data_loader.register_openimages()

# Train
trainer.train(train_loader, val_loader)

# Save model
torch.save(trainer.state_dict(), 'model_checkpoint.pt')
```

### Step 5: Run Training

```python
# In Colab notebook
!python train_colab.py
```

## 🎯 Kaggle Deployment

### Step 1: Create Kaggle Notebook

1. Go to [Kaggle](https://www.kaggle.com/)
2. Create new notebook
3. Enable GPU: `Settings > Accelerator > GPU`

### Step 2: Upload Dataset

```python
# Kaggle automatically mounts datasets
# Add dataset: Go to Add Data > New Dataset
# Upload your data files

# Access data
import os
data_path = '/kaggle/input/your-dataset-name'
```

### Step 3: Training Script

```python
# Similar to Colab, but use Kaggle paths
import torch
from implementations.v1.agi_unified_system import AGIUnifiedSystem

# Kaggle provides more GPU time
trainer = AGITrainer(
    hidden_dim=2048,
    batch_size=16,  # Can use larger batch
    num_epochs=50,
)

# Train
trainer.train(train_loader, val_loader)

# Save to output
torch.save(trainer.state_dict(), '/kaggle/working/model.pt')
```

## 🌐 Hugging Face Spaces Deployment

### Step 1: Create Space

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
2. Create new Space
3. Select "Gradio" as SDK
4. Select GPU if needed

### Step 2: Create `app.py`

```python
import gradio as gr
import torch
from implementations.v1.agi_unified_system import AGIUnifiedSystem

# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
system = AGIUnifiedSystem(device=device, enable_agi=True)

# Load checkpoint if available
# system.load_state_dict(torch.load('model.pt'))

def generate(prompt, modality):
    result = system.generate(prompt, modality=modality)
    return result

# Create Gradio interface
interface = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Prompt"),
        gr.Dropdown(["text", "image", "video", "audio"], label="Modality")
    ],
    outputs=gr.Textbox(label="Output"),
    title="AGI Unified System",
    description="Generate text, images, video, or audio"
)

interface.launch()
```

### Step 3: Create `requirements.txt`

```
torch>=2.0.0
transformers>=4.30.0
gradio>=3.0.0
accelerate>=0.20.0
```

### Step 4: Deploy

```bash
# Push to Hugging Face
git push origin main
```

## 📊 Optimized Training for Free Tiers

### Memory Optimization

```python
# Use gradient checkpointing
from torch.utils.checkpoint import checkpoint

class OptimizedModel(nn.Module):
    def forward(self, x):
        return checkpoint(self._forward, x)
    
    def _forward(self, x):
        # Your model code
        pass

# Use mixed precision
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Batch Size Optimization

```python
# Adaptive batch size
def get_optimal_batch_size(model, device):
    batch_size = 1
    while True:
        try:
            test_input = torch.randn(batch_size, 100, 2048).to(device)
            _ = model(test_input)
            batch_size *= 2
        except RuntimeError:
            return batch_size // 2

optimal_batch = get_optimal_batch_size(model, device)
```

### Data Efficiency

```python
# Use data compression
from lib.data_integration.multi_dataset_loader import EfficientDataRepresentation

compressor = EfficientDataRepresentation(
    hidden_dim=2048,
    compression_ratio=0.1  # 10% compression
)

# Compress data before training
compressed_data = compressor(training_data)
```

## 🔄 Continuous Training Strategy

### Strategy 1: Incremental Training

```python
# Train in stages
stages = [
    {'epochs': 10, 'batch_size': 8, 'lr': 1e-4},
    {'epochs': 10, 'batch_size': 16, 'lr': 5e-5},
    {'epochs': 10, 'batch_size': 32, 'lr': 1e-5},
]

for stage in stages:
    trainer.train_for_stage(stage)
    # Save checkpoint
    torch.save(trainer.state_dict(), f'checkpoint_stage_{stage}.pt')
```

### Strategy 2: Distributed Training

```python
# Use multiple free instances
# Train different components on different instances

# Instance 1: Train tokenization
# Instance 2: Train planner
# Instance 3: Train experts
# Then combine
```

## 💾 Model Saving & Loading

### Save Checkpoints

```python
# Save full model
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'loss': loss,
}, 'checkpoint.pt')

# Save to Google Drive (Colab)
from google.colab import drive
drive.mount('/content/drive')
torch.save(model.state_dict(), '/content/drive/MyDrive/model.pt')
```

### Load Checkpoints

```python
# Load checkpoint
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
```

## 📈 Monitoring Training

### Use Weights & Biases (Free)

```python
import wandb

wandb.init(project="agi-system", entity="your-username")

# Log metrics
wandb.log({
    'loss': loss.item(),
    'accuracy': accuracy,
    'epoch': epoch
})

# Log model
wandb.watch(model)
```

### Use TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment')

writer.add_scalar('Loss/Train', loss, epoch)
writer.add_scalar('Accuracy/Train', accuracy, epoch)
```

## 🎯 Recommended Workflow

### Week 1: Setup & Initial Training
1. Set up Colab/Kaggle environment
2. Download sample datasets
3. Train Phase I (multimodal alignment)
4. Save checkpoints

### Week 2: Continue Training
1. Load checkpoints
2. Train Phase II (reasoning)
3. Evaluate on benchmarks

### Week 3: AGI Training
1. Train AGI components
2. Self-improvement loops
3. Cross-domain transfer

### Week 4: Deployment
1. Deploy to Hugging Face Spaces
2. Create demo interface
3. Share and iterate

## 🔗 Useful Links

- [Google Colab](https://colab.research.google.com/)
- [Kaggle Notebooks](https://www.kaggle.com/code)
- [Hugging Face Spaces](https://huggingface.co/spaces)
- [Weights & Biases](https://wandb.ai/)
- [TensorBoard](https://www.tensorflow.org/tensorboard)

## 💡 Tips for Free Training

1. **Use Gradient Checkpointing**: Saves memory
2. **Mixed Precision Training**: Faster, less memory
3. **Smaller Models First**: Start small, scale up
4. **Incremental Training**: Train in stages
5. **Save Frequently**: Don't lose progress
6. **Use Multiple Platforms**: Maximize free resources
7. **Compress Data**: Use efficient data representation
8. **Monitor Resources**: Watch GPU/RAM usage

## 🚨 Troubleshooting

### Out of Memory
```python
# Reduce batch size
batch_size = 4

# Use gradient accumulation
accumulation_steps = 4
loss = loss / accumulation_steps
loss.backward()
if (step + 1) % accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()
```

### Slow Training
```python
# Use mixed precision
from torch.cuda.amp import autocast

with autocast():
    output = model(input)
    loss = criterion(output, target)
```

### Connection Issues
```python
# Save checkpoints frequently
if epoch % 5 == 0:
    torch.save(model.state_dict(), f'checkpoint_{epoch}.pt')
```

## 📝 Example Complete Training Script

See `train_free.py` for a complete example optimized for free platforms.

