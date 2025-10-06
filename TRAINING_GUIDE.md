# 🎓 Training Guide: Step-by-Step DimASR Training

## 📋 Table of Contents
- [Quick Start Training](#quick-start-training)
- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [Evaluation & Validation](#evaluation--validation)
- [Troubleshooting](#troubleshooting)
- [Advanced Training](#advanced-training)

## 🚀 Quick Start Training

### 1-Minute Training
```bash
# Setup
python setup.py

# Train the model
python quick_fix_training.py

# Evaluate results
python generate_fixed_predictions.py

# Check performance
cd evaluation_script
python metrics_subtask_1_2_3.py -p "sample data/subtask_1/eng/pred_eng_restaurant.jsonl" -g "sample data/subtask_1/eng/gold_eng_restaurant.jsonl" -t 1
```

### Expected Output
```
Final Results: {'PCC_V': 0.499, 'PCC_A': 0.558, 'RMSE': 0.203}
```

## 🛠️ Environment Setup

### Prerequisites
- Python 3.9+
- PyTorch 2.0+
- Transformers 4.20+
- 4GB+ RAM
- (Optional) CUDA-capable GPU

### Installation
```bash
# Clone repository
git clone https://github.com/adityabhalgat/DimASR.git
cd DimASR

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python setup.py
```

### Dependency Verification
```python
import torch
import transformers
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
```

## 📊 Data Preparation

### Understanding the Data Format

#### Input Data Structure (Trial Data)
```json
{
  "ID": "rest16_quad_dev_3",
  "Text": "the spicy tuna roll was unusually good and the rock shrimp tempura was awesome",
  "Quadruplet": [
    {
      "Aspect": "spicy tuna roll",
      "Opinion": "unusually good",
      "Category": "FOOD#QUALITY",
      "VA": "7.50#7.62"
    }
  ]
}
```

#### Processed Training Format
```python
{
  'text': "the spicy tuna roll was unusually good...",
  'aspect': "spicy tuna roll",
  'valence': 7.50,
  'arousal': 7.62
}
```

### Data Loading Pipeline
```python
def load_training_data():
    """Load and process training data from trial files"""
    
    trial_file = "task-dataset/trial/eng_restaurant_trial_alltasks.jsonl"
    
    data = []
    with open(trial_file, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            text = item['Text']
            
            for quad in item['Quadruplet']:
                aspect = quad['Aspect'] if quad['Aspect'] != 'NULL' else 'general'
                va_str = quad['VA']
                valence, arousal = map(float, va_str.split('#'))
                
                data.append({
                    'text': text,
                    'aspect': aspect,
                    'valence': valence,
                    'arousal': arousal
                })
    
    return data
```

### Data Augmentation Strategies

#### 1. Noise Injection
```python
def add_noise_augmentation(data, num_augments=2):
    """Add training samples with slight VA noise"""
    
    augmented = data.copy()
    
    for item in data:
        for _ in range(num_augments):
            # Add small random noise to VA values
            noise_v = np.random.normal(0, 0.2)
            noise_a = np.random.normal(0, 0.2)
            
            augmented.append({
                'text': item['text'],
                'aspect': item['aspect'],
                'valence': np.clip(item['valence'] + noise_v, 0, 10),
                'arousal': np.clip(item['arousal'] + noise_a, 0, 10)
            })
    
    return augmented
```

#### 2. Text Variations
```python
def add_text_variations(data):
    """Create text variations for augmentation"""
    
    variations = []
    text_transforms = [
        lambda t: t.replace('.', ' .'),
        lambda t: t.replace(',', ' ,'),
        lambda t: t.replace('!', ' !'),
        lambda t: t.lower(),
        lambda t: t.capitalize()
    ]
    
    for item in data[:5]:  # Apply to subset
        for transform in text_transforms:
            variations.append({
                'text': transform(item['text']),
                'aspect': item['aspect'],
                'valence': item['valence'],
                'arousal': item['arousal']
            })
    
    return data + variations
```

### Data Statistics
```python
def analyze_data(data):
    """Analyze training data statistics"""
    
    valences = [item['valence'] for item in data]
    arousals = [item['arousal'] for item in data]
    
    print(f"Dataset Size: {len(data)} samples")
    print(f"Valence - Mean: {np.mean(valences):.2f}, Std: {np.std(valences):.2f}")
    print(f"Valence - Range: [{np.min(valences):.2f}, {np.max(valences):.2f}]")
    print(f"Arousal - Mean: {np.mean(arousals):.2f}, Std: {np.std(arousals):.2f}")
    print(f"Arousal - Range: [{np.min(arousals):.2f}, {np.max(arousals):.2f}]")
```

## 🏋️ Model Training

### Training Configuration

#### Basic Configuration
```python
TRAINING_CONFIG = {
    'model_name': 'bert-base-uncased',
    'max_length': 256,
    'batch_size': 8,
    'learning_rate': 2e-5,
    'epochs': 12,
    'dropout_rate': 0.2,
    'weight_decay': 0.01,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}
```

#### Advanced Configuration
```python
ADVANCED_CONFIG = {
    # Model settings
    'model_name': 'bert-base-uncased',  # or 'roberta-base'
    'max_length': 256,
    'hidden_dropout': 0.2,
    'attention_dropout': 0.1,
    
    # Training settings
    'batch_size': 16,  # Increase if you have more memory
    'learning_rate': 2e-5,
    'epochs': 20,
    'warmup_steps': 100,
    'weight_decay': 0.01,
    'gradient_clipping': 1.0,
    
    # Loss settings
    'mse_weight': 1.0,
    'diversity_weight': 0.1,
    'range_penalty_weight': 0.05,
    
    # Validation
    'validation_split': 0.2,
    'early_stopping_patience': 5,
    'save_best_model': True
}
```

### Model Architecture Implementation

#### Core Model Class
```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class DimASRModel(nn.Module):
    def __init__(self, model_name='bert-base-uncased', dropout_rate=0.2):
        super().__init__()
        
        # BERT encoder
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        
        # Regression head
        self.dropout = nn.Dropout(dropout_rate)
        self.regressor = nn.Linear(hidden_size, 2)
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Initialize regression head weights"""
        nn.init.xavier_uniform_(self.regressor.weight)
        nn.init.constant_(self.regressor.bias, 5.0)  # Initialize to middle of range
    
    def forward(self, input_ids, attention_mask):
        # BERT encoding
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        
        # Regression
        x = self.dropout(pooled_output)
        x = self.regressor(x)
        
        # CRITICAL: Proper output scaling
        valence = torch.sigmoid(x[:, 0]) * 10.0
        arousal = torch.sigmoid(x[:, 1]) * 10.0
        
        return valence, arousal
```

#### Dataset Class
```python
from torch.utils.data import Dataset

class DimASRDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Enhanced input format
        enhanced_text = f"Review: {item['text']} Aspect: {item['aspect']}"
        
        # Tokenize
        encoding = self.tokenizer(
            enhanced_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'valence': torch.tensor(item['valence'], dtype=torch.float32),
            'arousal': torch.tensor(item['arousal'], dtype=torch.float32)
        }
```

### Training Loop Implementation

#### Basic Training Loop
```python
def train_model(model, train_loader, optimizer, criterion, device):
    """Basic training loop for one epoch"""
    
    model.train()
    epoch_losses = []
    
    for batch in tqdm(train_loader, desc="Training"):
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        true_valence = batch['valence'].to(device)
        true_arousal = batch['arousal'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        pred_valence, pred_arousal = model(input_ids, attention_mask)
        
        # Calculate loss
        loss = criterion(pred_valence, pred_arousal, true_valence, true_arousal)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_losses.append(loss.item())
    
    return np.mean(epoch_losses)
```

#### Advanced Training Loop with Validation
```python
def train_model_advanced(model, train_loader, val_loader, optimizer, 
                        scheduler, criterion, num_epochs, device):
    """Advanced training with validation and early stopping"""
    
    best_val_score = -1
    patience_counter = 0
    train_losses = []
    val_scores = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_losses = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # Training step (same as basic loop)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            true_valence = batch['valence'].to(device)
            true_arousal = batch['arousal'].to(device)
            
            optimizer.zero_grad()
            pred_valence, pred_arousal = model(input_ids, attention_mask)
            loss = criterion(pred_valence, pred_arousal, true_valence, true_arousal)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            if scheduler:
                scheduler.step()
            
            epoch_losses.append(loss.item())
        
        # Validation phase
        val_score = evaluate_model(model, val_loader, device)
        
        avg_train_loss = np.mean(epoch_losses)
        train_losses.append(avg_train_loss)
        val_scores.append(val_score)
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, "
              f"Val Pearson = {val_score:.4f}")
        
        # Early stopping and model saving
        if val_score > best_val_score:
            best_val_score = val_score
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_score': val_score
            }, 'results/best_model.pt')
            
        else:
            patience_counter += 1
            if patience_counter >= 5:  # Early stopping
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    return train_losses, val_scores
```

### Loss Functions

#### Enhanced Loss Function
```python
class EnhancedLoss(nn.Module):
    def __init__(self, mse_weight=1.0, diversity_weight=0.1):
        super().__init__()
        self.mse_weight = mse_weight
        self.diversity_weight = diversity_weight
    
    def forward(self, pred_valence, pred_arousal, true_valence, true_arousal):
        # MSE loss for accuracy
        mse_v = F.mse_loss(pred_valence, true_valence)
        mse_a = F.mse_loss(pred_arousal, true_arousal)
        mse_loss = (mse_v + mse_a) / 2
        
        # Diversity loss to encourage range usage
        diversity_loss_v = -torch.var(pred_valence)
        diversity_loss_a = -torch.var(pred_arousal)
        diversity_loss = (diversity_loss_v + diversity_loss_a) / 2
        
        # Combined loss
        total_loss = (self.mse_weight * mse_loss + 
                     self.diversity_weight * diversity_loss)
        
        return total_loss
```

### Complete Training Script

#### Full Training Example
```python
def main():
    # Configuration
    config = TRAINING_CONFIG
    device = torch.device(config['device'])
    
    # Load and prepare data
    print("Loading data...")
    data = load_training_data()
    data = add_noise_augmentation(data, num_augments=2)
    print(f"Training with {len(data)} samples")
    
    # Initialize model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    model = DimASRModel(config['model_name'], config['dropout_rate'])
    model.to(device)
    
    # Create dataset and dataloader
    dataset = DimASRDataset(data, tokenizer, config['max_length'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    # Setup training components
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    criterion = EnhancedLoss()
    
    # Training loop
    print("Starting training...")
    for epoch in range(config['epochs']):
        avg_loss = train_model(model, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, 'results/final_model.pt')
    
    print("Training completed!")

if __name__ == "__main__":
    main()
```

## 📊 Evaluation & Validation

### Evaluation Metrics
```python
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error

def calculate_metrics(predictions, targets):
    """Calculate comprehensive evaluation metrics"""
    
    pred_v, pred_a = predictions
    true_v, true_a = targets
    
    # Pearson correlations
    pearson_v = pearsonr(pred_v, true_v)[0] if len(pred_v) > 1 else 0
    pearson_a = pearsonr(pred_a, true_a)[0] if len(pred_a) > 1 else 0
    
    # Other metrics
    mse_v = mean_squared_error(true_v, pred_v)
    mse_a = mean_squared_error(true_a, pred_a)
    mae_v = mean_absolute_error(true_v, pred_v)
    mae_a = mean_absolute_error(true_a, pred_a)
    
    return {
        'pearson_v': pearson_v,
        'pearson_a': pearson_a,
        'pearson_avg': (pearson_v + pearson_a) / 2,
        'mse_v': mse_v,
        'mse_a': mse_a,
        'mae_v': mae_v,
        'mae_a': mae_a,
        'rmse_avg': np.sqrt((mse_v + mse_a) / 2)
    }
```

### Model Evaluation
```python
def evaluate_model(model, val_loader, device):
    """Evaluate model on validation set"""
    
    model.eval()
    predictions_v, predictions_a = [], []
    targets_v, targets_a = [], []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            true_valence = batch['valence'].to(device)
            true_arousal = batch['arousal'].to(device)
            
            pred_valence, pred_arousal = model(input_ids, attention_mask)
            
            predictions_v.extend(pred_valence.cpu().numpy())
            predictions_a.extend(pred_arousal.cpu().numpy())
            targets_v.extend(true_valence.cpu().numpy())
            targets_a.extend(true_arousal.cpu().numpy())
    
    metrics = calculate_metrics(
        (predictions_v, predictions_a),
        (targets_v, targets_a)
    )
    
    return metrics['pearson_avg']
```

### Prediction Generation
```python
def generate_predictions(model, tokenizer, test_file, output_file):
    """Generate predictions for test set"""
    
    model.eval()
    device = next(model.parameters()).device
    predictions = []
    
    with open(test_file, 'r') as f:
        for line in f:
            test_item = json.loads(line.strip())
            doc_id = test_item['ID']
            text = test_item['Text']
            aspects = test_item['Aspect']
            
            aspect_predictions = []
            for aspect in aspects:
                enhanced_text = f"Review: {text} Aspect: {aspect}"
                
                encoding = tokenizer(
                    enhanced_text,
                    truncation=True,
                    padding='max_length',
                    max_length=256,
                    return_tensors='pt'
                )
                
                with torch.no_grad():
                    input_ids = encoding['input_ids'].to(device)
                    attention_mask = encoding['attention_mask'].to(device)
                    
                    pred_valence, pred_arousal = model(input_ids, attention_mask)
                    
                    va_str = f"{pred_valence.item():.2f}#{pred_arousal.item():.2f}"
                    aspect_predictions.append({
                        "Aspect": aspect,
                        "VA": va_str
                    })
            
            predictions.append({
                "ID": doc_id,
                "Aspect_VA": aspect_predictions
            })
    
    # Save predictions
    with open(output_file, 'w') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')
    
    return output_file
```

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. Poor Performance (Low Pearson Correlation)
**Problem**: Model predictions have low correlation with ground truth

**Solutions**:
```python
# Check output scaling
def debug_output_scaling(model, sample_input):
    with torch.no_grad():
        raw_output = model.regressor(sample_input)
        scaled_output = torch.sigmoid(raw_output) * 10.0
    
    print(f"Raw output range: [{raw_output.min():.2f}, {raw_output.max():.2f}]")
    print(f"Scaled output range: [{scaled_output.min():.2f}, {scaled_output.max():.2f}]")

# Verify loss function
def check_loss_components(pred_v, pred_a, true_v, true_a):
    mse_loss = F.mse_loss(pred_v, true_v) + F.mse_loss(pred_a, true_a)
    diversity = torch.var(pred_v) + torch.var(pred_a)
    
    print(f"MSE Loss: {mse_loss:.4f}")
    print(f"Prediction Diversity: {diversity:.4f}")
```

#### 2. Range Compression
**Problem**: All predictions clustered in small range

**Solutions**:
1. Ensure proper sigmoid scaling: `torch.sigmoid(x) * 10.0`
2. Check initialization: Bias should be around 5.0
3. Add diversity loss to encourage range usage

#### 3. Overfitting
**Problem**: Training loss decreases but validation performance doesn't improve

**Solutions**:
```python
# Increase regularization
model = DimASRModel(dropout_rate=0.3)  # Increase dropout

# Add weight decay
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-5,
    weight_decay=0.1  # Increase weight decay
)

# Early stopping
if val_score <= best_val_score:
    patience_counter += 1
    if patience_counter >= 3:  # Stop early
        break
```

#### 4. Memory Issues
**Problem**: Out of memory errors during training

**Solutions**:
```python
# Reduce batch size
batch_size = 4  # Instead of 8

# Use gradient accumulation
accumulation_steps = 2
for i, batch in enumerate(dataloader):
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Debugging Checklist
- [ ] Output scaling implemented correctly
- [ ] Loss function includes all components
- [ ] Data loading returns correct format
- [ ] Model architecture matches expectations
- [ ] Learning rate is appropriate (2e-5)
- [ ] Gradient clipping is enabled
- [ ] Validation data is available

## 🚀 Advanced Training

### Hyperparameter Optimization
```python
import optuna

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_loguniform('learning_rate', 1e-6, 1e-3)
    dropout = trial.suggest_uniform('dropout_rate', 0.1, 0.5)
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])
    
    # Train model with suggested parameters
    model = DimASRModel(dropout_rate=dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # Training loop (simplified)
    val_score = train_and_evaluate(model, optimizer, batch_size)
    
    return val_score

# Run optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)
print(f"Best parameters: {study.best_params}")
```

### Multi-GPU Training
```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    print(f"Using {torch.cuda.device_count()} GPUs")
```

### Learning Rate Scheduling
```python
from transformers import get_linear_schedule_with_warmup

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=len(dataloader) * num_epochs
)
```

### Advanced Data Augmentation
```python
import nltk
from textblob import TextBlob

def advanced_augmentation(data):
    """Advanced data augmentation techniques"""
    
    augmented = data.copy()
    
    for item in data[:10]:  # Apply to subset
        text = item['text']
        
        # Synonym replacement
        words = text.split()
        if len(words) > 3:
            # Replace one word with synonym
            # (Implementation depends on synonym library)
            pass
        
        # Paraphrasing
        blob = TextBlob(text)
        # Apply simple transformations
        
        # Back-translation (if multilingual models available)
        # Translate to another language and back
        
    return augmented
```

## 📝 Training Logs and Monitoring

### Logging Setup
```python
import logging
import wandb  # For experiment tracking

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Weights & Biases
wandb.init(project="dimasr-training", config=config)
wandb.watch(model)

# Log training metrics
def log_metrics(epoch, train_loss, val_score):
    logger.info(f"Epoch {epoch}: Loss={train_loss:.4f}, Val Score={val_score:.4f}")
    wandb.log({
        'epoch': epoch,
        'train_loss': train_loss,
        'val_pearson': val_score
    })
```

### Training Visualization
```python
import matplotlib.pyplot as plt

def plot_training_progress(train_losses, val_scores):
    """Plot training progress"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    
    ax2.plot(val_scores)
    ax2.set_title('Validation Pearson')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Pearson Correlation')
    
    plt.tight_layout()
    plt.savefig('results/training_progress.png', dpi=300)
    plt.show()
```

---

This comprehensive training guide provides everything needed to successfully train the DimASR model from scratch. Follow the steps sequentially for best results, and refer to the troubleshooting section if you encounter issues.