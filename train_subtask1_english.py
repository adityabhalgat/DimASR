#!/usr/bin/env python3
"""
Quick start training script for DimASR Subtask 1 (English).
This script demonstrates training on English restaurant data.
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append('src')

# Import our modules
try:
    from transformers import AutoTokenizer, AutoModel
    from torch.utils.data import Dataset, DataLoader
    import torch.nn as nn
    from scipy.stats import pearsonr
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from tqdm import tqdm
    
    print("✅ All dependencies imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install dependencies: pip install -r requirements.txt")
    sys.exit(1)


class SimpleDimASRDataset(Dataset):
    """Simple dataset for DimASR Subtask 1."""
    
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Create input text: [CLS] text [SEP] aspect [SEP]
        text = item['text']
        aspect = item['aspect']
        input_text = f"{text} [SEP] {aspect}"
        
        # Tokenize
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        result = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'id': item['id']
        }
        
        # Add labels if available
        if 'valence' in item and 'arousal' in item:
            if item['valence'] is not None and item['arousal'] is not None:
                result['labels'] = torch.tensor([item['valence'], item['arousal']], dtype=torch.float)
        
        return result


class SimpleDimASRModel(nn.Module):
    """Simple BERT-based model for VA regression."""
    
    def __init__(self, model_name="bert-base-uncased"):
        super().__init__()
        
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.regressor = nn.Linear(self.backbone.config.hidden_size, 2)  # Valence, Arousal
        
        # Initialize regressor
        nn.init.normal_(self.regressor.weight, std=0.02)
        nn.init.zeros_(self.regressor.bias)
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.last_hidden_state[:, 0])  # [CLS] token
        predictions = self.regressor(pooled_output)
        
        result = {'predictions': predictions}
        
        if labels is not None:
            loss = nn.MSELoss()(predictions, labels)
            result['loss'] = loss
        
        return result


def load_and_process_data(file_path):
    """Load and process JSONL data for Subtask 1."""
    data = []
    
    print(f"📁 Loading data from {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    item = json.loads(line)
                    text = item.get('Text', '').strip()
                    
                    # Handle different data formats
                    if 'Aspect_VA' in item:  # Gold data format
                        for aspect_va in item['Aspect_VA']:
                            aspect = aspect_va['Aspect'].strip()
                            va_string = aspect_va['VA']
                            
                            # Parse VA scores
                            try:
                                valence_str, arousal_str = va_string.split('#')
                                valence = float(valence_str)
                                arousal = float(arousal_str)
                            except (ValueError, AttributeError):
                                print(f"⚠️  Invalid VA format at line {line_num}: {va_string}")
                                continue
                            
                            data.append({
                                'id': item['ID'],
                                'text': text,
                                'aspect': aspect,
                                'valence': valence,
                                'arousal': arousal
                            })
                    
                    elif 'Aspect' in item:  # Test data format
                        for aspect in item['Aspect']:
                            data.append({
                                'id': item['ID'],
                                'text': text,
                                'aspect': aspect.strip(),
                                'valence': None,
                                'arousal': None
                            })
                
                except json.JSONDecodeError as e:
                    print(f"⚠️  JSON error at line {line_num}: {e}")
                    continue
    
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return []
    
    print(f"✅ Loaded {len(data)} samples")
    return data


def train_model(model, train_loader, num_epochs=3, learning_rate=2e-5):
    """Train the model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    print(f"🚀 Training on {device} for {num_epochs} epochs")
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        
        for batch in progress_bar:
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            if 'labels' in batch:
                labels = batch['labels'].to(device)
                
                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs['loss']
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")


def evaluate_model(model, eval_loader):
    """Evaluate the model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    predictions = []
    targets = []
    
    print("📊 Evaluating model...")
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            predictions.extend(outputs['predictions'].cpu().numpy())
            
            if 'labels' in batch:
                targets.extend(batch['labels'].cpu().numpy())
    
    if targets:
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Calculate metrics
        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        
        # Pearson correlation for each dimension
        val_corr, _ = pearsonr(targets[:, 0], predictions[:, 0])
        aro_corr, _ = pearsonr(targets[:, 1], predictions[:, 1])
        avg_corr = (val_corr + aro_corr) / 2
        
        print(f"\n📈 Evaluation Results:")
        print(f"   MSE: {mse:.4f}")
        print(f"   MAE: {mae:.4f}")
        print(f"   Valence Pearson: {val_corr:.4f}")
        print(f"   Arousal Pearson: {aro_corr:.4f}")
        print(f"   Average Pearson: {avg_corr:.4f}")
        
        return {
            'mse': mse,
            'mae': mae,
            'valence_pearson': val_corr,
            'arousal_pearson': aro_corr,
            'avg_pearson': avg_corr
        }
    
    return {}


def main():
    """Main training function."""
    print("🚀 DimASR Subtask 1 Training - English Restaurant Data")
    print("=" * 60)
    
    # Configuration
    config = {
        'model_name': 'bert-base-uncased',
        'max_length': 128,
        'batch_size': 8,  # Small batch size for demo
        'num_epochs': 3,
        'learning_rate': 2e-5
    }
    
    print("⚙️  Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # Use sample data for training
    train_file = 'evaluation_script/sample data/subtask_1/eng/gold_eng_restaurant.jsonl'
    
    if not Path(train_file).exists():
        print(f"❌ Training file not found: {train_file}")
        print("Please ensure the evaluation_script sample data is available.")
        return
    
    # Load data
    train_data = load_and_process_data(train_file)
    
    if not train_data:
        print("❌ No training data loaded")
        return
    
    # Filter only samples with labels for training
    labeled_data = [item for item in train_data if item['valence'] is not None]
    
    if not labeled_data:
        print("❌ No labeled data found for training")
        return
    
    print(f"📊 Training samples: {len(labeled_data)}")
    
    # Create model and tokenizer
    print("🤖 Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    model = SimpleDimASRModel(config['model_name'])
    
    # Create dataset and dataloader
    dataset = SimpleDimASRDataset(labeled_data, tokenizer, config['max_length'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    print(f"📦 Created dataset with {len(dataset)} samples")
    
    # Train model
    train_model(model, dataloader, config['num_epochs'], config['learning_rate'])
    
    # Evaluate on the same data (just for demo)
    print("\n" + "=" * 60)
    eval_dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)
    metrics = evaluate_model(model, eval_dataloader)
    
    # Save model
    output_dir = Path("models/subtask1_english_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), output_dir / "model.pt")
    tokenizer.save_pretrained(output_dir)
    
    # Save training info
    training_info = {
        'config': config,
        'metrics': {k: float(v) for k, v in metrics.items()},  # Convert numpy floats to Python floats
        'num_samples': len(labeled_data)
    }
    
    with open(output_dir / "training_info.json", 'w') as f:
        json.dump(training_info, f, indent=2)
    
    print(f"\n✅ Model saved to {output_dir}")
    print("\n🎉 Training completed successfully!")
    
    # Test prediction
    print("\n" + "=" * 60)
    print("🧪 Testing prediction on sample data...")
    
    sample_text = "The food was excellent but the service was terrible."
    sample_aspect = "food"
    
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    with torch.no_grad():
        input_text = f"{sample_text} [SEP] {sample_aspect}"
        encoding = tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=config['max_length'],
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        prediction = outputs['predictions'].cpu().numpy()[0]
        
        print(f"Sample: '{sample_text}'")
        print(f"Aspect: '{sample_aspect}'")
        print(f"Predicted VA: valence={prediction[0]:.2f}, arousal={prediction[1]:.2f}")


if __name__ == "__main__":
    main()