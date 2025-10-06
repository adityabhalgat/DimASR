#!/usr/bin/env python3
"""
Quick Accuracy Fix - Simplified Version
Addresses the critical output scaling issue
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class QuickFixDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Better input format
        enhanced_text = f"Review: {item['text']} Aspect: {item['aspect']}"
        
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

class QuickFixModel(nn.Module):
    """Model with proper output scaling - this is the KEY FIX"""
    
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        
        self.dropout = nn.Dropout(0.2)
        self.regressor = nn.Linear(hidden_size, 2)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.regressor.weight)
        nn.init.constant_(self.regressor.bias, 5.0)  # Initialize bias to middle of range
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        
        x = self.dropout(pooled_output)
        x = self.regressor(x)
        
        # CRITICAL FIX: Proper scaling to 0-10 range
        # The original model was not scaling properly!
        valence = torch.sigmoid(x[:, 0]) * 10.0  # Scale sigmoid (0,1) to (0,10)
        arousal = torch.sigmoid(x[:, 1]) * 10.0  # Scale sigmoid (0,1) to (0,10)
        
        return valence, arousal

def load_data_corrected():
    """Load data with correct format"""
    
    trial_file = "/Users/adityabhalgat/Developer/DimASR/task-dataset/trial/eng_restaurant_trial_alltasks.jsonl"
    
    data = []
    with open(trial_file, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            text = item['Text']
            
            # Correct format: "Quadruplet" not "Aspect_VA"
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
    
    print(f"Loaded {len(data)} samples from trial data")
    
    # Simple augmentation: add slight variations
    augmented = data.copy()
    for item in data:
        # Add 2 variations with small noise
        for _ in range(2):
            noise_v = np.random.normal(0, 0.2)
            noise_a = np.random.normal(0, 0.2)
            
            augmented.append({
                'text': item['text'],
                'aspect': item['aspect'],
                'valence': np.clip(item['valence'] + noise_v, 0, 10),
                'arousal': np.clip(item['arousal'] + noise_a, 0, 10)
            })
    
    print(f"After augmentation: {len(augmented)} samples")
    return augmented

def quick_training():
    """Quick training with the key fix applied"""
    
    print("🚀 Quick Accuracy Fix Training")
    print("=" * 50)
    print("🎯 KEY FIX: Proper output scaling (sigmoid * 10)")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    data = load_data_corrected()
    
    # Setup
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    dataset = QuickFixDataset(data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    model = QuickFixModel()
    model.to(device)
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    
    def scaling_aware_loss(pred_v, pred_a, true_v, true_a):
        """Loss that encourages proper range usage"""
        mse_loss = (nn.functional.mse_loss(pred_v, true_v) + nn.functional.mse_loss(pred_a, true_a)) / 2
        
        # Encourage diversity in predictions (avoid all predictions being similar)
        diversity_loss_v = -torch.var(pred_v)  # Negative variance to encourage spread
        diversity_loss_a = -torch.var(pred_a)
        diversity_loss = (diversity_loss_v + diversity_loss_a) * 0.1
        
        return mse_loss + diversity_loss
    
    # Training
    num_epochs = 12
    print(f"Training for {num_epochs} epochs...")
    
    model.train()
    for epoch in range(num_epochs):
        epoch_losses = []
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            true_valence = batch['valence'].to(device)
            true_arousal = batch['arousal'].to(device)
            
            optimizer.zero_grad()
            
            pred_valence, pred_arousal = model(input_ids, attention_mask)
            loss = scaling_aware_loss(pred_valence, pred_arousal, true_valence, true_arousal)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'pred_v_range': f'{pred_valence.min().item():.1f}-{pred_valence.max().item():.1f}',
                'pred_a_range': f'{pred_arousal.min().item():.1f}-{pred_arousal.max().item():.1f}'
            })
        
        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
    
    # Save model
    os.makedirs('results', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer_name': 'bert-base-uncased'
    }, 'results/quick_fix_model.pt')
    
    print("✅ Quick fix training completed!")
    return model, tokenizer

def generate_fixed_predictions(model, tokenizer):
    """Generate predictions with fixed model"""
    
    print("🎯 Generating FIXED predictions...")
    
    device = next(model.parameters()).device
    model.eval()
    
    # Load test data
    test_file = "/Users/adityabhalgat/Developer/DimASR/evaluation_script/sample data/subtask_1/eng/test_eng_restaurant.jsonl"
    
    predictions = []
    with open(test_file, 'r') as f:
        for line in f:
            test_item = json.loads(line.strip())
            doc_id = test_item['ID']
            text = test_item['Text']
            aspects = test_item['Aspects']
            
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
    
    # Save fixed predictions
    output_file = 'results/pred_eng_restaurant_FIXED.jsonl'
    with open(output_file, 'w') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\\n')
    
    print(f"✅ Fixed predictions saved to: {output_file}")
    
    # Quick preview of predictions
    print("\\n📊 Preview of FIXED predictions:")
    for i, pred in enumerate(predictions[:3]):
        print(f"  Sample {i+1}: {pred['Aspect_VA'][0]['VA']}")
    
    return output_file

if __name__ == "__main__":
    print("🛠️ QUICK ACCURACY FIX")
    print("Problem: Model predictions too low (2.0 instead of 6.0)")
    print("Solution: Fix output scaling with sigmoid * 10")
    print("=" * 60)
    
    # Train fixed model
    model, tokenizer = quick_training()
    
    # Generate fixed predictions
    pred_file = generate_fixed_predictions(model, tokenizer)
    
    print("\\n" + "=" * 60)
    print("🎉 QUICK FIX COMPLETE!")
    print("=" * 60)
    print("🔧 Applied Critical Fix:")
    print("   - Changed output layer to: sigmoid(x) * 10")
    print("   - This scales predictions from 0-10 range properly")
    print("   - Previous model was stuck in 1-3 range")
    print()
    print("📁 Files:")
    print(f"   - Model: results/quick_fix_model.pt")
    print(f"   - Predictions: {pred_file}")
    print()
    print("🔍 Next: Run evaluation to see improvement!")
    print("   cd evaluation_script && python metrics_subtask_1_2_3.py")
    print("=" * 60)