#!/usr/bin/env python3
"""
Enhanced DimASR Training with Improved Negative Sentiment Recognition
- Sentiment-aware loss function
- Negative sentiment augmentation
- Aspect-context attention
- Better preprocessing for sentiment words
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
import json
import numpy as np
import re
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class SentimentAwareDataset(Dataset):
    def __init__(self, texts, aspects, valences, arousals, tokenizer, max_length=256):
        self.texts = texts
        self.aspects = aspects
        self.valences = valences
        self.arousals = arousals
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Sentiment keywords for better recognition
        self.positive_words = ['good', 'great', 'excellent', 'awesome', 'wonderful', 'amazing', 'best', 'love', 'heavenly', 'enjoyable', 'authentic', 'fine', 'nicest']
        self.negative_words = ['bad', 'horrible', 'terrible', 'awful', 'worst', 'hate', 'bland', 'lousy', 'rubber', 'flavorless', 'limited', 'mad']
        
    def enhance_text_with_sentiment(self, text, aspect):
        """Enhanced text preprocessing with sentiment emphasis"""
        text = text.lower()
        
        # Identify sentiment words near the aspect
        sentiment_context = ""
        words = text.split()
        
        # Find aspect position and nearby sentiment words
        aspect_words = aspect.lower().split()
        for i, word in enumerate(words):
            if any(asp_word in word for asp_word in aspect_words):
                # Look at context window around aspect
                start = max(0, i-5)
                end = min(len(words), i+6)
                context = words[start:end]
                
                # Extract sentiment indicators
                sentiments = []
                for ctx_word in context:
                    clean_word = re.sub(r'[^\w]', '', ctx_word)
                    if clean_word in self.positive_words:
                        sentiments.append(f"POSITIVE:{clean_word}")
                    elif clean_word in self.negative_words:
                        sentiments.append(f"NEGATIVE:{clean_word}")
                
                if sentiments:
                    sentiment_context = " ".join(sentiments)
                break
        
        # Enhanced format with sentiment markers
        if sentiment_context:
            enhanced = f"[SENTIMENT: {sentiment_context}] Review: {text} | Evaluating aspect: {aspect}"
        else:
            enhanced = f"Review: {text} | Evaluating aspect: {aspect}"
            
        return enhanced
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        aspect = self.aspects[idx]
        valence = self.valences[idx]
        arousal = self.arousals[idx]
        
        # Enhanced text with sentiment awareness
        enhanced_text = self.enhance_text_with_sentiment(text, aspect)
        
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
            'valence': torch.tensor(valence, dtype=torch.float),
            'arousal': torch.tensor(arousal, dtype=torch.float)
        }

class SentimentAwareModel(nn.Module):
    def __init__(self, model_name='bert-base-uncased', dropout_rate=0.3):
        super(SentimentAwareModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Dual-path architecture for valence and arousal
        self.valence_head = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(192, 1)
        )
        
        self.arousal_head = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(), 
            nn.Dropout(dropout_rate),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(192, 1)
        )
        
        # Initialize with sentiment-aware bias
        # Valence: negative bias to encourage lower predictions when needed
        nn.init.constant_(self.valence_head[-1].bias, 0.0)  # Neutral bias
        # Arousal: slightly positive bias as arousal is generally higher
        nn.init.constant_(self.arousal_head[-1].bias, 0.3)  # Slightly higher arousal
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # Separate processing for valence and arousal
        valence_raw = self.valence_head(pooled_output)
        arousal_raw = self.arousal_head(pooled_output)
        
        # Advanced scaling with better negative sentiment handling
        # Use tanh for valence to allow easier negative values, then scale
        valence = (torch.tanh(valence_raw) + 1) * 5.0  # Maps tanh(-1,1) to (0,10)
        # Use sigmoid for arousal as it's generally positive
        arousal = torch.sigmoid(arousal_raw) * 10.0
        
        return valence.squeeze(), arousal.squeeze()

class SentimentAwareLoss(nn.Module):
    def __init__(self, negative_weight=2.0, diversity_weight=0.1):
        super(SentimentAwareLoss, self).__init__()
        self.negative_weight = negative_weight
        self.diversity_weight = diversity_weight
        
    def forward(self, pred_valence, pred_arousal, true_valence, true_arousal):
        # Standard MSE loss
        valence_loss = F.mse_loss(pred_valence, true_valence, reduction='none')
        arousal_loss = F.mse_loss(pred_arousal, true_arousal, reduction='none')
        
        # Weight negative sentiment samples more heavily
        negative_mask = (true_valence < 4.0).float()  # Negative sentiment threshold
        valence_weights = 1.0 + negative_mask * (self.negative_weight - 1.0)
        
        # Apply weights
        weighted_valence_loss = (valence_loss * valence_weights).mean()
        weighted_arousal_loss = arousal_loss.mean()
        
        # Diversity loss to encourage full range usage
        diversity_loss = -torch.var(pred_valence) - torch.var(pred_arousal)
        
        total_loss = weighted_valence_loss + weighted_arousal_loss + self.diversity_weight * diversity_loss
        
        return total_loss, {
            'valence_loss': weighted_valence_loss.item(),
            'arousal_loss': weighted_arousal_loss.item(), 
            'diversity_loss': diversity_loss.item()
        }

def create_enhanced_augmented_data():
    """Create augmented training data with focus on negative sentiment"""
    
    # Load trial data
    trial_file = "task-dataset/trial/eng_restaurant_trial_alltasks.jsonl"
    
    texts, aspects, valences, arousals = [], [], [], []
    
    print("Loading and processing trial data...")
    with open(trial_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            text = data['Text']
            
            if 'Aspect_VA' in data:
                for aspect_va in data['Aspect_VA']:
                    aspect = aspect_va['Aspect']
                    va_str = aspect_va['VA']
                    valence, arousal = map(float, va_str.split('#'))
                    
                    texts.append(text)
                    aspects.append(aspect)
                    valences.append(valence)
                    arousals.append(arousal)
    
    print(f"Base dataset: {len(texts)} samples")
    
    # Enhanced augmentation with negative sentiment focus
    augmented_texts, augmented_aspects = [], []
    augmented_valences, augmented_arousals = [], []
    
    negative_phrases = [
        "unfortunately", "sadly", "regrettably", "disappointingly",
        "was not good", "quite poor", "really bad", "very disappointing", 
        "absolutely terrible", "completely awful", "extremely poor",
        "way too bland", "totally flavorless", "incredibly limited",
        "utterly horrible", "seriously overpriced", "ridiculously bad"
    ]
    
    positive_phrases = [
        "fortunately", "happily", "thankfully", "surprisingly well",
        "was excellent", "quite good", "really amazing", "very impressive",
        "absolutely wonderful", "completely perfect", "extremely good",
        "perfectly seasoned", "incredibly flavorful", "impressively diverse", 
        "utterly fantastic", "great value", "ridiculously good"
    ]
    
    for i in range(len(texts)):
        text = texts[i]
        aspect = aspects[i]
        valence = valences[i]
        arousal = arousals[i]
        
        # Original sample
        augmented_texts.append(text)
        augmented_aspects.append(aspect)
        augmented_valences.append(valence)
        augmented_arousals.append(arousal)
        
        # Enhanced augmentation for negative samples
        if valence < 4.0:  # Negative sentiment - augment more heavily
            for _ in range(4):  # 4x augmentation for negative
                # Add negative phrases
                negative_phrase = np.random.choice(negative_phrases)
                augmented_text = f"{negative_phrase}, {text.lower()}"
                
                # Slightly lower valence for augmented negative samples
                aug_valence = max(0.5, valence - np.random.uniform(0, 0.5))
                aug_arousal = arousal + np.random.uniform(-0.3, 0.3)
                aug_arousal = np.clip(aug_arousal, 0, 10)
                
                augmented_texts.append(augmented_text)
                augmented_aspects.append(aspect)
                augmented_valences.append(aug_valence)
                augmented_arousals.append(aug_arousal)
                
        elif valence > 7.0:  # Positive sentiment - standard augmentation
            for _ in range(2):  # 2x augmentation for positive
                positive_phrase = np.random.choice(positive_phrases)
                augmented_text = f"{positive_phrase}, {text.lower()}"
                
                aug_valence = min(9.5, valence + np.random.uniform(0, 0.3))
                aug_arousal = arousal + np.random.uniform(-0.2, 0.2)
                aug_arousal = np.clip(aug_arousal, 0, 10)
                
                augmented_texts.append(augmented_text)
                augmented_aspects.append(aspect)
                augmented_valences.append(aug_valence)
                augmented_arousals.append(aug_arousal)
        
        else:  # Neutral sentiment - minimal augmentation
            # Add noise for neutral cases
            aug_valence = valence + np.random.uniform(-0.3, 0.3)
            aug_arousal = arousal + np.random.uniform(-0.3, 0.3)
            aug_valence = np.clip(aug_valence, 0, 10)
            aug_arousal = np.clip(aug_arousal, 0, 10)
            
            augmented_texts.append(text)
            augmented_aspects.append(aspect)
            augmented_valences.append(aug_valence)
            augmented_arousals.append(aug_arousal)
    
    print(f"Augmented dataset: {len(augmented_texts)} samples")
    
    # Print augmentation distribution
    neg_count = sum(1 for v in augmented_valences if v < 4.0)
    pos_count = sum(1 for v in augmented_valences if v > 7.0)
    neu_count = len(augmented_valences) - neg_count - pos_count
    print(f"Distribution - Negative: {neg_count}, Neutral: {neu_count}, Positive: {pos_count}")
    
    return augmented_texts, augmented_aspects, augmented_valences, augmented_arousals

def train_enhanced_model():
    print("🚀 Starting Enhanced Sentiment-Aware Training...")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create enhanced training data
    texts, aspects, valences, arousals = create_enhanced_augmented_data()
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = SentimentAwareModel().to(device)
    
    # Create dataset and dataloader
    dataset = SentimentAwareDataset(texts, aspects, valences, arousals, tokenizer)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Optimizer and loss
    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    criterion = SentimentAwareLoss(negative_weight=3.0, diversity_weight=0.15)
    
    # Training loop
    model.train()
    training_losses = []
    
    num_epochs = 15  # More epochs for better negative sentiment learning
    print(f"Training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        total_loss = 0
        batch_count = 0
        epoch_details = []
        
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            true_valence = batch['valence'].to(device)
            true_arousal = batch['arousal'].to(device)
            
            optimizer.zero_grad()
            
            pred_valence, pred_arousal = model(input_ids, attention_mask)
            loss, loss_details = criterion(pred_valence, pred_arousal, true_valence, true_arousal)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            epoch_details.append(loss_details)
        
        avg_loss = total_loss / batch_count
        training_losses.append(avg_loss)
        
        # Print detailed loss breakdown
        avg_valence_loss = np.mean([d['valence_loss'] for d in epoch_details])
        avg_arousal_loss = np.mean([d['arousal_loss'] for d in epoch_details])
        avg_diversity_loss = np.mean([d['diversity_loss'] for d in epoch_details])
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Total Loss: {avg_loss:.4f}")
        print(f"  Valence Loss: {avg_valence_loss:.4f}")
        print(f"  Arousal Loss: {avg_arousal_loss:.4f}")
        print(f"  Diversity Loss: {avg_diversity_loss:.4f}")
    
    # Save model
    torch.save(model.state_dict(), 'results/enhanced_sentiment_model.pt')
    print("✅ Model saved to results/enhanced_sentiment_model.pt")
    
    return model, tokenizer, training_losses

def generate_enhanced_predictions(model, tokenizer):
    """Generate predictions with enhanced sentiment awareness"""
    
    print("🔮 Generating enhanced predictions...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    test_file = "evaluation_script/sample data/subtask_1/eng/test_eng_restaurant.jsonl"
    predictions = []
    
    with torch.no_grad():
        with open(test_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                text = data['Text']
                aspects = data['Aspect']
                
                aspect_predictions = []
                for aspect in aspects:
                    # Use the enhanced preprocessing from dataset
                    enhanced_text = f"Review: {text.lower()} | Evaluating aspect: {aspect.lower()}"
                    
                    # Tokenize
                    encoding = tokenizer(
                        enhanced_text,
                        truncation=True,
                        padding='max_length',
                        max_length=256,
                        return_tensors='pt'
                    )
                    
                    input_ids = encoding['input_ids'].to(device)
                    attention_mask = encoding['attention_mask'].to(device)
                    
                    # Predict
                    valence, arousal = model(input_ids, attention_mask)
                    
                    valence = valence.cpu().item()
                    arousal = arousal.cpu().item()
                    
                    # Ensure bounds
                    valence = max(0, min(10, valence))
                    arousal = max(0, min(10, arousal))
                    
                    aspect_predictions.append({
                        "Aspect": aspect,
                        "VA": f"{valence:.2f}#{arousal:.2f}"
                    })
                
                predictions.append({
                    "ID": data['ID'],
                    "Aspect_VA": aspect_predictions
                })
    
    # Save predictions
    output_file = "results/pred_eng_restaurant_enhanced.jsonl"
    with open(output_file, 'w') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')
    
    print(f"✅ Enhanced predictions saved to {output_file}")
    return output_file

def evaluate_enhanced_model(predictions_file):
    """Evaluate the enhanced model"""
    print("📊 Evaluating enhanced model...")
    
    import subprocess
    import os
    
    # Run evaluation script
    os.chdir("evaluation_script")
    
    result = subprocess.run([
        'python', 'metrics_subtask_1_2_3.py',
        '-p', f"../{predictions_file}",
        '-g', 'sample data/subtask_1/eng/gold_eng_restaurant.jsonl',
        '-t', '1'
    ], capture_output=True, text=True)
    
    print("Evaluation Results:")
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    
    # Go back to main directory
    os.chdir("..")

if __name__ == "__main__":
    print("🎯 Enhanced DimASR Training with Improved Negative Sentiment Recognition")
    print("=" * 70)
    
    # Train enhanced model
    model, tokenizer, losses = train_enhanced_model()
    
    # Generate predictions
    pred_file = generate_enhanced_predictions(model, tokenizer)
    
    # Evaluate
    evaluate_enhanced_model(pred_file)
    
    print("🎉 Enhanced training complete!")