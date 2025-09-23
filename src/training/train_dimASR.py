"""
Training script for DimASR models.
Supports training on multiple subtasks with configurable parameters.
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import numpy as np
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple, Optional
import argparse
from pathlib import Path

# Import custom modules (adjust paths as needed)
import sys
sys.path.append('../')
from data_preprocessing.data_processor import DimASRDataProcessor
from models.dimASR_transformer import DimASRTransformer, DimASRLoss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DimASRDataset(Dataset):
    """Dataset class for DimASR task."""
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_length: int = 128,
        task: int = 1
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task = task
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize input text
        encoding = self.tokenizer(
            item['input_text'],
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
        
        # Add token type ids if available
        if 'token_type_ids' in encoding:
            result['token_type_ids'] = encoding['token_type_ids'].squeeze()
        
        # Add labels if available
        if item['valence'] is not None and item['arousal'] is not None:
            result['labels'] = torch.tensor([item['valence'], item['arousal']], dtype=torch.float)
        
        return result


class DimASRTrainer:
    """Trainer class for DimASR models."""
    
    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        eval_dataset=None,
        learning_rate: float = 2e-5,
        batch_size: int = 16,
        num_epochs: int = 3,
        warmup_steps: int = 100,
        weight_decay: float = 0.01,
        device: str = None,
        output_dir: str = "./models",
        use_custom_loss: bool = True
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize loss function
        if use_custom_loss:
            self.loss_fn = DimASRLoss()
        else:
            self.loss_fn = nn.MSELoss()
        
        # Initialize optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        total_steps = len(train_dataset) * num_epochs // batch_size
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training history
        self.train_losses = []
        self.eval_losses = []
        self.eval_metrics = []
    
    def train_epoch(self, dataloader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                token_type_ids=batch.get('token_type_ids'),
                labels=batch.get('labels')
            )
            
            loss = outputs.get('loss')
            if loss is None and 'labels' in batch:
                # Use custom loss if model doesn't compute it
                loss = self.loss_fn(outputs['predictions'], batch['labels'])
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader) -> Dict[str, float]:
        """Evaluate model on validation set."""
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    token_type_ids=batch.get('token_type_ids'),
                    labels=batch.get('labels')
                )
                
                if 'labels' in batch:
                    loss = outputs.get('loss')
                    if loss is None:
                        loss = self.loss_fn(outputs['predictions'], batch['labels'])
                    
                    total_loss += loss.item()
                    
                    predictions.extend(outputs['predictions'].cpu().numpy())
                    targets.extend(batch['labels'].cpu().numpy())
        
        # Calculate metrics
        metrics = {}
        if predictions:
            predictions = np.array(predictions)
            targets = np.array(targets)
            
            # Overall metrics
            metrics['mse'] = mean_squared_error(targets, predictions)
            metrics['mae'] = mean_absolute_error(targets, predictions)
            
            # Valence metrics
            val_corr, _ = pearsonr(targets[:, 0], predictions[:, 0])
            metrics['valence_pearson'] = val_corr if not np.isnan(val_corr) else 0.0
            
            # Arousal metrics
            aro_corr, _ = pearsonr(targets[:, 1], predictions[:, 1])
            metrics['arousal_pearson'] = aro_corr if not np.isnan(aro_corr) else 0.0
            
            # Combined metric
            metrics['avg_pearson'] = (metrics['valence_pearson'] + metrics['arousal_pearson']) / 2
            
            metrics['loss'] = total_loss / len(dataloader)
        
        return metrics
    
    def train(self):
        """Main training loop."""
        logger.info(f"Starting training on {self.device}")
        logger.info(f"Training samples: {len(self.train_dataset)}")
        if self.eval_dataset:
            logger.info(f"Evaluation samples: {len(self.eval_dataset)}")
        
        # Create data loaders
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0  # Set to 0 for compatibility
        )
        
        eval_loader = None
        if self.eval_dataset:
            eval_loader = DataLoader(
                self.eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0
            )
        
        best_metric = float('-inf')
        
        for epoch in range(self.num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            # Training
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            logger.info(f"Training loss: {train_loss:.4f}")
            
            # Evaluation
            if eval_loader:
                eval_metrics = self.evaluate(eval_loader)
                self.eval_losses.append(eval_metrics['loss'])
                self.eval_metrics.append(eval_metrics)
                
                logger.info(f"Evaluation metrics:")
                for metric, value in eval_metrics.items():
                    logger.info(f"  {metric}: {value:.4f}")
                
                # Save best model
                current_metric = eval_metrics['avg_pearson']
                if current_metric > best_metric:
                    best_metric = current_metric
                    self.save_model(f"best_model_epoch_{epoch + 1}")
                    logger.info(f"New best model saved (avg_pearson: {best_metric:.4f})")
        
        # Save final model
        self.save_model("final_model")
        logger.info("Training completed!")
    
    def save_model(self, name: str):
        """Save model and tokenizer."""
        model_path = self.output_dir / name
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        torch.save(self.model.state_dict(), model_path / "model.pt")
        
        # Save tokenizer
        self.tokenizer.save_pretrained(model_path)
        
        # Save training config
        config = {
            'model_name': getattr(self.model, 'model_name', 'unknown'),
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'train_losses': self.train_losses,
            'eval_losses': self.eval_losses,
            'eval_metrics': self.eval_metrics
        }
        
        with open(model_path / "training_config.json", 'w') as f:
            json.dump(config, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Train DimASR model")
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data")
    parser.add_argument("--eval_data", type=str, help="Path to evaluation data")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Pre-trained model name")
    parser.add_argument("--task", type=int, default=1, choices=[1, 2, 3], help="Subtask number")
    parser.add_argument("--output_dir", type=str, default="./models", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--max_length", type=int, default=128, help="Max sequence length")
    
    args = parser.parse_args()
    
    # Initialize data processor
    processor = DimASRDataProcessor()
    
    # Load and process training data
    logger.info(f"Loading training data from {args.train_data}")
    train_raw = processor.load_jsonl(args.train_data)
    
    if args.task == 1:
        train_processed = processor.prepare_subtask1_data(train_raw)
    elif args.task == 2:
        train_processed = processor.prepare_subtask2_data(train_raw)
    else:
        train_processed = processor.prepare_subtask3_data(train_raw)
    
    train_inputs = processor.create_model_inputs(train_processed, task=args.task)
    
    # Load evaluation data if provided
    eval_inputs = None
    if args.eval_data:
        logger.info(f"Loading evaluation data from {args.eval_data}")
        eval_raw = processor.load_jsonl(args.eval_data)
        
        if args.task == 1:
            eval_processed = processor.prepare_subtask1_data(eval_raw)
        elif args.task == 2:
            eval_processed = processor.prepare_subtask2_data(eval_raw)
        else:
            eval_processed = processor.prepare_subtask3_data(eval_raw)
        
        eval_inputs = processor.create_model_inputs(eval_processed, task=args.task)
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = DimASRTransformer(model_name=args.model_name)
    
    # Create datasets
    train_dataset = DimASRDataset(train_inputs, tokenizer, args.max_length, args.task)
    eval_dataset = DimASRDataset(eval_inputs, tokenizer, args.max_length, args.task) if eval_inputs else None
    
    # Initialize trainer
    trainer = DimASRTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        output_dir=args.output_dir
    )
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()