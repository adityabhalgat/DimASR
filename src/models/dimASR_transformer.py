"""
Transformer-based model for DimASR task.
Implements BERT/RoBERTa-based regression model for VA score prediction.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DimASRTransformer(nn.Module):
    """
    Transformer-based model for dimensional aspect-based sentiment analysis.
    
    Architecture:
    - Pre-trained transformer (BERT/RoBERTa) backbone
    - Regression head for Valence-Arousal prediction
    - Support for aspect-specific encoding
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 2,  # Valence and Arousal
        dropout_rate: float = 0.1,
        freeze_backbone: bool = False
    ):
        super().__init__()
        
        self.model_name = model_name
        self.num_labels = num_labels
        
        # Load pre-trained transformer
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Regression head
        self.dropout = nn.Dropout(dropout_rate)
        self.regressor = nn.Linear(self.config.hidden_size, num_labels)
        
        # Initialize regression head
        self._init_weights(self.regressor)
    
    def _init_weights(self, module):
        """Initialize weights for new layers."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len] (optional)
            labels: Ground truth VA scores [batch_size, 2] (optional)
        
        Returns:
            Dictionary with predictions and loss (if labels provided)
        """
        # Forward through backbone
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0]  # [batch_size, hidden_size]
        
        # Apply dropout and regression head
        pooled_output = self.dropout(pooled_output)
        predictions = self.regressor(pooled_output)  # [batch_size, 2]
        
        result = {'predictions': predictions}
        
        # Calculate loss if labels provided
        if labels is not None:
            loss_fn = nn.MSELoss()
            loss = loss_fn(predictions, labels)
            result['loss'] = loss
        
        return result


class DimASRMultiTaskModel(nn.Module):
    """
    Multi-task model for handling different subtasks.
    Can be extended for joint training across subtasks.
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        dropout_rate: float = 0.1,
        freeze_backbone: bool = False
    ):
        super().__init__()
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # Task-specific heads
        self.va_regressor = nn.Linear(self.config.hidden_size, 2)  # Valence, Arousal
        self.aspect_classifier = nn.Linear(self.config.hidden_size, 1)  # Aspect detection
        self.opinion_classifier = nn.Linear(self.config.hidden_size, 1)  # Opinion detection
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for new layers."""
        for module in [self.va_regressor, self.aspect_classifier, self.opinion_classifier]:
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        task: str = "va_regression"
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for specific task.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs (optional)
            task: Task type ("va_regression", "aspect_detection", "opinion_detection")
        
        Returns:
            Task-specific predictions
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        pooled_output = self.dropout(outputs.last_hidden_state[:, 0])
        
        if task == "va_regression":
            predictions = self.va_regressor(pooled_output)
        elif task == "aspect_detection":
            predictions = torch.sigmoid(self.aspect_classifier(pooled_output))
        elif task == "opinion_detection":
            predictions = torch.sigmoid(self.opinion_classifier(pooled_output))
        else:
            raise ValueError(f"Unknown task: {task}")
        
        return {'predictions': predictions}


class DimASRLoss(nn.Module):
    """Custom loss function for DimASR task."""
    
    def __init__(
        self,
        mse_weight: float = 1.0,
        mae_weight: float = 0.5,
        cosine_weight: float = 0.3
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self.cosine_weight = cosine_weight
        
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.cosine_sim = nn.CosineSimilarity(dim=1)
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate combined loss.
        
        Args:
            predictions: Predicted VA scores [batch_size, 2]
            targets: Target VA scores [batch_size, 2]
        
        Returns:
            Combined loss
        """
        mse = self.mse_loss(predictions, targets)
        mae = self.mae_loss(predictions, targets)
        
        # Cosine similarity loss (1 - cosine similarity)
        cosine = 1 - self.cosine_sim(predictions, targets).mean()
        
        total_loss = (
            self.mse_weight * mse +
            self.mae_weight * mae +
            self.cosine_weight * cosine
        )
        
        return total_loss


def create_model(
    model_name: str = "bert-base-uncased",
    task_type: str = "single",
    **kwargs
) -> nn.Module:
    """
    Factory function to create DimASR models.
    
    Args:
        model_name: Pre-trained model name
        task_type: "single" or "multi" for single/multi-task models
        **kwargs: Additional model parameters
    
    Returns:
        Initialized model
    """
    if task_type == "single":
        return DimASRTransformer(model_name=model_name, **kwargs)
    elif task_type == "multi":
        return DimASRMultiTaskModel(model_name=model_name, **kwargs)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")


def main():
    """Example model creation and forward pass."""
    # Create model
    model = create_model(model_name="bert-base-uncased")
    
    # Example input
    batch_size, seq_len = 4, 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randn(batch_size, 2)  # Random VA scores
    
    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    
    print(f"Predictions shape: {outputs['predictions'].shape}")
    print(f"Loss: {outputs['loss'].item():.4f}")


if __name__ == "__main__":
    main()