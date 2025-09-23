"""
Evaluation utilities for DimASR models.
Provides inference and evaluation functions.
"""

import torch
import json
import numpy as np
from typing import List, Dict, Tuple
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DimASREvaluator:
    """Evaluator for DimASR models."""
    
    def __init__(self, model, tokenizer, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def predict_va_scores(self, texts: List[str], aspects: List[str], max_length: int = 128) -> List[Tuple[float, float]]:
        """
        Predict VA scores for text-aspect pairs.
        
        Args:
            texts: List of texts
            aspects: List of aspects
            max_length: Maximum sequence length
        
        Returns:
            List of (valence, arousal) tuples
        """
        predictions = []
        
        with torch.no_grad():
            for text, aspect in zip(texts, aspects):
                # Format input
                input_text = f"{text} [SEP] {aspect}"
                
                # Tokenize
                encoding = self.tokenizer(
                    input_text,
                    truncation=True,
                    padding='max_length',
                    max_length=max_length,
                    return_tensors='pt'
                )
                
                # Move to device
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                token_type_ids = encoding.get('token_type_ids')
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(self.device)
                
                # Predict
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                
                pred = outputs['predictions'].cpu().numpy()[0]
                predictions.append((float(pred[0]), float(pred[1])))
        
        return predictions
    
    def evaluate_predictions(
        self,
        predictions: List[Tuple[float, float]],
        targets: List[Tuple[float, float]]
    ) -> Dict[str, float]:
        """
        Evaluate predictions against targets.
        
        Args:
            predictions: List of (valence, arousal) predictions
            targets: List of (valence, arousal) targets
        
        Returns:
            Dictionary of evaluation metrics
        """
        pred_array = np.array(predictions)
        target_array = np.array(targets)
        
        metrics = {}
        
        # Overall metrics
        metrics['mse'] = mean_squared_error(target_array, pred_array)
        metrics['mae'] = mean_absolute_error(target_array, pred_array)
        
        # Valence metrics
        val_corr, val_p = pearsonr(target_array[:, 0], pred_array[:, 0])
        metrics['valence_pearson'] = val_corr if not np.isnan(val_corr) else 0.0
        metrics['valence_p_value'] = val_p if not np.isnan(val_p) else 1.0
        
        # Arousal metrics
        aro_corr, aro_p = pearsonr(target_array[:, 1], pred_array[:, 1])
        metrics['arousal_pearson'] = aro_corr if not np.isnan(aro_corr) else 0.0
        metrics['arousal_p_value'] = aro_p if not np.isnan(aro_p) else 1.0
        
        # Combined metric
        metrics['avg_pearson'] = (metrics['valence_pearson'] + metrics['arousal_pearson']) / 2
        
        # Cosine similarity
        cosine_similarities = []
        for pred, target in zip(pred_array, target_array):
            # Normalize vectors
            pred_norm = pred / (np.linalg.norm(pred) + 1e-8)
            target_norm = target / (np.linalg.norm(target) + 1e-8)
            cosine_sim = np.dot(pred_norm, target_norm)
            cosine_similarities.append(cosine_sim)
        
        metrics['avg_cosine_similarity'] = np.mean(cosine_similarities)
        
        return metrics
    
    def generate_predictions_subtask1(
        self,
        test_data: List[Dict],
        output_path: str = None
    ) -> List[Dict]:
        """
        Generate predictions for Subtask 1.
        
        Args:
            test_data: Test data with text and aspects
            output_path: Optional path to save predictions
        
        Returns:
            List of prediction dictionaries
        """
        predictions = []
        
        for item in test_data:
            text = item['Text']
            aspects = item['Aspect']
            
            # Predict VA for each aspect
            aspect_va_predictions = []
            for aspect in aspects:
                va_pred = self.predict_va_scores([text], [aspect])[0]
                aspect_va_predictions.append({
                    'Aspect': aspect,
                    'VA': f"{va_pred[0]:.2f}#{va_pred[1]:.2f}"
                })
            
            predictions.append({
                'ID': item['ID'],
                'Aspect_VA': aspect_va_predictions
            })
        
        # Save predictions if path provided
        if output_path:
            self._save_predictions(predictions, output_path)
        
        return predictions
    
    def _save_predictions(self, predictions: List[Dict], output_path: str):
        """Save predictions to JSONL file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for pred in predictions:
                f.write(json.dumps(pred, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(predictions)} predictions to {output_path}")
    
    def run_official_evaluation(
        self,
        pred_file: str,
        gold_file: str,
        task: int = 1,
        eval_script_path: str = "../evaluation_script/metrics_subtask_1_2_3.py"
    ) -> Dict[str, float]:
        """
        Run official evaluation script.
        
        Args:
            pred_file: Path to prediction file
            gold_file: Path to gold file
            task: Task number (1, 2, or 3)
            eval_script_path: Path to evaluation script
        
        Returns:
            Evaluation results
        """
        import subprocess
        import sys
        
        try:
            # Run evaluation script
            cmd = [
                sys.executable,
                eval_script_path,
                '-t', str(task),
                '-p', pred_file,
                '-g', gold_file
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(eval_script_path).parent)
            
            if result.returncode == 0:
                logger.info("Official evaluation completed successfully")
                logger.info(f"Output: {result.stdout}")
                return {'status': 'success', 'output': result.stdout}
            else:
                logger.error(f"Evaluation failed: {result.stderr}")
                return {'status': 'error', 'error': result.stderr}
        
        except Exception as e:
            logger.error(f"Error running evaluation: {e}")
            return {'status': 'error', 'error': str(e)}


def load_model_from_checkpoint(model_path: str, model_class, tokenizer_class):
    """
    Load model from checkpoint.
    
    Args:
        model_path: Path to model directory
        model_class: Model class to instantiate
        tokenizer_class: Tokenizer class to instantiate
    
    Returns:
        Tuple of (model, tokenizer)
    """
    model_path = Path(model_path)
    
    # Load tokenizer
    tokenizer = tokenizer_class.from_pretrained(model_path)
    
    # Load training config to get model parameters
    config_path = model_path / "training_config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        model_name = config.get('model_name', 'bert-base-uncased')
    else:
        model_name = 'bert-base-uncased'
    
    # Initialize model
    model = model_class(model_name=model_name)
    
    # Load model weights
    state_dict_path = model_path / "model.pt"
    if state_dict_path.exists():
        model.load_state_dict(torch.load(state_dict_path, map_location='cpu'))
        logger.info(f"Loaded model weights from {state_dict_path}")
    else:
        logger.warning(f"No model weights found at {state_dict_path}")
    
    return model, tokenizer


def main():
    """Example evaluation usage."""
    # This would be used after training a model
    # Example shows how to load and evaluate a trained model
    
    # Load model (placeholder)
    # model, tokenizer = load_model_from_checkpoint("./models/best_model", DimASRTransformer, AutoTokenizer)
    
    # Load test data
    # test_data = DimASRDataProcessor().load_jsonl("test_data.jsonl")
    
    # Create evaluator
    # evaluator = DimASREvaluator(model, tokenizer)
    
    # Generate predictions
    # predictions = evaluator.generate_predictions_subtask1(test_data, "predictions.jsonl")
    
    print("Evaluation utilities ready for use after model training.")


if __name__ == "__main__":
    main()