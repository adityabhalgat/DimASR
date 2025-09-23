"""
Utility functions for DimASR project.
"""

import json
import re
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def parse_va_string(va_string: str) -> Tuple[float, float]:
    """Parse VA scores from string format 'valence#arousal'."""
    try:
        valence_str, arousal_str = va_string.split('#')
        return float(valence_str), float(arousal_str)
    except (ValueError, AttributeError):
        logger.warning(f"Invalid VA format: {va_string}")
        return 0.0, 0.0


def format_va_string(valence: float, arousal: float) -> str:
    """Format VA scores to string format 'valence#arousal'."""
    return f"{valence:.2f}#{arousal:.2f}"


def normalize_text(text: str) -> str:
    """Normalize text for consistent processing."""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Handle common contractions
    contractions = {
        "n't": " not",
        "'re": " are", 
        "'ve": " have",
        "'ll": " will",
        "'d": " would",
        "'m": " am"
    }
    
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    
    return text


def calculate_va_statistics(va_scores: List[Tuple[float, float]]) -> Dict[str, float]:
    """Calculate statistics for VA scores."""
    if not va_scores:
        return {}
    
    valences = [v for v, a in va_scores]
    arousals = [a for v, a in va_scores]
    
    stats = {
        'valence_mean': np.mean(valences),
        'valence_std': np.std(valences),
        'valence_min': np.min(valences),
        'valence_max': np.max(valences),
        'arousal_mean': np.mean(arousals),
        'arousal_std': np.std(arousals),
        'arousal_min': np.min(arousals),
        'arousal_max': np.max(arousals),
        'count': len(va_scores)
    }
    
    return stats


def create_submission_file(
    predictions: List[Dict],
    output_path: str,
    task: int,
    language: str,
    domain: str
):
    """
    Create submission file in the correct format.
    
    Args:
        predictions: List of prediction dictionaries
        output_path: Base output directory
        task: Task number (1, 2, 3)
        language: Language code (e.g., 'eng', 'zho')
        domain: Domain (e.g., 'restaurant', 'laptop')
    """
    # Create output directory structure
    output_dir = Path(output_path) / f"subtask_{task}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename
    filename = f"pred_{language}_{domain}.jsonl"
    filepath = output_dir / filename
    
    # Write predictions
    with open(filepath, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + '\n')
    
    logger.info(f"Created submission file: {filepath}")
    return str(filepath)


def validate_submission_format(file_path: str, task: int) -> Dict[str, bool]:
    """
    Validate submission file format.
    
    Args:
        file_path: Path to submission file
        task: Task number (1, 2, 3)
    
    Returns:
        Dictionary with validation results
    """
    results = {
        'file_exists': False,
        'valid_json': True,
        'required_fields': True,
        'valid_va_format': True,
        'error_messages': []
    }
    
    if not Path(file_path).exists():
        results['file_exists'] = False
        results['error_messages'].append(f"File does not exist: {file_path}")
        return results
    
    results['file_exists'] = True
    
    required_fields = {
        1: ['ID', 'Aspect_VA'],
        2: ['ID', 'Triplet'],
        3: ['ID', 'Quadruplet']
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    results['valid_json'] = False
                    results['error_messages'].append(f"Invalid JSON at line {line_num}")
                    continue
                
                # Check required fields
                for field in required_fields[task]:
                    if field not in data:
                        results['required_fields'] = False
                        results['error_messages'].append(f"Missing field '{field}' at line {line_num}")
                
                # Validate VA format
                if task == 1 and 'Aspect_VA' in data:
                    for aspect_va in data['Aspect_VA']:
                        if 'VA' in aspect_va:
                            try:
                                parse_va_string(aspect_va['VA'])
                            except:
                                results['valid_va_format'] = False
                                results['error_messages'].append(f"Invalid VA format at line {line_num}")
    
    except Exception as e:
        results['error_messages'].append(f"Error reading file: {e}")
    
    return results


def get_supported_language_domain_pairs() -> Dict[str, List[str]]:
    """Get supported language-domain combinations."""
    return {
        'deu': ['stance'],
        'eng': ['restaurant', 'laptop', 'hotel', 'movie', 'stance', 'finance'],
        'hau': ['stance'],
        'ibo': ['stance'],
        'jpn': ['hotel', 'finance'],
        'kin': ['stance'],
        'ptb': ['restaurant'],
        'ptm': ['restaurant'],
        'rus': ['restaurant'],
        'swa': ['stance'],
        'tat': ['restaurant'],
        'twi': ['stance'],
        'ukr': ['restaurant'],
        'vmw': ['stance'],
        'xho': ['stance'],
        'zho': ['restaurant', 'laptop']
    }


def extract_language_domain_from_filename(filename: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract language and domain from filename."""
    pattern = r'pred_([a-z]{3})_([a-z]+)\.jsonl'
    match = re.match(pattern, filename)
    
    if match:
        return match.group(1), match.group(2)
    else:
        return None, None


def create_evaluation_report(
    metrics: Dict[str, float],
    output_path: str,
    model_name: str = "DimASR Model",
    task: int = 1,
    language: str = "unknown",
    domain: str = "unknown"
):
    """Create a formatted evaluation report."""
    report = {
        'model_name': model_name,
        'task': task,
        'language': language,
        'domain': domain,
        'metrics': metrics,
        'summary': {
            'primary_metric': metrics.get('avg_pearson', 0.0),
            'valence_performance': metrics.get('valence_pearson', 0.0),
            'arousal_performance': metrics.get('arousal_pearson', 0.0),
            'mse': metrics.get('mse', 0.0),
            'mae': metrics.get('mae', 0.0)
        }
    }
    
    # Save as JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Create human-readable summary
    summary_path = Path(output_path).with_suffix('.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"DimASR Evaluation Report\n")
        f.write(f"========================\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Task: {task}\n")
        f.write(f"Language: {language}\n")
        f.write(f"Domain: {domain}\n\n")
        f.write(f"Performance Metrics:\n")
        f.write(f"-------------------\n")
        f.write(f"Average Pearson Correlation: {metrics.get('avg_pearson', 0.0):.4f}\n")
        f.write(f"Valence Pearson Correlation: {metrics.get('valence_pearson', 0.0):.4f}\n")
        f.write(f"Arousal Pearson Correlation: {metrics.get('arousal_pearson', 0.0):.4f}\n")
        f.write(f"Mean Squared Error: {metrics.get('mse', 0.0):.4f}\n")
        f.write(f"Mean Absolute Error: {metrics.get('mae', 0.0):.4f}\n")
        
        if 'avg_cosine_similarity' in metrics:
            f.write(f"Average Cosine Similarity: {metrics['avg_cosine_similarity']:.4f}\n")
    
    logger.info(f"Evaluation report saved to {output_path}")


class ConfigManager:
    """Manage configuration for experiments."""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or "config.json"
        self.config = self._load_default_config()
        
        if Path(self.config_path).exists():
            self._load_config()
    
    def _load_default_config(self) -> Dict:
        """Load default configuration."""
        return {
            'model': {
                'name': 'bert-base-uncased',
                'max_length': 128,
                'dropout_rate': 0.1,
                'freeze_backbone': False
            },
            'training': {
                'learning_rate': 2e-5,
                'batch_size': 16,
                'num_epochs': 3,
                'warmup_steps': 100,
                'weight_decay': 0.01,
                'use_custom_loss': True
            },
            'data': {
                'train_data_path': '',
                'eval_data_path': '',
                'test_data_path': ''
            },
            'output': {
                'model_output_dir': './models',
                'results_output_dir': './results',
                'submission_output_dir': './submissions'
            }
        }
    
    def _load_config(self):
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                file_config = json.load(f)
            
            # Update default config with file config
            self._update_config(self.config, file_config)
            logger.info(f"Loaded configuration from {self.config_path}")
        
        except Exception as e:
            logger.warning(f"Could not load config from {self.config_path}: {e}")
    
    def _update_config(self, default: Dict, update: Dict):
        """Recursively update default config with new values."""
        for key, value in update.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._update_config(default[key], value)
            else:
                default[key] = value
    
    def save_config(self, path: str = None):
        """Save current configuration to file."""
        save_path = path or self.config_path
        
        with open(save_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"Configuration saved to {save_path}")
    
    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation."""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value):
        """Set configuration value using dot notation."""
        keys = key_path.split('.')
        config_ref = self.config
        
        for key in keys[:-1]:
            if key not in config_ref:
                config_ref[key] = {}
            config_ref = config_ref[key]
        
        config_ref[keys[-1]] = value


def main():
    """Example usage of utility functions."""
    # Test VA parsing
    va_string = "7.50#6.25"
    valence, arousal = parse_va_string(va_string)
    print(f"Parsed VA: valence={valence}, arousal={arousal}")
    
    # Test formatting
    formatted = format_va_string(valence, arousal)
    print(f"Formatted VA: {formatted}")
    
    # Test config manager
    config = ConfigManager()
    print(f"Default learning rate: {config.get('training.learning_rate')}")


if __name__ == "__main__":
    main()