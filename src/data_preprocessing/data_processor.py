"""
Data preprocessing utilities for DimASR task.
Handles data loading, cleaning, and preparation for model training.
"""

import json
import re
import pandas as pd
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DimASRDataProcessor:
    """Data processor for DimASR task."""
    
    def __init__(self):
        self.supported_languages = [
            'deu', 'eng', 'hau', 'ibo', 'jpn', 'kin', 'ptb', 
            'ptm', 'rus', 'swa', 'tat', 'twi', 'ukr', 'vmw', 'xho', 'zho'
        ]
        self.supported_domains = [
            'restaurant', 'laptop', 'hotel', 'movie', 'stance', 'finance'
        ]
    
    def load_jsonl(self, file_path: str) -> List[Dict]:
        """Load data from JSONL file."""
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON parsing error at line {line_num}: {e}")
                        continue
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
        return data
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        
        # Basic cleaning
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single space
        text = text.strip()
        
        # Handle contractions (basic)
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
    
    def parse_va_scores(self, va_string: str) -> Tuple[float, float]:
        """Parse VA scores from string format 'valence#arousal'."""
        try:
            valence, arousal = va_string.split('#')
            return float(valence), float(arousal)
        except (ValueError, AttributeError):
            logger.warning(f"Invalid VA format: {va_string}")
            return 0.0, 0.0
    
    def format_va_scores(self, valence: float, arousal: float) -> str:
        """Format VA scores to string format 'valence#arousal'."""
        return f"{valence:.2f}#{arousal:.2f}"
    
    def prepare_subtask1_data(self, data: List[Dict]) -> List[Dict]:
        """Prepare data for Subtask 1 (Aspect + VA prediction)."""
        processed_data = []
        
        for item in data:
            text = self.clean_text(item.get('Text', ''))
            
            # Handle both test and gold data formats
            if 'Aspect' in item:  # Test data format
                aspects = item['Aspect']
                for aspect in aspects:
                    processed_data.append({
                        'id': item['ID'],
                        'text': text,
                        'aspect': aspect.lower().strip(),
                        'valence': None,
                        'arousal': None
                    })
            elif 'Aspect_VA' in item:  # Gold data format
                for aspect_va in item['Aspect_VA']:
                    aspect = aspect_va['Aspect'].lower().strip()
                    valence, arousal = self.parse_va_scores(aspect_va['VA'])
                    processed_data.append({
                        'id': item['ID'],
                        'text': text,
                        'aspect': aspect,
                        'valence': valence,
                        'arousal': arousal
                    })
        
        return processed_data
    
    def prepare_subtask2_data(self, data: List[Dict]) -> List[Dict]:
        """Prepare data for Subtask 2 (Triplet extraction)."""
        processed_data = []
        
        for item in data:
            text = self.clean_text(item.get('Text', ''))
            
            if 'Quadruplet' in item:  # Gold data with full quadruplets
                for quad in item['Quadruplet']:
                    aspect = quad['Aspect'].lower().strip()
                    opinion = quad['Opinion'].lower().strip()
                    category = quad['Category'].lower().strip()
                    valence, arousal = self.parse_va_scores(quad['VA'])
                    
                    processed_data.append({
                        'id': item['ID'],
                        'text': text,
                        'aspect': aspect,
                        'opinion': opinion,
                        'category': category,
                        'valence': valence,
                        'arousal': arousal
                    })
            else:  # Test data (text only)
                processed_data.append({
                    'id': item['ID'],
                    'text': text,
                    'aspect': None,
                    'opinion': None,
                    'category': None,
                    'valence': None,
                    'arousal': None
                })
        
        return processed_data
    
    def prepare_subtask3_data(self, data: List[Dict]) -> List[Dict]:
        """Prepare data for Subtask 3 (Full quadruplet extraction)."""
        # Same as subtask 2 for now
        return self.prepare_subtask2_data(data)
    
    def create_model_inputs(self, processed_data: List[Dict], task: int = 1) -> List[Dict]:
        """Create model input format for transformer models."""
        model_inputs = []
        
        for item in processed_data:
            if task == 1:
                # Format: [CLS] text [SEP] aspect [SEP]
                input_text = f"{item['text']} [SEP] {item['aspect']}"
                model_inputs.append({
                    'input_text': input_text,
                    'text': item['text'],
                    'aspect': item['aspect'],
                    'valence': item['valence'],
                    'arousal': item['arousal'],
                    'id': item['id']
                })
            elif task in [2, 3]:
                # For subtasks 2 and 3, we need to handle full text analysis
                model_inputs.append({
                    'input_text': item['text'],
                    'text': item['text'],
                    'aspect': item['aspect'],
                    'opinion': item['opinion'],
                    'category': item['category'],
                    'valence': item['valence'],
                    'arousal': item['arousal'],
                    'id': item['id']
                })
        
        return model_inputs
    
    def save_processed_data(self, data: List[Dict], output_path: str):
        """Save processed data to file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(data)} processed items to {output_path}")


def main():
    """Example usage of the data processor."""
    processor = DimASRDataProcessor()
    
    # Example: Process trial data for subtask 1
    trial_data = processor.load_jsonl('../task-dataset/trial/eng_restaurant_trial_alltasks.jsonl')
    
    # Process for subtask 1
    processed_subtask1 = processor.prepare_subtask1_data(trial_data)
    model_inputs = processor.create_model_inputs(processed_subtask1, task=1)
    
    print(f"Loaded {len(trial_data)} trial items")
    print(f"Created {len(model_inputs)} model inputs for subtask 1")
    
    # Show example
    if model_inputs:
        print("\nExample model input:")
        print(json.dumps(model_inputs[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()