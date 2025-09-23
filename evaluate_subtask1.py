#!/usr/bin/env python3
"""
Generate predictions for Subtask 1 and run official evaluation.
"""

import json
import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append('src')

from transformers import AutoTokenizer
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class SimpleDimASRModel(nn.Module):
    """Simple BERT-based model for VA regression."""
    
    def __init__(self, model_name="bert-base-uncased"):
        super().__init__()
        
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.regressor = nn.Linear(self.backbone.config.hidden_size, 2)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.last_hidden_state[:, 0])
        predictions = self.regressor(pooled_output)
        return {'predictions': predictions}


def load_model(model_path, model_name="bert-base-uncased"):
    """Load trained model."""
    model = SimpleDimASRModel(model_name)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model


def generate_predictions(model, tokenizer, test_data, output_file):
    """Generate predictions in the required format."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    predictions = []
    
    # Group by ID
    id_to_aspects = {}
    for item in test_data:
        item_id = item['ID']
        if item_id not in id_to_aspects:
            id_to_aspects[item_id] = {
                'ID': item_id,
                'Text': item['Text'],
                'aspects': []
            }
        id_to_aspects[item_id]['aspects'].extend(item['Aspect'])
    
    print(f"🔮 Generating predictions for {len(id_to_aspects)} items...")
    
    with torch.no_grad():
        for item_id, data in id_to_aspects.items():
            text = data['Text']
            aspects = data['aspects']
            
            aspect_va_predictions = []
            
            for aspect in aspects:
                # Create input
                input_text = f"{text} [SEP] {aspect}"
                
                # Tokenize
                encoding = tokenizer(
                    input_text,
                    truncation=True,
                    padding='max_length',
                    max_length=128,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)
                
                # Predict
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                prediction = outputs['predictions'].cpu().numpy()[0]
                
                aspect_va_predictions.append({
                    'Aspect': aspect,
                    'VA': f"{prediction[0]:.2f}#{prediction[1]:.2f}"
                })
            
            predictions.append({
                'ID': item_id,
                'Aspect_VA': aspect_va_predictions
            })
    
    # Save predictions
    with open(output_file, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + '\n')
    
    print(f"✅ Predictions saved to {output_file}")
    return predictions


def run_official_evaluation(pred_file, gold_file):
    """Run the official evaluation script."""
    import subprocess
    import sys
    
    cmd = [
        sys.executable,
        'evaluation_script/metrics_subtask_1_2_3.py',
        '-t', '1',
        '-p', pred_file,
        '-g', gold_file
    ]
    
    print(f"🏃 Running official evaluation...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd='.')
        
        if result.returncode == 0:
            print("✅ Official evaluation completed!")
            print("\n📊 Results:")
            print(result.stdout)
        else:
            print("❌ Evaluation failed!")
            print("Error:", result.stderr)
            print("Output:", result.stdout)
        
        return result.returncode == 0
    
    except Exception as e:
        print(f"❌ Error running evaluation: {e}")
        return False


def main():
    """Main evaluation function."""
    print("🔍 DimASR Subtask 1 - Generate Predictions and Evaluate")
    print("=" * 60)
    
    # Paths
    model_path = "models/subtask1_english_demo/model.pt"
    tokenizer_path = "models/subtask1_english_demo"
    test_file = "evaluation_script/sample data/subtask_1/eng/test_eng_restaurant.jsonl"
    gold_file = "evaluation_script/sample data/subtask_1/eng/gold_eng_restaurant.jsonl"
    pred_file = "results/pred_eng_restaurant_subtask1.jsonl"
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print("Please run train_subtask1_english.py first to train a model.")
        return
    
    # Check if test file exists
    if not Path(test_file).exists():
        print(f"❌ Test file not found: {test_file}")
        return
    
    # Load model and tokenizer
    print("🤖 Loading trained model...")
    model = load_model(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Load test data
    print(f"📁 Loading test data from {test_file}")
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                test_data.append(json.loads(line))
    
    print(f"✅ Loaded {len(test_data)} test items")
    
    # Create results directory
    Path("results").mkdir(exist_ok=True)
    
    # Generate predictions
    predictions = generate_predictions(model, tokenizer, test_data, pred_file)
    
    # Show sample prediction
    if predictions:
        print(f"\n📋 Sample prediction:")
        print(json.dumps(predictions[0], indent=2, ensure_ascii=False))
    
    # Run official evaluation
    print("\n" + "=" * 60)
    if Path(gold_file).exists():
        success = run_official_evaluation(pred_file, gold_file)
        if success:
            print("\n🎉 Evaluation completed successfully!")
        else:
            print("\n⚠️  Evaluation had issues, but predictions were generated.")
    else:
        print(f"⚠️  Gold file not found: {gold_file}")
        print(f"✅ Predictions generated at: {pred_file}")


if __name__ == "__main__":
    main()