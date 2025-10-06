#!/usr/bin/env python3
"""
Generate Fixed Predictions - Corrected Version
"""

import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class QuickFixModel(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        
        self.dropout = nn.Dropout(0.2)
        self.regressor = nn.Linear(hidden_size, 2)
        
        nn.init.xavier_uniform_(self.regressor.weight)
        nn.init.constant_(self.regressor.bias, 5.0)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        
        x = self.dropout(pooled_output)
        x = self.regressor(x)
        
        # FIXED: Proper scaling to 0-10 range
        valence = torch.sigmoid(x[:, 0]) * 10.0
        arousal = torch.sigmoid(x[:, 1]) * 10.0
        
        return valence, arousal

def generate_fixed_predictions():
    """Generate predictions with the fixed model"""
    
    print("🎯 Generating FIXED predictions...")
    
    # Load the trained model
    model = QuickFixModel()
    checkpoint = torch.load('results/quick_fix_model.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model.eval()
    
    # Load test data (corrected format)
    test_file = "/Users/adityabhalgat/Developer/DimASR/evaluation_script/sample data/subtask_1/eng/test_eng_restaurant.jsonl"
    
    predictions = []
    with open(test_file, 'r') as f:
        for line in f:
            test_item = json.loads(line.strip())
            doc_id = test_item['ID']
            text = test_item['Text']
            aspects = test_item['Aspect']  # Corrected: 'Aspect' not 'Aspects'
            
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
                    input_ids = encoding['input_ids']
                    attention_mask = encoding['attention_mask']
                    
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
    import os
    os.makedirs('results', exist_ok=True)
    output_file = 'results/pred_eng_restaurant_FIXED.jsonl'
    
    with open(output_file, 'w') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\\n')
    
    print(f"✅ Fixed predictions saved to: {output_file}")
    
    # Show preview
    print("\\n📊 Preview of FIXED predictions:")
    for i, pred in enumerate(predictions[:5]):
        if pred['Aspect_VA']:
            va_value = pred['Aspect_VA'][0]['VA']
            valence, arousal = va_value.split('#')
            print(f"  Sample {i+1}: V={valence}, A={arousal} (was ~2.0#2.7 before)")
    
    return output_file

if __name__ == "__main__":
    pred_file = generate_fixed_predictions()
    
    print("\\n" + "=" * 60)
    print("🎉 FIXED PREDICTIONS GENERATED!")
    print("=" * 60)
    print("🔧 Key Fix Applied: Output scaling with sigmoid * 10")
    print(f"📁 File: {pred_file}")
    print("\\n🔍 Next Steps:")
    print("1. Compare with original predictions")
    print("2. Run official evaluation")
    print("3. Check if accuracy improved")
    print("=" * 60)