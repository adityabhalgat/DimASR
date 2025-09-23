# DimASR Quick Start Guide

This guide will help you get started with the DimASR project quickly.

## Prerequisites

- Python 3.7 or higher
- Git (for version control)

## Setup

1. **Clone/Navigate to the project directory**
   ```bash
   cd /path/to/DimASR
   ```

2. **Run the setup script**
   ```bash
   python setup.py
   ```
   This will:
   - Check your Python version
   - Install dependencies
   - Validate project structure
   - Create configuration files

3. **Run the demo**
   ```bash
   python demo.py
   ```
   This will show you how the data processing and utilities work.

## Quick Training Example

### Subtask 1: Aspect + VA Prediction

```bash
# Train on sample data (adjust paths as needed)
python src/training/train_dimASR.py \
    --train_data evaluation_script/sample_data/subtask_1/eng/gold_eng_restaurant.jsonl \
    --eval_data evaluation_script/sample_data/subtask_1/eng/gold_eng_restaurant.jsonl \
    --task 1 \
    --model_name bert-base-uncased \
    --batch_size 8 \
    --num_epochs 2 \
    --output_dir ./models/subtask1_demo
```

### Data Processing Pipeline

```python
from src.data_preprocessing.data_processor import DimASRDataProcessor

# Initialize processor
processor = DimASRDataProcessor()

# Load data
data = processor.load_jsonl('your_data.jsonl')

# Process for specific subtask
if task == 1:
    processed = processor.prepare_subtask1_data(data)
elif task == 2:
    processed = processor.prepare_subtask2_data(data)
else:
    processed = processor.prepare_subtask3_data(data)

# Create model inputs
model_inputs = processor.create_model_inputs(processed, task=task)
```

### Model Creation

```python
from src.models.dimASR_transformer import DimASRTransformer
from transformers import AutoTokenizer

# Create model and tokenizer
model = DimASRTransformer(model_name="bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# For inference
evaluator = DimASREvaluator(model, tokenizer)
predictions = evaluator.predict_va_scores(texts, aspects)
```

## Project Structure Overview

```
DimASR/
├── src/
│   ├── data_preprocessing/     # Data loading and preprocessing
│   ├── models/                 # Model architectures
│   ├── training/               # Training scripts
│   ├── evaluation/             # Evaluation utilities
│   └── utils/                  # Helper functions
├── evaluation_script/          # Official evaluation tools
├── task-dataset/              # Trial datasets
├── models/                    # Saved model checkpoints
├── results/                   # Experiment results
└── config.json               # Configuration file
```

## Key Files

- **`src/data_preprocessing/data_processor.py`**: Data loading and preprocessing
- **`src/models/dimASR_transformer.py`**: Transformer-based models
- **`src/training/train_dimASR.py`**: Main training script
- **`src/evaluation/evaluator.py`**: Model evaluation and inference
- **`src/utils/helpers.py`**: Utility functions
- **`demo.py`**: Demonstration script
- **`setup.py`**: Project setup

## Configuration

Edit `config.json` to customize:
- Model parameters (name, dropout, etc.)
- Training hyperparameters (learning rate, batch size, etc.)
- Data paths
- Output directories

## Common Tasks

### 1. Data Exploration
```bash
python demo.py  # See data exploration section
```

### 2. Training a Model
```bash
python src/training/train_dimASR.py \
    --train_data <train_file> \
    --eval_data <eval_file> \
    --task <1|2|3> \
    --output_dir ./models/my_model
```

### 3. Generating Predictions
```python
# Load trained model
model, tokenizer = load_model_from_checkpoint("./models/my_model")

# Create evaluator
evaluator = DimASREvaluator(model, tokenizer)

# Generate predictions
predictions = evaluator.generate_predictions_subtask1(test_data)
```

### 4. Running Official Evaluation
```bash
cd evaluation_script
python metrics_subtask_1_2_3.py \
    -t 1 \
    -p path/to/predictions.jsonl \
    -g path/to/gold.jsonl
```

### 5. Creating Submission Files
```python
from src.utils.helpers import create_submission_file

create_submission_file(
    predictions=predictions,
    output_path="./submissions",
    task=1,
    language="eng",
    domain="restaurant"
)
```

## Troubleshooting

### Import Errors
If you get import errors:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Missing Dependencies
```bash
pip install -r requirements.txt
```

### CUDA Issues
For GPU training, ensure you have the correct PyTorch version:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Next Steps

1. **Explore the data**: Run `python demo.py` to understand the data format
2. **Train a baseline**: Start with a simple BERT model on Subtask 1
3. **Experiment**: Try different models, hyperparameters, and preprocessing
4. **Evaluate**: Use the official evaluation script to measure performance
5. **Iterate**: Analyze errors and improve your approach

## Tips for Success

1. **Start Simple**: Begin with Subtask 1 and a basic BERT model
2. **Data Quality**: Spend time understanding and cleaning your data
3. **Hyperparameter Tuning**: Learning rate and batch size are crucial
4. **Error Analysis**: Look at specific examples where your model fails
5. **Multilingual**: Consider language-specific models for non-English tasks

## Support

- Check the main `README.md` for detailed documentation
- Look at the code comments for implementation details
- Use the demo script to understand the workflow
- Refer to the evaluation script for format requirements

Good luck with your DimASR project! 🚀