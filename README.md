# DimASR: Dimensional Aspect-Based Sentiment Analysis on Reviews

## Overview

DimASR is a shared task focusing on **Dimensional Aspect-Based Sentiment Analysis** with multilingual support. The task involves analyzing sentiment in reviews across multiple languages and domains using Valence-Arousal (VA) dimensional emotion modeling.

## Task Description

### Three Subtasks

1. **Subtask 1: Aspect Extraction with VA Prediction**
   - Input: Text + List of aspects
   - Output: Aspects with VA scores
   - Format: `Aspect_VA` containing aspect and VA values (e.g., "7.50#7.62")

2. **Subtask 2: Aspect-Opinion-Category Triplet Extraction with VA**
   - Input: Raw text only
   - Output: Triplets (Aspect, Opinion, Category) with VA scores
   - Format: `Triplet` data structure

3. **Subtask 3: Complete Quadruplet Extraction with VA**
   - Input: Raw text only
   - Output: Complete quadruplets (Aspect, Opinion, Category, VA)
   - Format: `Quadruplet` data structure

### Supported Languages & Domains

**Languages (16):** `deu`, `eng`, `hau`, `ibo`, `jpn`, `kin`, `ptb`, `ptm`, `rus`, `swa`, `tat`, `twi`, `ukr`, `vmw`, `xho`, `zho`

**Domains (6):** restaurant, laptop, hotel, movie, stance, finance

## Project Structure

```
DimASR/
├── README.md                           # This file
├── evaluation_script/                  # Official evaluation tools
│   ├── metrics_subtask_1_2_3.py       # Evaluation script
│   ├── requirements.txt                # Dependencies
│   └── sample_data/                    # Sample test data
├── task-dataset/                       # Trial datasets
│   └── trial/                          # Trial data for all tasks
├── sample_submission_files/            # Submission format examples
├── src/                                # Source code (to be created)
│   ├── data_preprocessing/             # Data cleaning and preprocessing
│   ├── models/                         # Model implementations
│   ├── training/                       # Training scripts
│   ├── evaluation/                     # Evaluation utilities
│   └── utils/                          # Helper functions
├── data/                               # Processed datasets
├── models/                             # Trained model checkpoints
├── results/                            # Experiment results
└── requirements.txt                    # Project dependencies
```

## Solution Approach

### 1. Data Understanding and Preprocessing
- **Data Analysis**: Examine VA score distributions and aspect patterns
- **Text Preprocessing**: Tokenization, normalization, handling multiple aspects
- **Feature Engineering**: Lexical features, syntactic features (POS, dependency parsing)

### 2. Model Architecture
- **Primary Approach**: Transformer-based models (BERT, RoBERTa, XLNet)
- **Input Format**: `[CLS] text [SEP] aspect [SEP]` for aspect-specific predictions
- **Output**: Regression layer predicting continuous VA scores
- **Loss Function**: MSE/MAE for regression task

### 3. Handling Multiple Aspects
For sentences with multiple aspects, create separate model inputs:
```
Input: {"Text": "Great food but terrible service", "Aspect": ["food", "service"]}
Model Inputs: 
- (text, "food") → VA scores for food
- (text, "service") → VA scores for service
```

### 4. Evaluation Metrics
- **Mean Squared Error (MSE)**
- **Pearson Correlation Coefficient**
- **Cosine Similarity** for VA vector alignment

## Data Format

### Valence-Arousal Scores
VA scores are formatted as `"valence#arousal"` (e.g., `"7.50#7.62"`)
- **Valence**: Emotional positivity/negativity
- **Arousal**: Emotional intensity/activation

### Sample Data Structure
```json
{
  "ID": "rest16_quad_dev_3",
  "Text": "the spicy tuna roll was unusually good and the rock shrimp tempura was awesome",
  "Quadruplet": [
    {
      "Aspect": "spicy tuna roll",
      "Opinion": "unusually good",
      "Category": "FOOD#QUALITY",
      "VA": "7.50#7.62"
    }
  ]
}
```

## Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running Evaluation
```bash
cd evaluation_script
python metrics_subtask_1_2_3.py -t 1 -p pred_file.jsonl -g gold_file.jsonl
```

### Submission Format
Files must be named: `pred_[lang_code]_[domain].jsonl`
- Example: `pred_eng_restaurant.jsonl`
- Organize in subtask folders and zip for submission

## Development Workflow

1. **Data Preprocessing**: Clean and prepare data
2. **Baseline Model**: Start with fine-tuned BERT
3. **Hyperparameter Tuning**: Optimize on development set
4. **Error Analysis**: Identify model weaknesses
5. **Iterative Improvement**: Enhance features/architecture
6. **Final Evaluation**: Test on held-out set

## License

This project is part of a shared task for computational linguistics research.

## Contributors

- Aditya Bhalgat (@adityabhalgat)