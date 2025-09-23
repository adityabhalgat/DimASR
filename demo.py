"""
Demo script for DimASR project.
Shows basic usage of data processing, model creation, and evaluation.
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.append('src')

# Import functions directly to avoid import issues
def parse_va_string(va_string):
    """Parse VA scores from string format 'valence#arousal'."""
    try:
        valence_str, arousal_str = va_string.split('#')
        return float(valence_str), float(arousal_str)
    except (ValueError, AttributeError):
        print(f"Warning: Invalid VA format: {va_string}")
        return 0.0, 0.0

def format_va_string(valence, arousal):
    """Format VA scores to string format 'valence#arousal'."""
    return f"{valence:.2f}#{arousal:.2f}"

def calculate_va_statistics(va_scores):
    """Calculate statistics for VA scores."""
    if not va_scores:
        return {}
    
    import numpy as np
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

# Simple data processor class for demo
class SimpleDimASRDataProcessor:
    """Simplified data processor for demo purposes."""
    
    def load_jsonl(self, file_path):
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
                        print(f"JSON parsing error at line {line_num}: {e}")
                        continue
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        return data

# Simple config class for demo
class SimpleConfigManager:
    """Simplified config manager for demo purposes."""
    
    def __init__(self):
        self.config = {
            'model': {
                'name': 'bert-base-uncased',
                'max_length': 128
            },
            'training': {
                'learning_rate': 2e-5,
                'batch_size': 16
            }
        }
    
    def get(self, key_path, default=None):
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def set(self, key_path, value):
        keys = key_path.split('.')
        config_ref = self.config
        for key in keys[:-1]:
            if key not in config_ref:
                config_ref[key] = {}
            config_ref = config_ref[key]
        config_ref[keys[-1]] = value
    
    def save_config(self, path):
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=2)

def explore_trial_data():
    """Explore the trial data to understand the task."""
    print("=" * 60)
    print("DimASR Project - Data Exploration")
    print("=" * 60)
    
    processor = SimpleDimASRDataProcessor()
    
    # Load trial data
    trial_files = [
        'task-dataset/trial/eng_restaurant_trial_alltasks.jsonl',
        'task-dataset/trial/zho_laptop_trial_alltasks.jsonl',
        'task-dataset/trial/ger_stance_trial_task1task2.jsonl'
    ]
    
    for file_path in trial_files:
        if not Path(file_path).exists():
            print(f"⚠️  File not found: {file_path}")
            continue
            
        print(f"\n📁 Loading: {file_path}")
        data = processor.load_jsonl(file_path)
        print(f"   Loaded {len(data)} items")
        
        if data:
            # Show first item
            print("   Sample item:")
            print(f"   ID: {data[0]['ID']}")
            print(f"   Text: {data[0]['Text'][:100]}...")
            
            # Extract VA scores for analysis
            va_scores = []
            for item in data:
                if 'Quadruplet' in item:
                    for quad in item['Quadruplet']:
                        if 'VA' in quad:
                            valence, arousal = parse_va_string(quad['VA'])
                            va_scores.append((valence, arousal))
            
            if va_scores:
                stats = calculate_va_statistics(va_scores)
                print(f"   VA Statistics:")
                print(f"     Valence: μ={stats['valence_mean']:.2f}, σ={stats['valence_std']:.2f}")
                print(f"     Arousal: μ={stats['arousal_mean']:.2f}, σ={stats['arousal_std']:.2f}")
                print(f"     Total VA pairs: {stats['count']}")


def demo_data_processing():
    """Demonstrate data preprocessing for different subtasks."""
    print("\n" + "=" * 60)
    print("Data Processing Demo")
    print("=" * 60)
    
    processor = SimpleDimASRDataProcessor()
    
    # Load sample data
    trial_file = 'task-dataset/trial/eng_restaurant_trial_alltasks.jsonl'
    if not Path(trial_file).exists():
        print("⚠️  Trial data not found. Please check the data path.")
        return
    
    data = processor.load_jsonl(trial_file)
    print(f"Loaded {len(data)} trial items")
    
    # Process for different subtasks
    for task in [1, 2, 3]:
        print(f"\n📊 Processing for Subtask {task}")
        
        # Simplified processing for demo
        print(f"   Would process {len(data)} items for subtask {task}")
        print("   (Full processing requires complete implementation)")
        
        if data:
            print("   Sample processing:")
            sample = data[0]
            print(f"     Input text: {sample['Text'][:80]}...")
            
            # Show different processing for each task
            if task == 1:
                print("     Task 1: Extract aspects and predict VA scores")
                if 'Quadruplet' in sample:
                    aspects = [q['Aspect'] for q in sample['Quadruplet']]
                    print(f"     Found aspects: {aspects[:3]}...")
            elif task == 2:
                print("     Task 2: Extract aspect-opinion-category triplets")
                if 'Quadruplet' in sample:
                    print(f"     Would extract triplets from quadruplets")
            else:
                print("     Task 3: Extract complete quadruplets")
                if 'Quadruplet' in sample:
                    print(f"     Found {len(sample['Quadruplet'])} quadruplets")


def demo_evaluation_data():
    """Demonstrate evaluation data formats."""
    print("\n" + "=" * 60)
    print("Evaluation Data Demo")
    print("=" * 60)
    
    processor = SimpleDimASRDataProcessor()
    
    # Check sample evaluation data
    eval_files = [
        'evaluation_script/sample data/subtask_1/eng/test_eng_restaurant.jsonl',
        'evaluation_script/sample data/subtask_1/eng/gold_eng_restaurant.jsonl'
    ]
    
    for file_path in eval_files:
        if not Path(file_path).exists():
            print(f"⚠️  File not found: {file_path}")
            continue
        
        print(f"\n📁 {file_path.split('/')[-1]}:")
        data = processor.load_jsonl(file_path)
        
        if data:
            print(f"   Items: {len(data)}")
            print(f"   Sample: {json.dumps(data[0], ensure_ascii=False, indent=4)}")


def demo_config_management():
    """Demonstrate configuration management."""
    print("\n" + "=" * 60)
    print("Configuration Management Demo")
    print("=" * 60)
    
    # Create config manager
    config = SimpleConfigManager()
    
    print("📋 Default Configuration:")
    print(f"   Model: {config.get('model.name')}")
    print(f"   Learning Rate: {config.get('training.learning_rate')}")
    print(f"   Batch Size: {config.get('training.batch_size')}")
    print(f"   Max Length: {config.get('model.max_length')}")
    
    # Modify config
    config.set('training.learning_rate', 1e-5)
    config.set('model.name', 'roberta-base')
    
    print("\n📝 Modified Configuration:")
    print(f"   Model: {config.get('model.name')}")
    print(f"   Learning Rate: {config.get('training.learning_rate')}")
    
    # Save config
    config.save_config('demo_config.json')
    print("   ✅ Configuration saved to demo_config.json")


def show_project_structure():
    """Show the project structure."""
    print("\n" + "=" * 60)
    print("Project Structure")
    print("=" * 60)
    
    print("""
📁 DimASR/
├── 📄 README.md                       # Project documentation
├── 📄 requirements.txt                # Python dependencies
├── 📁 src/                            # Source code
│   ├── 📁 data_preprocessing/         # Data processing utilities
│   │   └── 📄 data_processor.py       # Main data processor
│   ├── 📁 models/                     # Model implementations
│   │   └── 📄 dimASR_transformer.py   # Transformer-based models
│   ├── 📁 training/                   # Training scripts
│   │   └── 📄 train_dimASR.py         # Main training script
│   ├── 📁 evaluation/                 # Evaluation utilities
│   │   └── 📄 evaluator.py            # Model evaluation
│   └── 📁 utils/                      # Helper functions
│       └── 📄 helpers.py              # Utility functions
├── 📁 evaluation_script/              # Official evaluation
├── 📁 task-dataset/                   # Trial datasets
├── 📁 data/                           # Processed datasets
├── 📁 models/                         # Trained models
└── 📁 results/                        # Experiment results
    """)


def main():
    """Run the complete demo."""
    print("🚀 DimASR Project Demo")
    print("This demo shows the basic workflow for the DimASR project.")
    
    try:
        show_project_structure()
        explore_trial_data()
        demo_data_processing()
        demo_evaluation_data()
        demo_config_management()
        
        print("\n" + "=" * 60)
        print("Demo Complete! 🎉")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Explore the data in more detail")
        print("3. Train a baseline model using src/training/train_dimASR.py")
        print("4. Evaluate results and iterate")
        print("\nFor training:")
        print("python src/training/train_dimASR.py --train_data <path> --task 1")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        print("This is likely due to missing dependencies or data files.")
        print("Please install requirements: pip install -r requirements.txt")


if __name__ == "__main__":
    main()