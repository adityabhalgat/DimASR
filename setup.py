#!/usr/bin/env python3
"""
Setup script for DimASR project.
Initializes the project structure and validates the environment.
"""

import os
import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True


def install_dependencies():
    """Install project dependencies."""
    print("\n📦 Installing dependencies...")
    
    try:
        # Check if requirements.txt exists
        if not Path("requirements.txt").exists():
            print("❌ requirements.txt not found")
            return False
        
        # Install packages
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Dependencies installed successfully")
        return True
    
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        print("💡 Try manually: pip install -r requirements.txt")
        return False


def validate_project_structure():
    """Validate that the project structure is correct."""
    print("\n📁 Validating project structure...")
    
    required_dirs = [
        "src",
        "src/data_preprocessing",
        "src/models", 
        "src/training",
        "src/evaluation",
        "src/utils",
        "data",
        "models", 
        "results"
    ]
    
    required_files = [
        "README.md",
        "requirements.txt",
        "src/data_preprocessing/data_processor.py",
        "src/models/dimASR_transformer.py",
        "src/training/train_dimASR.py",
        "src/evaluation/evaluator.py",
        "src/utils/helpers.py"
    ]
    
    missing_dirs = []
    missing_files = []
    
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
    
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_dirs:
        print(f"❌ Missing directories: {', '.join(missing_dirs)}")
        return False
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        return False
    
    print("✅ Project structure is valid")
    return True


def check_trial_data():
    """Check if trial data is available."""
    print("\n📊 Checking trial data...")
    
    trial_files = [
        "task-dataset/trial/eng_restaurant_trial_alltasks.jsonl",
        "task-dataset/trial/zho_laptop_trial_alltasks.jsonl",
        "evaluation_script/metrics_subtask_1_2_3.py"
    ]
    
    missing_files = []
    for file_path in trial_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"⚠️  Missing trial data files: {', '.join(missing_files)}")
        print("💡 Some demo features may not work without trial data")
        return False
    
    print("✅ Trial data found")
    return True


def create_config_files():
    """Create default configuration files."""
    print("\n⚙️  Creating configuration files...")
    
    # Create .gitignore
    gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Project specific
models/*/
results/*/
submissions/*/
*.log
wandb/
demo_config.json

# Data files (large)
data/processed/
data/raw/

# Model checkpoints
*.pt
*.pth
*.pkl
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content.strip())
    
    # Create basic config.json
    config_content = """{
  "model": {
    "name": "bert-base-uncased",
    "max_length": 128,
    "dropout_rate": 0.1,
    "freeze_backbone": false
  },
  "training": {
    "learning_rate": 2e-5,
    "batch_size": 16,
    "num_epochs": 3,
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "use_custom_loss": true
  },
  "data": {
    "train_data_path": "",
    "eval_data_path": "",
    "test_data_path": ""
  },
  "output": {
    "model_output_dir": "./models",
    "results_output_dir": "./results", 
    "submission_output_dir": "./submissions"
  }
}"""
    
    if not Path("config.json").exists():
        with open("config.json", "w") as f:
            f.write(config_content)
        print("✅ Configuration files created")
    else:
        print("✅ Configuration files already exist")


def run_quick_test():
    """Run a quick test to verify the setup."""
    print("\n🧪 Running quick test...")
    
    try:
        # Test basic imports
        sys.path.append('src')
        
        # Test utility functions
        exec("""
import json
from pathlib import Path

def parse_va_string(va_string):
    try:
        valence_str, arousal_str = va_string.split('#')
        return float(valence_str), float(arousal_str)
    except:
        return 0.0, 0.0

# Test VA parsing
valence, arousal = parse_va_string("7.50#6.25")
assert valence == 7.50
assert arousal == 6.25
        """)
        
        print("✅ Basic functionality test passed")
        return True
    
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def main():
    """Main setup function."""
    print("🚀 DimASR Project Setup")
    print("=" * 50)
    
    # Run setup steps
    steps = [
        ("Python version", check_python_version),
        ("Project structure", validate_project_structure),
        ("Dependencies", install_dependencies),
        ("Trial data", check_trial_data),
        ("Config files", create_config_files),
        ("Quick test", run_quick_test)
    ]
    
    results = []
    for step_name, step_func in steps:
        try:
            result = step_func()
            results.append((step_name, result))
        except Exception as e:
            print(f"❌ {step_name} failed with error: {e}")
            results.append((step_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Setup Summary")
    print("=" * 50)
    
    all_passed = True
    for step_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{step_name:<20} {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Run demo: python demo.py")
        print("2. Explore the data and start training")
        print("3. Check the README.md for detailed instructions")
    else:
        print("\n⚠️  Setup completed with some issues")
        print("Please address the failed steps before proceeding")
    
    print("\n📚 Useful commands:")
    print("  python demo.py              # Run project demo")
    print("  python src/training/train_dimASR.py --help  # See training options")


if __name__ == "__main__":
    main()