
# Material Classifier

This module contains a trainer for a classifier for Minecraft material used in construction.

## Dataset
You need a dataset with a training and a validation directory.
Classes are determined by the directories inside each training and validation folder.

For example:
```bash
dataset
├── training
│   ├── bricks
│   ├── cobblestone
│   ├── deepslate_bricks
│   ├── end_stone_brick
│   └── stone_bricks
└── validation
    ├── bricks
    ├── cobblestone
    ├── deepslate_bricks
    ├── end_stone_brick
    └── stone_bricks
```

## Installation

```bash
# Clone repository
git clone https://github.com/ton_user/IA_minecraft.git
cd IA_minecraft

# Create venv
python -m venv venv

# Activate venv
source venv/bin/activate

# Installing dependencies
pip install -r requirements.txt

```

## Usage
```bash
# Train model
python3 -m material_classifier.material_classifier_launcher --train_dir /path/to/dataset/training --val_dir /path/to/dataset/validation

# Print help
python3 -m material_classifier.material_classifier_launcher --help
  -h, --help            show this help message and exit
  --epochs EPOCHS       Number of epochs for training
  --lr LR               Learning rate
  --train_dir TRAIN_DIR
                        Dataset training directory
  --val_dir VAL_DIR     Dataset validation directory
  --threshold_acc THRESHOLD_ACC
                        Threshold above which the model is registered
  --img_width IMG_WIDTH
                        Image width for training/validation transform
  --img_height IMG_HEIGHT
                        Image height for training/validation transform
  --workers WORKERS     Number of workers
  --batch_size BATCH_SIZE
                        Batch size
  --save_errors SAVE_ERRORS
                        Save error images in directory
                        
```

### Save model
Model with test/validation accuracy higher than THRESHOLD_ACC are saved in ./output/model
