# Building Analyzer

This module contains a trainer for a detection of elements on a building.
The trainer fine-tunes a YOLO object detection model.
The Ultranalytics Library is ised to fine-tune YOLO: https://docs.ultralytics.com/usage/cfg

## Prepare dataset

Download CMP facade datasets (base and extended) : https://cmp.felk.cvut.cz/~tylecr1/facade/
_from Tyleček, R., & Šára, R. (2013). Spatial Pattern Templates for Recognition of Objects with Regular Structure. Dans Proc. GCPR, Saarbrücken, Germany_


To train YOLO model, we need to transform .xml annotation files into .txt files.
Unzip dataset and launch prepare_dataset script:

```bash
cd IA_minecraft/
mkdir -p ./building_analyzer/dataset/cmp_facade
unzip -oj /path/to/CMP_facade_DB_base.zip -d dataset/cmp_facade && unzip -oj /path/to/CMP_facade_DB_extended.zip -d dataset/cmp_facade
python3 -m building_analyzer.prepare_dataset
```

### data.yml file

```
train: ./cmp/images/train
val: ./cmp/images/val
nc: 11  # Number of classes
names: ['facade', 'window', 'door', 'cornice', 'sill', 'balcony', 'blind', 'deco', 'molding', 'pillar', 'shop']  # Class names
```


## Usage
```bash
usage: building_analyzer_launcher.py [-h] [--epochs EPOCHS] [--data DATA] [--img_size IMG_SIZE] [--batch_size BATCH_SIZE] [--output_dir OUTPUT_DIR]

options:
  -h, --help            show this help message and exit
  --epochs EPOCHS       Number of epochs for training
  --data DATA           Path of yaml file containing classes, by default data.yml at the root directory of the module
  --img_size IMG_SIZE   Image size
  --batch_size BATCH_SIZE
                        Batch size
  --output_dir OUTPUT_DIR
                        Output directory for fine_tuned YOLO
```

Example:

```bash
python3 -m building_analyzer.building_analyzer_launcher --data ./path/to/data --batch_size 16 --epochs 100

```

