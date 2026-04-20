# Building Analyzer

This module contains a trainer for a detection of elements on a building.
The trainer fine-tunes a YOLO object detection model.

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

<<<<<<< HEAD
### data.yml file

```
=======
One the script is executed, you'll have a dataset such as:
```
└── cmp
    ├── images
    │   ├── train
    │   └── val
    ├── labels
    │   ├── train
    │   └── val
    └── mask
```
With a ratio 80-20 for training-validation.

### data.yml file
This file is used with YOLO fine-tuning:

```yaml
>>>>>>> 789eec5 (add building analyzer readme + data)
train: ./cmp/images/train
val: ./cmp/images/val
nc: 11  # Number of classes
names: ['facade', 'window', 'door', 'cornice', 'sill', 'balcony', 'blind', 'deco', 'molding', 'pillar', 'shop']  # Class names
```

## Usage

```bash
python3 -m building_analyzer.building_analyzer_launcher --data ./path/to/data --batch_size 32 --epochs 100

```