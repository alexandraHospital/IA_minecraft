# Building Analyzer

This module contains a trainer for a detection of elements on a building.
The trainer fine-tunes a YOLO object detection model.

## Prepare dataset
_WIP - transform CMP facade dataset annotation xml file into txt file for YOLO model_

### data.yml file
_TODO: example of data file_

## Usage

```bash
python3 -m building_analyzer.building_analyzer_launcher --data ./path/to/data --batch_size 32 --epochs 100

```