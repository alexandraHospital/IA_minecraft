import argparse
import torch
from pathlib import Path
import logging
from datetime import datetime
from ultralytics import YOLO
import os
import sys

from building_analyzer.building_analyzer_trainer import BuildingAnalyzerTrainer as BAT


def building_analyzer_launcher():
     # Parser with arguments
    parser = argparse.ArgumentParser()
    print(f"--------{os.path.abspath(os.getcwd())}")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--data", type=str, default="data.yml" ,help="Path of yaml file containing classes, by default data.yml at the root directory of the module")
    parser.add_argument("--img_size", type=int, default=640, help="Image size")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--output_dir", type=str, default="models", help="Output directory for fine_tuned YOLO")

    args = parser.parse_args()

    IMAGE_SIZE = args.img_size
    
    if args.data == "data.yml":
        module_dir = Path(__file__).resolve().parent
        DATA_FILE = os.path.join(module_dir, "data.yml")
    else:
        if not os.path.exists(args.data):
            print(f"{args.data} does not exist")
            sys.exit(1)

    BATCH_SIZE = args.batch_size

    epochs = args.epochs
    
    device = 0 if torch.cuda.is_available() else "cpu"

    ###########################
    # Output dir
    ###########################
    OUTPUT_DIR = args.output_dir
    if OUTPUT_DIR == "models":
        output_dir = Path("output/models").resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(OUTPUT_DIR).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    #####################
    # Log instantiation #
    #####################
    path_logs = Path("output/logs")
    path_logs.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(__name__)
    today = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"{path_logs}/building_analyzer_{today}.log"

    logging.basicConfig(filename=log_filename, level=logging.INFO,
                        format="%(asctime)s %(name)s %(funcName)s %(levelname)s: %(message)s")

    logger.info(f"Started")
    logger.info(f"Fine Tune YOLO with:\n\
                \t device={device} \n\
                \t image_size={IMAGE_SIZE} \n\
                \t batch={BATCH_SIZE} \n\
                \t data={DATA_FILE} \n\
                \t epochs={epochs} \n\
                \t output_dir={str(output_dir)}")
    
    ########################
    # Create YOLO          #
    ########################
    
    model = YOLO("yolov8n.pt")
    trainer = BAT(model=model,
                  device=device,
                  image_size=IMAGE_SIZE,
                  batch=BATCH_SIZE,
                  epochs=epochs,
                  data=DATA_FILE,
                  logger=logger,
                  project=str(output_dir))
    trainer.train()

    
if __name__ == "__main__":
    sys.exit(building_analyzer_launcher())