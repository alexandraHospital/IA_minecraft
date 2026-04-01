import argparse
import torch
from pathlib import Path
import logging
from datetime import datetime
from ultralytics import YOLO

from building_analyzer.building_analyzer_trainer import BuildingAnalyzerTrainer as BAT


def building_analyzer_launcher():
     # Parser with arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs for training")
    parser.add_argument("--data", type=float, default=1e-4, help="Path of yaml file containing classes")
    # parser.add_argument("--train_dir", type=str, required=True, help="Dataset training directory")
    # parser.add_argument("--val_dir", type=str, required=True, help="Dataset validation directory")
    # parser.add_argument("--threshold_acc",  type=float, default=0.90, help="Threshold above which the model is registered")
    parser.add_argument("--img_size", type=int, default=640, help="Image size")
    # parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    # parser.add_argument("--save_errors", type=bool, default=False, help="Save error images in directory")

    args = parser.parse_args()

    IMAGE_SIZE = args.img_size
    DATA_FILE = args.data

    BATCH_SIZE = args.batch_size

    epochs = args.epochs
    
    device = 0 if torch.cuda.is_available() else "cpu"
    
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
                  logger=logger)
    trainer.train()

    
if __name__ == "__main__":
    building_analyzer_launcher()