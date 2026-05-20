import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import random
from PIL import Image
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import logging
import sys
import seaborn as sns
from datetime import datetime
from torchinfo import summary
from tqdm import tqdm
from material_classifier.material_classifier import MaterialClassifier as MC
from material_classifier.classifier_trainer import ClassifierTrainer as CT
from timeit import default_timer as timer
from utils.plots import plot_loss_curves


def material_classifier_launcher():

    # Parser with arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--train_dir", type=str, required=True, help="Dataset training directory")
    parser.add_argument("--val_dir", type=str, required=True, help="Dataset validation directory")
    parser.add_argument("--threshold_acc",  type=float, default=0.90, help="Threshold above which the model is registered")
    parser.add_argument("--img_width", type=int, default=224, help="Image width for training/validation transform")
    parser.add_argument("--img_height", type=int, default=224, help="Image height for training/validation transform")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--save_errors", type=bool, default=False, help="Save error images in directory")

    args = parser.parse_args()

    IMAGE_WIDTH = args.img_width
    IMAGE_HEIGHT = args.img_height
    IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
    TRAIN_DIR = args.train_dir
    VAL_DIR = args.val_dir
    NUM_WORKERS = args.workers
    BATCH_SIZE = args.batch_size
    LR = args.lr

    epochs = args.epochs
    threshold = args.threshold_acc
    save_err = args.save_errors
    
    # Setup device-agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    ######################
    # Create output path #
    ######################
    output_path = Path("output")
    output_path.mkdir(parents=True, exist_ok=True)


    #####################
    # Log instantiation #
    #####################
    path_logs = Path("output/logs")
    path_logs.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(__name__)
    today = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"{path_logs}/material_recognizer_{today}.log"

    logging.basicConfig(filename=log_filename, level=logging.INFO,
                        format="%(asctime)s %(name)s %(funcName)s %(levelname)s: %(message)s")

    logger.info(f"Started")
    logger.info(f"\t executable: {sys.executable}\n\
                \t numpy version: {np.__version__}\n\
                \t pytorch version: {torch.__version__}\n\
                \t device: {device}")

    torch.manual_seed(42)

    ##################
    # Transform Data #
    ##################
    # Create training transform with TrivialAugment
    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor()])

    # Create testing transform (no data augmentation)
    validation_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()])

    #########################
    # Creating training set #
    #########################
    train_data_augmented = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    logger.info(f"{train_data_augmented}")

    ###########################
    # Creating validation set #
    ###########################
    validation_data_augmented = datasets.ImageFolder(VAL_DIR, transform=validation_transform)
    logger.info(f"{validation_data_augmented}")
    
    # Get class names as a list
    class_names = train_data_augmented.classes
    logger.info(f"Class names: {class_names}")

    # Can also get class names as a dict
    class_dict = train_data_augmented.class_to_idx
    logger.info(f"Class names as a dict: {class_dict}")

    # Check the lengths
    logger.info(f"The lengths of the training and test sets: {len(train_data_augmented)} {len(validation_data_augmented)}")

    ##############
    # DataLoader #
    ##############
    # Turn train and test Datasets into DataLoaders
    train_dataloader_augmented = DataLoader(dataset=train_data_augmented, 
                                batch_size=BATCH_SIZE,
                                num_workers=NUM_WORKERS,
                                shuffle=True)

    validation_dataloader_augmented = DataLoader(dataset=validation_data_augmented, 
                                batch_size=BATCH_SIZE,
                                num_workers=NUM_WORKERS, 
                                shuffle=False)

    logger.info(f"{train_dataloader_augmented}")
    logger.info(f"{validation_dataloader_augmented}")


    ###############
    # Create CNN #
    ##############

    # Instantiate an object.
    model = MC().to(device)

    # print summary
    mode_summary = summary(model, input_size=[1, 3, IMAGE_WIDTH ,IMAGE_HEIGHT])
    logger.info(f"{mode_summary}")

    # Set random seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Setup loss function and optimizer
    # Setup a trainer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)
    trainer = CT(model, 
                 train_dataloader_augmented, 
                 validation_dataloader_augmented, 
                 optimizer, 
                 loss_fn, 
                 device, 
                 logger)

    logger.info(f"Train model with:\n\tlr: {LR}\n\tepochs: {epochs}\n\tthreshold: {threshold}")

    # Start the timer
    start_time = timer()

    # Train model_0 
    model_results = trainer.train(epochs, threshold, save_err)

    # End the timer and print out how long it took
    end_time = timer()
    logger.info(f"Total training time: {end_time-start_time:.3f} seconds")

    # Plot curves
    out_plot_name = f"plots_{today}.png"
    plot_loss_curves(model_results, out_plot_name)

    logger.info('Finished')


if __name__ == "__main__":
    material_classifier_launcher()