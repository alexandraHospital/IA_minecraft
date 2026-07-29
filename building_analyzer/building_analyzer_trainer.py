from ultralytics import YOLO
import matplotlib.pyplot as plt
import os

class BuildingAnalyzerTrainer:
    def __init__(self, model, data, device, image_size, batch, logger, epochs, project, save_dir):
        self.model = model
        self.device = device
        self.image_size = image_size
        self.batch = batch
        self.logger = logger
        self.data = data
        self.epochs = epochs
        self.project = project
        self.save_dir = save_dir
    
    def train(self):
        self.model.train(data=self.data,
            epochs=self.epochs,
            imgsz=self.image_size,
            batch=self.batch,
            device=self.device, 
            project=self.project,
            save_dir = self.save_dir)
