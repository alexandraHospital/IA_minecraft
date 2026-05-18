from ultralytics import YOLO
from utils.img_drawing import *

class ImageProcessingController:
    def __init__(self):
        self.image_path = None


    def process_image(self):
        model_location = "./output/models/train2/weights/best.pt"
        model = YOLO(model_location) # best model location
        results = model(self.image_path)

        detections = sort_boxes(results[0])

        draw_boxes(self.image_path, detections, results[0])