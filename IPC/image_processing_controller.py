from ultralytics import YOLO
from utils.img_drawing import *
import cv2

class ImageProcessingController:
    def __init__(self):
        self.image_path = None
        self.mask_path = None





    def process_image(self):
        model_location = "./output/models/train2/weights/best.pt"
        model = YOLO(model_location) # best model location
        results = model(self.image_path)

        detections = sort_boxes(results[0])

        self.mask_path = draw_boxes(self.image_path, detections, results[0])
        
        extract_facade_for_material_detection(self.image_path, results[0])