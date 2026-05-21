from ultralytics import YOLO
from utils.img_drawing import *
import cv2
from glob import glob
import torch
from torch import jit
from torchvision import transforms

base_dir = "./output/results"

class ImageProcessingController:
    def __init__(self):
        self.image_path = None
        self.mask = None




    def process_image(self):
        yolo_location = "./output/models/train2/weights/best.pt"
        model = YOLO(yolo_location) # best model location
        results = model(self.image_path)

        detections = sort_boxes(results[0])

        self.mask = draw_boxes(self.image_path, detections, results[0])

        facade_sample = extract_region(self.mask, self.image_path, COLOR_PALETTE[0])
        
        
        
        # get CNN model
        # TODO: better way with objects?
        # TODO: torch.jit does it exist a better way?
        # cnn_location = glob("./output/models/MR/MR_*_best.pth")
        # cnn_model = torch.jit.load(cnn_location)
        # cnn_model.eval()
        
        # inference_transform = transforms.Compose([
        # transforms.Resize(224*224),
        # transforms.ToTensor()])
        # with torch.no_grad():
        #     output_inference = model(inference_transform)
        