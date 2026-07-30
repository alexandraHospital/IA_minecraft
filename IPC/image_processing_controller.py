from ultralytics import YOLO
from utils.img_drawing import *
import cv2
from glob import glob
import torch
from torch import jit
from torchvision import transforms, io
from buildings.elements import Element
from pathlib import Path
from PyQt5.QtCore import pyqtSignal, QObject


IMAGE_SIZE = (224, 224)

class ImageProcessingController(QObject):
    maskUpdated = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.image_path = None
        self.mask = None
        self.distance = None
        self.ratio = 1

    def process_image(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ############################################## 
        #    YOLO INFERENCE
        # TO HAVE ELEMENTS ON THE BUILDING
        ##############################################
        if device == "cuda":
            project_root = str(Path(__file__).resolve().parent.parent)
            
            yolo_location = project_root + "/output/models/BA/yolo/weights/best.pt"
            model = YOLO(yolo_location) # best model location
            results = model(self.image_path)

            detections = sort_boxes(results[0])

            self.mask = draw_boxes(self.image_path, detections)

            _, facade_path = extract_region(self.mask, self.image_path, COLOR_PALETTE[0])

            # get CNN model

            pattern_cnn_location = project_root + "/output/models/MC/MC_*_best.pth"
            matching_files = glob(pattern_cnn_location)
            
            if not matching_files:
                raise FileNotFoundError(f"Model at {pattern_cnn_location} not found.")

            print(f"cnn_location: {matching_files}")

            cnn_model = torch.jit.load(matching_files[0])

            ###############################################
            # INFERENCE ON FACADE SAMPLE TO HAVE MATERIAL
            ###############################################
            facade_sample_image = io.read_image(str(facade_path)).type(torch.float32)
            facade_sample_image = facade_sample_image / 255 # value bewtween 0 and 1
            
            # Print out image data
            print(f"Custom image tensor:\n{facade_sample_image}\n")
            print(f"Custom image shape: {facade_sample_image.shape}\n")
            print(f"Custom image dtype: {facade_sample_image.dtype}")

            facade_sample_image_transform = transforms.Compose([
                transforms.Resize(IMAGE_SIZE),
            ])
            
            facade_image_transformed = facade_sample_image_transform(facade_sample_image)
            
            # Print out original shape and new shape
            print(f"Original shape: {facade_sample_image.shape}")
            print(f"New shape: {facade_image_transformed.shape}")

            cnn_model.eval()
            with torch.inference_mode():
                # Add an extra dimension to image
                facade_image_transformed_with_batch_size = facade_image_transformed.unsqueeze(dim = 0)
                # Print out different shapes
                print(f"Custom image transformed shape: {facade_image_transformed.shape}")
                print(f"Unsqueezed custom image shape: {facade_image_transformed_with_batch_size.shape}")
            
            # Make a prediction on image with an extra dimension
                facade_image_pred = cnn_model(facade_image_transformed.unsqueeze(dim = 0).to(device))

                # Let's convert them from logits -> prediction probabilities -> prediction labels
                # Print out prediction logits
                print(f"Prediction logits: {facade_image_pred}")

                # Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
                facade_image_pred_probs = torch.softmax(facade_image_pred, dim = 1)
                print(f"Prediction probabilities: {facade_image_pred_probs}")

                # Convert prediction probabilities -> prediction labels
                facade_image_pred_label = torch.argmax(facade_image_pred_probs, dim = 1)
                print(f"Prediction label: {facade_image_pred_label}")
                
                facade_image_pred_class = CLASS_MATERIAL_NAME[facade_image_pred_label.cpu()] # put pred label to CPU, otherwise will error
                print(facade_image_pred_class)
        else:
            # cpu
            image_name = os.path.splitext(os.path.basename(self.image_path))[0]
            img = cv2.imread(self.image_path)
            img_height, img_width = img.shape[:2]
            
            print(f"img_height {img_height} img_width {img_width}")
        
            # generate random mask
            detections, mask, counts = generate_fake_detections_and_mask(img_height, img_width, 3)
            self.mask = draw_boxes(self.image_path, detections)
            # _, facade_path = extract_region(self.mask, self.image_path, COLOR_PALETTE[0])


    def set_distance(self, distance):
        print("Controller received distance:", distance)
        print("Controller ratio: ", self.ratio)
        self.distance = distance * self.ratio
        print("Controller computed distance:", self.distance)

    def process_draw_grid(self):
        """
            draw grid on image according to the pointed distance
        """

        if self.distance is None:
            raise Exception("distance cannot be undefined")
        else:
            mask_grid = draw_grid(self.mask, (255, 255, 255), self.distance)
            self.maskUpdated.emit(mask_grid)
            print("after emit")
