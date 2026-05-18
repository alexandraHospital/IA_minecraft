from ultralytics import YOLO
import cv2
from variables import *
import numpy as np
import os

class ImageData:
    def __init__(self):
        self.image_path = None
        
        
    def process_image(self):
        model_location = "./output/models/train2/weights/best.pt"
        model = YOLO(model_location) # best model location
        results = model(self.image_path)
        base_dir = "./output/results"
        image_name = os.path.splitext(os.path.basename(self.image_path))[0]
        r = results[0]
        img = cv2.imread(self.image_path)

        # Classes
        classes = r.boxes.cls.cpu().numpy().astype(int)

        for i, box in enumerate(r.boxes.xyxy.cpu().numpy()):
            x1, y1, x2, y2 = map(int, box)
            cls = classes[i]
            color = COLOR_PALETTE[cls]

            # draw bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), color=color, thickness=2)
            
            # draw mask
            if r.masks is not None:
                mask = r.masks.data[i].cpu().numpy()  # (H, W)
                mask_color = np.array(color, dtype=np.uint8).reshape(1,1,3)
                img = np.where(mask[..., None], (0.5*img + 0.5*mask_color).astype(np.uint8), img)
        # # add legend
        # start_y = 30
        # for i, (color, name) in enumerate(zip(COLOR_PALETTE, CLASS_NAME)):
        #     cv2.rectangle(img, (10, start_y + i*30), (30, start_y + i*30 + 20), color, -1)
        #     cv2.putText(img, name, (40, start_y + i*30 + 15),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
        # Save
        
        cv2.imwrite(base_dir + "/" + image_name + "_mask.jpg", img)