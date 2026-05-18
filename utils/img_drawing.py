from utils.variables import *
import os
import cv2
import numpy as np

def sort_boxes(r):
        classes = r.boxes.cls.cpu().numpy().astype(int)

        detections = []
        
        for i, box in enumerate(r.boxes.xyxy.cpu().numpy()):
            x1, y1, x2, y2 = map(int, box)
            cls = classes[i]

            area = (x2 - x1) * (y2 - y1)

            detections.append({
                "box": (x1, y1, x2, y2),
                "cls": cls,
                "area": area,
                "index": i
            })
        detections.sort(key=lambda d: d["area"], reverse=True)

        return detections

def draw_boxes(image_path, detections, r):
    base_dir = "./output/results"
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    img = cv2.imread(image_path)
    
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        cls = det["cls"]
        i = det["index"]
        color = COLOR_PALETTE[cls]

        # draw bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), color=color, thickness=-1)
        
        # calculates box area
        
        # draw mask
        if r.masks is not None:
            mask = r.masks.data[i].cpu().numpy()  # (H, W)
            mask_color = np.array(color, dtype=np.uint8).reshape(1,1,3)
            img = np.where(mask[..., None], (0.5*img + 0.5*mask_color).astype(np.uint8), img)
    # Save
    cv2.imwrite(base_dir + "/" + image_name + "_mask.jpg", img)