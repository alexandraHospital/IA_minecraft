from utils.variables import *
import os
import cv2
import numpy as np
from collections import deque


def sort_boxes(result):
        classes = result.boxes.cls.cpu().numpy().astype(int)

        detections = []
        
        for i, box in enumerate(result.boxes.xyxy.cpu().numpy()):
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

def draw_boxes(image_path, detections, result):
    base_dir = "./output/results"
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    img = cv2.imread(image_path)
    H, W = img.shape[:2]

    mask_img = np.zeros((H, W, 3), dtype=np.uint8)  # masque propre

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        cls = det["cls"]
        color = COLOR_PALETTE[cls]

        # visuel uniquement
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, thickness=-1)
        print(f"dessine {color}")

    cv2.imwrite(f"{base_dir}/{image_name}_mask.png", mask_img)
    cv2.imwrite(f"{base_dir}/{image_name}_visu.jpg", img)
    
    return mask_img

def extract_region(mask, image_path, target_color):
    base_dir = "./output/results"
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    image = cv2.imread(image_path)

    y_min, x_min, y_max, x_max = largest_monochrome_rectangle(mask, target_color)

    crop = image[y_min:y_max+1, x_min:x_max+1]
    crop_mask = mask[y_min:y_max+1, x_min:x_max+1]

    # for debug:
    cv2.imwrite(base_dir + "/" + image_name + "_facade_crop.jpg", crop)
    cv2.imwrite(base_dir + "/" + image_name + "_facade_crop_mask.jpg", crop_mask)
    
    return crop


def largest_monochrome_rectangle(image, target_color):
    """
    Finds the largest monochrome rectangle (where all pixels = target_color).
    Returns (y_min, x_min, y_max, x_max) or None.
    """
    # Create a boolean mask : True if pixel == target_color
    mask = np.all(image == target_color, axis=-1)  # Shape: (H, W)

    # For each line, calculate the height of consecutive columns
    H, W = mask.shape
    heights = np.zeros((H, W), dtype=int)
    for y in range(H):
        for x in range(W):
            if mask[y, x]:
                heights[y, x] = heights[y-1, x] + 1 if y > 0 else 1
            else:
                heights[y, x] = 0

    # Find the largest rectangle in 'heights'
    max_area = 0
    best_rect = None
    for y in range(H):
        stack = []
        for x in range(W + 1):
            # Ajouter une colonne virtuelle à droite pour vider le stack
            current_height = heights[y, x] if x < W else 0
            while stack and heights[y, stack[-1]] > current_height:
                h = heights[y, stack.pop()]
                w = x if not stack else x - stack[-1] - 1
                area = h * w
                if area > max_area:
                    max_area = area
                    best_rect = (y - h + 1, stack[-1] + 1 if stack else 0, y, x - 1)
            stack.append(x)

    return best_rect if best_rect else None

