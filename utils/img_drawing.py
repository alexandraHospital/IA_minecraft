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

def extract_region(mask, image_path, color):
    base_dir = "./output/results"
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    # load image
    image = cv2.imread(image_path)
    
    target_color = COLOR_PALETTE[0] # blue
    matches = np.all(mask == target_color, axis = -1)
    indices = np.where(matches)
    
    print(indices[:10])


    # for i from range(0, indices
    if len(indices[0] > 0):
        y, x = indices[0][0], indices[1][0]
        largest_color_rectangle(image, y, x, target_color)
        print(f"Premier pixel trouvé en ({y}, {x})")
    else:
        print("Couleur non trouvée")

    # if len(xs) != 0:
    #     crop = image[ys.min():ys.max()+1, xs.min():xs.max()+1]
    #     cv2.imwrite(base_dir + "/" + image_name + "_facade_crop.jpg", crop)

    # return crop
    
    
def largest_color_rectangle(image, start_y, start_x, target_color):
    """
    Trouve le plus grand rectangle de `target_color` contenu dans la région connexe
    autour de (start_y, start_x).

    Args:
        image: Tableau numpy de shape (H, W, 3) (RGB).
        start_y, start_x: Coordonnées du pixel de départ.
        target_color: Couleur cible sous forme [R, G, B].

    Returns:
        (y_min, x_min, y_max, x_max) : Coordonnées du rectangle englobant.
        None si la couleur ne correspond pas au pixel de départ.
    """
    H, W, _ = image.shape
    if not np.array_equal(image[start_y, start_x], target_color):
        return None

    # Masque des pixels de la couleur cible
    color_mask = np.all(image == target_color, axis=-1)

    # Flood fill (BFS) pour trouver la région connexe
    visited = np.zeros((H, W), dtype=bool)
    queue = deque([(start_y, start_x)])
    visited[start_y, start_x] = True
    region_pixels = []

    # Directions : haut, bas, gauche, droite
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        y, x = queue.popleft()
        region_pixels.append((y, x))
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                if color_mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    queue.append((ny, nx))

    if not region_pixels:
        return None

    # Extraire le rectangle englobant
    y_coords, x_coords = zip(*region_pixels)
    y_min, y_max = min(y_coords), max(y_coords)
    x_min, x_max = min(x_coords), max(x_coords)

    return (y_min, x_min, y_max, x_max)
