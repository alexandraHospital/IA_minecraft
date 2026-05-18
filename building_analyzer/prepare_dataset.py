import os
import re
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path
import cv2
import numpy as np

from utils.variables import *


DATASET_DIR = "./building_analyzer/dataset/cmp_facade"
EXTENDED_DIR = "./dataset/extended"
CMP_DIR = "./building_analyzer/dataset/cmp"
MASK_DIR = CMP_DIR + "/mask"
LABEL_DIR = CMP_DIR + "/labels"
IMAGE_DIR = CMP_DIR + "/images"
YOLO_TXT_DIR = CMP_DIR + "/YOLO"
SPLIT_RATIO = 0.8


# add annotation tag to be XML compliant in each xml file
def add_annotation_tag(xml_dir_path):
    print("Addind <annotation> to .xml files...")
    xml_dir = Path(xml_dir_path)
    for file in xml_dir.glob("*.xml"):
        with open(file, "r", encoding="utf-8") as f:
            content = f.read().strip()

        # éviter de rajouter deux fois la balise
        if not content.startswith("<annotation>"):
            new_content = "<annotation>\n" + content + "\n</annotation>"

            with open(file, "w", encoding="utf-8") as f:
                f.write(new_content)
    print("Done!")
                
def convert_to_yolo_detection(input_dir, output_dir):
    print("Converting .xml to .txt...")
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.endswith(".xml"):
            continue

        tree = ET.parse(os.path.join(input_dir, filename))
        root = tree.getroot()

        yolo_lines = []

        for obj in root.findall("object"):
            xs = obj.find("points").findall("x")
            ys = obj.find("points").findall("y")

            x_min = float(ys[0].text)
            x_max = float(ys[1].text)
            y_min = float(xs[0].text)
            y_max = float(xs[1].text)

            class_id = int(obj.find("label").text) - 2

            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min

            line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            yolo_lines.append(line)

        txt_name = filename.replace(".xml", ".txt")
        with open(os.path.join(output_dir, txt_name), "w") as f:
            f.write("\n".join(yolo_lines))
    print("Done!")


# print in a .png file the mask with all bbox on the photo
def print_mask(base_dir, label_dir, output_dir):
    print("Print bbox masks")
    os.makedirs(f"{output_dir}", exist_ok=True)
    image_dir = Path(base_dir)
    for image_path in image_dir.rglob("*.jpg"):
        img = cv2.imread(image_path)
        mask = np.zeros_like(img)

        # Corresponding .txt
        txt_file_name = os.path.splitext(os.path.basename(image_path))[0]
        split = image_path.parent.name
        txt_path = Path(label_dir) / split / f"{txt_file_name}.txt"
        output_path = f"{output_dir}/{txt_file_name}.png"

        if os.path.exists(txt_path):
            with open(txt_path, "r") as f:
                lines = f.readlines()

            # Dessiner tous les rectangles dans le mask
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id, x_center, y_center, width, height = map(float, parts)

                h, w = img.shape[:2]
                x_center *= w
                y_center *= h
                width *= w
                height *= h

                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)

                # -2 because from 2->12
                color = COLOR_PALETTE[int(class_id)-2]
                cv2.rectangle(mask, (x1, y1), (x2, y2), color, -1)

            # Overlay
            overlay = cv2.addWeighted(img, 0.6, mask, 0.4, 0)

            # Construire le panneau légende
            legend_width = 200
            legend_img = np.zeros((img.shape[0], legend_width, 3), dtype=np.uint8)
            start_y = 30
            for i, (color, name) in enumerate(zip(COLOR_PALETTE, CLASS_NAME)):
                cv2.rectangle(legend_img, (10, start_y + i*30), (30, start_y + i*30 + 20), color, -1)
                cv2.putText(legend_img, name, (40, start_y + i*30 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            # Combiner image et légende
            combined_img = np.hstack((overlay, legend_img))
            cv2.imwrite(output_path, combined_img)
        else:
            print(f"No .txt file found for {image_path}")
    print("Done!")


# ---- SPLIT TRAIN / VAL ----
def split_dataset(sources, output=CMP_DIR, split_ratio=0.8):
    print("Split dataset into train/val directories")
    img_train = Path(output) / "images/train"
    img_val = Path(output) / "images/val"
    lbl_train = Path(output) / "labels/train"
    lbl_val = Path(output) / "labels/val"

    for p in [img_train, img_val, lbl_train, lbl_val]:
        p.mkdir(parents=True, exist_ok=True)

    images = []
    for src in sources:
        images += list(Path(src).rglob("*.jpg"))

    random.shuffle(images)
    split = int(len(images) * split_ratio)

    for i, img in enumerate(images):
        label = img.with_suffix(".txt")
        subset = "train" if i < split else "val"

        img_dest = img_train if subset == "train" else img_val
        lbl_dest = lbl_train if subset == "train" else lbl_val

        shutil.copy(img, img_dest / img.name)

        if label.exists():
            shutil.copy(label, lbl_dest / label.name)

    print("Done!")
    
def prepare_dataset():
    add_annotation_tag(DATASET_DIR)

    convert_to_yolo_detection(DATASET_DIR, DATASET_DIR)

    split_dataset([DATASET_DIR, EXTENDED_DIR])

prepare_dataset()
print_mask(IMAGE_DIR, LABEL_DIR, MASK_DIR)