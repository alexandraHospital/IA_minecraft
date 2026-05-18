

from ultralytics import YOLO
from pathlib import Path
import sys
import os
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

from PyQt5.QtWidgets import QApplication
from image_selector import ImageSelector
from image_data import ImageData
from buildings.elements import Element
from buildings.building import Building

from variables import *

# def choose_material():
    # TODO call material recognizer

def create_building(results):
    r = results[0]

    # Classes correspondantes
    classes = r.boxes.cls.cpu().numpy().astype(int)
    confidences = r.boxes.conf.cpu().numpy()
    
    building = Building()
    elements = []

    for i, box in enumerate(r.boxes.xyxy.cpu().numpy()):
        x1, y1, x2, y2 = map(int, box)
        cls = classes[i]
        color = COLOR_PALETTE[cls]
        element = Element(class_id = cls, bbox = box, score = confidences[i])
        elements.add(element)


def main():
    #####################
    # Output results  #
    #####################
    path_logs = Path("output/results")
    path_logs.mkdir(parents=True, exist_ok=True)
    
    ##################################
    #            Create window
    ##################################
    app = QApplication(sys.argv)
    imageData = ImageData()
    imageSelector = ImageSelector(imageData)
    imageSelector.show()
    
    print(imageSelector.image_data.image_path)



    
    #######################################
    #            Call yolo on picture
    #######################################
    # # TDB: best model location
    # model_location = "./IA_minecraft/output/models/train2"
    # model = YOLO(model_location) # best model location
    # # calls model on this image
    # results = model(imageSelector.image_data.image_path)
    
    ##########################################
    #         Create building
    ##########################################
    
    
    
    ##########################################
    #            Call Minecraft API
    #########################################


    # output => minecraft structure
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()