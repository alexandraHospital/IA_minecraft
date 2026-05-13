

from ultralytics import YOLO
from pathlib import Path
import sys
import os
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

from PyQt5.QtWidgets import QApplication
from image_viewer import ImageViewer
from buildings.elements import Element
from buildings.building import Building

color_palette = [
    (255,0,0),     # facade # blue
    (0,255,0),     # window # green
    (0,0,255),     # door # red
    (255,255,0),   # cornice #cyan
    (255,0,255),   # sill # pink
    (0,255,255),   # balcony # yellow
    (128,0,0),     # blind #dark blue
    (0,128,0),     # deco # dark green
    (255,0,128),     # molding #
    (255,255,255),   # pillar
    (32,32,32),   # shop
]
class_names = ['facade', 'window', 'door', 'cornice', 'sill', 'balcony',
               'blind', 'deco', 'molding', 'pillar', 'shop']


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
        color = color_palette[cls]
        element = Element(class_id = cls, bbox = box, score = confidences[i])
        elements.add(element)


def main():
    ##################################
    #            Create window
    ##################################
    app = QApplication(sys.argv)
    imageViewer = ImageViewer()
    imageViewer.show()


    #######################################
    #            Call yolo on picture
    #######################################
    # TDB: best model location
    # model_location = "./output/models/train2/weights/best.pt"
    # model = YOLO(model_location) # best model location
    # calls model on this image
    # results = model(image_path)
    
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