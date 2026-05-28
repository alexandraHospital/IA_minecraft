
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from utils.img_drawing import numpy_to_qimage

class MaskViewer(QLabel):
    def __init__(self, mask_array):
        super().__init__()
        self.mask_array = mask_array

        qimg = numpy_to_qimage(mask_array)
        self.setPixmap(QPixmap.fromImage(qimg))
        self.setScaledContents(True)

        self.points = []

    def mousePressEvent(self, event):
        if len(self.points) < 2:
            pos = event.pos()
            self.points.append((pos.x(), pos.y()))

        if len(self.points) == 2:
            self.compute_distance()

    def compute_distance(self):
        (x1, y1), (x2, y2) = self.points
        dist = ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5
        print("Distance:", dist)
        
        
