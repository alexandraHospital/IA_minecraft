
from PyQt5.QtGui import (
    QPixmap,
    QPainter,
    QPen,
    QColor,
)

from PyQt5.QtCore import Qt
from utils.img_drawing import numpy_to_qimage

from views.base_image_view import BaseImageViewer

class MaskViewer(BaseImageViewer):
    def __init__(self, mask_array):
        pixmap = QPixmap.fromImage(numpy_to_qimage(mask_array))
        super().__init__(pixmap)

        self.mask_array = mask_array

        self.points = []

    def mousePressEvent(self, event):
        pos = event.pos()

        # Nouvelle paire de clics
        if len(self.points) >= 2:
            self.points = []

        self.points.append((pos.x(), pos.y()))

        if len(self.points) == 2:
            self.compute_distance()

        self.update()  # force redraw

    def compute_distance(self):
        (x1, y1), (x2, y2) = self.points

        dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        self.distanceComputed.emit(dist)
        print("Distance:", dist)

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)

        pen = QPen(QColor("red"))
        pen.setWidth(3)
        painter.setPen(pen)

        for x, y in self.points:
            size = 6

            # cross
            painter.drawLine(x - size, y, x + size, y)
            painter.drawLine(x, y - size, x, y + size)

        # Line between the 2 points
        if len(self.points) == 2:
            (x1, y1), (x2, y2) = self.points
            painter.drawLine(x1, y1, x2, y2)

        painter.end()
