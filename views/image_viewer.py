from PyQt5.QtWidgets import (
    QSizePolicy
)

from PyQt5.QtGui import (
    QPixmap,
)
from PyQt5.QtCore import Qt

from utils.img_drawing import numpy_to_qimage
from views.base_image_view import BaseImageViewer

class ImageViewer(BaseImageViewer):
    def __init__(self, image_path):
        super().__init__(QPixmap(image_path))

        self.pixmap_original = QPixmap(image_path)

        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        self.setPixmap(self.pixmap_original)
