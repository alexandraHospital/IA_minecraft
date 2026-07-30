from PyQt5.QtWidgets import (
    QLabel,
    QSizePolicy,
)

from PyQt5.QtCore import Qt


class BaseImageViewer(QLabel):
    def __init__(self, pixmap):
        super().__init__()
        self.base_pixmap = pixmap
        # self.setAlignment(Qt.AlignCenter)
        
        # viewer sized is fixed to image
        self.setFixedSize(pixmap.size())

        self.update_pixmap()

    def resizeEvent(self, event):
        self.update_pixmap()
        super().resizeEvent(event)

    def update_pixmap(self):
        self.setPixmap(
            self.base_pixmap.scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )