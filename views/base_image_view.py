from PyQt5.QtWidgets import (
    QLabel,
    QSizePolicy,
)

from PyQt5.QtCore import Qt


class BaseImageViewer(QLabel):
    def __init__(self, pixmap):
        super().__init__()
        self.base_pixmap = pixmap
        self.setAlignment(Qt.AlignCenter)
        
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        self.update_pixmap()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_pixmap()

    def update_pixmap(self):
        self.setPixmap(
            self.base_pixmap.scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )