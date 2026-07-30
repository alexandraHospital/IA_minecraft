import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel,
    QFileDialog, QVBoxLayout, QSizePolicy, QHBoxLayout
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

from views.distance_window import DistanceWindow


class ImageSelector(QWidget):
    def __init__(self, image_processing_controller):
        super().__init__()

        self.setWindowTitle("Choose picture")

        # ---------- IMAGE VIEW ----------
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # ---------- BUTTONS ----------
        self.button = QPushButton("Browse")
        self.button.clicked.connect(self.open_file)

        self.btn_process = QPushButton("Process")
        self.btn_process.clicked.connect(self.process_image)
        self.btn_process.setEnabled(False)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.button)
        button_layout.addWidget(self.btn_process)

        # ---------- MAIN LAYOUT ----------
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.label, stretch=1)
        self.layout.addLayout(button_layout)

        self.setLayout(self.layout)

        self.image_processing_controller = image_processing_controller

        # ---------- WINDOW SIZE ----------
        screen = QApplication.primaryScreen()
        geo = screen.availableGeometry()

        self.resize(int(geo.width() * 0.8), int(geo.height() * 0.8))

        self.move(
            geo.center().x() - self.width() // 2,
            geo.center().y() - self.height() // 2
        )

        self.current_pixmap = None

    # ---------- FILE LOADING ----------
    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose a picture",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )

        if file_path:
            self.image_processing_controller.image_path = file_path

            self.current_pixmap = QPixmap(file_path)
            self.update_image()

            self.btn_process.setEnabled(True)

    # ---------- RESIZE HANDLING ----------
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_image()

    def update_image(self):
        if self.current_pixmap:
            self.label.setPixmap(
                self.current_pixmap.scaled(
                    self.label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            )

    # ---------- PROCESS ----------
    def process_image(self):
        self.image_processing_controller.process_image()

        self.distanceWindow = DistanceWindow(self.image_processing_controller)

        self.distanceWindow.show()

        self.close()