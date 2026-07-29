
from views.mask_viewer import MaskViewer
from views.image_viewer import ImageViewer


from PyQt5.QtWidgets import (
    QWidget,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QApplication,
    QSizePolicy,
)

from PyQt5.QtCore import Qt


class DistanceWindow(QWidget):
    def __init__(self, image_path, mask_array):
        super().__init__()

        self.setWindowTitle("Distance measurement")

        # ---------- VIEWERS ----------
        self.mask_viewer = MaskViewer(mask_array)
        self.mask_viewer.distanceComputed.connect(self.validate_distance)
        self.image_viewer = ImageViewer(image_path)

        # IMPORTANT: let layout control size
        self.image_viewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.mask_viewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # ---------- BUTTON ----------
        self.validate_button = QPushButton("Validate")
        self.validate_button.clicked.connect(self.validate_distance)

        # ---------- LAYOUT ----------
        images_layout = QHBoxLayout()
        images_layout.addWidget(self.image_viewer, 1)
        images_layout.addWidget(self.mask_viewer, 1)

        main_layout = QVBoxLayout()
        main_layout.addLayout(images_layout)
        main_layout.addWidget(self.validate_button)

        self.setLayout(main_layout)

        # ---------- WINDOW SIZE CONTROL ----------
        screen = QApplication.primaryScreen()
        geo = screen.availableGeometry()

        width = int(geo.width() * 0.8)
        height = int(geo.height() * 0.8)

        self.resize(width, height)

        self.setMinimumSize(800, 500)
        self.setMaximumSize(geo.width(), geo.height())

        # Center window
        self.move(
            geo.center().x() - width // 2,
            geo.center().y() - height // 2
        )


    def validate_distance(self):
        distance = self.mask_viewer.current_distance

        if distance is None:
            print("No distance selected")
            return

        print("Validated distance:", distance)
