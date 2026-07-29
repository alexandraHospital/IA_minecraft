
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



class DistanceWindow(QWidget):
    def __init__(self, image_processing_controller):
        super().__init__()

        self.image_processing_controller = image_processing_controller
        self.setWindowTitle("Distance measurement")

        # ---------- VIEWERS ----------
        self.mask_viewer = MaskViewer(self.image_processing_controller.mask)
        self.mask_viewer.distanceComputed.connect(self.image_processing_controller.set_distance)
        self.image_processing_controller.maskUpdated.connect(self.mask_viewer.set_mask)
        self.image_viewer = ImageViewer(self.image_processing_controller.image_path)

        # let layout control size
        self.image_viewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.mask_viewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # ---------- BUTTON ----------
        self.validate_button = QPushButton("Validate")
        self.validate_button.clicked.connect(self.image_processing_controller.process_draw_grid)
        # self.mask_viewer.set_mask(self.image_processing_controller.mask)

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