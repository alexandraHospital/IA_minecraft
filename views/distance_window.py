
from views.mask_viewer import MaskViewer
from views.image_viewer import ImageViewer


from PyQt5.QtWidgets import (
    QWidget,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QApplication,
    QScrollArea,
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

        # ---------- BUTTON ----------
        self.validate_button = QPushButton("Validate")
        self.validate_button.clicked.connect(self.image_processing_controller.process_draw_grid)


        # ---------- SCROLL AERA ----------
        image_viewer_scroll = QScrollArea()
        image_viewer_scroll.setWidget(self.image_viewer)

        mask_viewer_scroll = QScrollArea()
        mask_viewer_scroll.setWidget(self.mask_viewer)

        # ---------- LAYOUT ----------
        images_layout = QHBoxLayout()
        images_layout.addWidget(image_viewer_scroll)
        images_layout.addWidget(mask_viewer_scroll)

        main_layout = QVBoxLayout()
        main_layout.addLayout(images_layout)
        main_layout.addWidget(self.validate_button)

        self.setLayout(main_layout)

        # ---------- WINDOW SIZE CONTROL ----------
        screen = QApplication.primaryScreen()
        geo = screen.availableGeometry()

        width = int(geo.width())
        height = int(geo.height())

        # resize window at maximum size of screen
        self.resize(width, height)

        self.setMinimumSize(800, 500)
        self.setMaximumSize(geo.width(), geo.height())

        original_width = self.image_viewer.base_pixmap.width()
        
        mask_viewer_width = self.image_viewer.width()
        
        print(f"*** mask_viewer_width {mask_viewer_width}")
        print(f"*** original_width {original_width}")
        print(f"*** ratio_w {original_width/mask_viewer_width}")
        
        original_height = self.image_viewer.base_pixmap.height()
        mask_viewer_height = self.image_viewer.height()
        ratio_h = {original_height/mask_viewer_height}
        print(f"*** mask_viewer_height {mask_viewer_height}")
        print(f"*** original_height {original_height}")
        print(f"*** ratio_h {original_height/mask_viewer_height}")

        ratio = ((original_width/mask_viewer_width) + (original_height/mask_viewer_height)) / 2
        
        # self.image_processing_controller.ratio = ratio

        # # Center window
        # self.move(
        #     geo.center().x() - width // 2,
        #     geo.center().y() - height // 2
        # )