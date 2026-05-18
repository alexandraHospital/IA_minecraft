import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel,
    QFileDialog, QVBoxLayout
)
from PyQt5.QtGui import QPixmap


class ImageSelector(QWidget):
    def __init__(self, image_data):
        super().__init__()

        self.setWindowTitle("Choose picture")

        self.layout = QVBoxLayout()

        # Bouton
        self.button = QPushButton("Browse")
        self.button.clicked.connect(self.open_file)

        # Label pour afficher l'image
        self.label = QLabel("")
        self.label.setScaledContents(True)
        
        self.btn_process = QPushButton("Process")
        self.btn_process.clicked.connect(self.process_image)
        self.btn_process.hide()


        self.layout.addWidget(self.button)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.btn_process)

        self.setLayout(self.layout)
        
        self.image_data = image_data

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose a picture",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )

        if file_path:
            self.image_data.image_path = file_path
            pixmap = QPixmap(file_path)
            self.label.setPixmap(pixmap)

            self.btn_process.show()

    def process_image(self):
        print(self.image_data.image_path)
        self.image_data.process_image()
        self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageSelector()
    window.resize(600, 400)
    window.show()
    sys.exit(app.exec_())