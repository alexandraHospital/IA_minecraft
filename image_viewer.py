import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel,
    QFileDialog, QVBoxLayout
)
from PyQt5.QtGui import QPixmap

class ImageViewer(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Sélecteur d'image")

        self.layout = QVBoxLayout()

        # Bouton
        self.button = QPushButton("Parcourir")
        self.button.clicked.connect(self.open_file)

        # Label pour afficher l'image
        self.label = QLabel("Aucune image")
        self.label.setScaledContents(True)

        self.layout.addWidget(self.button)
        self.layout.addWidget(self.label)

        self.setLayout(self.layout)

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Choisir une image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )

        if file_path:
            pixmap = QPixmap(file_path)
            self.label.setPixmap(pixmap)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageViewer()
    window.resize(600, 400)
    window.show()
    sys.exit(app.exec_())