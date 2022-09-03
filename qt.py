import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QMessageBox
from src.configs.arcface_config import ArcFaceConfig
from src.inferences.face_encode.arcface_ultils.models import get_model
from src.inferences.face_encode.arcface_torch import  ArcFaceTorch


def dialog():
    print("oke")
    config = ArcFaceConfig("configs\models\\arcface.yaml")
    arcface_torch = ArcFaceTorch(config)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = QWidget()
    w.resize(300,300)
    w.setWindowTitle("Guru99")
    
    label = QLabel(w)
    label.setText("Behold the Guru, Guru99")
    label.move(100,130)
    label.show()

    btn = QPushButton(w)
    btn.setText('Beheld')
    btn.move(110,150)
    btn.show()
    btn.clicked.connect(dialog)

    
    w.show()
    sys.exit(app.exec_())