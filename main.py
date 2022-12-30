from src.face_ui.face_recognition_ui import FaceRecognitionUI
from PyQt5 import QtWidgets
import sys

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    test = FaceRecognitionUI()
    test.show_window()
    sys.exit(app.exec_())