import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from src.face_ui.face_recognition_ui import FaceRecognitionUI
from PyQt5 import QtCore, QtGui, QtWidgets

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    test = FaceRecognitionUI()
    test.show_window()
    sys.exit(app.exec_())