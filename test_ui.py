import sys
sys.path.append("./src")
from PyQt5.QtWidgets import QApplication,QMainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer
from face_ui.face_ui import Ui_MainWindow
from queue import Queue
from video_stream.video_reader import VideoReader
from apps.data_queue import DataQueue

class Test:
    def __init__(self) -> None:
        self.MainWindow = QtWidgets.QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.MainWindow)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.viewCam)
        self.ui.pushButton.clicked.connect(self.controlTimer)
        self.frame_queue = DataQueue.__call__().get_frame_queue()
        self.video_reader = VideoReader("",self.frame_queue)

    
    def show_window(self):
        self.MainWindow.show()

    def viewCam(self):
        if self.frame_queue.qsize() != 0:
            frame_data = self.frame_queue.get()
            image = frame_data["image"]
            # ret, image = self.cap.read()

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, channel = image.shape
            step = channel * width
            qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
            self.ui.label.setPixmap(QPixmap.fromImage(qImg))
    
    def controlTimer(self):
        if not self.timer.isActive():
            self.video_reader.start()
            # self.cap = cv2.VideoCapture("rtsp://admin:ATDJTN@192.168.1.99:554/H.264")
            self.timer.start(20)
            self.ui.pushButton.setText("Stop")
        else:
            self.timer.stop()
            # self.cap.release()
            self.video_reader.join()
            self.ui.pushButton.setText("Start")

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    test = Test()
    test.show_window()
    sys.exit(app.exec_())