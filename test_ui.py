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
from apps.data_queue import DataQueue, ResultQueue
from apps.face_recognition_app import FaceRecognitionApp

class Test:
    def __init__(self) -> None:
        self.MainWindow = QtWidgets.QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.MainWindow)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.viewCam)
        self.ui.pushButton.clicked.connect(self.controlTimer)


        self.face_recognition_app = FaceRecognitionApp()
        self.task = self.face_recognition_app.get_task()
        self.frame_queue = DataQueue.__call__().get_frame_queue()
        self.result_queue = ResultQueue.__call__().get_result_queue()
        self.video_reader = VideoReader("",self.frame_queue)
        self.timer.start(20)
        self.viewCam()
        
        # self.video_reader.start()
        self.face_recognition_app.run()

        self.init()
    

    
    def init(self):
        _translate = QtCore.QCoreApplication.translate
        self.ui.tabWidget.setTabText(self.ui.tabWidget.indexOf(self.ui.tab), _translate("MainWindow", "Camera"))
        self.ui.tabWidget.setTabText(self.ui.tabWidget.indexOf(self.ui.tab_2), _translate("MainWindow", "Quản lý"))
        self.ui.label.setText(_translate("MainWindow", ""))

    
    def show_window(self):
        self.MainWindow.show()

    def viewCam(self):
        if self.result_queue.qsize() != 0:
            frame_data = self.frame_queue.get()
            image = frame_data["image"]

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, channel = image.shape
            step = channel * width
            qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
            self.ui.label.setPixmap(QPixmap.fromImage(qImg))
    
    def controlTimer(self):
        if not self.task["face_recognizer"].is_enable():
            self.ui.pushButton.setText("Pause")
            self.task["face_recognizer"].enable()
        else:
            self.ui.pushButton.setText("Start")
            self.task["face_recognizer"].disable()

    def __del__(self):
        self.video_reader.stop_thread()
        self.timer.stop()
        # print("Call")

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    test = Test()
    test.show_window()
    sys.exit(app.exec_())