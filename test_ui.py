import sys
from turtle import color
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
import time
from database import DatabaseInstance
class Test:
    def __init__(self) -> None:
        self.MainWindow = QtWidgets.QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.MainWindow)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.viewCam)
        self.ui.pushButton.clicked.connect(self.controlTimer)

        self.database = DatabaseInstance.__call__().get_database()
        # self.person_coll = self.database.get_person_coll()
        self.face_recognition_app = FaceRecognitionApp()
        self.task = self.face_recognition_app.get_task()
        self.frame_queue = DataQueue.__call__().get_frame_queue()
        self.result_queue = ResultQueue.__call__().get_result_queue()

        self.face = cv2.resize(cv2.imread("images/face.png"), (100,100))
        self.timer.start(0)
        self.viewCam()
        self.show_face()
        self.video_width = 1000
        self.video_height = 560
        
        self.face_recognition_app.run()

        self.init()
        self.load_data()

    def load_data(self):
        # print("oke")
        all_people = self.database.get_all_people()
        # print(self.database.number_of_people())
        row=0
        self.ui.table_people.setRowCount(self.database.number_of_people())
        for person in all_people:
            self.ui.table_people.setItem(row, 0, QtWidgets.QTableWidgetItem(person["name"]))
            self.ui.table_people.setItem(row, 1, QtWidgets.QTableWidgetItem(str(person["id"])))
            row=row+1
    
    def show_face(self):
        self.face_list = [self.ui.face_1,self.ui.face_2, self.ui.face_3, self.ui.face_4, self.ui.face_5]
        height, width, channel = self.face.shape
        step = channel * width
        qImg = QImage(self.face.data, width, height, step, QImage.Format_RGB888)
        for face in self.face_list:
            face.setPixmap(QPixmap.fromImage(qImg))
    
    def visualize(self, frame_data):
        image = frame_data["image"]
        if self.task["face_recognizer"].is_enable():
            # detection_results = frame_data["detection_results"]
            l_bbox = frame_data["largest_bbox"][0]
            person_dict = frame_data["person_dict"]
            text = ""
            if (person_dict["id"] != "unknown"):
                id = person_dict["id"]
                name = person_dict["name"]
                text = f"id: {id} Name: {name}"
                color = (0,255,0)
            else:
                text = f"unknow"
                color = (0,0,255)
            # print(person_dict)
            # bboxs = detection_results[0]
            # for bbox in bboxs:
            #     cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255,255,0), 1)
            cv2.putText(image, text, (int(l_bbox[0])-30, int(l_bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1, cv2.LINE_AA)
            cv2.rectangle(image, (int(l_bbox[0]), int(l_bbox[1])), (int(l_bbox[2]), int(l_bbox[3])), color, 1)
        return image


    

    
    def init(self):
        _translate = QtCore.QCoreApplication.translate
        self.ui.tabWidget.setTabText(self.ui.tabWidget.indexOf(self.ui.tab), _translate("MainWindow", "Camera"))
        self.ui.tabWidget.setTabText(self.ui.tabWidget.indexOf(self.ui.tab_2), _translate("MainWindow", "Quản lý"))
        self.ui.label.setText(_translate("MainWindow", ""))

    
    def show_window(self):
        self.MainWindow.show()

    def viewCam(self):
        if self.result_queue.qsize() != 0:
            frame_data = self.result_queue.get()
            image = self.visualize(frame_data)
            image = cv2.resize(image, (self.video_width, self.video_height))
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

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    test = Test()
    test.show_window()
    sys.exit(app.exec_())