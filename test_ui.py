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
from apps.person_management import PersonManagement
from apps.face_management import FaceManagement
from configs.config_instance import FaceRecognitionConfigInstance
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog
class Test:
    def __init__(self) -> None:
        self.MainWindow = QtWidgets.QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.MainWindow)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.viewCam)
        self.ui.pushButton.clicked.connect(self.controlTimer)
        self.ui.bt_add_person.clicked.connect(self.add_person)
        self.ui.bt_delete_person.clicked.connect(self.delete_person)
        self.ui.bt_chose_file.clicked.connect(self.open_file_name_dialog)
        self.ui.bt_add_face.clicked.connect(self.add_face)
        # self.ui.table_people.selectionModel().selectionChanged.connect(self.on_selection_changed)
        self.ui.table_people.cellClicked.connect(self.cell_was_clicked)

        self.config =  FaceRecognitionConfigInstance.__call__().get_config()
        self.database = DatabaseInstance.__call__().get_database()
        self.person_management =  PersonManagement(self.config, self.database)
        self.face_management = FaceManagement(self.config, self.database)
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
        self.load_all_people()

    def on_selection_changed(self, selected, deselected):
        # for ix in selected.indexes():
            # print(ix.row(), ix.column())
            # print(self.ui.table_people.takeVerticalHeaderItem(ix.row()))
        row = self.ui.table_people.currentRow()
        id = self.ui.table_people.itemAt(row, 1).text()
        name = self.ui.table_people.itemAt(row, 2).text()
        print(id, name)
     
    def cell_was_clicked(self, row, column):
        # row = self.ui.table_people.currentRow()
        id = self.ui.table_people.itemAt(2, 2).text()
        print(id)
        # name = self.ui.table_people.itemAt(1, 2).text()
        # print(id, name)
        # print("Row %d and Column %d was clicked" % (row, column))
        pass
    
    def load_all_faces(self):
        pass

    def load_all_people(self):
        all_people = self.database.get_all_people()
        row=0
        self.ui.table_people.setRowCount(self.database.number_of_people())
        for person in all_people:
            self.ui.table_people.setItem(row, 0, QtWidgets.QTableWidgetItem(person["name"]))
            self.ui.table_people.setItem(row, 1, QtWidgets.QTableWidgetItem(str(person["id"])))
            row=row+1

    def add_person(self):
        name = self.ui.tb_add_person_name.text()
        id = self.ui.tb_add_person_id.text()
        res = self.person_management.insert_person(id, name)
        print(res)
        self.load_all_people()
    
    def delete_person(self):
        id = self.ui.tb_add_person_id.text()
        res = self.person_management.delete_person_by_id(id)
        print(res)
        self.load_all_people()
    
    def update_person_by_id(self):
        name = self.ui.tb_add_person_name.text()
        id = self.ui.tb_add_person_id.text()
        res = self.person_management.update_person_name(id, name)
        print(res)
        self.load_all_people()
    
    def update_person_by_name(self):
        name = self.ui.tb_add_person_name.text()
        id = self.ui.tb_add_person_id.text()
        res = self.person_management.update_person_id(id, name)
        print(res)
        self.load_all_people()

    def open_file_name_dialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self.MainWindow,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
        self.ui.tb_path.setText(file_name)

    def add_face(self):
        # TODO checkpath
        image = cv2.imread(self.ui.tb_path.text())
        id = self.ui.tb_add_person_id.text()
        res = self.face_management.insert_face(id, "01", image)
        print(res)

    
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