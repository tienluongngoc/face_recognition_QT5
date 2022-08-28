import sys
from turtle import color, width
# sys.path.append("./src")
from PyQt5.QtWidgets import QApplication,QMainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
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
import numpy as np
from uuid import uuid4
from PyQt5.QtWidgets import QApplication, QDialog, QMainWindow, QPushButton
from schemas.validation import Validation
import cv2

from .ui_utils import ConfirmDialog, InfoDialog
from .update_person import UpDatePersonWindow


class FaceRecognitionUI:
    def __init__(self) -> None:
        self.MainWindow = QtWidgets.QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.MainWindow)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.viewCam)
        self.ui.pushButton.clicked.connect(self.controlTimer)

        self.ui.bt_add_person.clicked.connect(self.add_person)
        self.ui.bt_delete_person.clicked.connect(self.delete_person)
        self.ui.bt_update_person.clicked.connect(self.update_person)

        self.ui.bt_chose_file.clicked.connect(self.open_file_name_dialog)
        self.ui.bt_add_face.clicked.connect(self.add_face)
        self.ui.bt_delete_face.clicked.connect(self.delete_face)
        # self.ui.table_people.selectionModel().selectionChanged.connect(self.on_selection_changed)
        # self.ui.table_people.cellClicked.connect(self.cell_was_clicked)
        self.ui.table_people.cellClicked.connect(self.person_table_click)
        self.ui.table_face.cellClicked.connect(self.show_preview_face)
        

        self.config =  FaceRecognitionConfigInstance.__call__().get_config()
        self.ui_config = self.config.ui
        self.database = DatabaseInstance.__call__().get_database()
        self.person_management =  PersonManagement(self.config, self.database)
        self.face_management = FaceManagement(self.config, self.database)
        self.face_recognition_app = FaceRecognitionApp()
        self.task = self.face_recognition_app.get_task()
        self.frame_queue = DataQueue.__call__().get_frame_queue()
        self.result_queue = ResultQueue.__call__().get_result_queue()

        self.face = cv2.resize(cv2.imread("cr7.png"), (100,100))
        self.timer.start(0)
        self.viewCam()
        
        self.show_face()
        self.video_width = 1000
        self.video_height = 560
        
        self.face_recognition_app.run()

        self.init()
        self.load_all_people()
        self.init_face_table()
        # self.init_face_table()

    def update_person(self):
        self.update_person_window = UpDatePersonWindow(self)


    def person_table_click(self):
        current_row = self.ui.table_people.currentRow()
        name = self.ui.table_people.item(current_row, 0).text()
        id = self.ui.table_people.item(current_row, 1).text()
        self.ui.tb_add_person_id.setText(id)
        self.ui.tb_add_person_name.setText(name)
        self.load_all_faces(id)
        self.init_face_text_box()

    def face_table_click(self):
        current_row = self.ui.table_face.currentRow()
        try:
            id = self.ui.table_face.item(current_row, 0).text()
            path = self.ui.table_face.item(current_row, 1).text()
            self.ui.tb_path.setText(path)
            self.ui.tb_face_id.setText(id)
        except:
            pass
        


    def show_preview_face(self):
        self.face_table_click()
        try:
            current_row = self.ui.table_face.currentRow()
            img_path = self.ui.table_face.item(current_row, 1).text()
            image = cv2.imread(img_path)
            self.view_image(image, 491, 321, self.ui.face_preview)
        except:
            image = cv2.imread("libs/avata.jpg")
            self.view_image(image, 491, 321, self.ui.face_preview)

    
    def view_image(self, image: np.array, width, height, item):
        image = cv2.resize(image, (width, height))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channel = image.shape
        step = channel * width
        qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
        item.setPixmap(QPixmap.fromImage(qImg))

    
    def load_all_faces(self, persion_id):
        faces = self.face_management.select_all_face_of_person(persion_id, 0,10)
        self.ui.table_face.setRowCount(len(faces))
        for row,face in enumerate(faces):
            self.ui.table_face.setItem(row, 0, QtWidgets.QTableWidgetItem(face["id"]))
            self.ui.table_face.setItem(row, 1, QtWidgets.QTableWidgetItem(face["imgPath"]))

        self.show_preview_face()
        self.init_face_preview()

    def init_face_table(self):
        number_item = self.ui.table_people.rowCount()
        if number_item != 0:
            id = self.ui.table_people.item(0, 1).text()
            self.load_all_faces(id)
            self.init_face_text_box()
    
    def init_person_text_box(self):
        id = self.ui.table_people.item(0, 1).text()
        name = self.ui.table_people.item(0, 0).text()
        self.ui.tb_add_person_id.setText(id)
        self.ui.tb_add_person_name.setText(name)

    
    def init_face_text_box(self):
        number_items = self.ui.table_face.rowCount()
        if number_items != 0:
            path = self.ui.table_face.item(0, 1).text()
            id = self.ui.table_face.item(0, 0).text()
            self.ui.tb_face_id.setText(id)
            self.ui.tb_path.setText(path)
        else:
            self.ui.tb_face_id.setText("")
            self.ui.tb_path.setText("")



    def init_face_preview(self):
        number_item = self.ui.table_face.rowCount()
        if number_item != 0:
            img_path = self.ui.table_face.item(0, 1).text()
            image = cv2.imread(img_path)
            self.view_image(image, 491, 321, self.ui.face_preview)
        

    def load_all_people(self):
        all_people = self.database.get_all_people()
        row=0
        self.ui.table_people.setRowCount(self.database.number_of_people())
        for person in all_people:
            self.ui.table_people.setItem(row, 0, QtWidgets.QTableWidgetItem(person["name"]))
            self.ui.table_people.setItem(row, 1, QtWidgets.QTableWidgetItem(str(person["id"])))
            row=row+1
        self.init_person_text_box()

    def delete_face(self):
        person_id = self.ui.tb_add_person_id.text()
        person_name =  self.ui.tb_add_person_name.text()
        face_id = self.ui.tb_face_id.text()
        message = f"Do you want to delete face ID: {face_id} of {person_name}?"
        confirm_dlg = ConfirmDialog(self.MainWindow, message)
        if confirm_dlg.exec():
            self.face_management.delete_face_by_id(person_id, face_id)
        self.load_all_faces(person_id)

    def add_person(self):
        name = self.ui.tb_add_person_name.text()
        id = self.ui.tb_add_person_id.text()
        res = self.person_management.insert_person(id, name)
        print(res)
        self.load_all_people()
    
    def delete_person(self):
        id = self.ui.tb_add_person_id.text()
        message = f"Do you want to delete person id: {id}?"
        confirm_dlg = ConfirmDialog(self.MainWindow, message)
        if confirm_dlg.exec():
            res = self.person_management.delete_person_by_id(id)
            if res == Validation.DETETE_PERSON_SUCCESSFULY:
                info_dlg = InfoDialog(self.MainWindow, f"Delete person id: {id}, successfuly!")
                info_dlg.exec()
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
        person_id = self.ui.tb_add_person_id.text()
        face_id = self.ui.tb_face_id.text()
        all_faces = self.face_management.select_all_face_of_person(person_id,0,10)
        
        face_ids = [int(face["id"]) for face in all_faces]
        face_id = 0 if len(face_ids)==0 else  max(face_ids)+1
        res = self.face_management.insert_face(person_id, str(face_id), image)
        self.load_all_faces(person_id)

    
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
            if len (frame_data["largest_bbox"]) != 0:
                l_bbox = frame_data["largest_bbox"][0]
                person_dict = frame_data["person_dict"]
                text = ""
                face_id = int(l_bbox[5])
                if (person_dict["id"] != "unknown"):
                    id = person_dict["id"]
                    name = person_dict["name"]
                    
                    text = f"Hello {id}_{face_id}"
                    color = (0,255,0)
                else:
                    text = f"unknow_{face_id}"
                    color = (0,0,255)
                cv2.putText(image, text, (int(l_bbox[0]), int(l_bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1, cv2.LINE_AA)
                cv2.rectangle(image, (int(l_bbox[0]), int(l_bbox[1])), (int(l_bbox[2]), int(l_bbox[3])), color, 1)
        return image
    
    def init(self):
        _translate = QtCore.QCoreApplication.translate
        self.ui.tabWidget.setTabText(self.ui.tabWidget.indexOf(self.ui.tab), _translate("MainWindow", "Camera"))
        self.ui.tabWidget.setTabText(self.ui.tabWidget.indexOf(self.ui.tab_2), _translate("MainWindow", "Quản lý"))
        self.ui.label.setText(_translate("MainWindow", ""))
        self.people_tabale_properties()
        self.face_table_properties()
        
    
    def people_tabale_properties(self):
        self.ui.table_people.setColumnWidth(0,215)
        self.ui.table_people.setColumnWidth(1,110)

    def face_table_properties(self):
        self.ui.table_face.setColumnWidth(0,45)
        self.ui.table_face.setColumnWidth(1,265)

    
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

# if __name__ == "__main__":
#     import sys
#     app = QtWidgets.QApplication(sys.argv)
#     test = FaceRecognitionUI()
#     test.show_window()
#     sys.exit(app.exec_())