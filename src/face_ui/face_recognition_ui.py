
from asyncio import tasks
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import (QApplication, 
                            QDialog,
                             QMainWindow, 
                             QFileDialog,
                             QWidget, 
                             QMessageBox)
from datetime import datetime
from uuid import uuid4
import numpy as np
import time
import cv2


from src.database.database_instance import DatabaseInstance
from src.configs.config_instance import FaceRecognitionConfigInstance
from src.apps.face_recognition_app import FaceRecognitionApp
from src.apps.person_management import PersonManagement
from src.apps.data_queue import DataQueue, ResultQueue
from src.video_stream.video_reader import VideoReader
from .ui_utils import ConfirmDialog, InfoDialog
from src.apps.face_management import FaceManagement
from .update_person import UpDatePersonWindow
from src.schemas.validation import Validation
from src.face_ui.face_ui import Ui_MainWindow

class FaceRecognitionUI(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.ui_main_windown = Ui_MainWindow()
        self.ui_main_windown.setupUi(self)

        self.timer = QtCore.QTimer()
        self.timer.start(20)
        self.timer.timeout.connect(self.visualize_results_on_screen)
        self.ui_main_windown.pushButton.clicked.connect(self.controlTimer)

        self.ui_main_windown.bt_add_person.clicked.connect(self.add_person)
        self.ui_main_windown.bt_delete_person.clicked.connect(self.delete_person)
        self.ui_main_windown.bt_update_person.clicked.connect(self.update_person)

        self.ui_main_windown.bt_chose_file.clicked.connect(self.chose_face_image)
        self.ui_main_windown.bt_add_face.clicked.connect(self.add_face)
        self.ui_main_windown.bt_delete_face.clicked.connect(self.delete_face)

        self.ui_main_windown.table_people.cellClicked.connect(self.person_table_click_envent)
        self.ui_main_windown.table_face.cellClicked.connect(self.show_db_face_image_click_event)
        
        self.config =  FaceRecognitionConfigInstance.__call__().get_config()
        self.database = DatabaseInstance.__call__().get_database()
        self.person_management =  PersonManagement(self.config, self.database)
        self.face_management = FaceManagement(self.config, self.database)
        self.face_recognition_app = FaceRecognitionApp()
        self.tasks = self.face_recognition_app.get_task()
        self.frame_queue = DataQueue.__call__().get_frame_queue()
        self.result_queue = ResultQueue.__call__().get_result_queue()
        self.snapshot = {}

        self.video_width = 1000
        self.video_height = 560
        
        self.face_recognition_app.run()
        self.initialize()
        self.visualize_results_on_screen()
    #=====================================================================
    #======================== Table clicked event ========================
    #=====================================================================

    def person_table_click_envent(self):
        # Select row when clicked event
        # show info into person text box
        # load face table of this person
        current_row = self.ui_main_windown.table_people.currentRow()
        name = self.ui_main_windown.table_people.item(current_row, 0).text()
        id = self.ui_main_windown.table_people.item(current_row, 1).text()
        self.ui_main_windown.tb_add_person_id.setText(id)
        self.ui_main_windown.tb_add_person_name.setText(name)
        self.load_faces_db_to_table(id) # Load all faces of person id into face table
        self.init_face_text_box() # Load the first row of face table into text box

    #=====================================================================
    #========================  ========================
    #=====================================================================

    def load_faces_db_to_table(self, persion_id):
        """
        load face list into table: face_id and path of image in database
        """
        faces = self.face_management.select_all_face_of_person(persion_id, 0,10)
        self.ui_main_windown.table_face.setRowCount(len(faces))
        for row,face in enumerate(faces):
            self.ui_main_windown.table_face.setItem(row, 0, QtWidgets.QTableWidgetItem(face["id"]))
            self.ui_main_windown.table_face.setItem(row, 1, QtWidgets.QTableWidgetItem(face["imgPath"]))

        self.show_db_face_image_click_event()
        self.init_show_db_face_image() # load the first face of list face, 1st row
    
    def init_show_db_face_image(self):
        # Select the first row at the face table ==> face_id and path
        # Show this face to preview
        number_item = self.ui_main_windown.table_face.rowCount()
        if number_item != 0:
            img_path = self.ui_main_windown.table_face.item(0, 1).text()
            image = cv2.imread(img_path)
            self.view_image(image, 491, 321, self.ui_main_windown.face_preview)
        
    def show_db_face_image_click_event(self):
        # Check the current row when click at face table
        # show the face into face preview and face textbox
        try:
            # show face preview
            current_row = self.ui_main_windown.table_face.currentRow()
            id = self.ui_main_windown.table_face.item(current_row, 0).text()
            img_path = self.ui_main_windown.table_face.item(current_row, 1).text()
            image = cv2.imread(img_path)
            self.view_image(image, 491, 321, self.ui_main_windown.face_preview)
            
            # Show face info to textbox
            self.ui_main_windown.tb_path.setText(img_path)
            self.ui_main_windown.tb_face_id.setText(id)
        except:
            image = cv2.imread("libs/avata.jpg")
            self.view_image(image, 491, 321, self.ui_main_windown.face_preview)


    def init_face_table(self):
        # Select the first row at people ==> persopn_id
        # Load all faces in database of this person_id  into face_table
        number_item = self.ui_main_windown.table_people.rowCount()
        if number_item != 0:
            id = self.ui_main_windown.table_people.item(0, 1).text()
            self.load_faces_db_to_table(id)
            self.init_face_text_box()
    
    def init_person_text_box(self):
        # Load the first row at people table into textbox
        id = self.ui_main_windown.table_people.item(0, 1).text()
        name = self.ui_main_windown.table_people.item(0, 0).text()
        self.ui_main_windown.tb_add_person_id.setText(id)
        self.ui_main_windown.tb_add_person_name.setText(name)

    def init_face_text_box(self):
        # Init: if person id has face list, load the first row into textbox
        number_items = self.ui_main_windown.table_face.rowCount()
        if number_items != 0:
            path = self.ui_main_windown.table_face.item(0, 1).text()
            id = self.ui_main_windown.table_face.item(0, 0).text()
            self.ui_main_windown.tb_face_id.setText(id)
            self.ui_main_windown.tb_path.setText(path)
        else:
            self.ui_main_windown.tb_face_id.setText("")
            self.ui_main_windown.tb_path.setText("")

    def load_people_db_to_table(self):
        # Load all people in database into people table
        all_people = self.database.get_all_people()
        row=0
        self.ui_main_windown.table_people.setRowCount(self.database.number_of_people())
        for person in all_people:
            self.ui_main_windown.table_people.setItem(row, 0, QtWidgets.QTableWidgetItem(person["name"]))
            self.ui_main_windown.table_people.setItem(row, 1, QtWidgets.QTableWidgetItem(str(person["id"])))
            row=row+1
        try:
            self.init_person_text_box() # Load the first row at this table into textbox
        except:
            pass
        

    def initialize(self):
        _translate = QtCore.QCoreApplication.translate
        self.ui_main_windown.tabWidget.setTabText(self.ui_main_windown.tabWidget.indexOf(self.ui_main_windown.tab), _translate("MainWindow", "Camera"))
        self.ui_main_windown.tabWidget.setTabText(self.ui_main_windown.tabWidget.indexOf(self.ui_main_windown.tab_2), _translate("MainWindow", "Quản lý"))
        self.ui_main_windown.label.setText(_translate("MainWindow", ""))
        self.people_tabale_properties()
        self.face_table_properties()
        self.load_people_db_to_table()
        self.init_face_table()
        
    def people_tabale_properties(self):
        self.ui_main_windown.table_people.setColumnWidth(0,215)
        self.ui_main_windown.table_people.setColumnWidth(1,110)

    def face_table_properties(self):
        self.ui_main_windown.table_face.setColumnWidth(0,45)
        self.ui_main_windown.table_face.setColumnWidth(1,265)

    #=====================================================================
    #============================= face CRUD =============================
    #=====================================================================

    def chose_face_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()",
                     "","All Files (*);;Python Files (*.py)", options=options)
        self.ui_main_windown.tb_path.setText(file_name)

    def add_face(self):
        # TODO checkpath
        image = cv2.imread(self.ui_main_windown.tb_path.text())
        person_id = self.ui_main_windown.tb_add_person_id.text()
        face_id = self.ui_main_windown.tb_face_id.text()
        all_faces = self.face_management.select_all_face_of_person(person_id,0,10)
        
        face_ids = [int(face["id"]) for face in all_faces]
        face_id = 0 if len(face_ids)==0 else  max(face_ids)+1
        res = self.face_management.insert_face(person_id, str(face_id), image)
        self.load_faces_db_to_table(person_id)
        self.init_face_text_box()

    def delete_face(self):
        person_id = self.ui_main_windown.tb_add_person_id.text()
        person_name =  self.ui_main_windown.tb_add_person_name.text()
        face_id = self.ui_main_windown.tb_face_id.text()
        message = f"Do you want to delete face ID: {face_id} of {person_name}?"
        confirm_dlg = ConfirmDialog(self, message)
        if confirm_dlg.exec():
            self.face_management.delete_face_by_id(person_id, face_id)
        self.load_faces_db_to_table(person_id)
        self.init_face_text_box()

    #=====================================================================
    #============================ person CRUD ============================
    #=====================================================================

    def add_person(self):
        name = self.ui_main_windown.tb_add_person_name.text()
        id = self.ui_main_windown.tb_add_person_id.text()
        res = self.person_management.insert_person(id, name)
        self.load_people_db_to_table()

    def delete_person(self):
        id = self.ui_main_windown.tb_add_person_id.text()
        message = f"Do you want to delete person id: {id}?"
        confirm_dlg = ConfirmDialog(self, message)
        if confirm_dlg.exec():
            res = self.person_management.delete_person_by_id(id)
            if res == Validation.DETETE_PERSON_SUCCESSFULY:
                info_dlg = InfoDialog(self, f"Delete person id: {id}, successfuly!")
                info_dlg.exec()
        self.load_people_db_to_table()

    def update_person(self):
        self.update_person_window = UpDatePersonWindow(self)

    #=====================================================================
    #================= visualize_results_on_screen =======================
    #=====================================================================

    def view_image(self, image: np.array, width, height, item):
        image = cv2.resize(image, (width, height))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channel = image.shape
        step = channel * width
        qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
        item.setPixmap(QPixmap.fromImage(qImg))


    def visualize_results_on_screen(self):
        self.show_face_snapshot()
        if self.result_queue.qsize() != 0:
            frame_data = self.result_queue.get()
            image = self.visualize_recognition_results(frame_data)
            image = cv2.resize(image, (self.video_width, self.video_height))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, channel = image.shape
            step = channel * width
            qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
            self.ui_main_windown.label.setPixmap(QPixmap.fromImage(qImg))

    def visualize_recognition_results(self, frame_data):
        image = frame_data["image"]
        if self.tasks["face_recognizer"].is_enable():
            if "detection_results" in frame_data.keys():
                detection_results = frame_data["detection_results"]
                if len(detection_results) != 0:
                    bboxes, kpss = detection_results
                    person_dicts = frame_data["person_dict"]
                    for i,bbox in enumerate(bboxes):
                        #only visualize face which is recognize at least 5 frames
                        # if int(person_dicts[i]["number_frame"]) < 3:
                        #     continue 
                        person_dict = person_dicts[i]
                        face_id = int(bbox[5])
                        if (person_dict["id"] != "unknown"):
                            id = person_dict["id"]                    
                            text = f"{id} {face_id}"
                            color = (0,255,0)
                            snapshot_image = image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                            if face_id not in self.snapshot.keys():
                                self.snapshot[face_id] = {"image": snapshot_image,"person_id": id, "name": person_dict["name"], "time": time.time()}
                        else:
                            text = f"unknow {face_id}"
                            color = (0,0,255)
                        cv2.putText(image, text, (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1, cv2.LINE_AA)
                        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 1)
        return image

    def show_face_snapshot(self):
        face_ids = []
        time_list = []
        for key in self.snapshot.keys():
            face_ids.append(key)
            time_list.append(self.snapshot[key]["time"])
        face_ids, time_list = self.snap_sort(face_ids, time_list)
        
        self.face_list = [self.ui_main_windown.face_1,self.ui_main_windown.face_2, self.ui_main_windown.face_3, self.ui_main_windown.face_4, self.ui_main_windown.face_5]
        self.name_list = [self.ui_main_windown.name_1, self.ui_main_windown.name_2, self.ui_main_windown.name_3,self.ui_main_windown.name_4,self.ui_main_windown.name_5,]
        self.id_list = [self.ui_main_windown.id_1, self.ui_main_windown.id_2, self.ui_main_windown.id_3,self.ui_main_windown.id_4,self.ui_main_windown.id_5,]
        self.time_list = [self.ui_main_windown.time_1, self.ui_main_windown.time_2, self.ui_main_windown.time_3,self.ui_main_windown.time_4,self.ui_main_windown.time_5,]
        for i,face_id in enumerate(face_ids):
            if i > 4:
                self.snapshot.pop(face_id)
                continue
            img = cv2.resize(self.snapshot[face_id]["image"], (100,100))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, channel = img.shape
            step = channel * width
            qImg = QImage(img.data, width, height, step, QImage.Format_RGB888)
            self.face_list[i].setPixmap(QPixmap.fromImage(qImg))
            self.name_list[i].setText(self.snapshot[face_id]["name"])
            self.id_list[i].setText(self.snapshot[face_id]["person_id"])
            self.time_list[i].setText(str(datetime.fromtimestamp(self.snapshot[face_id]["time"])))
    
    def controlTimer(self):
        if not self.tasks["face_recognizer"].is_enable():
            self.ui_main_windown.pushButton.setText("Pause")
            self.tasks["face_recognizer"].enable()
        else:
            self.ui_main_windown.pushButton.setText("Start")
            self.tasks["face_recognizer"].disable()

    def show_window(self):
        self.show()

    def snap_sort(self, face_ids, time_list):
        """"
        Sort snapshot face by time
        """
        for i in range(len(time_list)):
            for j in range(len(time_list)-1):
                if time_list[i]>time_list[j]:
                    t_tmp = time_list[i]
                    id_tmp = face_ids[i]
                    time_list[i] = time_list[j]
                    face_ids[i] = face_ids[j]
                    time_list[j] = t_tmp
                    face_ids[j] = id_tmp
        return face_ids, time_list

    def closeEvent(self, event):
        close = QMessageBox()
        close.setText("You sure to close program?")
        close.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
        close = close.exec()
        if close == QMessageBox.Yes:
            event.accept()
            for task_key in self.tasks.keys():
                if "video_reader" in task_key:
                    self.tasks[task_key].stop_thread()
                if "face_recognizer" in task_key:
                    self.tasks[task_key].stop_thread()
        else:
            event.ignore()

    
    
