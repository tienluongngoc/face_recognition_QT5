from email import message
from .ui_update_person import QtWidgets, Ui_window_update_person
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget
from .ui_utils import ConfirmDialog, InfoDialog
from src.schemas.validation import Validation

class UpDatePersonWindow:
    def __init__(self, main_window) -> None:
        self.main_window  = main_window
        self.this_window = QtWidgets.QMainWindow()
        self.update_person_ui =  Ui_window_update_person()
        self.update_person_ui.setupUi(self.this_window)
        self.update_person_ui.bt_update.clicked.connect(self.update_person_info)
        
        self.current_name = self.main_window.ui_main_windown.tb_add_person_name.text()
        self.current_id = self.main_window.ui_main_windown.tb_add_person_id.text()
        self.update_person_ui.tb_current_id.setText(self.current_id)
        self.update_person_ui.tb_current_name.setText(self.current_name)
        self.update_person_ui.tb_new_id.setText(self.current_id)
        self.update_person_ui.tb_new_name.setText(self.current_name)

        self.this_window.show()  

    def update_person_info(self):
        new_id = self.update_person_ui.tb_new_id.text()
        new_name = self.update_person_ui.tb_new_name.text()
        message = f"Do you want to update person ID: {self.current_id} with name:  {self.current_name} to person ID: {new_id} with new name: {new_name}?"
        confirm_dlg = ConfirmDialog(self.this_window, message)
        self.person_management = self.main_window.person_management
        if confirm_dlg.exec():
            res_1 = self.person_management.update_person_name(self.current_id, new_name)
            res_2 = self.person_management.update_person_id(self.current_id, new_id)
            if res_1 ==  Validation.UPDATE_PERSON_NAME and res_2 == Validation.UPDATE_PERSON_ID:
                info_dlg = InfoDialog(self.this_window, f"Update successfuly!")
                info_dlg.exec()
            self.main_window.load_people_db_to_table()
            self.this_window.hide()
