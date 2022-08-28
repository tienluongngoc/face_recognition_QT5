


from PyQt5.QtWidgets import QDialog
from PyQt5 import QtWidgets
from PyQt5 import QtCore, QtGui, QtWidgets
class ConfirmDialog(QDialog):
    def __init__(self, parent=None, message= None):
        super().__init__(parent)

        self.setWindowTitle("HELLO!")

        QBtn = QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        self.buttonBox = QtWidgets.QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        
        self.layout = QtWidgets.QVBoxLayout()
        message = QtWidgets.QLabel(message)
        self.layout.addWidget(message)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)

        


class InfoDialog(QDialog):
    def __init__(self, parent=None, message=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Information")

        self.layout = QtWidgets.QVBoxLayout()
        message = QtWidgets.QLabel(message)
        self.layout.addWidget(message)
        self.setLayout(self.layout)

