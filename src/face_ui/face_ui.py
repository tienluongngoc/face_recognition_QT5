# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'src/face_ui/face_ui.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1385, 642)
        MainWindow.setMaximumSize(QtCore.QSize(1920, 1080))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(10, 10, 1361, 601))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.tabWidget = QtWidgets.QTabWidget(self.verticalLayoutWidget)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.label = QtWidgets.QLabel(self.tab)
        self.label.setGeometry(QtCore.QRect(10, 0, 1000, 560))
        self.label.setMaximumSize(QtCore.QSize(1280, 720))
        self.label.setLineWidth(1)
        self.label.setObjectName("label")
        self.verticalLayoutWidget_3 = QtWidgets.QWidget(self.tab)
        self.verticalLayoutWidget_3.setGeometry(QtCore.QRect(1030, 10, 321, 491))
        self.verticalLayoutWidget_3.setObjectName("verticalLayoutWidget_3")
        self.verticalLayout_14 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_3)
        self.verticalLayout_14.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_14.setObjectName("verticalLayout_14")
        self.horizontalLayout_15 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_15.setObjectName("horizontalLayout_15")
        self.face_1 = QtWidgets.QLabel(self.verticalLayoutWidget_3)
        self.face_1.setMaximumSize(QtCore.QSize(100, 100))
        self.face_1.setObjectName("face_1")
        self.horizontalLayout_15.addWidget(self.face_1)
        self.verticalLayout_15 = QtWidgets.QVBoxLayout()
        self.verticalLayout_15.setSpacing(0)
        self.verticalLayout_15.setObjectName("verticalLayout_15")
        self.name_1 = QtWidgets.QLabel(self.verticalLayoutWidget_3)
        self.name_1.setObjectName("name_1")
        self.verticalLayout_15.addWidget(self.name_1)
        self.id_1 = QtWidgets.QLabel(self.verticalLayoutWidget_3)
        self.id_1.setObjectName("id_1")
        self.verticalLayout_15.addWidget(self.id_1)
        self.time_1 = QtWidgets.QLabel(self.verticalLayoutWidget_3)
        self.time_1.setObjectName("time_1")
        self.verticalLayout_15.addWidget(self.time_1)
        self.horizontalLayout_15.addLayout(self.verticalLayout_15)
        self.verticalLayout_14.addLayout(self.horizontalLayout_15)
        self.horizontalLayout_16 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_16.setObjectName("horizontalLayout_16")
        self.face_2 = QtWidgets.QLabel(self.verticalLayoutWidget_3)
        self.face_2.setMaximumSize(QtCore.QSize(100, 100))
        self.face_2.setObjectName("face_2")
        self.horizontalLayout_16.addWidget(self.face_2)
        self.verticalLayout_18 = QtWidgets.QVBoxLayout()
        self.verticalLayout_18.setSpacing(0)
        self.verticalLayout_18.setObjectName("verticalLayout_18")
        self.name_2 = QtWidgets.QLabel(self.verticalLayoutWidget_3)
        self.name_2.setObjectName("name_2")
        self.verticalLayout_18.addWidget(self.name_2)
        self.id_2 = QtWidgets.QLabel(self.verticalLayoutWidget_3)
        self.id_2.setObjectName("id_2")
        self.verticalLayout_18.addWidget(self.id_2)
        self.time_2 = QtWidgets.QLabel(self.verticalLayoutWidget_3)
        self.time_2.setObjectName("time_2")
        self.verticalLayout_18.addWidget(self.time_2)
        self.horizontalLayout_16.addLayout(self.verticalLayout_18)
        self.verticalLayout_14.addLayout(self.horizontalLayout_16)
        self.horizontalLayout_17 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_17.setObjectName("horizontalLayout_17")
        self.face_3 = QtWidgets.QLabel(self.verticalLayoutWidget_3)
        self.face_3.setMaximumSize(QtCore.QSize(100, 100))
        self.face_3.setObjectName("face_3")
        self.horizontalLayout_17.addWidget(self.face_3)
        self.verticalLayout_20 = QtWidgets.QVBoxLayout()
        self.verticalLayout_20.setSpacing(0)
        self.verticalLayout_20.setObjectName("verticalLayout_20")
        self.name_3 = QtWidgets.QLabel(self.verticalLayoutWidget_3)
        self.name_3.setObjectName("name_3")
        self.verticalLayout_20.addWidget(self.name_3)
        self.id_3 = QtWidgets.QLabel(self.verticalLayoutWidget_3)
        self.id_3.setObjectName("id_3")
        self.verticalLayout_20.addWidget(self.id_3)
        self.time_3 = QtWidgets.QLabel(self.verticalLayoutWidget_3)
        self.time_3.setObjectName("time_3")
        self.verticalLayout_20.addWidget(self.time_3)
        self.horizontalLayout_17.addLayout(self.verticalLayout_20)
        self.verticalLayout_14.addLayout(self.horizontalLayout_17)
        self.horizontalLayout_18 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_18.setSpacing(6)
        self.horizontalLayout_18.setObjectName("horizontalLayout_18")
        self.face_4 = QtWidgets.QLabel(self.verticalLayoutWidget_3)
        self.face_4.setMaximumSize(QtCore.QSize(100, 100))
        self.face_4.setObjectName("face_4")
        self.horizontalLayout_18.addWidget(self.face_4)
        self.verticalLayout_21 = QtWidgets.QVBoxLayout()
        self.verticalLayout_21.setSpacing(0)
        self.verticalLayout_21.setObjectName("verticalLayout_21")
        self.name_4 = QtWidgets.QLabel(self.verticalLayoutWidget_3)
        self.name_4.setObjectName("name_4")
        self.verticalLayout_21.addWidget(self.name_4)
        self.id_4 = QtWidgets.QLabel(self.verticalLayoutWidget_3)
        self.id_4.setObjectName("id_4")
        self.verticalLayout_21.addWidget(self.id_4)
        self.time_4 = QtWidgets.QLabel(self.verticalLayoutWidget_3)
        self.time_4.setObjectName("time_4")
        self.verticalLayout_21.addWidget(self.time_4)
        self.horizontalLayout_18.addLayout(self.verticalLayout_21)
        self.verticalLayout_14.addLayout(self.horizontalLayout_18)
        self.horizontalLayout_19 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_19.setObjectName("horizontalLayout_19")
        self.face_5 = QtWidgets.QLabel(self.verticalLayoutWidget_3)
        self.face_5.setMaximumSize(QtCore.QSize(100, 100))
        self.face_5.setObjectName("face_5")
        self.horizontalLayout_19.addWidget(self.face_5)
        self.verticalLayout_22 = QtWidgets.QVBoxLayout()
        self.verticalLayout_22.setSpacing(0)
        self.verticalLayout_22.setObjectName("verticalLayout_22")
        self.name_5 = QtWidgets.QLabel(self.verticalLayoutWidget_3)
        self.name_5.setObjectName("name_5")
        self.verticalLayout_22.addWidget(self.name_5)
        self.id_5 = QtWidgets.QLabel(self.verticalLayoutWidget_3)
        self.id_5.setObjectName("id_5")
        self.verticalLayout_22.addWidget(self.id_5)
        self.time_5 = QtWidgets.QLabel(self.verticalLayoutWidget_3)
        self.time_5.setObjectName("time_5")
        self.verticalLayout_22.addWidget(self.time_5)
        self.horizontalLayout_19.addLayout(self.verticalLayout_22)
        self.verticalLayout_14.addLayout(self.horizontalLayout_19)
        self.pushButton = QtWidgets.QPushButton(self.tab)
        self.pushButton.setGeometry(QtCore.QRect(1030, 520, 321, 41))
        self.pushButton.setObjectName("pushButton")
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.table_people = QtWidgets.QTableWidget(self.tab_2)
        self.table_people.setGeometry(QtCore.QRect(0, 30, 441, 521))
        self.table_people.setObjectName("table_people")
        self.table_people.setColumnCount(3)
        self.table_people.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.table_people.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_people.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_people.setHorizontalHeaderItem(2, item)
        self.table_face = QtWidgets.QTableWidget(self.tab_2)
        self.table_face.setGeometry(QtCore.QRect(460, 330, 341, 221))
        self.table_face.setObjectName("table_face")
        self.table_face.setColumnCount(2)
        self.table_face.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.table_face.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_face.setHorizontalHeaderItem(1, item)
        self.face_preview = QtWidgets.QLabel(self.tab_2)
        self.face_preview.setGeometry(QtCore.QRect(820, 260, 521, 281))
        self.face_preview.setObjectName("face_preview")
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.tab_2)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(820, 20, 351, 121))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_2 = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.label_2.setMinimumSize(QtCore.QSize(50, 0))
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        self.tb_add_person_name = QtWidgets.QLineEdit(self.verticalLayoutWidget_2)
        self.tb_add_person_name.setMaxLength(525)
        self.tb_add_person_name.setObjectName("tb_add_person_name")
        self.horizontalLayout.addWidget(self.tb_add_person_name)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.horizontalLayout_21 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_21.setObjectName("horizontalLayout_21")
        self.label_8 = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.label_8.setMinimumSize(QtCore.QSize(50, 0))
        self.label_8.setObjectName("label_8")
        self.horizontalLayout_21.addWidget(self.label_8)
        self.tb_add_person_id = QtWidgets.QLineEdit(self.verticalLayoutWidget_2)
        self.tb_add_person_id.setObjectName("tb_add_person_id")
        self.horizontalLayout_21.addWidget(self.tb_add_person_id)
        self.verticalLayout_2.addLayout(self.horizontalLayout_21)
        self.horizontalLayout_22 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_22.setObjectName("horizontalLayout_22")
        self.label_14 = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.label_14.setMinimumSize(QtCore.QSize(50, 0))
        self.label_14.setObjectName("label_14")
        self.horizontalLayout_22.addWidget(self.label_14)
        self.tb_add_person_role = QtWidgets.QLineEdit(self.verticalLayoutWidget_2)
        self.tb_add_person_role.setObjectName("tb_add_person_role")
        self.horizontalLayout_22.addWidget(self.tb_add_person_role)
        self.verticalLayout_2.addLayout(self.horizontalLayout_22)
        self.verticalLayoutWidget_9 = QtWidgets.QWidget(self.tab_2)
        self.verticalLayoutWidget_9.setGeometry(QtCore.QRect(1200, 10, 131, 134))
        self.verticalLayoutWidget_9.setObjectName("verticalLayoutWidget_9")
        self.verticalLayout_23 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_9)
        self.verticalLayout_23.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_23.setObjectName("verticalLayout_23")
        self.bt_add_person = QtWidgets.QPushButton(self.verticalLayoutWidget_9)
        self.bt_add_person.setMinimumSize(QtCore.QSize(0, 40))
        self.bt_add_person.setObjectName("bt_add_person")
        self.verticalLayout_23.addWidget(self.bt_add_person)
        self.bt_update_person = QtWidgets.QPushButton(self.verticalLayoutWidget_9)
        self.bt_update_person.setMinimumSize(QtCore.QSize(0, 40))
        self.bt_update_person.setObjectName("bt_update_person")
        self.verticalLayout_23.addWidget(self.bt_update_person)
        self.bt_delete_person = QtWidgets.QPushButton(self.verticalLayoutWidget_9)
        self.bt_delete_person.setMinimumSize(QtCore.QSize(0, 40))
        self.bt_delete_person.setObjectName("bt_delete_person")
        self.verticalLayout_23.addWidget(self.bt_delete_person)
        self.tableWidget = QtWidgets.QTableWidget(self.tab_2)
        self.tableWidget.setGeometry(QtCore.QRect(460, 30, 341, 281))
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(2)
        self.tableWidget.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(1, item)
        self.verticalLayoutWidget_4 = QtWidgets.QWidget(self.tab_2)
        self.verticalLayoutWidget_4.setGeometry(QtCore.QRect(820, 180, 351, 80))
        self.verticalLayoutWidget_4.setObjectName("verticalLayoutWidget_4")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_4)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_4 = QtWidgets.QLabel(self.verticalLayoutWidget_4)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_4.addWidget(self.label_4)
        self.tb_path = QtWidgets.QLineEdit(self.verticalLayoutWidget_4)
        self.tb_path.setObjectName("tb_path")
        self.horizontalLayout_4.addWidget(self.tb_path)
        self.verticalLayout_3.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_3 = QtWidgets.QLabel(self.verticalLayoutWidget_4)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_5.addWidget(self.label_3)
        self.tb_face_id = QtWidgets.QLineEdit(self.verticalLayoutWidget_4)
        self.tb_face_id.setObjectName("tb_face_id")
        self.horizontalLayout_5.addWidget(self.tb_face_id)
        self.verticalLayout_3.addLayout(self.horizontalLayout_5)
        self.verticalLayoutWidget_5 = QtWidgets.QWidget(self.tab_2)
        self.verticalLayoutWidget_5.setGeometry(QtCore.QRect(1199, 170, 141, 89))
        self.verticalLayoutWidget_5.setObjectName("verticalLayoutWidget_5")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_5)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.bt_chose_file = QtWidgets.QPushButton(self.verticalLayoutWidget_5)
        self.bt_chose_file.setObjectName("bt_chose_file")
        self.verticalLayout_4.addWidget(self.bt_chose_file)
        self.bt_add_face = QtWidgets.QPushButton(self.verticalLayoutWidget_5)
        self.bt_add_face.setObjectName("bt_add_face")
        self.verticalLayout_4.addWidget(self.bt_add_face)
        self.bt_delete_face = QtWidgets.QPushButton(self.verticalLayoutWidget_5)
        self.bt_delete_face.setObjectName("bt_delete_face")
        self.verticalLayout_4.addWidget(self.bt_delete_face)
        self.tabWidget.addTab(self.tab_2, "")
        self.verticalLayout.addWidget(self.tabWidget)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1385, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "TextLabel"))
        self.face_1.setText(_translate("MainWindow", "TextLabel"))
        self.name_1.setText(_translate("MainWindow", "Name: Tien Luong Ngoc"))
        self.id_1.setText(_translate("MainWindow", "ID: tienln"))
        self.time_1.setText(_translate("MainWindow", "Time: 07:30:45"))
        self.face_2.setText(_translate("MainWindow", "TextLabel"))
        self.name_2.setText(_translate("MainWindow", "Name: Tien Luong Ngoc"))
        self.id_2.setText(_translate("MainWindow", "ID: tienln"))
        self.time_2.setText(_translate("MainWindow", "Time: 07:30:45"))
        self.face_3.setText(_translate("MainWindow", "TextLabel"))
        self.name_3.setText(_translate("MainWindow", "Name: Tien Luong Ngoc"))
        self.id_3.setText(_translate("MainWindow", "ID: tienln"))
        self.time_3.setText(_translate("MainWindow", "Time: 07:30:45"))
        self.face_4.setText(_translate("MainWindow", "TextLabel"))
        self.name_4.setText(_translate("MainWindow", "Name: Tien Luong Ngoc"))
        self.id_4.setText(_translate("MainWindow", "ID: tienln"))
        self.time_4.setText(_translate("MainWindow", "Time: 07:30:45"))
        self.face_5.setText(_translate("MainWindow", "TextLabel"))
        self.name_5.setText(_translate("MainWindow", "Name: Tien Luong Ngoc"))
        self.id_5.setText(_translate("MainWindow", "ID: tienln"))
        self.time_5.setText(_translate("MainWindow", "Time: 07:30:45"))
        self.pushButton.setText(_translate("MainWindow", "PushButton"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Tab 1"))
        item = self.table_people.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Name"))
        item = self.table_people.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "ID"))
        item = self.table_people.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "Role"))
        item = self.table_face.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Face ID"))
        item = self.table_face.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Path image"))
        self.face_preview.setText(_translate("MainWindow", "TextLabel"))
        self.label_2.setText(_translate("MainWindow", "Name"))
        self.label_8.setText(_translate("MainWindow", "ID"))
        self.label_14.setText(_translate("MainWindow", "Role"))
        self.bt_add_person.setText(_translate("MainWindow", "Insert Person"))
        self.bt_update_person.setText(_translate("MainWindow", "Update person"))
        self.bt_delete_person.setText(_translate("MainWindow", "Delete person"))
        item = self.tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Data"))
        item = self.tableWidget.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Time"))
        self.label_4.setText(_translate("MainWindow", "File path"))
        self.tb_path.setText(_translate("MainWindow", "...image/face/path/"))
        self.label_3.setText(_translate("MainWindow", "Face ID"))
        self.bt_chose_file.setText(_translate("MainWindow", "chose file path"))
        self.bt_add_face.setText(_translate("MainWindow", "Insert face"))
        self.bt_delete_face.setText(_translate("MainWindow", "Delete face"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Tab 2"))
