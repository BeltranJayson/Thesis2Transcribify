# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'ui_jaysonmYVNPl.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1489, 893)
        MainWindow.setMinimumSize(QSize(1000, 500))
        MainWindow.setStyleSheet(u"background-color: rgb(45, 45, 45);")
        MainWindow.setTabShape(QTabWidget.Rounded)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.Top_Bar = QFrame(self.centralwidget)
        self.Top_Bar.setObjectName(u"Top_Bar")
        self.Top_Bar.setMaximumSize(QSize(16777215, 40))
        self.Top_Bar.setStyleSheet(u"background-color: rgb(35, 35, 35);")
        self.Top_Bar.setFrameShape(QFrame.NoFrame)
        self.Top_Bar.setFrameShadow(QFrame.Raised)
        self.horizontalLayout = QHBoxLayout(self.Top_Bar)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.frame_toggle = QFrame(self.Top_Bar)
        self.frame_toggle.setObjectName(u"frame_toggle")
        self.frame_toggle.setMaximumSize(QSize(70, 40))
        self.frame_toggle.setStyleSheet(u"background-color: rgb(85, 170, 255);")
        self.frame_toggle.setFrameShape(QFrame.StyledPanel)
        self.frame_toggle.setFrameShadow(QFrame.Raised)
        self.verticalLayout_2 = QVBoxLayout(self.frame_toggle)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.Btn_Toggle = QPushButton(self.frame_toggle)
        self.Btn_Toggle.setObjectName(u"Btn_Toggle")
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Btn_Toggle.sizePolicy().hasHeightForWidth())
        self.Btn_Toggle.setSizePolicy(sizePolicy)
        self.Btn_Toggle.setStyleSheet(u"color: rgb(255, 255, 255);\n"
"background-color:rgb(43, 43, 43)")

        self.verticalLayout_2.addWidget(self.Btn_Toggle)


        self.horizontalLayout.addWidget(self.frame_toggle)

        self.frame_top = QFrame(self.Top_Bar)
        self.frame_top.setObjectName(u"frame_top")
        self.frame_top.setFrameShape(QFrame.StyledPanel)
        self.frame_top.setFrameShadow(QFrame.Raised)
        self.label_6 = QLabel(self.frame_top)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setGeometry(QRect(570, 10, 401, 21))
        font = QFont()
        font.setFamily(u"Segoe UI")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setStyleSheet(u"COLOR:WHITE;")

        self.horizontalLayout.addWidget(self.frame_top)


        self.verticalLayout.addWidget(self.Top_Bar)

        self.Content = QFrame(self.centralwidget)
        self.Content.setObjectName(u"Content")
        self.Content.setStyleSheet(u"background-color:white;")
        self.Content.setFrameShape(QFrame.NoFrame)
        self.Content.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_2 = QHBoxLayout(self.Content)
        self.horizontalLayout_2.setSpacing(0)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.frame_left_menu = QFrame(self.Content)
        self.frame_left_menu.setObjectName(u"frame_left_menu")
        self.frame_left_menu.setMinimumSize(QSize(150, 0))
        self.frame_left_menu.setMaximumSize(QSize(70, 16777215))
        self.frame_left_menu.setStyleSheet(u"background-color: rgb(35, 35, 35);")
        self.frame_left_menu.setFrameShape(QFrame.StyledPanel)
        self.frame_left_menu.setFrameShadow(QFrame.Raised)
        self.verticalLayout_3 = QVBoxLayout(self.frame_left_menu)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.frame_top_menus = QFrame(self.frame_left_menu)
        self.frame_top_menus.setObjectName(u"frame_top_menus")
        self.frame_top_menus.setFrameShape(QFrame.NoFrame)
        self.frame_top_menus.setFrameShadow(QFrame.Raised)
        self.verticalLayout_4 = QVBoxLayout(self.frame_top_menus)
        self.verticalLayout_4.setSpacing(0)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.btn_page_1 = QPushButton(self.frame_top_menus)
        self.btn_page_1.setObjectName(u"btn_page_1")
        self.btn_page_1.setMinimumSize(QSize(0, 40))
        font1 = QFont()
        font1.setFamily(u"Segoe UI")
        font1.setPointSize(11)
        self.btn_page_1.setFont(font1)
        self.btn_page_1.setStyleSheet(u"QPushButton {\n"
"	color: rgb(255, 255, 255);\n"
"	background-color: rgb(35, 35, 35);\n"
"	border: 0px solid;\n"
"}\n"
"QPushButton:hover {\n"
"	background-color: rgb(85, 170, 255);\n"
"}")

        self.verticalLayout_4.addWidget(self.btn_page_1)

        self.btn_page_2 = QPushButton(self.frame_top_menus)
        self.btn_page_2.setObjectName(u"btn_page_2")
        self.btn_page_2.setMinimumSize(QSize(0, 40))
        font2 = QFont()
        font2.setFamily(u"Segoe UI")
        font2.setPointSize(12)
        self.btn_page_2.setFont(font2)
        self.btn_page_2.setStyleSheet(u"QPushButton {\n"
"	color: rgb(255, 255, 255);\n"
"	background-color: rgb(35, 35, 35);\n"
"	border: 0px solid;\n"
"}\n"
"QPushButton:hover {\n"
"	background-color: rgb(85, 170, 255);\n"
"}")

        self.verticalLayout_4.addWidget(self.btn_page_2)

        self.btn_page_3 = QPushButton(self.frame_top_menus)
        self.btn_page_3.setObjectName(u"btn_page_3")
        self.btn_page_3.setMinimumSize(QSize(0, 40))
        self.btn_page_3.setFont(font2)
        self.btn_page_3.setStyleSheet(u"QPushButton {\n"
"	color: rgb(255, 255, 255);\n"
"	background-color: rgb(35, 35, 35);\n"
"	border: 0px solid;\n"
"}\n"
"QPushButton:hover {\n"
"	background-color: rgb(85, 170, 255);\n"
"}")

        self.verticalLayout_4.addWidget(self.btn_page_3)


        self.verticalLayout_3.addWidget(self.frame_top_menus, 0, Qt.AlignTop)


        self.horizontalLayout_2.addWidget(self.frame_left_menu)

        self.frame_pages = QFrame(self.Content)
        self.frame_pages.setObjectName(u"frame_pages")
        self.frame_pages.setStyleSheet(u"background-color:#4c4c4c;")
        self.frame_pages.setFrameShape(QFrame.StyledPanel)
        self.frame_pages.setFrameShadow(QFrame.Raised)
        self.verticalLayout_5 = QVBoxLayout(self.frame_pages)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.stackedWidget = QStackedWidget(self.frame_pages)
        self.stackedWidget.setObjectName(u"stackedWidget")
        self.stackedWidget.setStyleSheet(u"QCalendarWidget QAbstractItemView\n"
"{\n"
"background-color: rgb(192,192,192); /* \u0446\u0432\u0435\u0442 \u0444\u043e\u043d\u0430 \u0442\u0435\u043a\u0443\u0449\u0435\u0433\u043e \u043c\u0435\u0441\u044f\u0446\u0430 */\n"
"selection-background-color: yellow; /* \u0446\u0432\u0435\u0442 \u0444\u043e\u043d\u0430 \u0432\u044b\u0431\u0440\u0430\u043d\u043d\u043e\u0433\u043e \u0434\u043d\u044f */\n"
"selection-color: black; /* \u0446\u0432\u0435\u0442 \u0442\u0435\u043a\u0441\u0442\u0430 \u0432\u044b\u0431\u0440\u0430\u043d\u043d\u043e\u0433\u043e \u0434\u043d\u044f */\n"
"}")
        self.page_1 = QWidget()
        self.page_1.setObjectName(u"page_1")
        self.page_1.setMinimumSize(QSize(1240, 0))
        self.horizontalLayout_3 = QHBoxLayout(self.page_1)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.groupBox = QGroupBox(self.page_1)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setStyleSheet(u"background-color:#4c4c4c;")
        self.btnClear = QPushButton(self.groupBox)
        self.btnClear.setObjectName(u"btnClear")
        self.btnClear.setGeometry(QRect(520, 10, 111, 31))
        self.btnClear.setFont(font)
        self.btnClear.setStyleSheet(u"QPushButton {\n"
"	border: 2px solid rgb(52, 59, 72);\n"
"	border-radius: 5px;	\n"
"	background-color:blue;\n"
"	color:white;\n"
"}\n"
"QPushButton:hover {\n"
"	background-color:#356ca3;\n"
"	border: 2px solid rgb(61, 70, 86);\n"
"}\n"
"")
        self.outputText = QTextEdit(self.groupBox)
        self.outputText.setObjectName(u"outputText")
        self.outputText.setGeometry(QRect(40, 50, 591, 591))
        font3 = QFont()
        font3.setPointSize(10)
        self.outputText.setFont(font3)
        self.outputText.setStyleSheet(u"color:white;")
        self.frame_14 = QFrame(self.groupBox)
        self.frame_14.setObjectName(u"frame_14")
        self.frame_14.setGeometry(QRect(660, 10, 561, 81))
        self.frame_14.setFrameShape(QFrame.Box)
        self.frame_14.setFrameShadow(QFrame.Raised)
        self.frame_14.setLineWidth(3)
        self.frame_8 = QFrame(self.frame_14)
        self.frame_8.setObjectName(u"frame_8")
        self.frame_8.setGeometry(QRect(20, 20, 211, 41))
        self.frame_8.setStyleSheet(u"background-color:#118E0A;\n"
"border-radius:10px;")
        self.frame_8.setFrameShape(QFrame.StyledPanel)
        self.frame_8.setFrameShadow(QFrame.Raised)
        self.enableSpeakerDiarization = QCheckBox(self.frame_8)
        self.enableSpeakerDiarization.setObjectName(u"enableSpeakerDiarization")
        self.enableSpeakerDiarization.setGeometry(QRect(0, 10, 201, 21))
        self.enableSpeakerDiarization.setFont(font)
        self.enableSpeakerDiarization.setStyleSheet(u"QCheckBox{\n"
"color:white;}\n"
"QCheckBox::indicator{\n"
"                               width :30px;\n"
"                               height :30px;\n"
"\n"
"                               }\n"
"                               QCheckBox::indicator:unchecked:pressed\n"
"                               {\n"
"                               background-color : green;\n"
"                               }\n"
"")
        self.enableSpeakerDiarization.setChecked(True)
        self.inputSamplingRate = QLineEdit(self.frame_14)
        self.inputSamplingRate.setObjectName(u"inputSamplingRate")
        self.inputSamplingRate.setGeometry(QRect(410, 20, 111, 36))
        sizePolicy1 = QSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.inputSamplingRate.sizePolicy().hasHeightForWidth())
        self.inputSamplingRate.setSizePolicy(sizePolicy1)
        self.inputSamplingRate.setMinimumSize(QSize(0, 30))
        self.inputSamplingRate.setFont(font2)
        self.inputSamplingRate.setStyleSheet(u"	background-color:#5a5a5a;\n"
"	border: 2px solid white;\n"
"	color:#1aff29;\n"
"	text-align: center;")
        self.label_10 = QLabel(self.frame_14)
        self.label_10.setObjectName(u"label_10")
        self.label_10.setGeometry(QRect(260, 20, 141, 36))
        self.label_10.setFont(font2)
        self.label_10.setStyleSheet(u"color: white;")
        self.btnUpload = QPushButton(self.groupBox)
        self.btnUpload.setObjectName(u"btnUpload")
        self.btnUpload.setGeometry(QRect(660, 490, 131, 41))
        font4 = QFont()
        font4.setFamily(u"Segoe UI")
        font4.setPointSize(11)
        font4.setBold(True)
        font4.setWeight(75)
        self.btnUpload.setFont(font4)
        self.btnUpload.setStyleSheet(u"QPushButton {\n"
"	border: 2px solid rgb(52, 59, 72);\n"
"	border-radius: 5px;	\n"
"	background-color:rgb(204, 65, 68);\n"
"	color:white;\n"
"}\n"
"QPushButton:hover {\n"
"	background-color: rgb(204, 35, 72);\n"
"	border: 2px solid rgb(61, 70, 86);\n"
"}\n"
"QPushButton:pressed {	\n"
"	background-color: rgb(35, 40, 49);\n"
"	border: 2px solid rgb(43, 50, 61);\n"
"}")
        self.uploadFilename = QLineEdit(self.groupBox)
        self.uploadFilename.setObjectName(u"uploadFilename")
        self.uploadFilename.setGeometry(QRect(800, 490, 411, 41))
        font5 = QFont()
        font5.setPointSize(11)
        self.uploadFilename.setFont(font5)
        self.uploadFilename.setStyleSheet(u"color:white;")
        self.btnProcess = QPushButton(self.groupBox)
        self.btnProcess.setObjectName(u"btnProcess")
        self.btnProcess.setGeometry(QRect(220, 730, 211, 51))
        font6 = QFont()
        font6.setFamily(u"Segoe UI")
        font6.setPointSize(14)
        font6.setBold(True)
        font6.setWeight(75)
        self.btnProcess.setFont(font6)
        self.btnProcess.setStyleSheet(u"QPushButton {\n"
"	border: 2px solid rgb(52, 59, 72);\n"
"	border-radius: 5px;	\n"
"	background-color:rgb(12, 149, 53);\n"
"	color:white;\n"
"}\n"
"QPushButton:hover {\n"
"	background-color:#0c9535;\n"
"	border: 2px solid rgb(61, 70, 86);\n"
"}\n"
"QPushButton:pressed {	\n"
"	background-color: #0ebc40;\n"
"	border: 2px solid rgb(43, 50, 61);\n"
"}")
        self.consoleLog = QTextEdit(self.groupBox)
        self.consoleLog.setObjectName(u"consoleLog")
        self.consoleLog.setGeometry(QRect(660, 330, 561, 141))
        self.consoleLog.setFont(font3)
        self.consoleLog.setStyleSheet(u"background-color:black;\n"
"color:#20950e;")
        self.btnDocs = QPushButton(self.groupBox)
        self.btnDocs.setObjectName(u"btnDocs")
        self.btnDocs.setGeometry(QRect(90, 660, 131, 31))
        self.btnDocs.setFont(font)
        self.btnDocs.setStyleSheet(u"QPushButton {\n"
"	border: 2px solid rgb(52, 59, 72);\n"
"	border-radius: 5px;	\n"
"	background-color:#54a7ff;\n"
"	color:white;\n"
"}\n"
"QPushButton:hover {\n"
"	background-color:#356ca3;\n"
"	border: 2px solid rgb(61, 70, 86);\n"
"}\n"
"")
        self.btnPDF = QPushButton(self.groupBox)
        self.btnPDF.setObjectName(u"btnPDF")
        self.btnPDF.setGeometry(QRect(230, 660, 171, 31))
        self.btnPDF.setFont(font)
        self.btnPDF.setStyleSheet(u"QPushButton {\n"
"	border: 2px solid rgb(52, 59, 72);\n"
"	border-radius: 5px;	\n"
"	background-color:#9a0008;\n"
"	color:white;\n"
"}\n"
"QPushButton:hover {\n"
"	background-color:#d5000e;\n"
"	border: 2px solid rgb(61, 70, 86);\n"
"}\n"
"")
        self.label_7 = QLabel(self.groupBox)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setGeometry(QRect(40, 10, 81, 31))
        font7 = QFont()
        font7.setPointSize(14)
        self.label_7.setFont(font7)
        self.label_7.setStyleSheet(u"color:white;")
        self.btnSrt = QPushButton(self.groupBox)
        self.btnSrt.setObjectName(u"btnSrt")
        self.btnSrt.setGeometry(QRect(430, 660, 141, 31))
        self.btnSrt.setFont(font)
        self.btnSrt.setStyleSheet(u"QPushButton {\n"
"	border: 2px solid rgb(52, 59, 72);\n"
"	border-radius: 5px;	\n"
"	background-color:#bd7e00;\n"
"	color:white;\n"
"}\n"
"QPushButton:hover {\n"
"	background-color:#f1a100;\n"
"	border: 2px solid rgb(61, 70, 86);\n"
"}\n"
"")
        self.frame_6 = QFrame(self.groupBox)
        self.frame_6.setObjectName(u"frame_6")
        self.frame_6.setGeometry(QRect(660, 110, 561, 191))
        self.frame_6.setFrameShape(QFrame.Box)
        self.frame_6.setFrameShadow(QFrame.Raised)
        self.frame_6.setLineWidth(3)
        self.inputInit = QLineEdit(self.frame_6)
        self.inputInit.setObjectName(u"inputInit")
        self.inputInit.setGeometry(QRect(460, 60, 71, 36))
        sizePolicy1.setHeightForWidth(self.inputInit.sizePolicy().hasHeightForWidth())
        self.inputInit.setSizePolicy(sizePolicy1)
        self.inputInit.setMinimumSize(QSize(0, 30))
        self.inputInit.setFont(font2)
        self.inputInit.setStyleSheet(u"	background-color:#5a5a5a;\n"
"	border: 2px solid white;\n"
"	color:#1aff29;\n"
"	text-align: center;")
        self.inputGamma = QLineEdit(self.frame_6)
        self.inputGamma.setObjectName(u"inputGamma")
        self.inputGamma.setGeometry(QRect(160, 60, 111, 36))
        sizePolicy1.setHeightForWidth(self.inputGamma.sizePolicy().hasHeightForWidth())
        self.inputGamma.setSizePolicy(sizePolicy1)
        self.inputGamma.setMinimumSize(QSize(0, 30))
        self.inputGamma.setFont(font2)
        self.inputGamma.setStyleSheet(u"	background-color:#5a5a5a;\n"
"	border: 2px solid white;\n"
"	color:#1aff29;\n"
"	text-align: center;")
        self.label_15 = QLabel(self.frame_6)
        self.label_15.setObjectName(u"label_15")
        self.label_15.setGeometry(QRect(290, 60, 151, 36))
        self.label_15.setFont(font2)
        self.label_15.setStyleSheet(u"color: white;")
        self.label_16 = QLabel(self.frame_6)
        self.label_16.setObjectName(u"label_16")
        self.label_16.setGeometry(QRect(10, 60, 141, 36))
        self.label_16.setFont(font2)
        self.label_16.setStyleSheet(u"color: white;")
        self.label_17 = QLabel(self.frame_6)
        self.label_17.setObjectName(u"label_17")
        self.label_17.setGeometry(QRect(190, 20, 251, 31))
        self.label_17.setFont(font2)
        self.label_17.setStyleSheet(u"color: green;")
        self.inputJobs = QLineEdit(self.frame_6)
        self.inputJobs.setObjectName(u"inputJobs")
        self.inputJobs.setGeometry(QRect(460, 120, 71, 36))
        sizePolicy1.setHeightForWidth(self.inputJobs.sizePolicy().hasHeightForWidth())
        self.inputJobs.setSizePolicy(sizePolicy1)
        self.inputJobs.setMinimumSize(QSize(0, 30))
        self.inputJobs.setFont(font2)
        self.inputJobs.setStyleSheet(u"	background-color:#5a5a5a;\n"
"	border: 2px solid white;\n"
"	color:#1aff29;\n"
"	text-align: center;")
        self.label_18 = QLabel(self.frame_6)
        self.label_18.setObjectName(u"label_18")
        self.label_18.setGeometry(QRect(290, 120, 161, 36))
        self.label_18.setFont(font2)
        self.label_18.setStyleSheet(u"color: white;")
        self.InputEigen = QLineEdit(self.frame_6)
        self.InputEigen.setObjectName(u"InputEigen")
        self.InputEigen.setGeometry(QRect(160, 120, 111, 36))
        sizePolicy1.setHeightForWidth(self.InputEigen.sizePolicy().hasHeightForWidth())
        self.InputEigen.setSizePolicy(sizePolicy1)
        self.InputEigen.setMinimumSize(QSize(0, 30))
        self.InputEigen.setFont(font2)
        self.InputEigen.setStyleSheet(u"	background-color:#5a5a5a;\n"
"	border: 2px solid white;\n"
"	color:#1aff29;\n"
"	text-align: center;")
        self.label_19 = QLabel(self.frame_6)
        self.label_19.setObjectName(u"label_19")
        self.label_19.setGeometry(QRect(10, 120, 141, 36))
        self.label_19.setFont(font2)
        self.label_19.setStyleSheet(u"color: white;")
        self.selectModel = QComboBox(self.groupBox)
        self.selectModel.addItem("")
        self.selectModel.addItem("")
        self.selectModel.setObjectName(u"selectModel")
        self.selectModel.setGeometry(QRect(790, 580, 361, 41))
        font8 = QFont()
        font8.setFamily(u"Segoe UI")
        font8.setPointSize(14)
        self.selectModel.setFont(font8)
        self.selectModel.setStyleSheet(u"QComboBox{\n"
"	background-color:white;\n"
"	border-radius: 5px;\n"
"	padding: 5px;\n"
"	padding-left: 10px;\n"
"	color:black;\n"
"	border: 2px solid rgb(27, 29, 35);\n"
"}\n"
"QComboBox:hover{\n"
"	border: 2px solid rgb(64, 71, 88);\n"
"}\n"
"QComboBox QAbstractItemView {\n"
"	color: white;\n"
"	background-color:rgb(204, 65, 68);\n"
"	padding: 10px;\n"
"	selection-background-color: rgb(39, 44, 54);\n"
"}")
        self.label_2 = QLabel(self.groupBox)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(930, 630, 101, 16))
        self.label_2.setFont(font5)
        self.label_2.setStyleSheet(u"color:white;")
        self.label_14 = QLabel(self.groupBox)
        self.label_14.setObjectName(u"label_14")
        self.label_14.setGeometry(QRect(880, 530, 371, 31))
        self.label_14.setFont(font2)
        self.label_14.setStyleSheet(u"color: white;")
        self.label = QLabel(self.groupBox)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(470, 710, 811, 71))
        self.label.setStyleSheet(u"color:white;")

        self.horizontalLayout_3.addWidget(self.groupBox)

        self.stackedWidget.addWidget(self.page_1)
        self.page_2 = QWidget()
        self.page_2.setObjectName(u"page_2")
        self.verticalLayout_6 = QVBoxLayout(self.page_2)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.frame_3 = QFrame(self.page_2)
        self.frame_3.setObjectName(u"frame_3")
        self.frame_3.setFrameShape(QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QFrame.Raised)
        self.groupBox_2 = QGroupBox(self.frame_3)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.groupBox_2.setGeometry(QRect(0, 0, 1271, 801))
        self.groupBox_2.setStyleSheet(u"background-color:#4c4c4c;")
        self.btnClearLive = QPushButton(self.groupBox_2)
        self.btnClearLive.setObjectName(u"btnClearLive")
        self.btnClearLive.setGeometry(QRect(1080, 580, 111, 31))
        self.btnClearLive.setFont(font)
        self.btnClearLive.setStyleSheet(u"QPushButton {\n"
"	border: 2px solid rgb(52, 59, 72);\n"
"	border-radius: 5px;	\n"
"	background-color:blue;\n"
"	color:white;\n"
"}\n"
"QPushButton:hover {\n"
"	background-color:#356ca3;\n"
"	border: 2px solid rgb(61, 70, 86);\n"
"}\n"
"")
        self.outputText_Live = QTextEdit(self.groupBox_2)
        self.outputText_Live.setObjectName(u"outputText_Live")
        self.outputText_Live.setGeometry(QRect(110, 50, 1061, 521))
        self.outputText_Live.setFont(font3)
        self.outputText_Live.setStyleSheet(u"color:white;")
        self.btnStartLive = QPushButton(self.groupBox_2)
        self.btnStartLive.setObjectName(u"btnStartLive")
        self.btnStartLive.setGeometry(QRect(130, 700, 211, 51))
        self.btnStartLive.setFont(font)
        self.btnStartLive.setStyleSheet(u"QPushButton {\n"
"	border: 2px solid rgb(52, 59, 72);\n"
"	border-radius: 5px;	\n"
"	background-color:rgb(12, 149, 53);\n"
"	color:white;\n"
"}\n"
"QPushButton:hover {\n"
"	background-color:#0c9535;\n"
"	border: 2px solid rgb(61, 70, 86);\n"
"}\n"
"QPushButton:pressed {	\n"
"	background-color: #0ebc40;\n"
"	border: 2px solid rgb(43, 50, 61);\n"
"}")
        self.btnDocsLive = QPushButton(self.groupBox_2)
        self.btnDocsLive.setObjectName(u"btnDocsLive")
        self.btnDocsLive.setGeometry(QRect(900, 580, 161, 31))
        self.btnDocsLive.setFont(font)
        self.btnDocsLive.setStyleSheet(u"QPushButton {\n"
"	border: 2px solid rgb(52, 59, 72);\n"
"	border-radius: 5px;	\n"
"	background-color:#54a7ff;\n"
"	color:white;\n"
"}\n"
"QPushButton:hover {\n"
"	background-color:#356ca3;\n"
"	border: 2px solid rgb(61, 70, 86);\n"
"}\n"
"")
        self.btnPDFLive = QPushButton(self.groupBox_2)
        self.btnPDFLive.setObjectName(u"btnPDFLive")
        self.btnPDFLive.setGeometry(QRect(710, 580, 171, 31))
        self.btnPDFLive.setFont(font)
        self.btnPDFLive.setStyleSheet(u"QPushButton {\n"
"	border: 2px solid rgb(52, 59, 72);\n"
"	border-radius: 5px;	\n"
"	background-color:#9a0008;\n"
"	color:white;\n"
"}\n"
"QPushButton:hover {\n"
"	background-color:#d5000e;\n"
"	border: 2px solid rgb(61, 70, 86);\n"
"}\n"
"")
        self.label_8 = QLabel(self.groupBox_2)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setGeometry(QRect(520, 20, 231, 21))
        self.label_8.setFont(font7)
        self.label_8.setStyleSheet(u"color:white;")
        self.selectLiveModel = QComboBox(self.groupBox_2)
        self.selectLiveModel.addItem("")
        self.selectLiveModel.addItem("")
        self.selectLiveModel.setObjectName(u"selectLiveModel")
        self.selectLiveModel.setGeometry(QRect(120, 630, 211, 41))
        self.selectLiveModel.setFont(font8)
        self.selectLiveModel.setStyleSheet(u"QComboBox{\n"
"	background-color:white;\n"
"	border-radius: 5px;\n"
"	padding: 5px;\n"
"	padding-left: 10px;\n"
"	color:black;\n"
"	border: 2px solid rgb(27, 29, 35);\n"
"}\n"
"QComboBox:hover{\n"
"	border: 2px solid rgb(64, 71, 88);\n"
"}\n"
"QComboBox QAbstractItemView {\n"
"	color: white;\n"
"	background-color:rgb(204, 65, 68);\n"
"	padding: 10px;\n"
"	selection-background-color: rgb(39, 44, 54);\n"
"}")
        self.label_12 = QLabel(self.groupBox_2)
        self.label_12.setObjectName(u"label_12")
        self.label_12.setGeometry(QRect(150, 600, 171, 21))
        self.label_12.setFont(font5)
        self.label_12.setStyleSheet(u"color:white;")
        self.btnStopLive = QPushButton(self.groupBox_2)
        self.btnStopLive.setObjectName(u"btnStopLive")
        self.btnStopLive.setGeometry(QRect(360, 700, 211, 51))
        self.btnStopLive.setFont(font)
        self.btnStopLive.setStyleSheet(u"QPushButton {\n"
"	border: 2px solid rgb(52, 59, 72);\n"
"	border-radius: 5px;	\n"
"	background-color:red;\n"
"	color:white;\n"
"}\n"
"QPushButton:hover {\n"
"	background-color:#0c9535;\n"
"	border: 2px solid rgb(61, 70, 86);\n"
"}\n"
"QPushButton:pressed {	\n"
"	background-color: #0ebc40;\n"
"	border: 2px solid rgb(43, 50, 61);\n"
"}")
        self.selectMicrophone = QComboBox(self.groupBox_2)
        self.selectMicrophone.setObjectName(u"selectMicrophone")
        self.selectMicrophone.setGeometry(QRect(350, 630, 391, 41))
        self.selectMicrophone.setFont(font8)
        self.selectMicrophone.setStyleSheet(u"QComboBox{\n"
"	background-color:white;\n"
"	border-radius: 5px;\n"
"	padding: 5px;\n"
"	padding-left: 10px;\n"
"	color:black;\n"
"	border: 2px solid rgb(27, 29, 35);\n"
"}\n"
"QComboBox:hover{\n"
"	border: 2px solid rgb(64, 71, 88);\n"
"}\n"
"QComboBox QAbstractItemView {\n"
"	color: white;\n"
"	background-color:rgb(204, 65, 68);\n"
"	padding: 10px;\n"
"	selection-background-color: rgb(39, 44, 54);\n"
"}")
        self.label_13 = QLabel(self.groupBox_2)
        self.label_13.setObjectName(u"label_13")
        self.label_13.setGeometry(QRect(370, 600, 171, 21))
        self.label_13.setFont(font5)
        self.label_13.setStyleSheet(u"color:white;")

        self.verticalLayout_6.addWidget(self.frame_3)

        self.stackedWidget.addWidget(self.page_2)
        self.page_3 = QWidget()
        self.page_3.setObjectName(u"page_3")
        self.verticalLayout_8 = QVBoxLayout(self.page_3)
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.frame_7 = QFrame(self.page_3)
        self.frame_7.setObjectName(u"frame_7")
        self.frame_7.setFrameShape(QFrame.StyledPanel)
        self.frame_7.setFrameShadow(QFrame.Raised)

        self.verticalLayout_8.addWidget(self.frame_7)

        self.stackedWidget.addWidget(self.page_3)
        self.page_4 = QWidget()
        self.page_4.setObjectName(u"page_4")
        self.stackedWidget.addWidget(self.page_4)

        self.verticalLayout_5.addWidget(self.stackedWidget)


        self.horizontalLayout_2.addWidget(self.frame_pages)


        self.verticalLayout.addWidget(self.Content)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        self.stackedWidget.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Speech Diarization and Transcription", None))
        self.Btn_Toggle.setText(QCoreApplication.translate("MainWindow", u"Menu", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"SPEECH TRANSCRIPTION AND DIARIZATION", None))
        self.btn_page_1.setText(QCoreApplication.translate("MainWindow", u"Main", None))
        self.btn_page_2.setText(QCoreApplication.translate("MainWindow", u"Live Transcription", None))
        self.btn_page_3.setText(QCoreApplication.translate("MainWindow", u"Help", None))
        self.groupBox.setTitle("")
        self.btnClear.setText(QCoreApplication.translate("MainWindow", u"CLEAR", None))
        self.enableSpeakerDiarization.setText(QCoreApplication.translate("MainWindow", u"Speaker Diarization", None))
        self.inputSamplingRate.setText(QCoreApplication.translate("MainWindow", u"16000", None))
        self.inputSamplingRate.setPlaceholderText("")
        self.label_10.setText(QCoreApplication.translate("MainWindow", u"Sampling Rate", None))
        self.btnUpload.setText(QCoreApplication.translate("MainWindow", u"SELECT FILE", None))
        self.uploadFilename.setText("")
        self.btnProcess.setText(QCoreApplication.translate("MainWindow", u"PROCESS", None))
        self.btnDocs.setText(QCoreApplication.translate("MainWindow", u"EXPORT  DOCS", None))
        self.btnPDF.setText(QCoreApplication.translate("MainWindow", u"EXPORT TO PDF", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"Output :", None))
        self.btnSrt.setText(QCoreApplication.translate("MainWindow", u"CREATE SRT", None))
        self.inputInit.setText(QCoreApplication.translate("MainWindow", u"100", None))
        self.inputInit.setPlaceholderText("")
        self.inputGamma.setText(QCoreApplication.translate("MainWindow", u"1.0", None))
        self.inputGamma.setPlaceholderText("")
        self.label_15.setText(QCoreApplication.translate("MainWindow", u"n_init", None))
        self.label_16.setText(QCoreApplication.translate("MainWindow", u"Gamma", None))
        self.label_17.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p><span style=\" font-weight:600; color:#29c610;\">Advance Settings</span></p></body></html>", None))
        self.inputJobs.setText(QCoreApplication.translate("MainWindow", u"-1", None))
        self.inputJobs.setPlaceholderText("")
        self.label_18.setText(QCoreApplication.translate("MainWindow", u"n_jobs", None))
        self.InputEigen.setText(QCoreApplication.translate("MainWindow", u"1e-7", None))
        self.InputEigen.setPlaceholderText("")
        self.label_19.setText(QCoreApplication.translate("MainWindow", u"eigen_tol", None))
        self.selectModel.setItemText(0, QCoreApplication.translate("MainWindow", u"English 1", None))
        self.selectModel.setItemText(1, QCoreApplication.translate("MainWindow", u"Filipino", None))

        self.label_2.setText(QCoreApplication.translate("MainWindow", u"Select Model ", None))
        self.label_14.setText(QCoreApplication.translate("MainWindow", u"Supported file type (mp4, mp3, wav)", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'Courier New';\">n_jobs</span>: Number of CPU cores to use (-1 for all available cores)</li><li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'Courier New';\">gamma</span>: Width of the RBF kernel (higher value = more sparse affinity matrix)</li><li style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'Courier New';\">n_init</span>: Number of times to run clustering algorithm with different initializations</li><li style=\" margin-top:0px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-"
                        "family:'Courier New';\">eigen_tol</span>: Tolerance for eigenvalues when computing eigenvectors of affinity matrix (lower value can improve separation but increase computation time)</li></ul></body></html>", None))
        self.groupBox_2.setTitle("")
        self.btnClearLive.setText(QCoreApplication.translate("MainWindow", u"CLEAR", None))
        self.btnStartLive.setText(QCoreApplication.translate("MainWindow", u"START RECORDING", None))
        self.btnDocsLive.setText(QCoreApplication.translate("MainWindow", u"EXPORT  DOCS", None))
        self.btnPDFLive.setText(QCoreApplication.translate("MainWindow", u"EXPORT TO PDF", None))
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"Live Speech Transcription", None))
        self.selectLiveModel.setItemText(0, QCoreApplication.translate("MainWindow", u"English 1", None))
        self.selectLiveModel.setItemText(1, QCoreApplication.translate("MainWindow", u"Filipino", None))

        self.label_12.setText(QCoreApplication.translate("MainWindow", u"Select Language ", None))
        self.btnStopLive.setText(QCoreApplication.translate("MainWindow", u"STOP ", None))
        self.label_13.setText(QCoreApplication.translate("MainWindow", u"Microphone", None))
    # retranslateUi

