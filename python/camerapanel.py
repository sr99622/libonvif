import os
import sys
import numpy as np
import cv2
from time import sleep
from PyQt6.QtWidgets import QDialogButtonBox, QLineEdit, QPushButton, \
QGridLayout, QWidget, QDialog, QLabel, QMessageBox, QListWidget, \
QTabWidget
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QSettings
from videotab import VideoTab
from imagetab import ImageTab
from networktab import NetworkTab
from ptztab import PTZTab
from admintab import AdminTab

sys.path.append("../build/libonvif")
import onvif

class Signals(QObject):
    fill = pyqtSignal(onvif.Data)
    login = pyqtSignal(onvif.Data)


class LoginDialog(QDialog):
    def __init__(self, p):
        super().__init__(p)
        self.active = False
        self.lblCameraIP = QLabel()
        self.lblCameraName = QLabel()
        buttonBox = QDialogButtonBox( \
            QDialogButtonBox.StandardButton.Ok | \
            QDialogButtonBox.StandardButton.Cancel)
        self.txtUsername = QLineEdit()
        lblUsername = QLabel("Username")
        self.txtPassword = QLineEdit()
        lblPassword = QLabel("Password")

        lytMain = QGridLayout(self)
        lytMain.addWidget(self.lblCameraName,  0, 0, 1, 2, Qt.AlignmentFlag.AlignCenter)
        lytMain.addWidget(self.lblCameraIP,    1, 0, 1, 2, Qt.AlignmentFlag.AlignCenter)
        lytMain.addWidget(lblUsername,         2, 0, 1, 1)
        lytMain.addWidget(self.txtUsername,    2, 1, 1, 1)
        lytMain.addWidget(lblPassword,         3, 0, 1, 1)
        lytMain.addWidget(self.txtPassword,    3, 1, 1, 1)
        lytMain.addWidget(buttonBox,           4, 0, 1, 2)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def exec(self, onvif_data):
        self.lblCameraName.setText(onvif_data.camera_name())
        self.lblCameraIP.setText(onvif_data.host())
        self.txtUsername.setText("")
        self.txtPassword.setText("")
        self.txtUsername.setFocus()
        onvif_data.cancelled = not super().exec()
        onvif_data.setUsername(self.txtUsername.text())
        onvif_data.setPassword(self.txtPassword.text())
        self.active = False

class CameraList(QListWidget):
    def __init__(self):
        super().__init__()

    def keyPressEvent(self, event):
        print(event.key())
        if event.key() == Qt.Key.Key_Return:
            self.itemDoubleClicked.emit(self.currentItem())
        return super().keyPressEvent(event)

class CameraPanel(QWidget):
    def __init__(self, mw):
        super().__init__()
        self.devices = []
        self.mw = mw
        self.dlgLogin = LoginDialog(self)
        self.settings = QSettings("onvif", "alias")

        self.boss = onvif.Manager()
        self.boss.discovered = lambda : self.discovered()
        self.boss.getCredential = lambda D : self.getCredential(D)
        self.boss.getData = lambda D : self.getData(D)
        self.boss.filled = lambda D : self.filled(D)

        self.btnDiscover = QPushButton("Discover")
        self.btnDiscover.clicked.connect(self.btnDiscoverClicked)

        self.btnApply = QPushButton("Apply")
        self.btnApply.clicked.connect(self.btnApplyClicked)
        self.btnApply.setEnabled(False)

        self.lstCamera = CameraList()
        self.lstCamera.currentRowChanged.connect(self.onCurrentRowChanged)
        self.lstCamera.itemDoubleClicked.connect(self.onItemDoubleClicked)

        self.tabOnvif = QTabWidget()
        self.tabVideo = VideoTab(self)
        self.tabImage = ImageTab(self)
        self.tabNetwork = NetworkTab(self)
        self.ptzTab = PTZTab(self)
        self.adminTab = AdminTab(self)
        self.tabOnvif.addTab(self.tabVideo, "Video")
        self.tabOnvif.addTab(self.tabImage, "Image")
        self.tabOnvif.addTab(self.tabNetwork, "Network")
        self.tabOnvif.addTab(self.ptzTab, "PTZ")
        self.tabOnvif.addTab(self.adminTab, "Admin")

        self.signals = Signals()
        self.signals.fill.connect(self.tabVideo.fill)
        self.signals.fill.connect(self.tabImage.fill)
        self.signals.fill.connect(self.tabNetwork.fill)
        self.signals.fill.connect(self.ptzTab.fill)
        self.signals.fill.connect(self.adminTab.fill)
        self.signals.login.connect(self.onShowLogin)

        lytMain = QGridLayout(self)
        lytMain.addWidget(self.lstCamera,   0, 0, 1, 4)
        lytMain.addWidget(self.tabOnvif,    1, 0, 1, 4)
        lytMain.addWidget(self.btnDiscover, 2, 2, 1, 1, Qt.AlignmentFlag.AlignCenter)
        lytMain.addWidget(self.btnApply,    2, 3, 1, 1, Qt.AlignmentFlag.AlignCenter)
        lytMain.setRowStretch(0, 10)

    def btnDiscoverClicked(self):
        self.boss.startPyDiscover()
        self.btnDiscover.setEnabled(False)

    def filled(self, onvif_data):
        if len(onvif_data.last_error()) > 0:
            print("ERROR:", onvif_data.last_error())
        self.devices[self.lstCamera.currentRow()] = onvif_data
        self.signals.fill.emit(onvif_data)
        self.btnApply.setEnabled(False)

    def discovered(self):
        print("discover finished")
        self.btnDiscover.setEnabled(True)

    def getCredential(self, onvif_data):
        for d in self.devices:
            if d == onvif_data:
                onvif_data.cancelled = True
                return onvif_data
            
        if len(self.mw.settingsPanel.txtPassword.text()) > 0 and \
                len(onvif_data.last_error()) == 0:
            onvif_data.setUsername(self.mw.settingsPanel.txtUsername.text())
            onvif_data.setPassword(self.mw.settingsPanel.txtPassword.text())
        else:
            onvif_data.clearLastError()
            self.dlgLogin.active = True
            self.signals.login.emit(onvif_data)
            while self.dlgLogin.active:
                sleep(0.01)

        return onvif_data
    
    def onShowLogin(self, onvif_data):
        self.dlgLogin.exec(onvif_data)
    
    def getData(self, onvif_data):
        onvif_data.alias = self.settings.value(onvif_data.serial_number(), onvif_data.camera_name())
        self.devices.append(onvif_data)
        self.lstCamera.addItem(onvif_data.alias)

    def onCurrentRowChanged(self, row):
        print("row changed", row)
        onvif_data = self.devices[self.lstCamera.currentRow()]
        if onvif_data.filled:
            self.setTabsEnabled(False)
            self.signals.fill.emit(onvif_data)
        else:
            self.boss.onvif_data = onvif_data
            self.setTabsEnabled(False)
            self.boss.startPyFill()

    def onItemDoubleClicked(self, item):
        onvif_data = self.devices[self.lstCamera.currentRow()]
        uri = onvif_data.stream_uri()[0 : 7] + onvif_data.username() + ":" \
            + onvif_data.password() + "@" + onvif_data.stream_uri()[7:]
        self.mw.playMedia(uri)

    def setTabsEnabled(self, enabled):
        self.tabVideo.setEnabled(enabled)
        self.tabImage.setEnabled(enabled)
        self.tabNetwork.setEnabled(enabled)
        self.ptzTab.setEnabled(enabled)
        self.adminTab.setEnabled(enabled)

    def btnApplyClicked(self):
        onvif_data = self.devices[self.lstCamera.currentRow()]
        self.tabVideo.update(onvif_data)
        self.tabImage.update(onvif_data)
        self.tabNetwork.update(onvif_data)
        self.adminTab.update(onvif_data)

    def onEdit(self):
        onvif_data = self.devices[self.lstCamera.currentRow()]
        if self.tabVideo.edited(onvif_data) or \
                self.tabImage.edited(onvif_data) or \
                self.tabNetwork.edited(onvif_data) or \
                self.adminTab.edited(onvif_data):
            self.btnApply.setEnabled(True)
        else:
            self.btnApply.setEnabled(False)

       
