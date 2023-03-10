import os
import sys
import numpy as np
import cv2
import platform
from time import sleep
from PyQt6.QtWidgets import QComboBox, QLineEdit, QSpinBox, \
QGridLayout, QWidget, QCheckBox, QLabel, QPushButton, QMessageBox
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QProcess

sys.path.append("../build/libonvif")
import onvif

class AdminTab(QWidget):
    def __init__(self, cp):
        super().__init__()
        self.cp = cp
        self.process = QProcess()

        self.txtCameraName = QLineEdit()
        self.txtCameraName.textEdited.connect(self.cp.onEdit)
        lblCameraName = QLabel("Camera Name")
        self.txtAdminPassword = QLineEdit()
        self.txtAdminPassword.textEdited.connect(self.cp.onEdit)
        lblAdminPassword = QLabel("Admin Password")
        self.btnSyncTime = QPushButton("Sync Time")
        self.btnSyncTime.clicked.connect(self.btnSyncTimeClicked)
        self.btnReboot = QPushButton("Reboot")
        self.btnReboot.clicked.connect(self.btnRebootClicked)
        self.btnHardReset = QPushButton("Hard Reset")
        self.btnHardReset.clicked.connect(self.btnHardResetClicked)
        self.btnHardReset.setEnabled(False)
        self.btnBrowser = QPushButton("Browser")
        self.btnBrowser.clicked.connect(self.btnBrowserClicked)
        self.chkEnableReset = QCheckBox("Enable Reset")
        self.chkEnableReset.stateChanged.connect(self.chkEnableResetChanged)

        lytMain = QGridLayout(self)
        lytMain.addWidget(lblCameraName,         0, 0, 1, 1)
        lytMain.addWidget(self.txtCameraName,    0, 1, 1, 2)
        lytMain.addWidget(lblAdminPassword,      1, 0, 1, 1)
        lytMain.addWidget(self.txtAdminPassword, 1, 1, 1, 2)
        lytMain.addWidget(self.btnSyncTime,      2, 0, 1, 1)
        lytMain.addWidget(self.btnReboot,        2, 1, 1, 1)
        lytMain.addWidget(self.btnHardReset,     2, 2, 1, 1)
        lytMain.addWidget(self.btnBrowser,       3, 1, 1, 1)
        lytMain.addWidget(self.chkEnableReset,   3, 2, 1, 1)

    def fill(self, onvif_data):
        self.txtCameraName.setText(onvif_data.alias)
        self.setEnabled(True)

    def edited(self, onvif_data):
        result = False
        if self.isEnabled():
            if not onvif_data.alias == self.txtCameraName.text():
                result = True
            if len(self.txtAdminPassword.text()) > 0:
                result = True
        return result
    
    def update(self, onvif_data):
        if self.edited(onvif_data):
            print("admin tab update")
            if not onvif_data.alias == self.txtCameraName.text():
                onvif_data.alias = self.txtCameraName.text()
                self.cp.devices[self.cp.lstCamera.currentRow()] = onvif_data
                self.cp.lstCamera.currentItem().setText(onvif_data.alias)
                self.cp.settings.setValue(onvif_data.serial_number(), onvif_data.alias)
                print("editing camera alias")
            if len(self.txtAdminPassword.text()) > 0:
                result = QMessageBox.question(self, "Warning", "Please confirm camera password change")
                if result == QMessageBox.StandardButton.Yes:
                    self.cp.boss.onvif_data = onvif_data
                    self.cp.boss.new_password = self.txtAdminPassword.text()
                    self.cp.boss.startPySetUser()
                    self.txtAdminPassword.clear()
                    print("editing admin password")

    def btnRebootClicked(self):
        result = QMessageBox.question(self, "Warning", "Please confirm reboot")
        if result == QMessageBox.StandardButton.Yes:
            self.cp.boss.onvif_data = self.cp.devices[self.cp.lstCamera.currentRow()]
            self.cp.boss.startPyReboot()
        print("btnRebootClicked")

    def btnSyncTimeClicked(self):
        print("btnSyncTimeClicked")
        self.cp.boss.onvif_data = self.cp.devices[self.cp.lstCamera.currentRow()]
        self.cp.boss.startPyUpdateTime()

    def btnHardResetClicked(self):
        result = QMessageBox.question(self, "Warning", "** THIS WILL ERASE ALL SETTINGS **\nAre you sure you want to do this?")
        if result == QMessageBox.StandardButton.Yes:
            self.cp.boss.onvif_data = self.cp.devices[self.cp.lstCamera.currentRow()]
            self.cp.boss.startPyReset()
        print("btnHardResetClicked")

    def btnBrowserClicked(self):
        print("btnBrowserClicked")
        onvif_data = self.cp.devices[self.cp.lstCamera.currentRow()]
        if platform.system() == "Linux":
            cmd = "xdg-open"
        if platform.system() == "Windows":
            cmd = "\"C:\\Program Files\\Internet Explorer\\iexplore.exe\""
        args = "http://" + onvif_data.host()
        print(args)
        self.process.start(cmd, [args,])

    def chkEnableResetChanged(self, state):
        print("chkEnableResetChanged", state)
        self.btnHardReset.setEnabled(state)