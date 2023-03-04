import os
import sys
import numpy as np
import cv2
from time import sleep
from PyQt6.QtWidgets import QComboBox, QLineEdit, QSpinBox, \
QGridLayout, QWidget, QCheckBox, QLabel, QPushButton, QMessageBox
from PyQt6.QtCore import Qt, pyqtSignal, QObject

sys.path.append("../build/libonvif")
import onvif

class AdminTab(QWidget):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw

        self.txtCameraName = QLineEdit()
        lblCameraName = QLabel("Camera Name")
        self.txtAdminPassword = QLineEdit()
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
        self.txtCameraName.setText(onvif_data.camera_name())
        self.setEnabled(True)

    def btnRebootClicked(self):
        result = QMessageBox.question(self, "Warning", "Please confirm reboot")
        if result == QMessageBox.StandardButton.Yes:
            print("do it")
        print("btnRebootClicked")

    def btnSyncTimeClicked(self):
        print("btnSyncTimeClicked")

    def btnHardResetClicked(self):
        result = QMessageBox.question(self, "Warning", "** THIS WILL ERASE ALL SETTINGS **\nAre you sure you want to do this?")
        if result == QMessageBox.StandardButton.Yes:
            print("do it")
        print("btnHardResetClicked")

    def btnBrowserClicked(self):
        print("btnBrowserClicked")

    def chkEnableResetChanged(self, state):
        print("chkEnableResetChanged", state)
        self.btnHardReset.setEnabled(state)