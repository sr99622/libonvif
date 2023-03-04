import os
import sys
import numpy as np
import cv2
from time import sleep
from PyQt6.QtWidgets import QPushButton, QLineEdit, QSpinBox, \
QGridLayout, QWidget, QCheckBox, QLabel, QMessageBox, QListWidget
from PyQt6.QtCore import Qt, pyqtSignal, QObject

sys.path.append("../build/libonvif")
import onvif

class SettingsPanel(QWidget):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw
        self.autoDiscoverKey = "settings/autoDiscover"
        self.usernameKey = "settings/username"
        self.passwordKey = "settings/password"

        self.chkAutoDiscover = QCheckBox("Enable Auto Discovery")
        self.chkAutoDiscover.setChecked(int(mw.settings.value(self.autoDiscoverKey)))
        self.chkAutoDiscover.stateChanged.connect(self.autoDiscoverChecked)
        self.txtUsername = QLineEdit()
        self.txtUsername.setText(mw.settings.value(self.usernameKey))
        self.txtUsername.textChanged.connect(self.usernameChanged)
        lblUsername = QLabel("Common Username")
        self.txtPassword = QLineEdit()
        self.txtPassword.setText(mw.settings.value(self.passwordKey))
        self.txtPassword.textChanged.connect(self.passwordChanged)
        lblPassword = QLabel("Common Password")

        lytMain = QGridLayout(self)
        lytMain.addWidget(self.chkAutoDiscover,   0, 0, 1, 2)
        lytMain.addWidget(lblUsername,            1, 0, 1, 1)
        lytMain.addWidget(self.txtUsername,       1, 1, 1, 1)
        lytMain.addWidget(lblPassword,            2, 0, 1, 1)
        lytMain.addWidget(self.txtPassword,       2, 1, 1, 1)

    def autoDiscoverChecked(self, state):
        self.mw.settings.setValue(self.autoDiscoverKey, state)

    def usernameChanged(self, username):
        self.mw.settings.setValue(self.usernameKey, username)

    def passwordChanged(self, password):
        self.mw.settings.setValue(self.passwordKey, password)
