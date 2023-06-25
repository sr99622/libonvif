#/********************************************************************
# onvif-gui/gui/onvif/admintab.py 
#
# Copyright (c) 2023  Stephen Rhodes
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#*********************************************************************/

import platform
import webbrowser
from PyQt6.QtWidgets import QLineEdit, QGridLayout, QWidget, \
QCheckBox, QLabel, QPushButton, QMessageBox
from PyQt6.QtCore import QProcess

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
        self.setEnabled(len(onvif_data.alias))
        self.cp.onEdit()

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
            if not onvif_data.alias == self.txtCameraName.text():
                onvif_data.alias = self.txtCameraName.text()
                self.cp.devices[self.cp.lstCamera.currentRow()] = onvif_data
                self.cp.lstCamera.currentItem().setText(onvif_data.alias)
                self.cp.settings.setValue(onvif_data.serial_number(), onvif_data.alias)
                self.cp.boss.onvif_data = onvif_data
                self.cp.boss.startFill()
            if len(self.txtAdminPassword.text()) > 0:
                result = QMessageBox.question(self, "Warning", "Please confirm camera password change")
                if result == QMessageBox.StandardButton.Yes:
                    self.cp.boss.onvif_data = onvif_data
                    self.cp.boss.new_password = self.txtAdminPassword.text()
                    self.cp.boss.startSetUser()
                    self.txtAdminPassword.clear()

    def btnRebootClicked(self):
        result = QMessageBox.question(self, "Warning", "Please confirm reboot")
        if result == QMessageBox.StandardButton.Yes:
            onvif_data = self.cp.devices[self.cp.lstCamera.currentRow()]
            if self.cp.mw.player is not None:
                if self.cp.mw.player.uri == self.cp.getStreamURI(onvif_data):
                    self.cp.mw.stopMedia()
            self.cp.boss.onvif_data = onvif_data
            self.cp.boss.startReboot()
            self.cp.removeCurrent()

    def btnSyncTimeClicked(self):
        self.cp.boss.onvif_data = self.cp.devices[self.cp.lstCamera.currentRow()]
        self.cp.boss.startUpdateTime()

    def btnHardResetClicked(self):
        result = QMessageBox.question(self, "Warning", "** THIS WILL ERASE ALL SETTINGS **\nAre you sure you want to do this?")
        if result == QMessageBox.StandardButton.Yes:
            onvif_data = self.cp.devices[self.cp.lstCamera.currentRow()]
            if self.cp.mw.player is not None:
                if self.cp.mw.player.uri == self.cp.getStreamURI(onvif_data):
                    self.cp.mw.stopMedia()
            self.cp.boss.onvif_data = onvif_data
            self.cp.boss.startReset()

    def btnBrowserClicked(self):
        onvif_data = self.cp.devices[self.cp.lstCamera.currentRow()]
        args = "http://" + onvif_data.host()
        webbrowser.open(args)

    def chkEnableResetChanged(self, state):
        self.btnHardReset.setEnabled(state)