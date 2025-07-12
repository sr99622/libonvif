#/********************************************************************
# libonvif/onvif-gui/onvif_gui/panels/cameras/logindialog.py 
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

from PyQt6.QtWidgets import QDialogButtonBox, QLineEdit, QGridLayout, QDialog, QLabel
from PyQt6.QtCore import Qt
import sys

class LoginDialog(QDialog):
    def __init__(self, p):
        super().__init__(p)
        self.active = False
        if sys.platform == "linux":
            self.setMinimumWidth(320)
            self.setWindowTitle("Login")
        else:
            self.setWindowTitle("Camera Login")
            self.setMinimumWidth(280)
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

