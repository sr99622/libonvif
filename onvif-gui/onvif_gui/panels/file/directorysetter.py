#/********************************************************************
# libonvif/onvif-gui/onvif_gui/panels/file/directorysetter.py 
#
# Copyright (c) 2025  Stephen Rhodes
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
from PyQt6.QtWidgets import QLineEdit, QPushButton, \
    QGridLayout, QWidget, QFileDialog

class DirectorySetter(QWidget):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw
        self.txtDirectory = QLineEdit()
        self.btnSelect = QPushButton("...")
        self.btnSelect.clicked.connect(self.btnSelectClicked)
        self.dlgFile = QFileDialog()
        lytMain = QGridLayout(self)
        lytMain.setContentsMargins(0, 0, 0, 0)
        lytMain.addWidget(self.txtDirectory,   0, 0, 1, 1)
        lytMain.addWidget(self.btnSelect,      0, 1, 1, 1)
        lytMain.setColumnStretch(0, 10)
        self.setContentsMargins(0, 0, 0, 0)

    def showEvent(self, event):
        self.btnSelect.setFocus()

    def btnSelectClicked(self):
        path = None
        if platform.system() == "Linux":
            path = QFileDialog.getExistingDirectory(self, "Select Directory", self.txtDirectory.text(), QFileDialog.Option.DontUseNativeDialog)
        else:
            path = QFileDialog.getExistingDirectory(self, "Select Directory", self.txtDirectory.text())
        if path:
            self.txtDirectory.setText(path)
            self.mw.filePanel.dirChanged(path)

