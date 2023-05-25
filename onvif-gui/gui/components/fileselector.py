#*******************************************************************************
# onvif-gui/gui/components/fileselector.py
#
# Copyright (c) 2023 Stephen Rhodes 
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
#******************************************************************************/

from PyQt6.QtWidgets import QWidget, QLineEdit, QPushButton, \
    QLabel, QGridLayout, QFileDialog

class FileSelector(QWidget):
    def __init__(self, mw, name):
        super().__init__()
        self.mw = mw
        self.filenameKey = "Module/" + name + "/filename"

        self.txtFilename = QLineEdit()
        self.txtFilename.setText(self.mw.settings.value(self.filenameKey))
        self.txtFilename.textEdited.connect(self.txtFilenameChanged)
        self.btnSelect = QPushButton("...")
        self.btnSelect.clicked.connect(self.btnSelectClicked)
        lblSelect = QLabel("Model")

        lytMain = QGridLayout(self)
        lytMain.addWidget(lblSelect,          0, 0, 1, 1)
        lytMain.addWidget(self.txtFilename,   0, 1, 1, 1)
        lytMain.addWidget(self.btnSelect,     0, 2, 1, 1)
        lytMain.setColumnStretch(1, 10)
        lytMain.setContentsMargins(0, 0, 0, 0)

    def btnSelectClicked(self):
        filename = QFileDialog.getOpenFileName(self, "Select File", self.txtFilename.text())[0]

        if len(filename) > 0:
            self.txtFilename.setText(filename)
            self.mw.settings.setValue(self.filenameKey, filename)

    def txtFilenameChanged(self, text):
        self.mw.settings.setValue(self.filenameKey, text)

    def text(self):
        return self.txtFilename.text()
