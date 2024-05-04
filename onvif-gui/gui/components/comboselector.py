#*******************************************************************************
# libonvif/onvif-gui/gui/components/comboselector.py
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

from PyQt6.QtWidgets import QWidget, QLabel, QComboBox, QGridLayout

class ComboSelector(QWidget):
    def __init__(self, mw, name, list, select, id=None):
        super().__init__()
        self.mw = mw
        self.selectKey = "Module/"
        if id is not None:
            self.selectKey += id + "/"
        self.selectKey += name + "/filename"

        self.cmbBox = QComboBox()
        self.cmbBox.addItems(list)
        self.cmbBox.setCurrentText(self.mw.settings.value(self.selectKey, select))
        self.cmbBox.currentTextChanged.connect(self.cmbBoxChanged)

        lblBox = QLabel(name)
        lytBox = QGridLayout(self)
        lytBox.addWidget(lblBox,       0, 0, 1, 1)
        lytBox.addWidget(self.cmbBox,  0, 1, 1, 1)
        lytBox.setColumnStretch(1, 10)
        lytBox.setContentsMargins(0, 0, 0, 0)

    def cmbBoxChanged(self, text):
        self.mw.settings.setValue(self.selectKey, text)

    def currentText(self):
        return self.cmbBox.currentText()

    def clear(self):
        self.cmbBox.clear()

    def addItems(self, items):
        self.cmbBox.addItems(items)