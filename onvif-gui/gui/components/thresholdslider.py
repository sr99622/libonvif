#********************************************************************
# onvif-gui/gui/components/thresholdslider.py
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

from PyQt6.QtWidgets import QWidget, QSlider, QLabel, QGridLayout
from PyQt6.QtCore import Qt

class ThresholdSlider(QWidget):
    def __init__(self, mw, name, title, initValue):
        super().__init__()
        self.mw = mw
        self.thresholdKey = "Module/" + name + "/threshold"
        self.sldThreshold = QSlider(Qt.Orientation.Horizontal)
        value = int(self.mw.settings.value(self.thresholdKey, initValue))
        self.sldThreshold.setValue(value)
        self.sldThreshold.valueChanged.connect(self.sldThresholdChanged)
        lblThreshold = QLabel(title)
        self.lblValue = QLabel(str(self.sldThreshold.value()))
        lytThreshold = QGridLayout(self)
        lytThreshold.addWidget(lblThreshold,          0, 0, 1, 1)
        lytThreshold.addWidget(self.sldThreshold,     0, 1, 1, 1)
        lytThreshold.addWidget(self.lblValue,         0, 2, 1, 1)
        lytThreshold.setContentsMargins(0, 0, 0, 0)

    def sldThresholdChanged(self, value):
        self.lblValue.setText(str(value))
        self.mw.settings.setValue(self.thresholdKey, value)

    def value(self):
        return self.sldThreshold.value() / 100