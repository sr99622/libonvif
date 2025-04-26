#/********************************************************************
# libonvif/onvif-gui/onvif_gui/panels/cameras/imagetab.py 
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

from PyQt6.QtWidgets import QGridLayout, QWidget, QSlider, QLabel
from PyQt6.QtCore import Qt
from onvif_gui.enums import ProxyType

class ImageTab(QWidget):
    def __init__(self, cp):
        super().__init__()
        self.cp = cp

        self.sldBrightness = QSlider(Qt.Orientation.Horizontal)
        self.sldBrightness.valueChanged.connect(cp.onEdit)
        self.sldSaturation = QSlider(Qt.Orientation.Horizontal)
        self.sldSaturation.valueChanged.connect(cp.onEdit)
        self.sldContrast = QSlider(Qt.Orientation.Horizontal)
        self.sldContrast.valueChanged.connect(cp.onEdit)
        self.sldSharpness = QSlider(Qt.Orientation.Horizontal)
        self.sldSharpness.valueChanged.connect(cp.onEdit)

        lblBrightness = QLabel("Brightness")
        lblSaturation = QLabel("Saturation")
        lblContrast = QLabel("Contrast")
        lblSharpness = QLabel("Sharpness")

        lytMain = QGridLayout(self)
        lytMain.addWidget(lblBrightness,      0, 0, 1, 1)
        lytMain.addWidget(self.sldBrightness, 0, 1, 1, 3)        
        lytMain.addWidget(lblSaturation,      1, 0, 1, 1)
        lytMain.addWidget(self.sldSaturation, 1, 1, 1, 3)        
        lytMain.addWidget(lblContrast,        2, 0, 1, 1)
        lytMain.addWidget(self.sldContrast,   2, 1, 1, 3)        
        lytMain.addWidget(lblSharpness,       3, 0, 1, 1)
        lytMain.addWidget(self.sldSharpness,  3, 1, 1, 3)        
        
    def fill(self, onvif_data):
        self.sldBrightness.setMaximum(onvif_data.brightness_max())
        self.sldBrightness.setMinimum(onvif_data.brightness_min())
        self.sldBrightness.setValue(onvif_data.brightness())

        self.sldContrast.setMaximum(onvif_data.contrast_max())
        self.sldContrast.setMinimum(onvif_data.contrast_min())
        self.sldContrast.setValue(onvif_data.contrast())

        self.sldSaturation.setMaximum(onvif_data.saturation_max())
        self.sldSaturation.setMinimum(onvif_data.saturation_min())
        self.sldSaturation.setValue(onvif_data.saturation())

        self.sldSharpness.setMaximum(onvif_data.sharpness_max())
        self.sldSharpness.setMinimum(onvif_data.sharpness_min())
        self.sldSharpness.setValue(onvif_data.sharpness())

        self.setEnabled(True)
        self.cp.onEdit()

    def edited(self, onvif_data):
        result = False
        if self.isEnabled():
            if not onvif_data.brightness() == self.sldBrightness.value():
                result = True
            if not onvif_data.contrast() == self.sldContrast.value():
                result = True
            if not onvif_data.saturation() == self.sldSaturation.value():
                result = True
            if not onvif_data.sharpness() == self.sldSharpness.value():
                result = True

        return result
    
    def update(self, onvif_data):
        if self.edited(onvif_data):
            onvif_data.setBrightness(self.sldBrightness.value())
            onvif_data.setSaturation(self.sldSaturation.value())
            onvif_data.setContrast(self.sldContrast.value())
            onvif_data.setSharpness(self.sldSharpness.value())
            if self.cp.mw.settingsPanel.proxy.proxyType == ProxyType.CLIENT:
                arg = "UPDATE IMAGE\n\n" + onvif_data.toJSON() + "\r\n"
                self.cp.mw.client.transmit(arg)
            else:
                onvif_data.startUpdateImage()

