#/********************************************************************
# libonvif/onvif-gui/onvif_gui/panels/cameras/ptztab.py 
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

from PyQt6.QtWidgets import QPushButton, QGridLayout, QWidget, QCheckBox
from onvif_gui.enums import ProxyType

class PTZTab(QWidget):
    def __init__(self, cp):
        super().__init__()
        self.cp = cp

        self.btn1 = QPushButton("1")
        self.btn1.setMaximumWidth(52)
        self.btn1.pressed.connect(lambda val=1: self.presetButtonClicked(val))
        self.btn2 = QPushButton("2")
        self.btn2.setMaximumWidth(52)
        self.btn2.pressed.connect(lambda val=2: self.presetButtonClicked(val))
        self.btn3 = QPushButton("3")
        self.btn3.setMaximumWidth(52)
        self.btn3.pressed.connect(lambda val=3: self.presetButtonClicked(val))
        self.btn4 = QPushButton("4")
        self.btn4.setMaximumWidth(52)
        self.btn4.pressed.connect(lambda val=4: self.presetButtonClicked(val))
        self.btn5 = QPushButton("5")
        self.btn5.setMaximumWidth(52)
        self.btn5.pressed.connect(lambda val=5: self.presetButtonClicked(val))

        self.btnLeft = QPushButton("<")
        self.btnLeft.setMaximumWidth(52)
        self.btnLeft.pressed.connect(   lambda x=-0.5, y=0.0,  z=0.0 : self.move(x, y, z))
        self.btnLeft.released.connect(self.stopPanTilt)
        self.btnRight = QPushButton(">")
        self.btnRight.setMaximumWidth(52)
        self.btnRight.pressed.connect(  lambda x=0.5,  y=0.0,  z=0.0 : self.move(x, y, z))
        self.btnRight.released.connect(self.stopPanTilt)
        self.btnUp = QPushButton("^")
        self.btnUp.setMaximumWidth(52)
        self.btnUp.pressed.connect(     lambda x=0.0,  y=0.5,  z=0.0 : self.move(x, y, z))
        self.btnUp.released.connect(self.stopPanTilt)
        self.btnDown = QPushButton("v")
        self.btnDown.setMaximumWidth(52)
        self.btnDown.pressed.connect(   lambda x=0.0,  y=-0.5, z=0.0 : self.move(x, y, z))
        self.btnDown.released.connect(self.stopPanTilt)
        self.btnZoomIn = QPushButton("+")
        self.btnZoomIn.setMaximumWidth(52)
        self.btnZoomIn.pressed.connect( lambda x=0.0,  y=0.0,  z=0.5 : self.move(x, y, z))
        self.btnZoomIn.released.connect(self.stopZoom)
        self.btnZoomOut = QPushButton("-")
        self.btnZoomOut.setMaximumWidth(52)
        self.btnZoomOut.pressed.connect( lambda x=0.0,  y=0.0, z=-0.5 : self.move(x, y, z))
        self.btnZoomOut.released.connect(self.stopZoom)

        self.chkSet = QCheckBox("Set Preset Position")

        lytMain = QGridLayout(self)
        lytMain.addWidget(self.btn1,   0, 0, 1, 1)
        lytMain.addWidget(self.btn2,   1, 0, 1, 1)
        lytMain.addWidget(self.btn3,   2, 0, 1, 1)
        lytMain.addWidget(self.btn4,   3, 0, 1, 1)
        lytMain.addWidget(self.btn5,   4, 0, 1, 1)

        lytMain.addWidget(self.btnLeft,    1, 2, 1, 1)
        lytMain.addWidget(self.btnUp,      0, 3, 1, 1)
        lytMain.addWidget(self.btnDown,    2, 3, 1, 1)
        lytMain.addWidget(self.btnRight,   1, 4, 1, 1)

        lytMain.addWidget(self.btnZoomIn,  3, 4, 1, 1)
        lytMain.addWidget(self.btnZoomOut, 4, 4, 1, 1)

        lytMain.addWidget(self.chkSet,     4, 1, 1, 3)

    def presetButtonClicked(self, n):
        camera = self.cp.getCurrentCamera()
        if camera:
            if self.chkSet.isChecked():
                camera.onvif_data.preset = n
                if self.cp.mw.settingsPanel.proxy.proxyType == ProxyType.CLIENT:
                    arg = "SET PRESET\n\n" + camera.onvif_data.toJSON() + "\r\n"
                    self.cp.mw.client.transmit(arg)
                else:
                    camera.onvif_data.startSetGotoPreset()
            else:
                camera.onvif_data.preset = n
                if self.cp.mw.settingsPanel.proxy.proxyType == ProxyType.CLIENT:
                    arg = "GOTO PRESET\n\n" + camera.onvif_data.toJSON() + "\r\n"
                    self.cp.mw.client.transmit(arg)
                else:
                    camera.onvif_data.startSet()

    def move(self, x, y, z):
        camera = self.cp.getCurrentCamera()
        if camera:
            camera.onvif_data.x = x
            camera.onvif_data.y = y
            camera.onvif_data.z = z
            if self.cp.mw.settingsPanel.proxy.proxyType == ProxyType.CLIENT:
                arg = "MOVE\n\n" + camera.onvif_data.toJSON() + "\r\n"
                self.cp.mw.client.transmit(arg)
            else:
                camera.onvif_data.startMove()

    def stopPanTilt(self):
        camera = self.cp.getCurrentCamera()
        if camera:
            camera.onvif_data.stop_type = 0
            if self.cp.mw.settingsPanel.proxy.proxyType == ProxyType.CLIENT:
                arg = "STOP\n\n" + camera.onvif_data.toJSON() + "\r\n"
                self.cp.mw.client.transmit(arg)
            else:
                camera.onvif_data.startStop()

    def stopZoom(self):
        camera = self.cp.getCurrentCamera()
        if camera:
            camera.onvif_data.stop_type = 1
            if self.cp.mw.settingsPanel.proxy.proxyType == ProxyType.CLIENT:
                arg = "STOP\n\n" + camera.onvif_data.toJSON() + "\r\n"
                self.cp.mw.client.transmit(arg)
            else:
                camera.onvif_data.startStop()

    def fill(self, onvif_data):
        self.setEnabled(True)
        self.chkSet.setChecked(False)