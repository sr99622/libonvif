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

class PTZTab(QWidget):
    def __init__(self, cp):
        super().__init__()
        self.cp = cp

        self.btn1 = QPushButton("1")
        self.btn1.pressed.connect(lambda val=1: self.presetButtonClicked(val))
        self.btn2 = QPushButton("2")
        self.btn2.pressed.connect(lambda val=2: self.presetButtonClicked(val))
        self.btn3 = QPushButton("3")
        self.btn3.pressed.connect(lambda val=3: self.presetButtonClicked(val))
        self.btn4 = QPushButton("4")
        self.btn4.pressed.connect(lambda val=4: self.presetButtonClicked(val))
        self.btn5 = QPushButton("5")
        self.btn5.pressed.connect(lambda val=5: self.presetButtonClicked(val))

        self.btnLeft = QPushButton("<")
        self.btnLeft.pressed.connect(   lambda x=-0.5, y=0.0,  z=0.0 : self.move(x, y, z))
        self.btnLeft.released.connect(self.stopPanTilt)
        self.btnRight = QPushButton(">")
        self.btnRight.pressed.connect(  lambda x=0.5,  y=0.0,  z=0.0 : self.move(x, y, z))
        self.btnRight.released.connect(self.stopPanTilt)
        self.btnUp = QPushButton("^")
        self.btnUp.pressed.connect(     lambda x=0.0,  y=0.5,  z=0.0 : self.move(x, y, z))
        self.btnUp.released.connect(self.stopPanTilt)
        self.btnDown = QPushButton("v")
        self.btnDown.pressed.connect(   lambda x=0.0,  y=-0.5, z=0.0 : self.move(x, y, z))
        self.btnDown.released.connect(self.stopPanTilt)
        self.btnZoomIn = QPushButton("+")
        self.btnZoomIn.pressed.connect( lambda x=0.0,  y=0.0,  z=0.5 : self.move(x, y, z))
        self.btnZoomIn.released.connect(self.stopZoom)
        self.btnZoomOut = QPushButton("-")
        self.btnZoomOut.pressed.connect(lambda x=0.0,  y=0.0, z=-0.5 : self.move(x, y, z))
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

        lytMain.addWidget(self.btnZoomOut, 4, 4, 1, 1)
        lytMain.addWidget(self.btnZoomIn,  3, 4, 1, 1)

        lytMain.addWidget(self.chkSet,     4, 1, 1, 3)

    def presetButtonClicked(self, n):
        row = self.cp.lstCamera.currentRow()
        if row > -1:
            if self.chkSet.isChecked():
                print(self.cp.devices[row].stream_uri())
                self.cp.boss.onvif_data = self.cp.devices[row]
                self.cp.boss.preset = n
                self.cp.boss.startPySetPreset()
            else:
                self.cp.boss.onvif_data = self.cp.devices[row]
                self.cp.boss.preset = n
                self.cp.boss.startPySet()

    def move(self, x, y, z):
        row = self.cp.lstCamera.currentRow()
        if row > -1:
            self.cp.boss.onvif_data = self.cp.devices[row]
            self.cp.boss.x = x
            self.cp.boss.y = y
            self.cp.boss.z = z
            self.cp.boss.startPyMove()

    def stopPanTilt(self):
        row = self.cp.lstCamera.currentRow()
        if row > -1:
            self.cp.boss.onvif_data = self.cp.devices[row]
            self.cp.boss.stop_type = 0
            self.cp.boss.startPyStop()

    def stopZoom(self):
        row = self.cp.lstCamera.currentRow()
        if row > -1:
            self.cp.boss.onvif_data = self.cp.devices[row]
            self.cp.boss.stop_type = 1
            self.cp.boss.startPyStop()

    def fill(self, onvif_data):
        self.setEnabled(True)