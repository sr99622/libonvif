import os
import sys
import numpy as np
import cv2
from time import sleep
from PyQt6.QtWidgets import QComboBox, QLineEdit, QSpinBox, \
QGridLayout, QWidget, QSlider, QLabel, QMessageBox, QListWidget
from PyQt6.QtCore import Qt, pyqtSignal, QObject

sys.path.append("../build/libonvif")
import onvif

class SpinBox(QSpinBox):
    def __init__(self, qle):
        super().__init__()
        self.setLineEdit(qle)

class VideoTab(QWidget):
    def __init__(self, cp):
        super().__init__()
        self.cp = cp

        self.cmbResolutions = QComboBox()
        self.cmbResolutions.currentTextChanged.connect(self.cp.onEdit)
        lblResolutions = QLabel("Resolution")

        txtFrameRate = QLineEdit()
        self.spnFrameRate = SpinBox(txtFrameRate)
        self.spnFrameRate.textChanged.connect(self.cp.onEdit)
        lblFrameRate = QLabel("Frame Rate")

        txtGovLength = QLineEdit()
        self.spnGovLength = SpinBox(txtGovLength)
        self.spnGovLength.textChanged.connect(self.cp.onEdit)
        lblGovLength = QLabel("GOP Length")

        txtBitrate = QLineEdit()
        self.spnBitrate = SpinBox(txtBitrate)
        self.spnBitrate.textChanged.connect(self.cp.onEdit)
        lblBitrate = QLabel("Bitrate")

        lytMain = QGridLayout(self)
        lytMain.addWidget(lblResolutions,      0, 0, 1, 1)
        lytMain.addWidget(self.cmbResolutions, 0, 1, 1, 1)
        lytMain.addWidget(lblFrameRate,        1, 0, 1, 1)
        lytMain.addWidget(self.spnFrameRate,   1, 1, 1, 1)
        lytMain.addWidget(lblGovLength,        2, 0, 1, 1)
        lytMain.addWidget(self.spnGovLength,   2, 1, 1, 1)
        lytMain.addWidget(lblBitrate,          3, 0, 1, 1)
        lytMain.addWidget(self.spnBitrate,     3, 1, 1, 1)

    def fill(self, onvif_data):
        self.cmbResolutions.clear()
        i = 0
        while len(onvif_data.resolutions_buf(i)) > 0:
            self.cmbResolutions.addItem(onvif_data.resolutions_buf(i))
            i += 1

        current_resolution = str(onvif_data.width()) + " x " + str(onvif_data.height())
        self.cmbResolutions.setCurrentText(current_resolution)

        self.spnFrameRate.setMaximum(onvif_data.frame_rate_max())
        self.spnFrameRate.setMinimum(onvif_data.frame_rate_min())
        self.spnFrameRate.setValue(onvif_data.frame_rate())

        self.spnGovLength.setMaximum(onvif_data.gov_length_max())
        self.spnGovLength.setMinimum(onvif_data.gov_length_min())
        self.spnGovLength.setValue(onvif_data.gov_length())

        self.spnBitrate.setMaximum(onvif_data.bitrate_max())
        self.spnBitrate.setMinimum(onvif_data.bitrate_min())
        self.spnBitrate.setValue(onvif_data.bitrate())

        self.setEnabled(True)

    def edited(self, onvif_data):
        result = False
        if self.isEnabled():
            current_resolution = str(onvif_data.width()) + " x " + str(onvif_data.height())
            if not current_resolution == self.cmbResolutions.currentText():
                result = True
            if not onvif_data.frame_rate() == self.spnFrameRate.value():
                result = True
            if not onvif_data.gov_length() == self.spnGovLength.value():
                result = True
            if not onvif_data.bitrate() == self.spnBitrate.value():
                result = True
        return result

    def update(self, onvif_data):
        if self.edited(onvif_data):
            self.setEnabled(False)
            onvif_data.setFrameRate(self.spnFrameRate.value())
            onvif_data.setGovLength(self.spnGovLength.value())
            onvif_data.setBitrate(self.spnBitrate.value())
            dims = self.cmbResolutions.currentText().split('x')
            onvif_data.setWidth(int(dims[0]))
            onvif_data.setHeight(int(dims[1]))
            self.cp.boss.onvif_data = onvif_data
            self.cp.boss.startPyUpdateVideo()




