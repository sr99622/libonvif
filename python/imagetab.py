import os
import sys
import numpy as np
from time import sleep
from PyQt6.QtWidgets import QComboBox, QLineEdit, QSpinBox, \
QGridLayout, QWidget, QSlider, QLabel, QMessageBox, QListWidget
from PyQt6.QtCore import Qt, pyqtSignal, QObject

sys.path.append("../build/libonvif")
import onvif

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
        lytMain.addWidget(self.sldBrightness, 0, 1, 1, 1)        
        lytMain.addWidget(lblSaturation,      1, 0, 1, 1)
        lytMain.addWidget(self.sldSaturation, 1, 1, 1, 1)        
        lytMain.addWidget(lblContrast,        2, 0, 1, 1)
        lytMain.addWidget(self.sldContrast,   2, 1, 1, 1)        
        lytMain.addWidget(lblSharpness,       3, 0, 1, 1)
        lytMain.addWidget(self.sldSharpness,  3, 1, 1, 1)        
        
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
            print("image tab update")
            onvif_data.setBrightness(self.sldBrightness.value())
            onvif_data.setSaturation(self.sldSaturation.value())
            onvif_data.setContrast(self.sldContrast.value())
            onvif_data.setSharpness(self.sldSharpness.value())
            self.cp.boss.onvif_data = onvif_data
            self.cp.boss.startPyUpdateImage()