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

class ImageTab(QWidget):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw

        self.sldBrightness = QSlider(Qt.Orientation.Horizontal)
        self.sldSaturation = QSlider(Qt.Orientation.Horizontal)
        self.sldContrast = QSlider(Qt.Orientation.Horizontal)
        self.sldSharpness = QSlider(Qt.Orientation.Horizontal)

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
        