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
    def __init__(self):
        super().__init__()

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
        
    def fill(self, D):
        self.sldBrightness.setMaximum(D.brightness_max())
        self.sldBrightness.setMinimum(D.brightness_min())
        self.sldBrightness.setValue(D.brightness())
        self.sldContrast.setMaximum(D.contrast_max())
        self.sldContrast.setMinimum(D.contrast_min())
        self.sldContrast.setValue(D.contrast())
        self.sldSaturation.setMaximum(D.saturation_max())
        self.sldSaturation.setMinimum(D.saturation_min())
        self.sldSaturation.setValue(D.saturation())
        self.sldSharpness.setMaximum(D.sharpness_max())
        self.sldSharpness.setMinimum(D.sharpness_min())
        self.sldSharpness.setValue(D.sharpness())
        