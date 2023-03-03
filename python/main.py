import os
import sys
import numpy as np
import cv2
from time import sleep
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, \
QGridLayout, QWidget, QSlider, QLabel, QMessageBox, QListWidget, \
QTabWidget
from PyQt6.QtCore import Qt, pyqtSignal, QObject
from videotab import VideoTab
from imagetab import ImageTab
from networktab import NetworkTab
from ptztab import PTZTab

sys.path.append("../build/libonvif")
import onvif

class Signals(QObject):
    fill = pyqtSignal(onvif.Data)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.boss = onvif.Manager()
        self.devices = []

        self.btnDiscover = QPushButton("discover")
        self.btnDiscover.clicked.connect(self.btnDiscoverClicked)

        self.btnFill = QPushButton("fill")
        self.btnFill.clicked.connect(self.btnFillClicked)

        self.lstCamera = QListWidget()

        self.tabOnvif = QTabWidget()
        self.tabVideo = VideoTab()
        self.tabImage = ImageTab()
        self.tabNetwork = NetworkTab()
        self.ptzTab = PTZTab(self)
        self.tabOnvif.addTab(self.tabVideo, "Video")
        self.tabOnvif.addTab(self.tabImage, "Image")
        self.tabOnvif.addTab(self.tabNetwork, "Network")
        self.tabOnvif.addTab(self.ptzTab, "PTZ")

        self.signals = Signals()
        self.signals.fill.connect(self.tabVideo.fill)
        self.signals.fill.connect(self.tabImage.fill)
        self.signals.fill.connect(self.tabNetwork.fill)

        pnlMain = QWidget()
        lytMain = QGridLayout(pnlMain)
        lytMain.addWidget(self.lstCamera,   0, 0, 1, 4)
        lytMain.addWidget(self.tabOnvif,    1, 0, 1, 4)
        lytMain.addWidget(self.btnDiscover, 2, 0, 1, 1, Qt.AlignmentFlag.AlignCenter)
        lytMain.addWidget(self.btnFill,     2, 1, 1, 1, Qt.AlignmentFlag.AlignCenter)
        lytMain.setRowStretch(0, 10)
        self.setCentralWidget(pnlMain)

    def btnFillClicked(self):
        print("btnFillClicked")
        print("current row", self.lstCamera.currentRow())
        print("devices length", len(self.devices))
        self.boss.onvif_data = self.devices[self.lstCamera.currentRow()]
        self.boss.filled = lambda D : self.filled(D)
        self.boss.startPyFill()

    def btnDiscoverClicked(self):
        print("btnDiscoverClicked")
        self.boss.discovered = lambda : self.discovered()
        self.boss.getCredential = lambda D : self.getCredential(D)
        self.boss.getData = lambda D : self.getData(D)
        self.boss.startPyDiscover()

    def filled(self, D):
        print("filled", D.resolutions_buf(0))
        i = 0
        while len(D.resolutions_buf(i)) > 0:
            print("res", D.resolutions_buf(i))
            i += 1
        print("res print done")
        #self.tabVideo.fill(D)
        self.signals.fill.emit(D)

    def discovered(self):
        print("python discovered")

    def getCredential(self, D):
        print("getCredential", D.xaddrs())
        D.setUsername("admin")
        D.setPassword("admin123")
        return D
    
    def getData(self, D):
        print("stream_uri", D.stream_uri())
        print("width", D.width())
        print("height", D.height())
        self.devices.append(D)
        self.lstCamera.addItem(D.camera_name())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()