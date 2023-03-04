import os
import sys
import numpy as np
import cv2
from time import sleep
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, \
QGridLayout, QWidget, QSlider, QLabel, QMessageBox, QSplitter, \
QTabWidget
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QSettings
from camerapanel import CameraPanel
from settingspanel import SettingsPanel
from glwidget import GLWidget

sys.path.append("../build/libonvif")
import onvif
sys.path.append("../build/libavio")
import avio

class Signals(QObject):
    fill = pyqtSignal(onvif.Data)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.settings = QSettings()

        self.player = avio.Player()
        self.playing = False

        self.tab = QTabWidget()
        self.cameraPanel = CameraPanel(self)
        self.settingsPanel = SettingsPanel(self)
        self.tab.addTab(self.cameraPanel, "Cameras")
        self.tab.addTab(self.settingsPanel, "Settings")
        self.glWidget = GLWidget()
        split = QSplitter()
        split.addWidget(self.glWidget)
        split.addWidget(self.tab)
        split.setStretchFactor(0, 4)
        self.setCentralWidget(split)

        if self.settingsPanel.chkAutoDiscover.isChecked():
            self.cameraPanel.btnDiscoverClicked()

    def playMedia(self, uri):
        print("main window play", uri)
        self.stopMedia()
        self.player = avio.Player()
        self.player.uri = uri
        self.player.width = lambda : self.glWidget.width()
        self.player.height = lambda : self.glWidget.height()
        #self.player.progressCallback = lambda f : self.progressCallback(f)
        #self.player.hWnd = self.glWidget.winId()
        self.player.video_filter = "format=rgb24"
        self.player.renderCallback = lambda F : self.glWidget.renderCallback(F)
        #self.player.pythonCallback = lambda F : self.pythonCallback(F)
        self.player.cbMediaPlayingStarted = lambda n : self.mediaPlayingStarted(n)
        self.player.cbMediaPlayingStopped = lambda : self.mediaPlayingStopped()
        self.player.errorCallback = lambda s : self.errorCallback(s)
        self.player.infoCallback = lambda s : self.infoCallback(s)
        #self.player.setVolume(self.sldVolume.value())
        #self.player.setMute(self.mute)
        #self.player.disable_video = True
        #self.player.hw_device_type = avio.AV_HWDEVICE_TYPE_CUDA
        self.player.start()
        self.cameraPanel.setEnabled(False)
        print("oops")

    def stopMedia(self):
        print("stopMedia")
        self.player.running = False
        while self.playing:
            sleep(0.01)

    def mediaPlayingStarted(self, n):
        print("mediaPlayingStarted", n)
        self.playing = True
        self.cameraPanel.setEnabled(True)
        self.cameraPanel.lstCamera.setFocus()

    def mediaPlayingStopped(self):
        print("mediaPlayingStopped")
        self.playing = False

    def infoCallback(self, s):
        print("info", s)

    def errorCallback(self, s):
        print("error", s)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()