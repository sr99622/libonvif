import os
import numpy as np
import cv2
from time import sleep
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, \
QGridLayout, QWidget, QSlider, QLabel, QMessageBox, QSplitter, \
QTabWidget
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QSettings, QDir
from PyQt6.QtGui import QIcon
from camerapanel import CameraPanel
from filepanel import FilePanel
from settingspanel import SettingsPanel
from glwidget import GLWidget

import sys
sys.argv += ['-platform', 'windows:darkmode=2']

sys.path.append("../build/libonvif")
import onvif
sys.path.append("../build/libavio")
import avio

class MainWindowSignals(QObject):
    started = pyqtSignal(int)
    stopped = pyqtSignal()
    progress = pyqtSignal(float)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        QDir.addSearchPath('image', 'resources/')
        self.setWindowTitle("onvif gui version 2.0.0")
        self.setWindowIcon(QIcon('image:onvif-gui.png'))

        self.settings = QSettings("onvif", "gui")
        self.volumeKey = "MainWindow/volume"
        self.muteKey = "MainWindow/mute"

        self.player = avio.Player()
        self.playing = False
        self.connecting = False
        self.volume = self.settings.value(self.volumeKey, 80)

        if self.settings.value(self.muteKey, 0) == 0:
            self.mute = False
        else:
            self.mute = True

        self.signals = MainWindowSignals()

        self.tab = QTabWidget()
        self.cameraPanel = CameraPanel(self)
        self.signals.started.connect(self.cameraPanel.onMediaStarted)
        self.filePanel = FilePanel(self)
        self.signals.started.connect(self.filePanel.onMediaStarted)
        self.signals.stopped.connect(self.filePanel.onMediaStopped)
        self.signals.progress.connect(self.filePanel.onMediaProgress)
        self.settingsPanel = SettingsPanel(self)
        self.tab.addTab(self.cameraPanel, "Cameras")
        self.tab.addTab(self.filePanel, "Files")
        self.tab.addTab(self.settingsPanel, "Settings")
        self.glWidget = GLWidget()
        split = QSplitter()
        split.addWidget(self.glWidget)
        split.addWidget(self.tab)
        split.setStretchFactor(0, 4)
        self.setCentralWidget(split)

        self.signals.stopped.connect(self.onMediaStopped)

        if self.settingsPanel.chkAutoDiscover.isChecked():
            self.cameraPanel.btnDiscoverClicked()

    def playMedia(self, uri):
        self.stopMedia()
        self.player = avio.Player()
        self.player.uri = uri
        self.player.width = lambda : self.glWidget.width()
        self.player.height = lambda : self.glWidget.height()
        self.player.vpq_size = 100
        self.player.apq_size = 100
        self.player.progressCallback = lambda f : self.mediaProgress(f)
        #self.player.progressCallback = lambda f : self.filePanel.progress.updateProgress(f)
        #self.player.hWnd = self.glWidget.winId()
        self.player.video_filter = "format=rgb24"
        self.player.renderCallback = lambda F : self.glWidget.renderCallback(F)
        #self.player.pythonCallback = lambda F : self.pythonCallback(F)
        self.player.cbMediaPlayingStarted = lambda n : self.mediaPlayingStarted(n)
        self.player.cbMediaPlayingStopped = lambda : self.mediaPlayingStopped()
        self.player.errorCallback = lambda s : self.errorCallback(s)
        self.player.infoCallback = lambda s : self.infoCallback(s)
        self.player.setVolume(self.volume)
        self.player.setMute(self.mute)
        #self.player.disable_video = True
        #self.player.hw_device_type = avio.AV_HWDEVICE_TYPE_CUDA
        self.player.start()
        self.cameraPanel.setEnabled(False)

    def stopMedia(self):
        self.player.running = False
        while self.playing:
            sleep(0.01)

    def toggleMute(self):
        self.mute = not self.mute
        if self.mute:
            self.settings.setValue(self.muteKey, 1)
        else:
            self.settings.setValue(self.muteKey, 0)
        self.player.setMute(self.mute)

    def setVolume(self, value):
        self.volume = value
        self.settings.setValue(self.volumeKey, value)
        self.player.setVolume(value)

    def closeEvent(self, e):
        self.stopMedia()

    def mediaPlayingStarted(self, n):
        self.playing = True
        self.connecting = False
        self.signals.started.emit(n)

    def mediaPlayingStopped(self):
        self.playing = False
        self.signals.stopped.emit()

    def mediaProgress(self, f):
        self.signals.progress.emit(f)

    def onMediaStopped(self):
        self.glWidget.clear()

    def infoCallback(self, msg):
        print(msg)

    def errorCallback(self, s):
        print("error", s)

    def onMediaStopped(self):
        self.glWidget.clear()
        self.setWindowTitle("onvif gui version 2.0.0")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    app.exec()