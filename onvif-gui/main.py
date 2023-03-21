#/********************************************************************
# libonvif/python/main.py 
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

import os
import sys
from time import sleep
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QSplitter, \
QTabWidget, QMessageBox
from PyQt6.QtCore import pyqtSignal, QObject, QSettings, QDir, QSize
from PyQt6.QtGui import QIcon
from camerapanel import CameraPanel
from filepanel import FilePanel
from settingspanel import SettingsPanel
from glwidget import GLWidget
from modules.sample import Sample

sys.path.append("../build/libonvif")
import onvif
sys.path.append("../build/libavio")
import avio

class MainWindowSignals(QObject):
    started = pyqtSignal(int)
    stopped = pyqtSignal()
    progress = pyqtSignal(float)
    error = pyqtSignal(str)

class ViewLabel(QLabel):
    def __init__(self):
        super().__init__()

    def sizeHint(self):
        return QSize(640, 480)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        QDir.addSearchPath('image', 'resources/')
        self.program_name = "onvif gui version 2.0.0"
        self.setWindowTitle(self.program_name)
        self.setWindowIcon(QIcon('image:onvif-gui.png'))
        self.settings = QSettings("onvif", "gui")
        self.volumeKey = "MainWindow/volume"
        self.muteKey = "MainWindow/mute"

        self.player = None
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
        self.signals.stopped.connect(self.cameraPanel.onMediaStopped)
        self.filePanel = FilePanel(self)
        self.signals.started.connect(self.filePanel.onMediaStarted)
        self.signals.stopped.connect(self.filePanel.onMediaStopped)
        self.signals.progress.connect(self.filePanel.onMediaProgress)
        self.signals.stopped.connect(self.onMediaStopped)
        self.signals.error.connect(self.onError)
        self.settingsPanel = SettingsPanel(self)
        self.tab.addTab(self.cameraPanel, "Cameras")
        self.tab.addTab(self.filePanel, "Files")
        self.tab.addTab(self.settingsPanel, "Settings")

        if self.settings.value(self.settingsPanel.renderKey, 0) == 0:
            self.glWidget = GLWidget()
        else:
            self.glWidget = ViewLabel()

        split = QSplitter()
        split.addWidget(self.glWidget)
        split.addWidget(self.tab)
        split.setStretchFactor(0, 4)
        self.setCentralWidget(split)

        savedGeometry = self.settings.value("geometry")
        if savedGeometry.isValid():
            self.setGeometry(savedGeometry)

        if self.settingsPanel.chkAutoDiscover.isChecked():
            self.cameraPanel.btnDiscoverClicked()

        self.sample = None

    def playMedia(self, uri):
        self.stopMedia()
        self.player = avio.Player()
        self.player.uri = uri
        self.player.width = lambda : self.glWidget.width()
        self.player.height = lambda : self.glWidget.height()
        if not self.settingsPanel.chkLowLatency.isChecked():
            self.player.vpq_size = 100
            self.player.apq_size = 100
        self.player.progressCallback = lambda f : self.mediaProgress(f)

        video_filter = self.settingsPanel.txtVideoFilter.text()
        if self.settingsPanel.chkConvert2RGB.isChecked():
            self.player.video_filter = "format=rgb24"
            if len(video_filter) > 0:
                self.player.video_filter += "," + video_filter
        else:
            if len(video_filter) > 0:
                self.player.video_filter = video_filter
            else:
                self.player.video_filter = "null"

        if self.settings.value(self.settingsPanel.renderKey, 0) == 0:
            self.player.renderCallback = lambda F : self.glWidget.renderCallback(F)
        else:
            self.player.hWnd = self.glWidget.winId()

        self.player.disable_audio = self.settingsPanel.chkDisableAudio.isChecked()
        self.player.disable_video = self.settingsPanel.chkDisableVideo.isChecked()

        self.player.pythonCallback = lambda F : self.pythonCallback(F)
        self.player.cbMediaPlayingStarted = lambda n : self.mediaPlayingStarted(n)
        self.player.cbMediaPlayingStopped = lambda : self.mediaPlayingStopped()
        self.player.errorCallback = lambda s : self.errorCallback(s)
        self.player.infoCallback = lambda s : self.infoCallback(s)
        self.player.setVolume(self.volume)
        self.player.setMute(self.mute)
        self.player.keyframe_cache_size = self.settingsPanel.spnCacheSize.value()
        self.player.hw_device_type = self.settingsPanel.getDecoder()
        self.player.hw_encoding = self.settingsPanel.chkHardwareEncode.isChecked()
        self.player.post_encode = self.settingsPanel.chkPostEncode.isChecked()
        self.player.start()
        self.cameraPanel.setEnabled(False)

    def stopMedia(self):
        if self.player is not None:
            self.player.running = False
        while self.playing:
            sleep(0.01)

    def toggleMute(self):
        self.mute = not self.mute
        if self.mute:
            self.settings.setValue(self.muteKey, 1)
        else:
            self.settings.setValue(self.muteKey, 0)
        if self.player is not None:
            self.player.setMute(self.mute)

    def setVolume(self, value):
        self.volume = value
        self.settings.setValue(self.volumeKey, value)
        if self.player is not None:
            self.player.setVolume(value)

    def closeEvent(self, e):
        self.stopMedia()
        self.settings.setValue("geometry", self.geometry())

    def mediaPlayingStarted(self, n):
        self.playing = True
        self.connecting = False
        self.signals.started.emit(n)

    def mediaPlayingStopped(self):
        self.playing = False
        self.player = None
        self.signals.stopped.emit()

    def onMediaStopped(self):
        self.glWidget.clear()
        self.setWindowTitle(self.program_name)

    def mediaProgress(self, f):
        self.signals.progress.emit(f)

    def infoCallback(self, msg):
        print(msg)

    def pythonCallback(self, F):
        if self.settingsPanel.chkProcessFrame.isChecked():
            if self.sample is None:
                self.sample = Sample(self)
            self.sample(F)
        return F

    def errorCallback(self, s):
        self.signals.error.emit(s)

    def onError(self, msg):
        msgBox = QMessageBox(self)
        msgBox.setText(msg)
        msgBox.setWindowTitle(self.program_name)
        msgBox.setIcon(QMessageBox.Icon.Warning)
        msgBox.exec()
        self.cameraPanel.setBtnRecord()
        self.filePanel.control.setBtnRecord()
        self.cameraPanel.setEnabled(True)


if __name__ == '__main__':
    os.environ["QT_FILESYSTEMMODEL_WATCH_FILES"] = "ON"
    if sys.platform == "win32":
        sys.argv += ['-platform', 'windows:darkmode=2']
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    app.exec()