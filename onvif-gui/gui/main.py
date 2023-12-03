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
from loguru import logger
if sys.platform == "win32":
    filename = os.environ['HOMEPATH'] + "/.cache/onvif-gui/errors.txt"
else:
    filename = os.environ['HOME'] + "/.cache/onvif-gui/errors.txt"
logger.add(filename, retention="10 days")

import time
from datetime import datetime
import importlib.util
from pathlib import Path
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QSplitter, \
    QTabWidget, QMessageBox
from PyQt6.QtCore import pyqtSignal, QObject, QSettings, QDir, QSize, QTimer
from PyQt6.QtGui import QIcon
from gui.panels import CameraPanel, FilePanel, SettingsPanel, VideoPanel, AudioPanel
from gui.glwidget import GLWidget
from gui.components import WaitDialog
from collections import deque

import avio

VERSION = "1.2.11"

class MainWindowSignals(QObject):
    started = pyqtSignal(int)
    stopped = pyqtSignal()
    progress = pyqtSignal(float)
    error = pyqtSignal(str)
    showWait = pyqtSignal()
    hideWait = pyqtSignal()

class ViewLabel(QLabel):
    def __init__(self):
        super().__init__()

    def sizeHint(self):
        return QSize(640, 480)

class MainWindow(QMainWindow):
    def __init__(self, clear_settings=False):
        super().__init__()
        os.environ["QT_FILESYSTEMMODEL_WATCH_FILES"] = "ON"
        QDir.addSearchPath("image", self.getLocation() + "/gui/resources/")
        self.style()

        self.program_name = f'onvif gui version {VERSION}'
        self.setWindowTitle(self.program_name)
        self.setWindowIcon(QIcon('image:onvif-gui.png'))
        self.settings = QSettings("onvif", "gui")
        if clear_settings:
            self.settings.clear()
        self.volumeKey = "MainWindow/volume"
        self.muteKey = "MainWindow/mute"
        self.geometryKey = "MainWindow/geometry"
        self.tabIndexKey = "MainWindow/tabIndex"
        self.splitKey = "MainWindow/split"

        self.uri = ""
        self.reconnectTimer = QTimer()
        self.reconnectTimer.setSingleShot(True)
        self.reconnectTimer.timeout.connect(self.reconnect)
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
        self.videoPanel = VideoPanel(self)
        self.audioPanel = AudioPanel(self)
        self.signals.stopped.connect(self.onMediaStopped)
        self.signals.error.connect(self.onError)
        self.settingsPanel = SettingsPanel(self)
        self.signals.started.connect(self.settingsPanel.onMediaStarted)
        self.signals.stopped.connect(self.settingsPanel.onMediaStopped)
        self.tab.addTab(self.cameraPanel, "Cameras")
        self.tab.addTab(self.filePanel, "Files")
        self.tab.addTab(self.settingsPanel, "Settings")
        self.tab.addTab(self.videoPanel, "Video")
        self.tab.addTab(self.audioPanel, "Audio")

        if self.settings.value(self.settingsPanel.renderKey, 0) == 0:
            self.glWidget = GLWidget()
        else:
            self.glWidget = ViewLabel()

        self.split = QSplitter()
        self.split.addWidget(self.glWidget)
        self.split.addWidget(self.tab)
        self.split.setStretchFactor(0, 10)
        self.split.splitterMoved.connect(self.splitterMoved)
        splitterState = self.settings.value(self.splitKey)

        self.setCentralWidget(self.split)

        rect = self.settings.value(self.geometryKey)
        if rect is not None:
            if rect.width() > 0 and rect.height() > 0 and rect.x() > 0 and rect.y() > 0:
                self.setGeometry(rect)

        if self.settingsPanel.chkAutoDiscover.isChecked():
            self.cameraPanel.btnDiscoverClicked()

        self.dlgWait = WaitDialog(self)
        self.signals.showWait.connect(self.dlgWait.show)
        self.signals.hideWait.connect(self.dlgWait.hide)

        self.videoRuntimes = deque()
        self.videoWorkerHook = None
        self.videoWorker = None
        self.videoConfigureHook = None
        self.videoConfigure = None
        videoWorkerName = self.videoPanel.cmbWorker.currentText()
        if len(videoWorkerName) > 0:
            self.loadVideoConfigure(videoWorkerName)

        self.audioRuntimes = deque()
        self.audioWorkerHook = None
        self.audioWorker = None
        self.audioConfigureHook = None
        self.audioConfigure = None
        audioWorkerName = self.audioPanel.cmbWorker.currentText()
        if len(audioWorkerName) > 0:
            self.loadAudioConfigure(audioWorkerName)

        if splitterState is not None:
            self.split.restoreState(splitterState)
        self.splitterCollapsed = False
        if not self.split.sizes()[1]:
            self.splitterCollapsed = True

        self.tab.setCurrentIndex(4)
        self.tab.setCurrentIndex(int(self.settings.value(self.tabIndexKey, 0)))

    def loadVideoConfigure(self, workerName):
        spec = importlib.util.spec_from_file_location("VideoConfigure", self.videoPanel.dirModules.text() + "/" + workerName)
        videoConfigureHook = importlib.util.module_from_spec(spec)
        sys.modules["VideoConfigure"] = videoConfigureHook
        spec.loader.exec_module(videoConfigureHook)
        self.configure = videoConfigureHook.VideoConfigure(self)
        self.videoPanel.setPanel(self.configure)

    def loadVideoWorker(self, workerName):
        spec = importlib.util.spec_from_file_location("VideoWorker", self.videoPanel.dirModules.text() + "/" + workerName)
        self.videoWorkerHook = importlib.util.module_from_spec(spec)
        sys.modules["VideoWorker"] = self.videoWorkerHook
        spec.loader.exec_module(self.videoWorkerHook)
        self.worker = None

    def pyVideoCallback(self, F):
        if self.videoPanel.chkEngage.isChecked():

            if self.videoWorkerHook is None:
                videoWorkerName = self.videoPanel.cmbWorker.currentText()
                if len(videoWorkerName) > 0:
                    self.loadVideoWorker(videoWorkerName)

            if self.videoWorkerHook is not None:
                if self.worker is None:
                    self.worker = self.videoWorkerHook.VideoWorker(self)

                start = time.perf_counter()
                self.worker(F)
                finish = time.perf_counter()
                elapsed = int((finish - start) * 1000)
                self.videoRuntimes.append(elapsed)
                if len(self.videoRuntimes) > 60:
                    self.videoRuntimes.popleft()
                sum = 0
                for x in self.videoRuntimes:
                    sum += x
                display = str(int(sum / len(self.videoRuntimes)))
                self.videoPanel.lblElapsed.setText(f'Avg Rumtime (ms)  {display}')

        else:
            self.videoPanel.lblElapsed.setText("")
        return F
    
    def loadAudioConfigure(self, workerName):
        spec = importlib.util.spec_from_file_location("AudioConfigure", self.audioPanel.dirModules.text() + "/" + workerName)
        audioConfigureHook = importlib.util.module_from_spec(spec)
        sys.modules["AudioConfigure"] = audioConfigureHook
        spec.loader.exec_module(audioConfigureHook)
        self.audioConfigure = audioConfigureHook.AudioConfigure(self)
        self.audioPanel.setPanel(self.audioConfigure)
    
    def loadAudioWorker(self, workerName):
        spec = importlib.util.spec_from_file_location("AudioWorker", self.audioPanel.dirModules.text() + "/" + workerName)
        self.audioWorkerHook = importlib.util.module_from_spec(spec)
        sys.modules["AudioWorker"] = self.audioWorkerHook
        spec.loader.exec_module(self.audioWorkerHook)
        self.audioWorker = None

    def pyAudioCallback(self, F):
        if self.audioPanel.chkEngage.isChecked():

            if self.audioWorkerHook is None:
                audioWorkerName = self.audioPanel.cmbWorker.currentText()
                if len(audioWorkerName) > 0:
                    self.loadAudioWorker(audioWorkerName)

            if self.audioWorkerHook is not None:
                if self.audioWorker is None:
                    self.audioWorker = self.audioWorkerHook.AudioWorker(self)
                
                start = time.perf_counter()
                self.audioWorker(F)
                finish = time.perf_counter()
                elapsed = int((finish - start) * 1000)
                self.audioRuntimes.append(elapsed)
                if len(self.audioRuntimes) > 100:
                    self.audioRuntimes.popleft()
                sum = 0
                for x in self.audioRuntimes:
                    sum += x
                display = str(int(sum / len(self.audioRuntimes)))
                self.audioPanel.lblElapsed.setText(f'Avg Runtime (ms)  {display}')

        else:
            self.audioPanel.lblElapsed.setText("")
        return F

    def playMedia(self, uri):
        self.stopMedia()
        self.player = avio.Player()
        self.uri = uri
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
        if "fps=" in self.player.video_filter:
            self.filePanel.progress.setEnabled(False)

        audio_filter = self.settingsPanel.txtAudioFilter.text()
        if len(audio_filter) > 0:
            self.player.audio_filter = audio_filter

        if self.settings.value(self.settingsPanel.renderKey, 0) == 0:
            self.player.renderCallback = lambda F : self.glWidget.renderCallback(F)
        else:
            self.player.hWnd = self.glWidget.winId()

        self.player.disable_audio = self.settingsPanel.chkDisableAudio.isChecked()
        self.player.disable_video = self.settingsPanel.chkDisableVideo.isChecked()

        self.player.pythonCallback = lambda F : self.pyVideoCallback(F)
        self.player.pyAudioCallback = lambda F: self.pyAudioCallback(F)
        self.player.cbMediaPlayingStarted = lambda n : self.mediaPlayingStarted(n)
        self.player.cbMediaPlayingStopped = lambda : self.mediaPlayingStopped()
        self.player.errorCallback = lambda s : self.errorCallback(s)
        self.player.infoCallback = lambda s : self.infoCallback(s)
        self.player.setVolume(int(self.volume))
        self.player.setMute(self.mute)
        self.player.keyframe_cache_size = self.settingsPanel.spnCacheSize.value()
        self.player.hw_device_type = self.settingsPanel.getDecoder()
        self.player.hw_encoding = self.settingsPanel.chkHardwareEncode.isChecked()
        self.player.post_encode = self.settingsPanel.chkPostEncode.isChecked()
        self.player.process_pause = self.settingsPanel.chkProcessPause.isChecked()
        self.player.start()
        self.cameraPanel.setEnabled(False)

    def stopMedia(self):
        if self.player is not None:
            self.player.running = False
        while self.playing:
            time.sleep(0.01)

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
        self.settings.setValue(self.geometryKey, self.geometry())
        self.settings.setValue(self.tabIndexKey, self.tab.currentIndex())

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

    def errorCallback(self, msg):
        self.signals.error.emit(msg)

    def onError(self, msg):
        logger.debug(f'Error processing stream: {self.uri} - {msg}')
        if self.settingsPanel.chkAutoReconnect.isChecked():
            self.reconnectTimer.start(10000)
            self.signals.showWait.emit()
        else:
            msgBox = QMessageBox(self)
            msgBox.setText(msg)
            msgBox.setWindowTitle(self.program_name)
            msgBox.setIcon(QMessageBox.Icon.Warning)
            msgBox.exec()
            self.cameraPanel.setBtnRecord()
            self.filePanel.control.setBtnRecord()
            self.cameraPanel.setEnabled(True)

    def reconnect(self):
        self.signals.hideWait.emit()
        logger.debug("Attempting to re-connnect")
        self.playMedia(self.uri)

    def splitterMoved(self, pos, index):
        if self.split.sizes()[1]:
            if self.splitterCollapsed:
                self.splitterCollapsed = False
                ci = self.tab.currentIndex()
                self.tab.setCurrentIndex(0)
                self.tab.setCurrentIndex(ci)
                ci = self.cameraPanel.tabOnvif.currentIndex()
                self.cameraPanel.tabOnvif.setCurrentIndex(0)
                self.cameraPanel.tabOnvif.setCurrentIndex(ci)
        else:
            self.splitterCollapsed = True

        self.settings.setValue(self.splitKey, self.split.saveState())

    def getLocation(self):
        path = Path(os.path.dirname(__file__))
        return str(path.parent.absolute())
    
    def getVideoFrameRate(self):
        result = 0
        if self.player is not None:
            result = self.player.getVideoFrameRate()
        return result
    
    def get_log_filename(self):
        source = self.windowTitle()
        source = source.replace(".", "_")
        source = source.replace(" ", "_")
        datestamp = datetime.now().strftime("%Y%m%d")
        timestamp = datetime.now().strftime("%H%M%S")
        log_dir = ""
        if sys.platform == "win32":
            log_dir = os.environ["HOMEPATH"]
        else:
            log_dir = os.environ["HOME"]
        log_dir += os.path.sep + "logs" + os.path.sep + "onvif-gui" + os.path.sep + datestamp
        return log_dir + os.path.sep + source + "_" + timestamp + ".csv"
    
    def style(self):
        blDefault = "#5B5B5B"
        bmDefault = "#4B4B4B"
        bdDefault = "#3B3B3B"
        flDefault = "#C6D9F2"
        fmDefault = "#9DADC2"
        fdDefault = "#808D9E"
        slDefault = "#FFFFFF"
        smDefault = "#DDEEFF"
        sdDefault = "#306294"
        strStyle = open(self.getLocation() + "/gui/resources/darkstyle.qss", "r").read()
        strStyle = strStyle.replace("background_light",  blDefault)
        strStyle = strStyle.replace("background_medium", bmDefault)
        strStyle = strStyle.replace("background_dark",   bdDefault)
        strStyle = strStyle.replace("foreground_light",  flDefault)
        strStyle = strStyle.replace("foreground_medium", fmDefault)
        strStyle = strStyle.replace("foreground_dark",   fdDefault)
        strStyle = strStyle.replace("selection_light",   slDefault)
        strStyle = strStyle.replace("selection_medium",  smDefault)
        strStyle = strStyle.replace("selection_dark",    sdDefault)
        self.setStyleSheet(strStyle)

def run():
    clear_settings = False

    if sys.platform == "linux":
        os.environ["QT_QPA_PLATFORM"] = "xcb"

    if len(sys.argv) > 1:
        if str(sys.argv[1]) == "--clear":
            clear_settings = True

        if str(sys.argv[1]) == "--icon":
            if sys.platform == "win32":
                icon = f'{os.path.split(__file__)[0]}\\resources\\onvif-gui.ico'
                working_dir = f'{os.path.split(sys.executable)[0]}'
                executable = f'{working_dir}\\onvif-gui.exe'
                try:
                    import winshell
                    link_filepath = f'{Path(winshell.desktop())}\\onvif-gui.lnk'
                    with winshell.shortcut(link_filepath) as link:
                        link.path = executable
                        link.description = "onvif-gui"
                        link.arguments = ""
                        link.icon_location = (icon, 0)
                        link.working_directory = working_dir

                    logger.debug(f'Desktop icon created for executable {executable}')
                except Exception as ex:
                    logger.debug(f'Error attempting to create desktop icon {str(ex)}')

            else:
                icon = f'{os.path.split(__file__)[0]}/resources/onvif-gui.png'
                executable = f'{os.path.split(sys.executable)[0]}/onvif-gui %U'

                contents = (f'[Desktop Entry]\n'
                            f'Version={VERSION}\n'
                            f'Name=onvif-gui\n'
                            f'Comment=onvif-gui\n'
                            f'Exec={executable}\n'
                            f'Terminal=false\n'
                            f'Icon={icon}\n'
                            f'StartupWMClass=onvif-gui\n'
                            f'Type=Application\n'
                            f'Categories=Application;Network\n')
                
                try:
                    with open('/usr/share/applications/onvif-gui.desktop', 'w') as f:
                        f.write(contents)
                    print("Desktop icon created, please log out to update")
                except Exception as ex:
                    print("Error attempting to create desktop icon " + str(ex))

            sys.exit()

    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow(clear_settings)
    window.style()
    window.show()
    app.exec()

if __name__ == '__main__':
    run()