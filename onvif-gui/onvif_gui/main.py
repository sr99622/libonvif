#/********************************************************************
# libonvif/onvif-gui/onvif_gui/main.py 
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

if sys.platform == "darwin":
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from loguru import logger

from time import sleep
from datetime import datetime
import importlib.util
from pathlib import Path
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QSplitter, \
    QTabWidget, QMessageBox, QDialog, QGridLayout
from PyQt6.QtCore import pyqtSignal, QObject, QSettings, QDir, QSize, QTimer, Qt
from PyQt6.QtGui import QIcon, QGuiApplication, QMovie
from onvif_gui.panels import CameraPanel, FilePanel, SettingsPanel, VideoPanel, \
    AudioPanel
from onvif_gui.enums import ProxyType, Style
from onvif_gui.glwidget import GLWidget
from onvif_gui.manager import Manager
from onvif_gui.player import Player
from onvif_gui.enums import StreamState
from onvif_gui.protocols import ServerProtocols, ClientProtocols, ListenProtocols
from onvif_gui.components.diskmanager import DiskManager
import avio
import kankakee
import platform
import subprocess
import requests
import onvif_gui
import threading

if sys.platform == "win32":
    from zipfile import ZipFile
else:
    import tarfile

VERSION = "3.0.10"

class TimerSignals(QObject):
    timeoutPlayer = pyqtSignal(str)

class Timer(QTimer):
    def __init__(self, mw, uri):
        super().__init__()
        self.signals = TimerSignals()
        self.mw = mw
        self.uri = uri
        self.size = None
        self.attempting_reconnect = False
        self.disconnected_time = None
        self.thread_lock = False
        self.timeout.connect(self.createPlayer)
        self.signals.timeoutPlayer.connect(mw.playMedia)
        self.start(10000)

    def __str__(self):
        s = self.uri 
        s += "\nattempting_reconnect: " + str(self.attempting_reconnect)
        s += "\ndisconnected_time: " + str(self.disconnected_time)
        s += "\nisActive: " + str(self.isActive())
        return s
    
    def lock(self):
        # lock protects timer spinner render
        while self.thread_lock:
            sleep(0.001)
        self.thread_lock = True

    def unlock(self):
        self.thread_lock = False
        
    def createPlayer(self):
        self.signals.timeoutPlayer.emit(self.uri)

    def start(self, interval):
        self.attempting_reconnect = True
        if self.disconnected_time is None:
            self.disconnected_time = datetime.now()
        super().start(interval)

    def stop(self):
        self.attempting_reconnect = False
        self.disconnected_time = None
        super().stop()

class WaitDialog(QDialog):
    def __init__(self, p):
        super().__init__(p)
        self.lblMessage = QLabel("Please wait for proxy server to download")
        self.lblProgress = QLabel()
        self.movie = QMovie("image:spinner.gif")
        self.movie.setScaledSize(QSize(50, 50))
        self.lblProgress.setMovie(self.movie)
        self.setWindowTitle("Downloading")

        lytMain = QGridLayout(self)
        lytMain.addWidget(self.lblMessage,  0, 1, 1, 1, Qt.AlignmentFlag.AlignCenter)
        lytMain.addWidget(self.lblProgress, 1, 1, 1, 1, Qt.AlignmentFlag.AlignCenter)

        self.movie.start()
        self.setModal(True)

    def sizeHint(self):
        return QSize(300, 100)

class MainWindowSignals(QObject):
    started = pyqtSignal(str)
    stopped = pyqtSignal(str)
    progress = pyqtSignal(float, str)
    error = pyqtSignal(str)
    reconnect = pyqtSignal(str)
    stopReconnect = pyqtSignal(str)
    setTabIndex = pyqtSignal(int)
    showWaitDialog = pyqtSignal(str)
    hideWaitDialog = pyqtSignal()

class MainWindow(QMainWindow):
    def __init__(self, clear_settings=False, settings_profile="gui", parent_role="None"):
        super().__init__()

        self.version = VERSION        

        if sys.platform == "win32":
            filename = os.environ['HOMEPATH'] + "/.cache/onvif-gui/logs/logs.txt"
        else:
            filename = os.environ['HOME'] + "/.cache/onvif-gui/logs/logs.txt"
        self.logger_id = logger.add(filename, rotation="1 MB")
        logger.debug(f'starting onvif-gui version: {VERSION}')

        self.settings_profile = settings_profile
        self.parent_role = parent_role

        QDir.addSearchPath("image", self.getLocation() + "/onvif_gui/resources/")
        self.STD_FILE_DURATION = 900 # duration in seconds (15 * 60)
        self.focus_window = None
        self.reader_window = None
        self.external_windows = []
        self.audioStatus = avio.AudioStatus.UNINITIALIZED
        self.audioLock = False
        self.mediamtx_process = None
        self.viewer_cameras_filled = False
        self.alarm_ordinals = {}
        self.alarm_states = []
        self.last_alarm = None
        self.diskManager = DiskManager(self)

        self.program_name = f'Onvif GUI version {VERSION}'
        self.setWindowTitle(self.program_name)
        self.setWindowIcon(QIcon('image:onvif-gui.png'))
        QGuiApplication.setWindowIcon(QIcon('image:onvif-gui.png'))

        self.settings = QSettings("onvif-gui", settings_profile)
        logger.debug(f'Settings loaded from file {self.settings.fileName()} using format {self.settings.format()}')
        if clear_settings:
            self.settings.clear()
        self.geometryKey = "MainWindow/geometry"
        self.splitKey = "MainWindow/split"
        self.collapsedKey = "MainWindow/collapsed"
        self.nvidiaKey = "MainWindow/haveNVIDIA"
        self.closing = False
        self.dlgWait = WaitDialog(self)
        self.signals = MainWindowSignals()

        self.pm = Manager(self)
        self.timers = {}

        self.proxies = {}
        self.proxy = None
        self.server = None
        self.serverProtocols = ServerProtocols(self)
        self.client = None
        self.clientProtocols = ClientProtocols(self)

        self.broadcaster = None
        self.listener = None
        self.listenProtocols = ListenProtocols(self)

        self.settingsPanel = SettingsPanel(self)
        self.signals.started.connect(self.settingsPanel.onMediaStarted)
        self.signals.stopped.connect(self.settingsPanel.onMediaStopped)
        self.glWidget = GLWidget(self)
        self.cameraPanel = CameraPanel(self)
        self.signals.started.connect(self.cameraPanel.onMediaStarted)
        self.signals.stopped.connect(self.cameraPanel.onMediaStopped)
        self.filePanel = FilePanel(self)
        self.filePanel.control.setBtnMute()
        self.filePanel.control.setSldVolume()
        self.signals.started.connect(self.filePanel.onMediaStarted)
        self.signals.stopped.connect(self.filePanel.onMediaStopped)
        self.signals.progress.connect(self.filePanel.onMediaProgress)
        self.videoPanel = VideoPanel(self)
        self.audioPanel = AudioPanel(self)
        self.signals.error.connect(self.onError)
        self.signals.reconnect.connect(self.startReconnectTimer)
        self.signals.stopReconnect.connect(self.stopReconnectTimer)
        self.signals.showWaitDialog.connect(self.showWaitDialog)
        self.signals.hideWaitDialog.connect(self.hideWaitDialog)

        self.tab = QTabWidget()
        if not bool(int(self.settings.value(self.filePanel.control.hideCameraKey, 0))):
            self.tab.addTab(self.cameraPanel, "Cameras")
        self.tab.addTab(self.filePanel, "Files")
        self.tab.addTab(self.settingsPanel, "Settings")
        if self.settingsPanel.proxy.generateAlarmsLocally():
            self.tab.addTab(self.videoPanel, "Video")
            self.tab.addTab(self.audioPanel, "Audio")
        self.tabVisible = True
        self.tabIndex = 0
        self.tabBugFix = False

        self.split = QSplitter()
        self.split.addWidget(self.glWidget)
        self.split.addWidget(self.tab)
        self.split.setStretchFactor(0, 10)
        self.split.splitterMoved.connect(self.splitterMoved)
        self.setCentralWidget(self.split)

        rect = self.settings.value(self.geometryKey)
        if rect is not None:
            screen = QGuiApplication.screenAt(rect.topLeft())
            if screen:
                self.setGeometry(rect)

        if remote := self.settingsPanel.proxy.proxyRemote:
            comps = remote[:len(remote)-1][7:].split(":")
            ip_addr = comps[0]
            try:
                self.initializeClient(ip_addr)
            except Exception as ex:
                logger.error(f'Unable to configure client {ex}')

        self.videoWorkerHook = None
        self.videoWorker = None
        self.videoConfigureHook = None
        self.videoConfigure = None
        if self.settingsPanel.proxy.generateAlarmsLocally():
            videoWorkerName = self.videoPanel.cmbWorker.currentText()
            if len(videoWorkerName) > 0:
                self.loadVideoConfigure(videoWorkerName)

        self.audioWorkerHook = None
        self.audioWorker = None
        self.audioConfigureHook = None
        self.audioConfigure = None
        if self.settingsPanel.proxy.generateAlarmsLocally():
            audioWorkerName = self.audioPanel.cmbWorker.currentText()
            if len(audioWorkerName) > 0:
                self.loadAudioConfigure(audioWorkerName)

        try:
            if_addrs = self.settingsPanel.proxy.getInterfaces()
            if self.settingsPanel.proxy.proxyType == ProxyType.CLIENT:
                if self.settingsPanel.proxy.chkListen.isChecked():
                    self.startListener(if_addrs)
        except Exception as ex:
            logger.error(f'Unable to initialize multicast {ex}')

        appearance = self.settingsPanel.general.cmbAppearance.currentText()
        if appearance == "Dark":
            self.setStyleSheet(self.style(Style.DARK))
        if appearance == "Light":
            self.setStyleSheet(self.style(Style.LIGHT))

        collapsed = int(self.settings.value(self.collapsedKey, 0))
        if collapsed:
            self.collapseSplitter()
        else:
            self.restoreSplitter()

        logger.debug(f'FFMPEG VERSION: {Player("", self).getFFMPEGVersions()}')

    def loadVideoConfigure(self, workerName):
        spec = importlib.util.spec_from_file_location("VideoConfigure", self.videoPanel.stdLocation + "/" + workerName)
        videoConfigureHook = importlib.util.module_from_spec(spec)        
        sys.modules["VideoConfigure"] = videoConfigureHook
        spec.loader.exec_module(videoConfigureHook)
        self.videoConfigure = videoConfigureHook.VideoConfigure(self)
        self.cameraPanel.lstCamera.currentItemChanged.connect(self.videoConfigure.setCamera)
        self.filePanel.tree.signals.selectionChanged.connect(self.videoConfigure.setFile)
        self.videoPanel.setPanel(self.videoConfigure)

    def pyVideoCallback(self, frame, player):
        if not self.videoWorkerHook:
            file = self.videoPanel.stdLocation + "/" + self.videoPanel.cmbWorker.currentText()
            spec = importlib.util.spec_from_file_location("VideoWorker", file)
            self.videoWorkerHook = importlib.util.module_from_spec(spec)
            sys.modules["VideoWorker"] = self.videoWorkerHook
            spec.loader.exec_module(self.videoWorkerHook)

        if self.videoWorkerHook:
            if not self.videoWorker:
                self.videoWorker = self.videoWorkerHook.VideoWorker(self)
                if player.isCameraStream():
                    player.clearCache()

        if self.videoWorker:
            # motion detector has the option to return the diff frame for viewing
            result = self.videoWorker(frame, player)
            if result is not None:
                frame = result

        return frame

    def loadAudioConfigure(self, workerName):
        spec = importlib.util.spec_from_file_location("AudioConfigure", self.audioPanel.stdLocation + "/" + workerName)
        audioConfigureHook = importlib.util.module_from_spec(spec)
        sys.modules["AudioConfigure"] = audioConfigureHook
        spec.loader.exec_module(audioConfigureHook)
        self.audioConfigure = audioConfigureHook.AudioConfigure(self)
        self.cameraPanel.lstCamera.currentItemChanged.connect(self.audioConfigure.setCamera)
        self.filePanel.tree.signals.selectionChanged.connect(self.audioConfigure.setFile)
        self.audioPanel.setPanel(self.audioConfigure)
    
    def pyAudioCallback(self, frame, player):
        if player.analyze_audio:

            while self.audioLock:
                sleep(0.001)
            self.audioLock = True

            if self.audioWorkerHook is None:
                workerName = self.audioPanel.cmbWorker.currentText()
                spec = importlib.util.spec_from_file_location("AudioWorker", self.audioPanel.stdLocation + "/" + workerName)
                self.audioWorkerHook = importlib.util.module_from_spec(spec)
                sys.modules["AudioWorker"] = self.audioWorkerHook
                spec.loader.exec_module(self.audioWorkerHook)
                self.audioWorker = None

            if self.audioWorkerHook:
                if not self.audioWorker:
                    self.audioWorker = self.audioWorkerHook.AudioWorker(self)
                
            if self.audioWorker:
                self.audioWorker(frame, player)
            
            self.audioLock = False

        else:
            if player.uri == self.glWidget.focused_uri:
                if self.audioWorker:
                    self.audioWorker(None, None)

        return frame
    
    def playMedia(self, uri, file_start_from_seek=-1.0):

        if not uri:
            logger.debug(f'Attempt to create player with null uri')
            return

        count = 0

        if self.settings_profile == "Focus":
            self.closeAllStreams()

        while player := self.pm.getPlayer(uri):
            sleep(0.01)
            count += 1
            if count > 20:
                logger.debug(f'Duplicate media uri from {self.getCameraName(uri)}:{uri} is blocking launch of new player, requesting shutdown')
                player.requestShutdown()
                return
        
        player = Player(uri, self)
        player.file_start_from_seek = file_start_from_seek

        player.pyAudioCallback = self.pyAudioCallback
        player.video_filter = "format=rgb24"
        player.packetDrop = self.packetDrop
        player.renderCallback = self.glWidget.renderCallback
        player.mediaPlayingStarted = self.mediaPlayingStarted
        player.mediaPlayingStopped = self.mediaPlayingStopped
        player.errorCallback = self.errorCallback
        player.infoCallback = self.infoCallback
        player.getAudioStatus = self.getAudioStatus
        player.setAudioStatus = self.setAudioStatus
        player.hw_device_type = self.settingsPanel.general.getDecoder()
        player.audio_driver_index = self.settingsPanel.general.cmbAudioDriver.currentIndex()

        if player.isCameraStream():
            if profile := self.cameraPanel.getProfile(uri):
                player.vpq_size = self.settingsPanel.general.spnCacheMax.value()
                player.apq_size = self.settingsPanel.general.spnCacheMax.value()
                if profile.audio_encoding() == "AAC" and profile.audio_sample_rate() and profile.frame_rate():
                    player.apq_size = int(player.vpq_size * profile.audio_sample_rate() / profile.frame_rate())
                player.buffer_size_in_seconds = self.settings.value(self.settingsPanel.alarm.bufferSizeKey, 10)
                player.onvif_frame_rate.num = profile.frame_rate()
                player.onvif_frame_rate.den = 1
                player.disable_audio = profile.getDisableAudio()
                player.disable_video = profile.getDisableVideo()
                player.hidden = profile.getHidden()
                if not player.hidden:
                    player.last_render = datetime.now()
                player.desired_aspect = profile.getDesiredAspect()
                player.analyze_video = profile.getAnalyzeVideo()
                player.analyze_audio = profile.getAnalyzeAudio()
                player.sync_audio = profile.getSyncAudio()
                camera = self.cameraPanel.getCamera(uri)
                if camera:
                    player.systemTabSettings = camera.systemTabSettings
                    player.setVolume(camera.volume)
                    player.setMute(camera.mute)
        else:
            player.request_reconnect = False
            player.analyze_video = self.filePanel.getAnalyzeVideo()
            player.setVolume(self.filePanel.getVolume())
            player.setMute(self.filePanel.getMute())
            player.analyze_audio = self.filePanel.getAnalyzeAudio()
            player.progressCallback = self.mediaProgress

        self.pm.startPlayer(player)

    def keyPressEvent(self, event):
        match event.key():
            case Qt.Key.Key_Escape:
                if self.settings_profile == "gui":
                    self.showNormal()
                else:
                    self.close()
            case Qt.Key.Key_F12:
                if self.isFullScreen():
                    self.showNormal()
                else:
                    self.showFullScreen()
            case Qt.Key.Key_F11:
                if self.isSplitterCollapsed():
                    self.restoreSplitter()
                    camera = self.cameraPanel.getCurrentCamera()
                    if camera:
                        self.glWidget.focused_uri = camera.uri()
                else:
                    self.glWidget.focused_uri = None
                    self.collapseSplitter()
        super().keyPressEvent(event)

    def showEvent(self, event):
        splitterState = self.settings.value(self.splitKey)
        if not splitterState:
            self.splitterMoved(0, 0)

        if self.settingsPanel.general.chkStartFullScreen.isChecked():
            self.showFullScreen()

        if not self.split.sizes()[0]:
            self.settingsPanel.general.btnHideDisplay.setText("Show Display")

        if self.settingsPanel.discover.chkAutoDiscover.isChecked():
            self.cameraPanel.btnDiscoverClicked()

        super().showEvent(event)

    def resizeEvent(self, event):
        if self.split.sizes()[0]:
            self.settingsPanel.general.btnHideDisplay.setText("Hide Display")
        else:
            self.settingsPanel.general.btnHideDisplay.setText("Show Display")
        super().resizeEvent(event)

    def startAllCameras(self):
        try:
            lstCamera = self.cameraPanel.lstCamera
            if lstCamera:
                cameras = [lstCamera.item(x) for x in range(lstCamera.count())]
                for camera in cameras:
                    self.cameraPanel.setCurrentCamera(camera.uri())
                    self.cameraPanel.onItemDoubleClicked(camera)
        except Exception as ex:
            logger.error(f'Start all cameras error : {ex}')

    def closeAllStreams(self):
        try:
            for timer in self.timers.values():
                self.signals.stopReconnect.emit(timer.uri)
                timer.stop()
            for player in self.pm.players:
                player.requestShutdown()

            self.pm.auto_start_mode = False
            lstCamera = self.cameraPanel.lstCamera
            if lstCamera:
                cameras = [lstCamera.item(x) for x in range(lstCamera.count())]
                for camera in cameras:
                    camera.setIconIdle()

            count = 0
            while len(self.pm.players):
                sleep(0.1)
                count += 1
                if count > 50:
                    logger.error("not all players closed within the allotted time, flushing player manager")
                    for player in self.pm.players:
                        name = ""
                        if player.isCameraStream():
                            if camera := self.cameraPanel.getCamera(player.uri):
                                name = camera.name()
                        else:
                            name = player.uri
                        logger.debug(f'{name} failed orderly shutdown')
                        # This can cause crashing
                        # self.pm.removePlayer(player.uri)
                        # logger.debug(f'{name} was removed from the list after failing orderly shutdown')
                    break

            self.pm.ordinals.clear()
            self.pm.sizes.clear()

            if not self.closing:
                self.cameraPanel.syncGUI()
                
                if self.settingsPanel:
                    if self.settingsPanel.general:
                        self.settingsPanel.general.btnCloseAll.setText("Start All")
        except Exception as ex:
            logger.error(f'Close all streams error : {ex}')

    def closeEvent(self, event):
        try:
            self.closing = True
            self.closeAllStreams()
            self.stopProxyServer()
            self.stopOnvifServer()

            self.settings.setValue(self.geometryKey, self.geometry())
            super().closeEvent(event)

            if self.focus_window:
                self.focus_window.close()

            for window in self.external_windows:
                window.close()

        except Exception as ex:
            logger.error(f'Window close error : {ex}')

    def showWaitDialog(self, str):
        self.dlgWait.lblMessage.setText(str)
        self.dlgWait.exec()

    def hideWaitDialog(self):
        self.dlgWait.hide()


    def mediaPlayingStarted(self, uri):
        if self.isCameraStreamURI(uri):
            if profile := self.cameraPanel.getProfile(uri):
                profile_type = ""
                if camera := self.cameraPanel.getCamera(uri):
                    if camera.isDisplayProfile(uri):
                        profile_type = "Display Profile"
                    else:
                        profile_type = "Record Profile"
                
                window_name = self.settings_profile
                if self.settings_profile == "gui":
                    window_name = "Main"

                logger.debug(f'Camera stream opened {self.getCameraName(uri)}, stream_uri : {profile.stream_uri()}, resolution : {profile.width()} x {profile.height()}, fps: {profile.frame_rate()}, {profile_type}, Window: {window_name}')

            if self.pm.auto_start_mode:
                finished = True
                cameras = [self.cameraPanel.lstCamera.item(x) for x in range(self.cameraPanel.lstCamera.count())]
                for camera in cameras:
                    state = camera.getStreamState(camera.displayProfileIndex())
                    if state == StreamState.IDLE:
                        finished = False
                if finished:
                    self.pm.auto_start_mode = False

            if player := self.pm.getPlayer(uri):
                player.clearCache()
                if player.systemTabSettings:
                    if player.systemTabSettings.record_enable and player.systemTabSettings.record_always:
                        camera = self.cameraPanel.getCamera(uri)
                        if camera:
                            record = False
                            if camera.displayProfileIndex() != camera.recordProfileIndex():
                                if camera.isRecordProfile(uri):
                                    record = True
                            else:
                                if camera.profiles[camera.displayProfileIndex()].uri() == uri:
                                    record = True
                            if record:
                                d = self.settingsPanel.storage.dirArchive.txtDirectory.text()
                                if self.settingsPanel.storage.chkManageDiskUsage.isChecked():
                                    self.diskManager.manageDirectory(d, player.uri)
                                if filename := player.getPipeOutFilename():
                                    player.toggleRecording(filename)

        self.signals.stopReconnect.emit(uri)
        self.signals.started.emit(uri)

    def stopReconnectTimer(self, uri):
        if timer := self.timers.get(uri, None):
            timer.lock()
            timer.stop()
            timer.unlock()

    def mediaPlayingStopped(self, uri):
        if player := self.pm.getPlayer(uri):
            self.pm.removePlayer(uri)

            if player.request_reconnect:
                if camera := self.cameraPanel.getCamera(uri):
                    logger.debug(f'Camera stream closed with reconnect requested {self.getCameraName(uri)}')
                    self.signals.reconnect.emit(uri)
            else:
                if self.isCameraStreamURI(uri):
                    logger.debug(f'Stream closed {self.getCameraName(uri)}')

            if self.signals:
                self.signals.stopped.emit(uri)

    def startReconnectTimer(self, uri):
        if uri in self.timers:
            self.timers[uri].start(10000)
        else:
            self.timers[uri] = Timer(self, uri)
        self.cameraPanel.syncGUI()

    def infoCallback(self, msg, uri):
        if msg == "player audio disabled":
            return
        if msg == "player video disabled":
            return
        if msg == "dropping frames due to buffer overflow":
            return
        if msg.startswith("Pipe opened write file:"):
            return
        if msg.startswith("Pipe closed file:"):
            return
        if msg == "NO AUDIO STREAM FOUND":
            return
        
        name = ""
        if self.isCameraStreamURI(uri):
            camera = self.cameraPanel.getCamera(uri)
            if camera:
                name = f'Camera: {self.getCameraName(uri)}'
        else:
            name = f'File: {uri}'

        if msg.startswith("Output file creation failure") or \
           msg.startswith("Record to file close error"):
            logger.error(f'{name}, Message: {msg}')
            return

        if msg.startswith("SDL_OpenAudioDevice exception"):
            if profile := self.cameraPanel.getProfile(uri):
                profile.setDisableAudio(True)
                self.cameraPanel.tabVideo.syncGUI()
                self.signals.error.emit("Error: Audio output device initialization has failed, audio for this stream has been disabled")
                return

        if msg.startswith("Using SDL audio driver"):
            logger.debug(msg)
            return

        print(f'{name}, Message: {msg}')

    def errorCallback(self, msg, uri, reconnect):
        if reconnect:
            camera_name = ""
            last_msg = ""

            if camera := self.cameraPanel.getCamera(uri):
                camera_name = self.getCameraName(uri)
                last_msg = camera.last_msg
                camera.last_msg = msg

                self.signals.reconnect.emit(uri)

            if msg != last_msg:
                logger.error(f'Error from camera: {camera_name} : {msg}, attempting to re-connect')
        else:
            name = ""
            last_msg = ""
            if self.isCameraStreamURI(uri):
                if player := self.pm.getPlayer(uri):
                    player.requestShutdown()
                    last_msg = player.last_msg
                    player.last_msg = msg

                if camera := self.cameraPanel.getCamera(uri):
                    if c_uri := camera.companionURI(uri):
                        if c_player := self.pm.getPlayer(c_uri):
                            c_player.requestShutdown()

                    name = f'Camera: {self.getCameraName(uri)}'
                    camera.setIconIdle()
                    self.cameraPanel.syncGUI()
                    self.cameraPanel.setTabsEnabled(True)
                
                if msg != last_msg:
                    self.signals.error.emit(msg)

            else:
                self.signals.error.emit(msg)

            logger.error(f'{name}, Error: {msg}')
                
    def mediaProgress(self, pct, uri):
        self.signals.progress.emit(pct, uri)

    def packetDrop(self, uri):
        player = self.pm.getPlayer(uri)
        if player:
            frames = 10
            if player.onvif_frame_rate.num and player.onvif_frame_rate.den:
                frames = int((player.onvif_frame_rate.num / player.onvif_frame_rate.den) * 2)
            player.packet_drop_frame_counter = frames

    def getAudioStatus(self):
        return self.audioStatus
    
    def setAudioStatus(self, status):
        self.audioStatus = status

    def onError(self, msg):
        if not self.closing:
            msgBox = QMessageBox(self)
            msgBox.setText(msg)
            msgBox.setWindowTitle(self.program_name)
            msgBox.setIcon(QMessageBox.Icon.Warning)
            msgBox.exec()
            self.cameraPanel.syncGUI()
            self.cameraPanel.setEnabled(True)

    def isSplitterCollapsed(self):
        return self.split.sizes()[1] == 0

    def collapseSplitter(self):
        self.split.setSizes([self.split.frameSize().width(), 0])
        self.settings.setValue(self.collapsedKey, 1)
        self.tabVisible = False
    
    def splitterMoved(self, pos, index):
        if self.split.sizes()[1]:
            self.settings.setValue(self.collapsedKey, 0)
            self.settings.setValue(self.splitKey, self.split.saveState())
            self.tabVisible = True
        else:
            self.settings.setValue(self.collapsedKey, 1)
            self.tabVisible = False

    def restoreSplitter(self):
        self.settings.setValue(self.collapsedKey, 0)
        splitterState = self.settings.value(self.splitKey)
        if splitterState is not None:
            self.split.restoreState(splitterState)
        else:
            dims = self.size()
            self.split.setSizes([int(dims.width()*0.75), int(dims.width()*0.25)])

    def getCameraName(self, uri):
        result = ""
        camera = self.cameraPanel.getCamera(uri)
        if camera:
            result = camera.text() + " (" + camera.profileName(uri) + ")"
        return result
    
    def getLocation(self):
        path = Path(os.path.dirname(__file__))
        return str(path.parent.absolute())
    
    def initializeFocusWindowSettings(self):
        proxy = None
        match self.settingsPanel.proxy.proxyType:
            case ProxyType.CLIENT:
                proxy = self.settingsPanel.proxy.txtRemote.text()
            case ProxyType.SERVER:
                proxy = self.settingsPanel.proxy.lblServer.text().split()[0]
        focus_settings = QSettings("onvif-gui", "Focus")
        focus_settings.setValue("settings/proxyType", ProxyType.CLIENT)
        focus_settings.setValue("settings/proxyRemote", proxy)
        focus_settings.setValue("settings/autoDiscover", 1)
        focus_settings.setValue(self.collapsedKey, 1)
        self.focus_window = onvif_gui.main.MainWindow(settings_profile="Focus")
        self.focus_window.show()
        while not self.focus_window.viewer_cameras_filled:
            sleep(0.001)

    def initializeReaderWindowSettings(self):
        reader_settings = QSettings("onvif-gui", "Reader")
        reader_settings.setValue("filePanel/hideCameraPanel", 1)
        self.reader_window = onvif_gui.main.MainWindow(settings_profile="Reader")
        self.reader_window.show()

    def isCameraStreamURI(self, uri):
        result = False
        if uri:
            result = uri.lower().startswith("rtsp") or uri.lower().startswith("http")
        return result
    
    def manageBroadcaster(self, if_addrs):
        # an empty list for if_addrs will disable broadcaster
        if self.broadcaster:
            del self.broadcaster
            self.broadcaster = None
        try:
            if len(if_addrs):
                self.broadcaster = kankakee.Broadcaster(if_addrs)
                self.broadcaster.errorCallback = self.listenProtocols.error
                self.broadcaster.enableLoopback(False)
        except Exception as ex:
            logger.error(f'Error initializing broadcaster : {ex}')

    def startListener(self, if_addrs):
        if not self.settings_profile == "gui":
            return
                        
        ip_addr = None
        if len(if_addrs):
            ip_addr = if_addrs[0]

        if remote := self.settingsPanel.proxy.proxyRemote:
            comps = remote[:len(remote)-1][7:].split(":")
            rmt_addr = comps[0]
            rmt = rmt_addr.split(".")
            if ip_addr:
                found = False
                lcl = ip_addr.split(".")
                if len(rmt) == len(lcl):
                    for addr in if_addrs:
                        lcl = addr.split(".")
                        if rmt[0] == lcl[0] and rmt[1] == lcl[1] and rmt[2] == lcl[2]:
                            found = True
                            ip_addr = addr
                            break
                if not found:
                    QMessageBox.warning(self, "Listener Error", "Unable to Start Event Listener\nPlease check proxy configuration")
                    return

        try:
            if self.listener:
                logger.debug("Found existing Alarm Listener, terminating")
                self.stopListener()
            self.listener = kankakee.Listener([ip_addr])
            self.listener.listenCallback = self.listenProtocols.callback
            self.listener.errorCallback = self.listenProtocols.error
            if not self.listener.running:
                self.listener.start()
                logger.debug("Alarm Listener was started successfully")
        except Exception as ex:
            logger.error(f'Error starting Alarm Listener : {ex}')

    def stopListener(self):
        if self.listener:
            try:
                self.listener.stop()
            except Exception as ex:
                logger.error(f'Error stopping Alarm Listener : {ex}')
            
    
    def initializeClient(self, ip_addr):
        try:
            self.client = kankakee.Client(f'{ip_addr}:8550')
            self.client.clientCallback = self.clientProtocols.callback
            self.client.errorCallback = self.clientProtocols.error
        except Exception as ex:
            logger.error(f'Error initializing Onvif Client : {ex}')

    def startOnvifServer(self, ip):
        # if ip is an empty string, bind server to IPADDR_ANY, otherwise bind to ip address
        try:
            if not self.server:
                self.server = kankakee.Server(ip, 8550)
                self.server.serverCallback = self.serverProtocols.callback
                self.server.errorCallback = self.serverProtocols.error
            if not self.server.running:
                self.server.start()
        except Exception as ex:
            logger.error(f'Error starting Onvif Server : {ex}')

    def stopOnvifServer(self):
        try:
            if self.server:
                self.server.stop()
        except Exception as ex:
            logger.error(f'Error stopping Onvif Server : {ex}')
    
    def downloadProxyServer(self):
        try:
            dir = self.settingsPanel.proxy.getProxyServerDir()
            if not os.path.exists(dir):
                os.makedirs(dir)

            logger.debug('Attempting to download MediaMTX server to directory {dir}')
            
            architecture = None
            match platform.machine():
                case "AMD64":
                    architecture = "amd64"
                case "x86_64":
                    architecture = "amd64"
                case "arm64":
                    architecture = "arm64"
                case "aarch64":
                    architecture = "arm64v8"

            operating_system = None
            match sys.platform:
                case "linux":
                    operating_system = "linux"
                case "darwin":
                    operating_system = "darwin"
                case "win32":
                    operating_system = "windows"

            version = "v1.11.2"
            home = "https://github.com/bluenviron/mediamtx/releases/download"
            suffix = "tar.gz"
            if operating_system == "windows":
                suffix = "zip"

            url = None
            if architecture and operating_system:
                url = f'{home}/{version}/mediamtx_{version}_{operating_system}_{architecture}.{suffix}'
            else:
                raise AttributeError(f'Unable to determine MediaMTX server for operating system for {sys.platform} and architecture {platform.machine()}')

            if url:
                download_filename = os.path.join(dir, url.rsplit('/', 1)[1])
                logger.debug(f'Downloading MediaMTX from {url} to {download_filename}')
            
                response = requests.get(url, allow_redirects=True, timeout=(10, 120))
                if not response:
                    raise RuntimeError(f'Error downloading {url}: {response.status_code}')
                
                with open(download_filename, 'wb') as content:
                    content.write(response.content)

                if os.path.isfile(download_filename):
                    logger.debug(f'MediaMTX {url} compressed file was downloaded successfully')

                    archive = None
                    if sys.platform == "win32":
                        archive = ZipFile(download_filename, 'r')
                    else:
                        archive = tarfile.open(download_filename)
                    if archive:
                        archive.extractall(dir)
                        archive.close()
                    else:
                        raise RuntimeError("Unable to open decompression utility for MediaMTX")

        except Exception as ex:
            self.signals.hideWaitDialog.emit()
            raise RuntimeError(f'Unable download MediaMTX {ex}')
        
        self.signals.hideWaitDialog.emit()

    def startProxyServer(self, autoDownload, dir):
        try:
            executable_filename = f'{dir}/mediamtx'
            if sys.platform == "win32":
                executable_filename += ".exe"
            config_filename = f'{dir}/mediamtx.yml'

            if not os.path.isfile(executable_filename) or not os.path.isfile(config_filename):
                if not autoDownload:
                    self.signals.error.emit(f'Error: cannot find MediaMTX in {dir}, please use auto download selection in Settings -> Proxy')
                    return

                thread = threading.Thread(target=self.downloadProxyServer)
                thread.start()
                self.signals.showWaitDialog.emit("Please wait for MediaMTX binary to download")

            if os.path.isfile(executable_filename) and os.path.isfile(config_filename):
                if not self.mediamtx_process:
                    self.mediamtx_process = subprocess.Popen([executable_filename, config_filename], start_new_session=True)
                    sleep(1)
            else:
                raise RuntimeError("Unknown error has occurred in starting MediaMTX proxy server")

        except Exception as ex:
            logger.error(f'Error starting proxy server: {ex}')
            self.signals.error.emit(f'Error starting proxy server: {ex}')

    def stopProxyServer(self):
        if self.mediamtx_process:
            self.mediamtx_process.terminate()
            self.mediamtx_process = None
            logger.debug("Proxy server stopped")

    def getProxyURI(self, arg):
        return self.proxies.get(arg, arg)
    
    def addCameraProxy(self, camera):
        match self.settingsPanel.proxy.proxyType:
            case ProxyType.SERVER:
                for profile in camera.profiles:
                    if_addr = None
                    if len(self.settingsPanel.proxy.if_addrs):
                        if_addr = self.settingsPanel.proxy.if_addrs[0]
                    self.proxies[profile.stream_uri()] = f'rtsp://{if_addr}:8554/{camera.serial_number()}/{profile.profile()}'
            case ProxyType.CLIENT:
                for profile in camera.profiles:
                    server = self.settingsPanel.proxy.txtRemote.text()
                    self.proxies[profile.stream_uri()] = f'{server}{camera.serial_number()}/{profile.profile()}'

    def style(self, appearance):
        match appearance:
            case Style.DARK:
                blDefault = "#5B5B5B"
                bmDefault = "#4B4B4B"
                bdDefault = "#3B3B3B"
                flDefault = "#C6D9F2"
                fmDefault = "#9DADC2"
                fdDefault = "#808D9E"
                slDefault = "#FFFFFF"
                smDefault = "#DDEEFF"
                sdDefault = "#306294"
                isDefault = "#323232"
                strStyle = open(self.getLocation() + "/onvif_gui/resources/darkstyle.qss", "r").read()
                strStyle = strStyle.replace("background_light",  blDefault)
                strStyle = strStyle.replace("background_medium", bmDefault)
                strStyle = strStyle.replace("background_dark",   bdDefault)
                strStyle = strStyle.replace("foreground_light",  flDefault)
                strStyle = strStyle.replace("foreground_medium", fmDefault)
                strStyle = strStyle.replace("foreground_dark",   fdDefault)
                strStyle = strStyle.replace("selection_light",   slDefault)
                strStyle = strStyle.replace("selection_medium",  smDefault)
                strStyle = strStyle.replace("selection_dark",    sdDefault)
                strStyle = strStyle.replace("selection_item",    isDefault)
            case Style.LIGHT:
                blDefault = "#AAAAAA"
                bmDefault = "#CCCCCC"
                bdDefault = "#FFFFFF"
                flDefault = "#111111"
                fmDefault = "#222222"
                fdDefault = "#999999"
                slDefault = "#111111"
                smDefault = "#222222"
                sdDefault = "#999999"
                isDefault = "#888888"
                strStyle = open(self.getLocation() + "/onvif_gui/resources/darkstyle.qss", "r").read()
                strStyle = strStyle.replace("background_light",  blDefault)
                strStyle = strStyle.replace("background_medium", bmDefault)
                strStyle = strStyle.replace("background_dark",   bdDefault)
                strStyle = strStyle.replace("foreground_light",  flDefault)
                strStyle = strStyle.replace("foreground_medium", fmDefault)
                strStyle = strStyle.replace("foreground_dark",   fdDefault)
                strStyle = strStyle.replace("selection_light",   slDefault)
                strStyle = strStyle.replace("selection_medium",  smDefault)
                strStyle = strStyle.replace("selection_dark",    sdDefault)
                strStyle = strStyle.replace("selection_item",    isDefault)

        return strStyle

    def haveNvidia(self):
        output = self.run_command("source $HOME/.local/share/onvif-gui-env/bin/activate && pip list | grep nvidia")
        if len(output.strip()):
            return True
        else:
            return False

    def run_command(self, command, env=None, exit_on_error=False, silent=True):
        if env is None:
            env = ""
            env_dict = os.environ.copy()
            for key in env_dict:
                env += f'{key}={env_dict[key]}\n'

        stdout_r, stdout_w = os.pipe()
        stderr_r, stderr_w = os.pipe()
        pid = os.fork()

        if pid == 0:
            os.close(stdout_r)  
            os.close(stderr_r)
            os.dup2(stdout_w, 1)
            os.dup2(stderr_w, 2)
            os.close(stdout_w)
            os.close(stderr_w)
            os.execlp("bash", "bash", "-c", command, env)
        else:
            os.close(stdout_w)
            os.close(stderr_w)
            with os.fdopen(stdout_r) as stdout_pipe, os.fdopen(stderr_r) as stderr_pipe:
                output = stdout_pipe.read().strip()
                error_output = stderr_pipe.read().strip()
            pid, self.status = os.waitpid(pid, 0)

            if os.WIFEXITED(self.status) and os.WEXITSTATUS(self.status) != 0:
                if not silent:
                    print(f"Command failed: {command}", file=sys.stderr)
                    print(f"Error message: {error_output}", file=sys.stderr)
                if exit_on_error:
                    sys.exit(os.WEXITSTATUS(self.status))

            return output + error_output
        
def run():
    clear_settings = False

    if len(sys.argv) > 1:
        if str(sys.argv[1]) == "--clear":
            clear_settings = True

        if str(sys.argv[1]) == "--icon":
            if sys.platform == "win32":
                filename = os.environ['HOMEPATH'] + "/.cache/onvif-gui/logs/logs.txt"
            else:
                filename = os.environ['HOME'] + "/.cache/onvif-gui/logs/logs.txt"
            logger.add(filename, rotation="1 MB")

            if sys.platform == "win32":
                icon = f'{os.path.split(__file__)[0]}\\resources\\onvif-gui.ico'
                working_dir = f'{os.path.split(sys.executable)[0]}'
                if "conda" in working_dir:
                    working_dir = f'{working_dir}\\Scripts'
                executable = f'{working_dir}\\onvif-gui.exe'
                try:
                    import winshell
                    link_filepath = f'{Path(winshell.desktop())}\\Onvif GUI.lnk'
                    with winshell.shortcut(link_filepath) as link:
                        link.path = executable
                        link.description = "Onvif GUI"
                        link.arguments = ""
                        link.icon_location = (icon, 0)
                        link.working_directory = working_dir

                    logger.debug(f'Desktop icon created for executable {executable}')
                except Exception as ex:
                    logger.debug(f'Error attempting to create desktop icon : {ex}')
            else:
                icon = f'{os.path.split(__file__)[0]}/resources/onvif-gui.png'
                executable = f'{os.path.split(sys.executable)[0]}/onvif-gui %U'

                contents = (f'[Desktop Entry]\n'
                            f'Version={VERSION}\n'
                            f'Name=Onvif GUI\n'
                            f'Comment=IP Camera Application\n'
                            f'Exec={executable}\n'
                            f'Terminal=false\n'
                            f'Icon={icon}\n'
                            f'StartupWMClass=python3\n'
                            f'Type=Application\n'
                            f'Categories=Utility\n')
                
                try:
                    with open('/usr/share/applications/onvif-gui.desktop', 'w') as f:
                        f.write(contents)
                    #logger.debug("Desktop icon created successfully")
                    print("Desktop Icon Created Successfully, look in the system Applications to find Onvif GUI\n")
                except Exception as ex:
                    logger.error(f'Error attempting to create desktop icon : {ex}')

            sys.exit()

    app = QApplication(sys.argv)

    app.setStyle('Fusion')
    window = MainWindow(clear_settings)
    window.show()
    app.exec()

if __name__ == '__main__':
    run()
