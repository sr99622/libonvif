#/********************************************************************
# libonvif/onvif-gui/gui/main.py 
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

if sys.platform == "linux":
    os.environ["QT_QPA_PLATFORM"] = "xcb"
if sys.platform == "darwin":
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from loguru import logger

if sys.platform == "win32":
    filename = os.environ['HOMEPATH'] + "/.cache/onvif-gui/logs.txt"
else:
    filename = os.environ['HOME'] + "/.cache/onvif-gui/logs.txt"
logger.add(filename, rotation="1 MB")

from time import sleep
from datetime import datetime
import importlib.util
from pathlib import Path
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QSplitter, \
    QTabWidget, QMessageBox
from PyQt6.QtCore import pyqtSignal, QObject, QSettings, QDir, QSize, QTimer, Qt
from PyQt6.QtGui import QIcon
from gui.panels import CameraPanel, FilePanel, SettingsPanel, VideoPanel, \
    AudioPanel
from gui.enums import ProxyType
from gui.glwidget import GLWidget
from gui.manager import Manager
from gui.player import Player
from gui.onvif import StreamState
import avio
import liblivemedia

VERSION = "2.2.9"

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
        self.rendering = False
        self.timeout.connect(self.createPlayer)
        self.signals.timeoutPlayer.connect(mw.playMedia)
        self.start(10000)

    def __str__(self):
        s = self.uri 
        s += "\nattempting_reconnect: " + str(self.attempting_reconnect)
        s += "\ndisconnected_time: " + str(self.disconnected_time)
        s += "\nisActive: " + str(self.isActive())
        return s
        
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

class ViewLabel(QLabel):
    def __init__(self):
        super().__init__()

    def sizeHint(self):
        return QSize(640, 480)
    
class MainWindowSignals(QObject):
    started = pyqtSignal(str)
    stopped = pyqtSignal(str)
    progress = pyqtSignal(float, str)
    error = pyqtSignal(str)
    reconnect = pyqtSignal(str)
    stopReconnect = pyqtSignal(str)
    setTabIndex = pyqtSignal(int)

class MainWindow(QMainWindow):
    def __init__(self, clear_settings=False):
        super().__init__()
        os.environ["QT_FILESYSTEMMODEL_WATCH_FILES"] = "ON"
        QDir.addSearchPath("image", self.getLocation() + "/gui/resources/")
        self.style()
        self.STD_FILE_DURATION = 900 # duration in seconds (15 * 60)

        self.audioStatus = avio.AudioStatus.UNINITIALIZED
        self.audioLock = False

        self.program_name = f'onvif gui version {VERSION}'
        self.setWindowTitle(self.program_name)
        self.setWindowIcon(QIcon('image:onvif-gui.png'))
        self.settings = QSettings("onvif", "gui")
        if clear_settings:
            self.settings.clear()
        self.geometryKey = "MainWindow/geometry"
        self.splitKey = "MainWindow/split"
        self.collapsedKey = "MainWindow/collapsed"

        self.signals = MainWindowSignals()

        self.pm = Manager(self)
        self.timers = {}
        self.audioPlayer = None

        self.proxies = {}
        self.proxy = None

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

        self.tab = QTabWidget()
        self.tab.addTab(self.cameraPanel, "Cameras")
        self.tab.addTab(self.filePanel, "Files")
        self.tab.addTab(self.settingsPanel, "Settings")
        self.tab.addTab(self.videoPanel, "Video")
        self.tab.addTab(self.audioPanel, "Audio")
        self.signals.setTabIndex.connect(self.tab.setCurrentIndex)
        self.tab.currentChanged.connect(self.tabIndexChanged)
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
            if rect.width() > 0 and rect.height() > 0 and rect.x() >= 0 and rect.y() >= 0:
                self.setGeometry(rect)

        self.discoverTimer = None
        if self.settingsPanel.discover.chkAutoDiscover.isChecked():
            self.cameraPanel.btnDiscoverClicked()

        self.videoWorkerHook = None
        self.videoWorker = None
        self.videoConfigureHook = None
        self.videoConfigure = None
        videoWorkerName = self.videoPanel.cmbWorker.currentText()
        if len(videoWorkerName) > 0:
            self.loadVideoConfigure(videoWorkerName)

        self.audioWorkerHook = None
        self.audioWorker = None
        self.audioConfigureHook = None
        self.audioConfigure = None
        audioWorkerName = self.audioPanel.cmbWorker.currentText()
        if len(audioWorkerName) > 0:
            self.loadAudioConfigure(audioWorkerName)

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
    
    def playMedia(self, uri, alarm_sound=False):
        if not uri:
            logger.debug(f'Attempt to create player with null uri')
            return

        count = 0
        while self.pm.getPlayer(uri) is not None:
            sleep(0.01)
            count += 1
            if count > 300:
                logger.debug(f'Duplicate media uri from {self.getCameraName(uri)} is blocking launch of new player')
                return

        player = Player(uri, self)

        player.pyAudioCallback = lambda frame, player: self.pyAudioCallback(frame, player)
        player.video_filter = "format=rgb24"
        player.packetDrop = lambda uri : self.packetDrop(uri)
        player.renderCallback = lambda frame, player : self.glWidget.renderCallback(frame, player)
        player.mediaPlayingStarted = lambda uri : self.mediaPlayingStarted(uri)
        player.mediaPlayingStopped = lambda uri : self.mediaPlayingStopped(uri)
        player.errorCallback = lambda msg, uri, reconnect : self.errorCallback(msg, uri, reconnect)
        player.infoCallback = lambda msg, uri : self.infoCallback(msg, uri)
        player.getAudioStatus = lambda : self.getAudioStatus()
        player.setAudioStatus = lambda status : self.setAudioStatus(status)
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
            if alarm_sound:
                player.disable_video = True
                player.setMute(False)
                player.setVolume(self.settingsPanel.alarm.sldAlarmVolume.value())
                player.analyze_audio = False
            else:
                player.setVolume(self.filePanel.getVolume())
                player.setMute(self.filePanel.getMute())
                player.analyze_audio = self.filePanel.getAnalyzeAudio()
                player.progressCallback = lambda pct, uri : self.mediaProgress(pct, uri)

        self.pm.startPlayer(player)

    def keyPressEvent(self, event):
        match event.key():
            case Qt.Key.Key_Escape:
                self.showNormal()
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

        super().showEvent(event)

    def startAllCameras(self):
        try:
            lstCamera = self.cameraPanel.lstCamera
            if lstCamera:
                cameras = [lstCamera.item(x) for x in range(lstCamera.count())]
                for camera in cameras:
                    self.cameraPanel.setCurrentCamera(camera.uri())
                    self.cameraPanel.onItemDoubleClicked(camera)
        except Exception as ex:
            logger.error(f'start all cameras error {ex}')

    def closeAllStreams(self):
        try:
            for player in self.pm.players:
                player.requestShutdown()
            for timer in self.timers.values():
                timer.stop()
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
                if count > 20:
                    logger.debug("not all players closed within the allotted time, flushing player manager")
                    self.pm.players.clear()
                    break

            self.pm.ordinals.clear()
            self.pm.sizes.clear()
            self.cameraPanel.syncGUI()
            if self.settingsPanel:
                if self.settingsPanel.general:
                    self.settingsPanel.general.btnCloseAll.setText("Start All Cameras")
        except Exception as ex:
            logger.error(f'close all streams error {ex}')

    def closeEvent(self, event):
        try:
            self.closeAllStreams()

            self.settings.setValue(self.geometryKey, self.geometry())
            super().closeEvent(event)
        except Exception as ex:
            logger.error(f'window close error: {ex}')

    def mediaPlayingStarted(self, uri):
        if self.isCameraStreamURI(uri):
            #logger.debug(f'camera stream opened {self.getCameraName(uri)} : {uri}')
            logger.debug(f'camera stream opened {self.getCameraName(uri)}')

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
                                    player.manageDirectory(d)
                                filename = player.getPipeOutFilename(d)
                                if filename:
                                    player.toggleRecording(filename)

        self.signals.stopReconnect.emit(uri)
        self.signals.started.emit(uri)

    def stopReconnectTimer(self, uri):
        if timer := self.timers.get(uri, None):
            while timer.rendering:
                sleep(0.001)
            timer.stop()

    def mediaPlayingStopped(self, uri):
        if player := self.pm.getPlayer(uri):
            if player.request_reconnect:
                if camera := self.cameraPanel.getCamera(uri):
                    logger.debug(f'Camera stream closed with reconnect requested {self.getCameraName(uri)}')
                    self.signals.reconnect.emit(uri)
            else:
                if self.isCameraStreamURI(uri):
                    logger.debug(f'Stream closed by user {self.getCameraName(uri)}')

            player.rendering = False
            self.pm.removePlayer(uri)
            self.glWidget.update()
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
           msg.startswith("Record to file close error") or \
           msg.startswith("SDL_OpenAudioDevice exception"):
            logger.error(f'{name}, Message: {msg}')

        if msg.startswith("Using SDL audio driver"):
            logger.debug(msg)

        else: 
            print(f'{name}, Message: {msg}')

    def errorCallback(self, msg, uri, reconnect):
        if reconnect:
            camera_name = ""
            last_msg = ""

            if camera := self.cameraPanel.getCamera(uri):
                camera_name = self.getCameraName(uri)
                last_msg = camera.last_msg
                camera.last_msg = msg

                if player := self.pm.getPlayer(uri):
                    if not player.getVideoWidth():
                        self.pm.removePlayer(uri)

                self.signals.reconnect.emit(uri)

            if msg != last_msg:
                logger.debug(f'Error from camera: {camera_name} : {msg}, attempting to re-connect')
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
                self.closeAllStreams()
                #sleep(0.5)
                #self.filePanel.control.btnPlay.setStyleSheet(self.filePanel.control.getButtonStyle("play"))
                #sleep(0.5)
                #self.signals.error.emit(msg)
                #sleep(0.5)
                #self.pm.removePlayer(uri)

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
        msgBox = QMessageBox(self)
        msgBox.setText(msg)
        msgBox.setWindowTitle(self.program_name)
        msgBox.setIcon(QMessageBox.Icon.Warning)
        msgBox.exec()
        self.cameraPanel.syncGUI()
        self.cameraPanel.setEnabled(True)

    def fixTabIndex(self):
        self.tabBugFix = True
        self.signals.setTabIndex.emit(0)
        shown = self.cameraPanel.tabOnvif.currentIndex()
        self.cameraPanel.tabOnvif.setCurrentIndex(0)
        self.cameraPanel.tabOnvif.setCurrentIndex(shown)

    def tabIndexChanged(self, index):
        if self.tabBugFix and self.tabIndex:
            self.tabBugFix = False
            self.signals.setTabIndex.emit(self.tabIndex)
        else:
            self.tabIndex = index

        if index == 0:
            if camera := self.cameraPanel.getCurrentCamera():
                if self.videoConfigure:
                    self.videoConfigure.setCamera(camera)

    def isSplitterCollapsed(self):
        return self.split.sizes()[1] == 0

    def collapseSplitter(self):
        self.split.setSizes([self.split.frameSize().width(), 0])
        self.settings.setValue(self.collapsedKey, 1)
        self.tabVisible = False
    
    def splitterMoved(self, pos, index):
        if self.split.sizes()[1]:
            if not self.tabVisible:
                self.fixTabIndex()
            self.settings.setValue(self.splitKey, self.split.saveState())
            self.tabVisible = True
        else:
            self.tabVisible = False

    def restoreSplitter(self):
        self.settings.setValue(self.collapsedKey, 0)
        splitterState = self.settings.value(self.splitKey)
        if splitterState is not None:
            self.split.restoreState(splitterState)
        self.fixTabIndex()

    def getCameraName(self, uri):
        result = ""
        camera = self.cameraPanel.getCamera(uri)
        if camera:
            result = camera.text() + " (" + camera.profileName(uri) + ")"
        return result
    
    def getLocation(self):
        path = Path(os.path.dirname(__file__))
        return str(path.parent.absolute())

    def isCameraStreamURI(self, uri):
        result = False
        if uri:
            result = uri.lower().startswith("rtsp") or uri.lower().startswith("http")
        return result
    
    def getAlarmSound(self):
        return f'{self.getLocation()}/gui/resources/drops.mp3'
    
    def getLogFilename(self):
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
    
    def startProxyServer(self):
        try:
            if not self.proxy:
                self.proxy = liblivemedia.ProxyServer()
                self.proxy.init(554)
            self.proxy.startLoop()

        except Exception as ex:
            logger.error(f'Error starting proxy server {str(ex)}')

    def stopProxyServer(self):
        if self.proxy:
            self.proxy.stopLoop()

    def getProxyURI(self, arg):
        match self.settingsPanel.proxy.proxyType:
            case ProxyType.CLIENT:
                return self.proxies[arg]
            case ProxyType.SERVER:
                return self.proxy.getProxyURI(arg)
    
    def addCameraProxy(self, camera):
        match self.settingsPanel.proxy.proxyType:
            case ProxyType.SERVER:
                for profile in camera.profiles:
                    key = f'{camera.serial_number()}/{profile.profile()}'
                    existing_uri = self.proxy.getProxyURI(profile.stream_uri())
                    if not len(existing_uri):
                        self.proxy.addURI(profile.stream_uri(), key, camera.onvif_data.username(), camera.onvif_data.password())
            case ProxyType.CLIENT:
                for profile in camera.profiles:
                    server = self.settingsPanel.proxy.txtRemote.text()
                    self.proxies[profile.stream_uri()] = f'{server}{camera.serial_number()}/{profile.profile()}'

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
                    print("Desktop icon created successfully")
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
