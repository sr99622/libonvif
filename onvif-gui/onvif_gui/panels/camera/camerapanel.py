#/********************************************************************
# onvif-gui/onvif_gui/panels/camera/camerapanel.py 
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

from time import sleep
from PyQt6.QtWidgets import QPushButton, QGridLayout, QWidget, QSlider, \
    QListWidget, QTabWidget, QMessageBox, QMenu, QFileDialog
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QTimer, QSettings
from . import NetworkTab, ImageTab, VideoTab, PTZTab, SystemTab, LoginDialog, \
    Session, Camera, CameraList, Snapshot
from loguru import logger
import libonvif as onvif
from pathlib import Path
import os
import subprocess
from datetime import datetime
from onvif_gui.enums import ProxyType, SnapshotAuth
import platform
import webbrowser
import requests
from requests.auth import HTTPDigestAuth
from urllib.parse import urlparse, parse_qs
import threading
import sys
import time

class CameraPanelSignals(QObject):
    fill = pyqtSignal(onvif.Data)
    login = pyqtSignal(onvif.Data)
    collapseSplitter = pyqtSignal()
    guiSync = pyqtSignal()

class CameraPanel(QWidget):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw
        self.dlgLogin = LoginDialog(self)
        self.fillers = []
        self.sync_lock = False
        self.snapshot = Snapshot(mw)

        self.autoTimeSyncer = None
        self.enableAutoTimeSync(self.mw.settingsPanel.general.chkAutoTimeSync.isChecked())
       
        self.lstCamera = CameraList(mw)
        self.lstCamera.currentItemChanged.connect(self.onCurrentItemChanged)
        self.lstCamera.itemDoubleClicked.connect(self.onItemDoubleClicked)
        self.lstCamera.itemClicked.connect(self.onItemClicked)
        self.lstCamera.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.lstCamera.customContextMenuRequested.connect(self.showContextMenu)

        self.sldVolume = QSlider(Qt.Orientation.Horizontal)
        self.sldVolume.setValue(80)
        self.sldVolume.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.sldVolume.valueChanged.connect(self.sldVolumeChanged)
        self.sldVolume.setEnabled(False)

        self.btnStop = QPushButton()
        self.btnStop.setMinimumWidth(40)
        self.btnStop.setMaximumHeight(20)
        self.btnStop.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btnStop.clicked.connect(self.btnStopClicked)

        self.btnRecord = QPushButton()
        self.btnRecord.setMinimumWidth(40)
        self.btnRecord.setMaximumHeight(20)
        self.btnRecord.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btnRecord.clicked.connect(self.btnRecordClicked)

        self.btnMute = QPushButton()
        self.btnMute.setMinimumWidth(40)
        self.btnMute.setMaximumHeight(20)
        self.btnMute.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btnMute.clicked.connect(self.btnMuteClicked)

        self.btnDiscover = QPushButton()
        self.btnDiscover.setMinimumWidth(40)
        self.btnDiscover.setMaximumHeight(20)
        self.btnDiscover.setStyleSheet(self.getButtonStyle("discover"))
        self.btnDiscover.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btnDiscover.clicked.connect(self.btnDiscoverClicked)

        self.btnApply = QPushButton()
        self.btnApply.setMinimumWidth(40)
        self.btnApply.setMaximumHeight(20)
        self.btnApply.setStyleSheet(self.getButtonStyle("apply"))
        self.btnApply.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btnApply.clicked.connect(self.btnApplyClicked)
        self.btnApply.setEnabled(False)
        
        self.btnSnapshot = QPushButton()
        self.btnSnapshot.setMinimumWidth(40)
        self.btnSnapshot.setMaximumHeight(20)
        self.btnSnapshot.setStyleSheet(self.getButtonStyle("snapshot"))
        self.btnSnapshot.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btnSnapshot.clicked.connect(self.btnSnapshotClicked)
        self.btnSnapshot.setEnabled(False)        

        self.btnHelp = QPushButton()
        self.btnHelp.setMinimumWidth(40)
        self.btnHelp.setMaximumHeight(20)
        self.btnHelp.setStyleSheet(self.getButtonStyle("help"))
        self.btnHelp.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btnHelp.clicked.connect(self.btnHelpClicked)
        
        self.btnStopAll = QPushButton()
        self.btnStopAll.setMinimumWidth(40)
        self.btnStopAll.setMaximumHeight(20)
        self.btnStopAll.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btnStopAll.clicked.connect(self.btnStopAllClicked)

        self.btnHistory = QPushButton()
        self.btnHistory.setMinimumWidth(40)
        self.btnHistory.setMaximumHeight(20)
        self.btnHistory.setStyleSheet(self.getButtonStyle("history"))
        self.btnHistory.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btnHistory.clicked.connect(self.btnHistoryClicked)

        self.btnFullScreen = QPushButton()
        self.btnFullScreen.setMinimumWidth(40)
        self.btnFullScreen.setMaximumHeight(20)
        self.btnFullScreen.setStyleSheet(self.getButtonStyle("full_screen"))
        self.btnFullScreen.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btnFullScreen.clicked.connect(self.btnFullScreenClicked)

        self.tabOnvif = QTabWidget()
        self.tabOnvif.setUsesScrollButtons(False)
        self.tabVideo = VideoTab(self)
        self.tabImage = ImageTab(self)
        self.tabNetwork = NetworkTab(self)
        self.ptzTab = PTZTab(self)
        self.tabSystem = SystemTab(self)
        self.tabOnvif.addTab(self.tabVideo, "Media")
        self.tabOnvif.addTab(self.tabImage, "Image")
        self.tabOnvif.addTab(self.tabNetwork, "Network")
        self.tabOnvif.addTab(self.ptzTab, "PTZ")
        self.tabOnvif.addTab(self.tabSystem, "System")

        self.signals = CameraPanelSignals()
        self.signals.fill.connect(self.tabVideo.fill)
        self.signals.fill.connect(self.tabImage.fill)
        self.signals.fill.connect(self.tabNetwork.fill)
        self.signals.fill.connect(self.ptzTab.fill)
        self.signals.fill.connect(self.tabSystem.fill)
        self.signals.fill.connect(self.syncGUI)
        self.signals.login.connect(self.onShowLogin)
        self.signals.collapseSplitter.connect(self.mw.collapseSplitter)
        self.signals.guiSync.connect(self.syncGUI)

        lytMain = QGridLayout(self)
        lytMain.addWidget(self.lstCamera,     0, 0, 1, 6)
        lytMain.addWidget(self.tabOnvif,      1, 0, 1, 6)
        lytMain.addWidget(self.btnStopAll,    2, 0, 1, 1)
        lytMain.addWidget(self.btnHistory,    2, 1, 1, 1)
        lytMain.addWidget(self.btnSnapshot,   2, 2, 1, 1)
        lytMain.addWidget(self.btnFullScreen, 2, 3, 1, 1)
        lytMain.addWidget(self.btnHelp,       2, 4, 1, 1)
        lytMain.addWidget(self.btnStop,       3, 0, 1, 1)
        lytMain.addWidget(self.btnRecord,     3, 1, 1, 1)
        lytMain.addWidget(self.btnDiscover,   3, 2, 1, 1)
        lytMain.addWidget(self.btnApply,      3, 3, 1, 1)
        lytMain.addWidget(self.btnMute,       3, 4, 1, 1)
        lytMain.addWidget(self.sldVolume,     3, 5, 1, 1)
        lytMain.setColumnStretch(5, 10)
        lytMain.setRowStretch(0, 10)

        self.menu = QMenu("Context Menu", self)
        self.remove = QAction("Delete", self)
        self.rename = QAction("Rename", self)
        self.info = QAction("Info", self)
        self.password = QAction("Password", self)
        self.start = QAction("Start", self)
        self.stop = QAction("Stop", self)
        self.remove.triggered.connect(self.onMenuRemove)
        self.rename.triggered.connect(self.onMenuRename)
        self.info.triggered.connect(self.onMenuInfo)
        self.password.triggered.connect(self.onMenuPassword)
        self.start.triggered.connect(self.onMenuStart)
        self.stop.triggered.connect(self.onMenuStop)
        self.menu.addAction(self.remove)
        self.menu.addAction(self.rename)
        self.menu.addAction(self.info)
        self.menu.addAction(self.password)
        self.menu.addAction(self.start)
        self.menu.addAction(self.stop)

        self.syncGUI()
        self.setTabsEnabled(False)
        self.sessions = []
        self.closing = False

    def showContextMenu(self, pos):
        index = self.lstCamera.indexAt(pos)
        if index.isValid():
            self.menu.exec(self.mapToGlobal(pos))

    def onMenuRemove(self):
        self.lstCamera.remove()
    
    def onMenuRename(self):
        self.lstCamera.rename()

    def onMenuInfo(self):
        self.lstCamera.info()

    def onMenuPassword(self):
        self.lstCamera.password()

    def onMenuStart(self):
        self.lstCamera.startCamera()

    def onMenuStop(self):
        self.lstCamera.stopCamera()

    def btnDiscoverClicked(self):
        if self.mw.settingsPanel.proxy.proxyType == ProxyType.CLIENT:
            if self.mw.client:
                self.mw.client.transmit(bytearray("GET CAMERAS\r\n", 'utf-8'))
            return
        
        if self.mw.settingsPanel.discover.radDiscover.isChecked():
            logger.debug("Using broadcast discovery")
            interfaces = []
            self.sessions.clear()

            if self.mw.settingsPanel.discover.chkScanAllNetworks.isChecked():
                for i in range(self.mw.settingsPanel.discover.cmbInterfaces.count()):
                    interfaces.append(self.mw.settingsPanel.discover.cmbInterfaces.itemText(i))
            else:
                interfaces.append(self.mw.settingsPanel.discover.cmbInterfaces.currentText())

            for interface in interfaces:
                session = Session(self, interface)
                session.start()
                self.sessions.append(session)
                self.btnDiscover.setEnabled(False)
                if len(interfaces) > 1:
                    sleep(1)
        else:
            logger.debug("Using cached camera addresses for discovery")
            self.fillers.clear()
            tmp = self.mw.settings.value(self.mw.settingsPanel.discover.cameraListKey)
            if tmp:
                numbers = tmp.strip().split("\n")
                for serial_number in numbers:
                    key = f'{serial_number}/XAddrs'
                    xaddrs = self.mw.settings.value(key)
                    alias = self.mw.settings.value(f'{serial_number}/Alias')
                    data = onvif.Data()
                    data.errorCallback = self.errorCallback
                    data.infoCallback = self.infoCallback
                    data.setSetting = self.mw.settings.setValue
                    data.getSetting = self.mw.settings.value
                    data.getData = self.getData
                    data.getCredential = self.getCredential
                    data.setXAddrs(xaddrs)
                    data.alias = alias
                    data.setCameraName(alias)
                    data.setDeviceService("POST /onvif/device_service HTTP/1.1\r\n")
                    self.fillers.append(data)
                    data.startManualFill()

    def errorCallback(self, msg):
        try:
            self.mw.signals.error.emit(msg)
        except Exception as ex:
            logger.error("camera panel error callback exception: {ex}")

    def infoCallback(self, msg):
        try:
            if msg.startswith("Set System Date and Time Error"):
                logger.error(msg)
            else:
                logger.debug(msg)
        except Exception as ex:
            logger.error("camera panel info callback exception: {ex}")

    def discoveryTimeout(self):
        self.setEnabled(True)
        self.btnDiscover.setEnabled(True)

    def discovered(self):
        finished = True
        for session in self.sessions:
            if session.active:
                finished = False
                break

        if finished:
            self.sessions.clear()
            self.btnDiscover.setEnabled(True)
            if self.mw.settingsPanel.proxy.proxyType == ProxyType.SERVER:
                self.mw.settingsPanel.proxy.setMediaMTXProxies()

    def getCredential(self, onvif_data):
        if not onvif_data: return
        
        if self.getCameraByXAddrs(onvif_data.xaddrs()) and not len(self.fillers):
            onvif_data.cancelled = True
            return onvif_data
                
        alternateUsername = self.mw.settings.value(f'{onvif_data.xaddrs()}/alternateUsername', None)
        alternatePassword = self.mw.settings.value(f'{onvif_data.xaddrs()}/alternatePassword', None)

        if (len(self.mw.settingsPanel.general.txtPassword.text()) or alternatePassword) and len(onvif_data.last_error()) == 0:
            if alternateUsername:
                onvif_data.setUsername(alternateUsername)
            else:
                onvif_data.setUsername(self.mw.settingsPanel.general.txtUsername.text())
            if alternatePassword:
                onvif_data.setPassword(alternatePassword)
            else:
                onvif_data.setPassword(self.mw.settingsPanel.general.txtPassword.text())
        else:
            if onvif_data.last_error().startswith("Network error, unable to connect"):
                logger.debug(f'Unable to connect with {onvif_data.xaddrs()}')
                onvif_data.cancelled = True
                self.mw.signals.error.emit(f'Unable to connect with {onvif_data.xaddrs()}')
            else:
                while self.dlgLogin.active:
                    sleep(0.01)

                self.dlgLogin.active = True
                self.signals.login.emit(onvif_data)
                while self.dlgLogin.active:
                    sleep(0.01)

        return onvif_data
    
    def onShowLogin(self, onvif_data):
        self.dlgLogin.exec(onvif_data)

    def getProxyData(self, onvif_data):
        if not onvif_data: return
        
        onvif_data.getProxyURI = self.mw.getProxyURI

        alias = self.mw.settings.value(f'{onvif_data.serial_number()}/Alias')
        if not alias:
            name = onvif_data.camera_name()
            if len(name):
                alias = name
                self.mw.settings.setValue(f'{onvif_data.serial_number()}/Alias', name)
            else:
                alias = onvif_data.host()
        onvif_data.alias = alias

        if not self.getCameraBySerialNumber(onvif_data.serial_number()):

            self.mw.alarm_ordinals[len(self.mw.alarm_ordinals)] = onvif_data.serial_number()

            add_camera = True
            if self.mw.settingsPanel.discover.radCached.isChecked() and self.mw.settingsPanel.proxy.proxyType == ProxyType.CLIENT:
                if tmp := self.mw.settings.value(self.mw.settingsPanel.discover.cameraListKey):
                    numbers = tmp.strip().split("\n")
                    if onvif_data.serial_number() not in numbers:
                        add_camera = False

            if add_camera:
                camera = Camera(onvif_data, self.mw)
                camera.setIconIdle()
                camera.dimForeground()
                self.mw.addCameraProxy(camera)
                self.lstCamera.addItem(camera)
                self.lstCamera.sortItems()
                camera.setDisplayProfile(camera.getDisplayProfileSetting())
                logger.debug(f'Discovery completed for Camera: {onvif_data.alias}, Stream URI: {onvif_data.stream_uri()}, xaddrs: {onvif_data.xaddrs()}')

                self.filled(onvif_data)

    def getData(self, onvif_data):
        if not onvif_data: return
        
        if onvif_data.last_error().startswith("Error initializing camera data during manual fill:"):
            logger.debug(onvif_data.last_error())
            return

        onvif_data.filled = self.filled
        onvif_data.infoCallback = self.infoCallback
        onvif_data.errorCallback = self.errorCallback

        alias = self.mw.settings.value(f'{onvif_data.serial_number()}/Alias')
        if not alias:
            name = onvif_data.camera_name()
            if len(name):
                alias = name
                self.mw.settings.setValue(f'{onvif_data.serial_number()}/Alias', name)
            else:
                alias = onvif_data.host()
        onvif_data.alias = alias

        if existing := self.getCameraBySerialNumber(onvif_data.serial_number()):
            synchronizeTime = self.mw.settingsPanel.general.chkAutoTimeSync.isChecked()
            if not self.closing:
                existing.onvif_data.setXAddrs(onvif_data.xaddrs())
                for profile in existing.profiles:
                    profile.setXAddrs(onvif_data.xaddrs())
                existing.onvif_data.startFill(synchronizeTime)
        else:
            camera = Camera(onvif_data, self.mw)
            camera.setIconIdle()
            camera.dimForeground()
            self.mw.addCameraProxy(camera)
            
            self.lstCamera.addItem(camera)
            self.lstCamera.sortItems()
            camera.setDisplayProfile(camera.getDisplayProfileSetting())
            self.saveCameraList()
            logger.debug(f'Discovery completed for Camera: {onvif_data.alias}, Stream URI: {onvif_data.stream_uri()}, xaddrs: {onvif_data.xaddrs()}, {onvif_data.camera_name()}')

            synchronizeTime = self.mw.settingsPanel.general.chkAutoTimeSync.isChecked()
            if not self.closing:
                onvif_data.startFill(synchronizeTime)

    def filled(self, onvif_data):
        if not onvif_data: return
        
        if camera := self.getCamera(onvif_data.uri()):
            camera.restoreForeground()
            key = f'{camera.serial_number()}/XAddrs'
            self.mw.settings.setValue(key, camera.xaddrs())

            if camera.manual_fill:
                camera.assignData(onvif_data)
                self.mw.addCameraProxy(camera)
                camera.setDisplayProfile(camera.getDisplayProfileSetting())

            if self.lstCamera is not None:
                current_camera = self.getCurrentCamera()
                if current_camera:
                    if current_camera.xaddrs() == onvif_data.xaddrs():
                        self.signals.fill.emit(onvif_data)
                        self.setEnabled(True)
                        self.setTabsEnabled(True)

            camera.filled = True
            if self.allCamerasFilled():
                self.signals.guiSync.emit()
            if self.mw.settingsPanel.proxy.proxyType == ProxyType.SERVER and \
                    self.mw.settingsPanel.discover.radCached.isChecked() and \
                    self.allCamerasFilled():
                
                self.mw.settingsPanel.proxy.setMediaMTXProxies()

            # auto start after fill, recording needs onvif frame rate
            if self.mw.settingsPanel.discover.chkAutoStart.isChecked():
                if not camera.isRunning():
                    while not self.mw.isVisible():
                        sleep(0.1)
                    self.lstCamera.itemClicked.emit(camera)
                    self.lstCamera.itemDoubleClicked.emit(camera)
                    sleep(0.1)

        if len(onvif_data.last_error()):
            logger.debug(f'Error from {onvif_data.alias} : {onvif_data.last_error()}')

    def saveCameraList(self):
        serial_numbers = ""
        cameras = [self.mw.cameraPanel.lstCamera.item(x) for x in range(self.mw.cameraPanel.lstCamera.count())]
        for camera in cameras:
            serial_numbers += camera.serial_number() + "\n"
        self.mw.settings.setValue(self.mw.settingsPanel.discover.cameraListKey, serial_numbers)

    def onCurrentItemChanged(self, current, previous):
        if current:
            if self.mw.pm.getPlayer(current.uri()):
                self.mw.glWidget.focused_uri = current.uri()
            else:
                self.mw.glWidget.focused_uri = None
            self.signals.fill.emit(current.onvif_data)
            self.syncGUI()

    def onItemClicked(self, camera):
        if self.mw.videoConfigure:
            self.mw.videoConfigure.setCamera(camera)

    def onItemDoubleClicked(self, camera):
        if not camera: return
        profiles = self.mw.pm.getStreamPairProfiles(camera.uri())
        players = self.mw.pm.getStreamPairPlayers(camera.uri())
        timers = self.mw.pm.getStreamPairTimers(camera.uri())

        activeTimer = False
        for timer in timers:
            if timer.isActive():
                activeTimer = True

        if activeTimer:
            for timer in timers:
                self.mw.signals.stopReconnect.emit(timer.uri)
            for player in players:
                player.requestShutdown()
            camera.setIconIdle()
        else:
            if len(players):
                if player := self.mw.cameraPanel.getCurrentPlayer():
                    if profile := camera.getRecordProfile():
                        self.mw.openFocusWindow()
                        count = 0
                        while not self.mw.focus_window.cameraPanel.getCamera(profile.uri()):
                            time.sleep(0.01)
                            count += 1
                            if count > 200:
                                logger.error("timeout error opening focus window")
                                break
                        if camera := self.mw.focus_window.cameraPanel.getCamera(profile.uri()):
                            self.mw.focus_window.cameraPanel.onItemDoubleClicked(camera)
            else:
                for i, profile in enumerate(profiles):
                    if i == 0:
                        profile.setHidden(False)
                        self.mw.playMedia(profile.uri())
                    else:
                        if camera.displayProfileIndex() != camera.recordProfileIndex():
                            profile.setHidden(True)
                            self.mw.playMedia(profile.uri())

        self.syncGUI()

    def setTabsEnabled(self, enabled):
        self.tabVideo.setEnabled(enabled)
        self.tabImage.setEnabled(enabled)
        self.tabNetwork.setEnabled(enabled)
        self.ptzTab.setEnabled(enabled)
        self.tabSystem.setEnabled(enabled)

    def btnApplyClicked(self):
        if camera := self.getCurrentCamera():
            self.btnApply.setEnabled(False)
            self.tabVideo.update(camera.onvif_data)
            self.tabImage.update(camera.onvif_data)
            self.tabNetwork.update(camera.onvif_data)

    def onEdit(self):
        if camera := self.getCurrentCamera():
            if self.tabVideo.edited(camera.onvif_data) or \
                    self.tabImage.edited(camera.onvif_data) or \
                    self.tabNetwork.edited(camera.onvif_data):
                self.btnApply.setEnabled(True)
            else:
                self.btnApply.setEnabled(False)

    def sldVolumeChanged(self, value):
        if player := self.getCurrentPlayer():
            player.setVolume(value)
        if camera := self.getCurrentCamera():
            camera.setVolume(value)

    def btnMuteClicked(self):
        player = self.getCurrentPlayer()
        if player:
            player.setMute(not player.isMuted())
            camera = self.getCurrentCamera()
            if camera:
                camera.setMute(player.isMuted())
        else:
            camera = self.getCurrentCamera()
            if camera:
                camera.setMute(not camera.mute)
        self.syncGUI()

    def btnRecordClicked(self):
        player = self.getCurrentPlayer()
        camera = self.getCurrentCamera()
        if camera.displayProfileIndex() != camera.recordProfileIndex():
            recordProfile = camera.getRecordProfile()
            if recordProfile:
                record_uri = recordProfile.uri()
                player = self.mw.pm.getPlayer(record_uri)

        if player:
            if player.isRecording():
                player.output_file_start_time = None
                player.toggleRecording("")
                self.mw.settingsPanel.storage.signals.updateDiskUsage.emit()
                if camera:
                    camera.manual_recording = False
            else:
                #d = self.mw.settingsPanel.storage.dirArchive.txtDirectory.text()
                #if self.mw.settingsPanel.storage.chkManageDiskUsage.isChecked():
                #    self.mw.diskManager.manageDirectory(d)
                #else:
                #    self.mw.settingsPanel.storage.signals.updateDiskUsage.emit()
                if filename := player.getOutputFilename():
                    player.toggleRecording(filename)
                    if camera:
                        camera.manual_recording = True

        self.syncGUI()

    def btnHistoryClicked(self):
        try:
            reader_settings = QSettings("onvif-gui", "Reader")
            reader_settings.setValue("filePanel/hideCameraPanel", 1)
            main_file = Path(__file__).parent.parent.parent / "main.py"
            #subprocess.Popen(["python", str(main_file), "--profile", "reader"], env=os.environ.copy(), start_new_session=True, shell=True)
            if platform.system() == "Windows":
                subprocess.Popen([sys.executable, str(main_file), "--profile", "Reader"], env=os.environ.copy(), start_new_session=True, shell=True)
            else:
                subprocess.Popen([sys.executable, str(main_file), "--profile", "Reader"], env=os.environ.copy(), start_new_session=True)
        except Exception as ex:
            logger.error(f'Error starting file browser: {ex}')
            self.mw.signals.error.emit(f'Error starting file browser: {ex}')

    def btnStopClicked(self):
        if camera := self.getCurrentCamera():
            #profiles = self.mw.pm.getStreamPairProfiles(camera.uri())
            players = self.mw.pm.getStreamPairPlayers(camera.uri())
            timers = self.mw.pm.getStreamPairTimers(camera.uri())

            activeTimer = False
            for timer in timers:
                if timer.isActive():
                    activeTimer = True

            if activeTimer:
                for timer in timers:
                    self.mw.signals.stopReconnect.emit(timer.uri)
                for player in players:
                    player.requestShutdown()
                camera.setIconIdle()
                self.syncGUI()
                return

            if len(players):
                for player in players:
                    player.requestShutdown()
                camera.setIconIdle()
                self.syncGUI()
                return
            else:
                self.onItemDoubleClicked(camera)
                self.syncGUI()

    def btnHelpClicked(self):
        result = webbrowser.get().open("https://github.com/sr99622/libonvif#readme-ov-file")
        if not result:
            webbrowser.get().open("https://github.com/sr99622/libonvif")

    def btnSnapshotClicked(self):
        if player := self.getCurrentPlayer():
            root = self.mw.settingsPanel.storage.dirPictures.txtDirectory.text() + "/" + self.getCamera(player.uri).text()
            Path(root).mkdir(parents=True, exist_ok=True)
            filename = '{0:%Y%m%d%H%M%S.jpg}'.format(datetime.now())
            filename = str(root + "/" + filename)

            if self.mw.settingsPanel.general.chkSnapshotDlg.isChecked():
                if platform.system() == "Linux":
                    filename = QFileDialog.getSaveFileName(self, "Save File As", filename, options=QFileDialog.Option.DontUseNativeDialog)[0]
                else:
                    filename = QFileDialog.getSaveFileName(self, "Save File As", filename)[0]

            if len(Path(filename).stem):
                answer = QMessageBox.StandardButton.Yes
                if not filename.endswith(".jpg"):
                    filename += ".jpg"
                    if Path(filename).is_file():
                        answer = QMessageBox.question(self.mw, "File Exists", "You are about to overwrite an existing file, are you sure you wnat to do this?")
                if answer == QMessageBox.StandardButton.Yes:
                    if camera := self.getCamera(player.uri):
                        profile = camera.getRecordProfile()
                        if not profile:
                            profile = camera.getProfile(player.uri)

                        thread = threading.Thread(target=self.snapshot, args=(profile, filename, camera, player))
                        thread.start()

    def btnStopAllClicked(self):
        if len(self.mw.pm.players):
            self.mw.closeAllStreams()
        else:
            self.mw.startAllCameras()
        self.syncGUI()

    def btnFullScreenClicked(self):
        if self.mw.isFullScreen():
            self.mw.showNormal()
            self.btnFullScreen.setStyleSheet(self.getButtonStyle("full_screen"))
        else:
            self.mw.showFullScreen()
            self.btnFullScreen.setStyleSheet(self.getButtonStyle("normal"))
       
    def onMediaStarted(self, uri):
        if self.mw.tabVisible:
            if self.mw.glWidget.focused_uri is None:
                if camera := self.getCurrentCamera():
                    self.mw.glWidget.focused_uri = camera.uri()

        #self.tabVideo.btnSnapshot.setEnabled(True)
        
        self.syncGUI()

    def onMediaStopped(self, uri):
        if camera := self.getCamera(uri):
            camera.setIconIdle()
            if profile := camera.getProfile(camera.uri()):
                if profile.getAnalyzeVideo():
                    if self.mw.videoWorker:
                        self.mw.videoWorker(None, None)
                if profile.getAnalyzeAudio():
                    if self.mw.audioWorker:
                        self.mw.audioWorker(None, None)
        #self.tabVideo.btnSnapshot.setEnabled(False)
        self.syncGUI()

    def syncGUI(self):
        while (self.sync_lock):
            sleep(0.001)
        self.sync_lock = True
        
        if camera := self.getCurrentCamera():
            self.btnStop.setEnabled(True)
            if player := self.mw.pm.getPlayer(camera.uri()):
                self.btnStop.setStyleSheet(self.getButtonStyle("stop"))
                self.btnSnapshot.setEnabled(True)
                #self.tabVideo.btnSnapshot.setEnabled(True)

                if ps := player.systemTabSettings():
                    self.btnRecord.setEnabled(not (ps.record_enable and ps.record_always))

                if player.hasAudio() and not player.disable_audio:
                    self.btnMute.setEnabled(True)
                    self.sldVolume.setEnabled(True)
                    self.sldVolume.setValue(camera.volume)
                    if camera.mute:
                        self.btnMute.setStyleSheet(self.getButtonStyle("mute"))
                    else:
                        self.btnMute.setStyleSheet(self.getButtonStyle("audio"))
                else:
                    self.btnMute.setEnabled(False)
                    self.sldVolume.setEnabled(False)

                self.btnRecord.setEnabled(True)
                if camera.isRecording():
                    self.btnRecord.setStyleSheet(self.getButtonStyle("recording"))
                    record_always = player.systemTabSettings().record_always if player.systemTabSettings() else False
                    record_alarm = player.systemTabSettings().record_alarm if player.systemTabSettings() else False
                    record_enable = player.systemTabSettings().record_enable if player.systemTabSettings() else False
                    if record_enable and ((camera.isAlarming() and record_alarm) or record_always):
                        self.btnRecord.setEnabled(False)
                else:
                    self.btnRecord.setStyleSheet(self.getButtonStyle("record"))
            else:
                reconnecting = False
                #self.tabVideo.btnSnapshot.setEnabled(False)
                timers = self.mw.pm.getStreamPairTimers(camera.uri())
                for timer in timers:
                    if timer.isActive():
                        reconnecting = True
                
                if reconnecting:
                    self.btnStop.setStyleSheet(self.getButtonStyle("stop"))
                else:
                    self.btnStop.setStyleSheet(self.getButtonStyle("play"))
                    self.setTabsEnabled(True)

                if profile := camera.getProfile(camera.uri()):
                    if camera.mute:
                        self.btnMute.setStyleSheet(self.getButtonStyle("mute"))
                    else:
                        self.btnMute.setStyleSheet(self.getButtonStyle("audio"))

                    if profile.audio_bitrate() and not profile.getDisableAudio():
                        self.btnMute.setEnabled(True)
                        self.sldVolume.setEnabled(True)
                    else:
                        self.btnMute.setEnabled(False)
                        self.sldVolume.setEnabled(False)

                self.btnRecord.setStyleSheet(self.getButtonStyle("record"))
                self.btnRecord.setEnabled(False)
                self.btnSnapshot.setEnabled(False)
        else:
            self.sldVolume.setEnabled(False)
            self.btnMute.setStyleSheet(self.getButtonStyle("audio"))
            self.btnMute.setEnabled(False)
            self.btnRecord.setStyleSheet(self.getButtonStyle("record"))
            self.btnRecord.setEnabled(False)
            self.btnStop.setStyleSheet(self.getButtonStyle("play"))
            self.btnStop.setEnabled(False)
            self.btnSnapshot.setEnabled(False)

        if self.lstCamera.count():
            self.btnStopAll.setEnabled(True) 
        else:
            self.btnStopAll.setEnabled(False)
        if len(self.mw.pm.players):
            self.btnStopAll.setStyleSheet(self.getButtonStyle("stop_all"))
        else:
            self.btnStopAll.setStyleSheet(self.getButtonStyle("play_all"))

        self.sync_lock = False


    def getButtonStyle(self, name):
        strStyle = "QPushButton { image : url(image:%1.png); } \
                    QPushButton:hover { image : url(image:%1_hi.png); } \
                    QPushButton:pressed { image : url(image:%1_lo.png); } \
                    QPushButton:disabled { image : url(image:%1_lo.png); }"
        strStyle = strStyle.replace("%1", name)
        return strStyle

    def getCurrentPlayer(self):
        result = None
        if self.lstCamera:
            camera = self.getCurrentCamera()
            if camera:
                result = self.mw.pm.getPlayer(camera.uri())
        return result

    def getCamera(self, uri):
        if not uri: return None
        if not self.lstCamera: return None

        cameras = [self.lstCamera.item(x) for x in range(self.lstCamera.count())]
        for camera in cameras:
            for profile in camera.profiles:
                if profile.uri() == uri:
                    return camera

        return None
    
    def getCameraByName(self, name):
        if not name: return None
        if not self.lstCamera: return None

        cameras = [self.lstCamera.item(x) for x in range(self.lstCamera.count())]
        for camera in cameras:
            if camera.name() == name:
                return camera
                
        return None

    
    def getCameraBySerialNumber(self, serial_number):
        if not serial_number: return None
        if not self.lstCamera: return None

        cameras = [self.lstCamera.item(x) for x in range(self.lstCamera.count())]
        for camera in cameras:
            if camera.serial_number() == serial_number:
                return camera
                
        return None
    
    def getCameraByXAddrs(self, xaddrs):
        if not xaddrs: return None
        if not self.lstCamera: return None

        cameras = [self.lstCamera.item(x) for x in range(self.lstCamera.count())]
        for camera in cameras:
            if camera.xaddrs() == xaddrs:
                return camera
            
        return None
    
    def getProfile(self, uri):
        result = None
        if camera := self.getCamera(uri):
            result = camera.getProfile(uri)
        return result
    
    def getCurrentProfile(self):
        result = None
        if camera := self.getCurrentCamera():
            result = camera.getProfile(camera.uri())
        return result
    
    def getCurrentCamera(self):
        result = None
        if self.lstCamera:
            result = self.lstCamera.currentItem()
        return result
    
    def setCurrentCamera(self, uri):
        if camera := self.getCamera(uri):
            self.lstCamera.setCurrentItem(camera)
            self.signals.fill.emit(camera.onvif_data)
            self.syncGUI()

            if self.mw.videoConfigure:
                    self.mw.videoConfigure.setCamera(camera)

            if self.mw.audioConfigure:
                self.mw.audioConfigure.setCamera(camera)

    def enableAutoTimeSync(self, state):
        AUTO_TIME_SYNC_INTERVAL = 3600000
        if int(state) == 0:
            logger.debug("Auto time sync has been turned off")
            if self.autoTimeSyncer:
                self.autoTimeSyncer.stop()
        else:
            logger.debug("Auto time sync has been turned on")
            if not self.autoTimeSyncer:
                self.autoTimeSyncer = QTimer()
                self.autoTimeSyncer.setInterval(AUTO_TIME_SYNC_INTERVAL)
                self.autoTimeSyncer.timeout.connect(self.timeSync)
                self.autoTimeSyncer.start()
    
    def timeSync(self):
        if not self.lstCamera: return
        logger.debug("Synchronizing camera times")
        cameras = [self.lstCamera.item(x) for x in range(self.lstCamera.count())]
        for camera in cameras:
            camera.onvif_data.startUpdateTime()

    def activeSessions(self):
        for session in self.sessions:
            if session.active:
                return True
            
        return False

    def closeEvent(self):
        self.closing = True

        if self.lstCamera:
            cameras = [self.lstCamera.item(x) for x in range(self.lstCamera.count())]
            for camera in cameras:
                camera.onvif_data.filled = None

        if self.activeSessions():
            for session in self.sessions:
                session.abort = True

            sleep(1)
            
            waiting = True
            count = 0
            while waiting:
                count += 1
                tmp = False
                for session in self.sessions:
                    if session.active:
                        tmp = True
                        break
                    sleep(0.1)
                waiting = tmp
                if count > 10:
                    break

    def allCamerasFilled(self):
        if not self.lstCamera: return True

        cameras = [self.lstCamera.item(x) for x in range(self.lstCamera.count())]
        for camera in cameras:
            if not camera.filled:
                return False

        return True