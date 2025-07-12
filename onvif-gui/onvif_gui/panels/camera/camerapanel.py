#/********************************************************************
# libonvif/onvif-gui/onvif_gui/panels/camera/camerapanel.py 
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
    QListWidget, QTabWidget, QMessageBox, QMenu
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QTimer
from . import NetworkTab, ImageTab, VideoTab, PTZTab, SystemTab, LoginDialog, \
    Session, Camera
from onvif_gui.enums import MediaSource
from loguru import logger
import libonvif as onvif
from onvif_gui.enums import ProxyType

class CameraList(QListWidget):
    def __init__(self, mw):
        super().__init__()
        self.signals = CameraPanelSignals()
        self.setSortingEnabled(True)
        self.mw = mw

    def focusInEvent(self, event):
        if self.currentRow() == -1:
            self.setCurrentRow(0)
        super().focusInEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Return:
            if camera := self.currentItem():
                if not camera.editing():
                    self.itemDoubleClicked.emit(camera)

        if event.key() == Qt.Key.Key_Delete:
            self.remove()

        if event.key() == Qt.Key.Key_F2:
            self.rename()

        if event.key() == Qt.Key.Key_F1:
            self.info()

        return super().keyPressEvent(event)
    
    def remove(self):
        if camera := self.currentItem():
            if self.mw.pm.getPlayer(camera.uri()):
                ret = QMessageBox.warning(self, camera.name(),
                                            "Camera is currently playing. Please stop before deleting.",
                                            QMessageBox.StandardButton.Ok)

                return
            else:
                ret = QMessageBox.warning(self, camera.name(),
                                            "Removing this camera from the list.\n"
                                            "Are you sure you want to continue?",
                                            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)

                if ret != QMessageBox.StandardButton.Ok:
                    return

            row = self.currentRow()
            if row > -1:
                if camera.filled:
                    camera = self.takeItem(row)
                    for profile in camera.profiles:
                        self.mw.proxies.pop(profile.stream_uri(), None)

                    if self.mw.settingsPanel.proxy.proxyType == ProxyType.SERVER:
                        self.mw.settingsPanel.proxy.setMediaMTXProxies()

                else:
                    ret = QMessageBox.warning(self, camera.name(),
                                                "The program is currently communicating with the camera. Please wait before deleting.",
                                                QMessageBox.StandardButton.Ok)

        if not self.count():
            data = onvif.Data()
            self.mw.cameraPanel.signals.fill.emit(data)

        self.mw.cameraPanel.saveCameraList()

    def info(self):
        msg = ""
        if camera := self.currentItem():
            players = self.mw.pm.getStreamPairPlayers(camera.uri())
            if not len(players):
                msg = "Start camera to get stream info"
            for i, player in enumerate(players):
                if i == 0:
                    msg += "<h1>Display Stream</h1>"
                    msg += player.getStreamInfo()
                    msg += "\n"
                if i == 1:
                    msg += "<h1>Record Stream</h1>"
                    msg += player.getStreamInfo()
                    msg += "\n"
        msgBox = QMessageBox(self)
        msgBox.setWindowTitle("Stream Info")
        msgBox.setText(msg)
        msgBox.setTextFormat(Qt.TextFormat.RichText)
        msgBox.exec()
    
    def rename(self):
        if camera := self.currentItem():
            camera.setFlags(camera.flags() | Qt.ItemFlag.ItemIsEditable)
            index = self.currentIndex()
            if index.isValid():
                self.edit(index)

    def password(self):
        if camera := self.currentItem():
            self.mw.settings.setValue(f'{camera.xaddrs()}/alternateUsername', camera.onvif_data.username())
            self.mw.settings.setValue(f'{camera.xaddrs()}/alternatePassword', camera.onvif_data.password())
            logger.debug(f'Alternate password set for camera {camera.name()}')

    def closeEditor(self, editor, hint):
        if camera := self.currentItem():
            camera.onvif_data.alias = editor.text()
            self.mw.settings.setValue(f'{camera.serial_number()}/Alias', editor.text())
            camera.setFlags(camera.flags() & ~Qt.ItemFlag.ItemIsEditable)
        return super().closeEditor(editor, hint)
    
    def startCamera(self):
        if camera := self.currentItem():
            self.mw.cameraPanel.onItemDoubleClicked(camera)

    def stopCamera(self):
        if camera := self.currentItem():
            self.mw.cameraPanel.onItemDoubleClicked(camera)

class CameraPanelSignals(QObject):
    fill = pyqtSignal(onvif.Data)
    login = pyqtSignal(onvif.Data)
    collapseSplitter = pyqtSignal()

class CameraPanel(QWidget):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw
        self.dlgLogin = LoginDialog(self)
        self.fillers = []
        self.sync_lock = False

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

        lytMain = QGridLayout(self)
        lytMain.addWidget(self.lstCamera,   0, 0, 1, 6)
        lytMain.addWidget(self.tabOnvif,    1, 0, 1, 6)
        lytMain.addWidget(self.btnStop,     2, 0, 1, 1)
        lytMain.addWidget(self.btnRecord,   2, 1, 1, 1)
        lytMain.addWidget(self.btnDiscover, 2, 2, 1, 1)
        lytMain.addWidget(self.btnApply,    2, 3, 1, 1)
        lytMain.addWidget(self.btnMute,     2, 4, 1, 1)
        lytMain.addWidget(self.sldVolume,   2, 5, 1, 1)
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
                self.mw.client.transmit("GET CAMERAS\r\n")
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
        if not onvif_data:
            return
        
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
        if not onvif_data:
            return
        
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
                tmp = self.mw.settings.value(self.mw.settingsPanel.discover.cameraListKey)
                if tmp:
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
        if not onvif_data:
            return
        
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

        if not onvif_data:
            return
        
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
        if not camera:
            return
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
                for player in players:
                    player.requestShutdown()
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
        camera = self.getCurrentCamera()
        if camera:
            self.btnApply.setEnabled(False)
            self.tabVideo.update(camera.onvif_data)
            self.tabImage.update(camera.onvif_data)
            self.tabNetwork.update(camera.onvif_data)

    def onEdit(self):
        camera = self.getCurrentCamera()
        if camera:
            if self.tabVideo.edited(camera.onvif_data) or \
                    self.tabImage.edited(camera.onvif_data) or \
                    self.tabNetwork.edited(camera.onvif_data):
                self.btnApply.setEnabled(True)
            else:
                self.btnApply.setEnabled(False)

    def sldVolumeChanged(self, value):
        player = self.getCurrentPlayer()
        if player:
            player.setVolume(value)
        camera = self.getCurrentCamera()
        if camera:
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
                player.pipe_output_start_time = None
                player.toggleRecording("")
                self.mw.diskManager.getDirectorySize(self.mw.settingsPanel.storage.dirArchive.txtDirectory.text())
                if camera:
                    camera.manual_recording = False
            else:
                d = self.mw.settingsPanel.storage.dirArchive.txtDirectory.text()
                if self.mw.settingsPanel.storage.chkManageDiskUsage.isChecked():
                    self.mw.diskManager.manageDirectory(d, player.uri)
                #else:
                #    self.mw.diskManager.getDirectorySize(d)
                if filename := player.getPipeOutFilename():
                    player.toggleRecording(filename)
                    if camera:
                        camera.manual_recording = True

        self.syncGUI()

    def btnStopClicked(self):
        camera = self.getCurrentCamera()
        if camera:
            self.onItemDoubleClicked(camera)
        self.syncGUI()
       
    def onMediaStarted(self, uri):
        if self.mw.tabVisible:
            if self.mw.glWidget.focused_uri is None:
                if camera := self.getCurrentCamera():
                    self.mw.glWidget.focused_uri = camera.uri()

        self.tabVideo.btnSnapshot.setEnabled(True)
        
        self.syncGUI()

    def onMediaStopped(self, uri):
        camera = self.getCamera(uri)
        if camera:
            camera.setIconIdle()
            profile = camera.getProfile(camera.uri())
            if profile:
                if profile.getAnalyzeVideo():
                    if self.mw.videoWorker:
                        self.mw.videoWorker(None, None)
                if profile.getAnalyzeAudio():
                    if self.mw.audioWorker:
                        self.mw.audioWorker(None, None)
        self.tabVideo.btnSnapshot.setEnabled(False)
        self.syncGUI()

    def syncGUI(self):

        while (self.sync_lock):
            sleep(0.001)
        self.sync_lock = True
        
        if camera := self.getCurrentCamera():
            self.btnStop.setEnabled(True)
            if player := self.mw.pm.getPlayer(camera.uri()):
                self.btnStop.setStyleSheet(self.getButtonStyle("stop"))
                if player.running:
                    self.tabVideo.btnSnapshot.setEnabled(True)

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
                self.tabVideo.btnSnapshot.setEnabled(False)
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
        else:
            self.sldVolume.setEnabled(False)
            self.btnMute.setStyleSheet(self.getButtonStyle("audio"))
            self.btnMute.setEnabled(False)
            self.btnRecord.setStyleSheet(self.getButtonStyle("record"))
            self.btnRecord.setEnabled(False)
            self.btnStop.setStyleSheet(self.getButtonStyle("play"))
            self.btnStop.setEnabled(False)

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
        result = None
        if self.lstCamera:
            cameras = [self.lstCamera.item(x) for x in range(self.lstCamera.count())]
            for camera in cameras:
                found = False
                for profile in camera.profiles:
                    #print("profile uri", profile.uri())
                    if profile.uri() == uri:
                        result = camera
                        found = True
                        break
                if found:
                    break
                else:
                    if camera.uri() == uri:
                        result = camera
                        break

        return result
    
    def getCameraBySerialNumber(self, serial_number):
        result = None
        if self.lstCamera:
            cameras = [self.lstCamera.item(x) for x in range(self.lstCamera.count())]
            for camera in cameras:
                if camera.serial_number() == serial_number:
                    result = camera
                    break
        return result
    
    def getCameraByXAddrs(self, xaddrs):
        result = None
        if self.lstCamera:
            cameras = [self.lstCamera.item(x) for x in range(self.lstCamera.count())]
            for camera in cameras:
                if camera.xaddrs() == xaddrs:
                    result = camera
                    break
        return result
    
    def getProfile(self, uri):
        result = None
        camera = self.getCamera(uri)
        if camera:
            result = camera.getProfile(uri)
        return result
    
    def getCurrentProfile(self):
        result = None
        camera = self.getCurrentCamera()
        if camera:
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
                if self.mw.videoConfigure.source != MediaSource.CAMERA:
                    self.mw.videoConfigure.setCamera(camera)

            if self.mw.audioConfigure:
                if self.mw.audioConfigure.source != MediaSource.CAMERA:
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
        logger.debug("Synchronizing camera times")
        if self.lstCamera:
            cameras = [self.lstCamera.item(x) for x in range(self.lstCamera.count())]
            for camera in cameras:
                camera.onvif_data.startUpdateTime()

    def activeSessions(self):
        result = False
        for session in self.sessions:
            if session.active:
                result = True
                break
        return result

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
        result = True
        if self.lstCamera:
            cameras = [self.lstCamera.item(x) for x in range(self.lstCamera.count())]
            for camera in cameras:
                if not camera.filled:
                    result = False
                    break
        return result