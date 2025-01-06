#/********************************************************************
# libonvif/onvif-gui/gui/panels/options/proxy.py 
#
# Copyright (c) 2024  Stephen Rhodes
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

from PyQt6.QtWidgets import QMessageBox, QLineEdit, QPushButton, \
    QGridLayout, QWidget, QLabel, QMessageBox, QRadioButton, \
    QGroupBox, QCheckBox, QComboBox, QDialog, QDialogButtonBox
from PyQt6.QtCore import Qt
from gui.components import DirectorySelector
from gui.enums import ProxyType
from loguru import logger
import libonvif as onvif
import ipaddress
import os
import sys

class ServerManageDialog(QDialog):
    def __init__(self, mw):
        super().__init__(mw)
        self.mw = mw

        self.setWindowTitle("Server Management")
        self.setMinimumWidth(480)

        self.btnIterateClients = QPushButton("Clients")
        self.btnIterateClients.clicked.connect(self.iterateClients)
        self.btnIterateMedia = QPushButton("Media")
        self.btnIterateMedia.clicked.connect(self.iterateMedia)
        self.btnIterateConnections = QPushButton("Connections")
        self.btnIterateConnections.clicked.connect(self.iterateConnections)

        buttonBox = QDialogButtonBox( \
            QDialogButtonBox.StandardButton.Ok | \
            QDialogButtonBox.StandardButton.Cancel)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

        lytMain = QGridLayout(self)
        lytMain.addWidget(self.btnIterateClients,      0, 0, 1, 1)
        lytMain.addWidget(self.btnIterateMedia,        0, 1, 1, 1)
        lytMain.addWidget(self.btnIterateConnections,  0, 2, 1, 1)
        lytMain.addWidget(buttonBox,                   0, 3, 1, 1)

    def iterateClients(self):
        clients = self.mw.proxy.iterateClients()
        for client in clients:
            print(client)

    def iterateMedia(self):
        media = self.mw.proxy.iterateMedia()
        media.sort()
        for medium in media:
            print(medium)

    def iterateConnections(self):
        connections = self.mw.proxy.iterateConnections()
        for connection in connections:
            print(connection)

    def accept(self):
        print("ACCEPT")
        #self.close()
        super().accept()

    def reject(self):
        print("REJECT")
        #self.close()
        super().reject()

    def exec(self):
        super().exec()

class ProxyOptions(QWidget):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw

        self.proxyTypeKey = "settings/proxyType"
        self.proxyRemoteKey = "settings/proxyRemote"
        self.autoDownloadKey = "settings/autoDownloadMTX"
        self.listenKey = "settings/alarmsFromServer"
        self.broadcastInterfaceKey = "settings/proxyBroadcastInterface"
        self.broadcastEnableKey = "setings/enableBroadcasting"
        self.broadcastAllInterfacesKey = "settings/broadcastAllInterfaces"

        self.if_addrs = []
        session = onvif.Session()
        session.getActiveInterfaces()
        i = 0
        while len(session.active_interface(i)) > 0 and i < 16:
            self.if_addrs.append(session.active_interface(i))
            i += 1

        self.grpProxyType = QGroupBox("Select Proxy Type")
        self.radStandAlone = QRadioButton("Stand Alone", self.grpProxyType)
        self.radServer = QRadioButton("Server", self.grpProxyType)
        self.radClient = QRadioButton("Client", self.grpProxyType)

        self.lblServer = QLabel("\n\n")
        self.lblConnect = QLabel("Connect url for clients")
        self.lblConnect.setEnabled(False)


        self.dlgServerManage = ServerManageDialog(self.mw)
        self.btnServerManage = QPushButton("Manage")
        self.btnServerManage.clicked.connect(self.btnServerManageClicked)
        self.btnServerManage.setEnabled(False)

        self.chkAutoDownload = QCheckBox("Auto Download Media MTX")
        self.chkAutoDownload.setEnabled(False)
        self.chkAutoDownload.setChecked(int(mw.settings.value(self.autoDownloadKey, 1)))
        self.chkAutoDownload.stateChanged.connect(self.chkAutoDownloadChecked)

        self.txtDirectoryMTX = DirectorySelector(mw, "MTXDir", "Dir", os.path.dirname(sys.executable))
        self.txtDirectoryMTX.setEnabled(not self.chkAutoDownload.isChecked())

        #self.btnTest = QPushButton("TEST")
        #self.btnTest.clicked.connect(self.btnTestClicked)

        self.grpMediaMTX = QGroupBox("Media MTX Settings")
        lytMediaMTX = QGridLayout(self.grpMediaMTX)
        lytMediaMTX.addWidget(self.chkAutoDownload,       0, 0, 1, 1)
        lytMediaMTX.addWidget(self.txtDirectoryMTX,       1, 0, 1, 1)
        #lytMediaMTX.addWidget(self.btnTest,               2, 0, 1, 1, Qt.AlignmentFlag.AlignCenter)

        self.cmbInterfaces = QComboBox()
        self.lblInterfaces = QLabel("Network")

        # Windows can only broadcast on the network interface with highest priority
        if sys.platform == "win32":
            self.cmbInterfaces.addItem(session.primary_network_interface())
        else:
            for if_addr in self.if_addrs:
                self.cmbInterfaces.addItem(if_addr)
        
        self.cmbInterfaces.setCurrentText(self.mw.settings.value(self.broadcastInterfaceKey, self.if_addrs[0]))
        self.cmbInterfaces.currentTextChanged.connect(self.cmbInterfacesChanged)
        
        #self.chkBroadcastAllInterfaces = QCheckBox("Broadcast on all interfaces")
        #self.chkBroadcastAllInterfaces.setChecked(int(self.mw.settings.value(self.broadcastAllInterfacesKey, 0)))
        #self.chkBroadcastAllInterfaces.clicked.connect(self.chkBroadcastAllInterfacesChecked)
        #self.cmbInterfaces.setEnabled(not self.chkBroadcastAllInterfaces.isChecked())
        #self.lblInterfaces.setEnabled(not self.chkBroadcastAllInterfaces.isChecked())

        self.grpAlarmBroadcast = QGroupBox("Alarm Broadcasting")
        self.grpAlarmBroadcast.setCheckable(True)
        self.grpAlarmBroadcast.setChecked(int(self.mw.settings.value(self.broadcastEnableKey, 1)))
        lytAlarmBroadcast = QGridLayout(self.grpAlarmBroadcast)
        #lytAlarmBroadcast.addWidget(self.chkBroadcastAllInterfaces,  0, 0, 1, 2)
        lytAlarmBroadcast.addWidget(self.lblInterfaces,              1, 0, 1, 1)
        lytAlarmBroadcast.addWidget(self.cmbInterfaces,              1, 1, 1, 1)

        self.proxyRemote = self.mw.settings.value(self.proxyRemoteKey)

        self.txtRemote = QLineEdit()
        self.txtRemote.setText(self.proxyRemote)
        self.txtRemote.textEdited.connect(self.txtRemoteEdited)
        self.txtRemote.setEnabled(False)
        self.lblRemote = QLabel("Connect url from server")
        self.lblRemote.setEnabled(False)

        self.btnUpdate = QPushButton("Update")
        self.btnUpdate.clicked.connect(self.btnUpdateClicked)
        self.btnUpdate.setEnabled(False)

        self.chkListen = QCheckBox("Get alarm events from server")
        self.chkListen.setChecked(int(mw.settings.value(self.listenKey, 1)))
        self.chkListen.stateChanged.connect(self.chkListenChecked)
        self.chkListen.setEnabled(False)

        self.proxyType = (self.mw.settings.value(self.proxyTypeKey, ProxyType.STAND_ALONE))
        
        match self.proxyType:
            case ProxyType.STAND_ALONE:
                self.radStandAlone.setChecked(True)
                self.radStandAloneToggled(True)
            case ProxyType.SERVER:
                self.radServer.setChecked(True)
                self.radServerToggled(True)
            case ProxyType.CLIENT:
                self.radClient.setChecked(True)
                self.radClientToggled(True)

        self.radStandAlone.toggled.connect(self.radStandAloneToggled)
        self.radServer.toggled.connect(self.radServerToggled)
        self.radClient.toggled.connect(self.radClientToggled)

        if self.mw.settings_profile != "gui":
            self.radServer.setEnabled(False)

        lytProxyType = QGridLayout(self.grpProxyType)
        lytProxyType.addWidget(self.radStandAlone,      0, 0, 1, 1)
        lytProxyType.addWidget(QLabel(),                1, 0, 1, 2)
        lytProxyType.addWidget(self.radClient,          2, 0, 1, 1)
        lytProxyType.addWidget(self.lblRemote,          2, 1, 1, 1, Qt.AlignmentFlag.AlignCenter)
        lytProxyType.addWidget(self.btnUpdate,          3, 0, 1, 1, Qt.AlignmentFlag.AlignCenter)
        lytProxyType.addWidget(self.txtRemote,          3, 1, 1, 1)
        lytProxyType.addWidget(self.chkListen,          4, 0, 1, 2)
        lytProxyType.addWidget(QLabel(),                6, 0, 1, 1)
        lytProxyType.addWidget(self.radServer,          7, 0, 1, 1)
        lytProxyType.addWidget(self.lblConnect,         7, 1, 1, 1)
        lytProxyType.addWidget(self.lblServer,          8, 0, 1, 2, Qt.AlignmentFlag.AlignCenter)
        lytProxyType.addWidget(self.grpMediaMTX,        9, 0, 1, 2)
        lytProxyType.addWidget(self.grpAlarmBroadcast,  10, 0, 1, 2)
        lytProxyType.addWidget(QLabel(),                11, 0, 1, 1)

        lytMain = QGridLayout(self)
        lytMain.addWidget(self.grpProxyType,  0, 0, 1, 2)

        lytMain.addWidget(QLabel(),           2, 0, 1, 2)
        lytMain.setRowStretch(2, 10)

    def btnServerManageClicked(self):
        print("manage server")
        self.dlgServerManage.exec()

    def setProxyType(self, type):
        self.proxyType = type
        self.mw.settings.setValue(self.proxyTypeKey, type)
        self.mw.stopProxyServer()
        self.mw.stopOnvifServer()
        if len(self.mw.pm.players):
            QMessageBox.information(self.mw, "Closing Streams", "All current streams will be closed")
            self.mw.closeAllStreams()

        if type == ProxyType.SERVER and self.mw.settings_profile == "gui":
            self.setServersLabel()

        self.grpAlarmBroadcast.setEnabled(type == ProxyType.SERVER)
        self.grpMediaMTX.setEnabled(type == ProxyType.SERVER)

        if not hasattr(self.mw, "cameraPanel"):
            return

        if type == ProxyType.SERVER and self.mw.settings_profile == "gui":
            self.mw.manageBroadcaster(self.getInterfaces())
            self.mw.startProxyServer(self.chkAutoDownload.isChecked())
            self.mw.startOnvifServer("")

        if lstCamera := self.mw.cameraPanel.lstCamera:
            cameras = [lstCamera.item(x) for x in range(lstCamera.count())]
            for camera in cameras:
                self.mw.addCameraProxy(camera)
                if type == ProxyType.STAND_ALONE:
                    camera.onvif_data.nullifyGetProxyURI()
                else:
                    camera.onvif_data.getProxyURI = self.mw.getProxyURI
                for profile in camera.profiles:
                    if type == ProxyType.STAND_ALONE:
                        profile.nullifyGetProxyURI()
                    else:
                        profile.getProxyURI = self.mw.getProxyURI

        if type == ProxyType.CLIENT:
            self.manageAnalyzers(self.chkListen.isChecked())
            if len(self.mw.alarm_ordinals) != self.mw.cameraPanel.lstCamera.count():
                if not self.mw.cameraPanel.allCamerasFilled():
                    QMessageBox.warning(self.mw, "Application Restart Required", "It is necessary to re-start Onvif GUI in order to enable this change", QMessageBox.StandardButton.Close)
                    self.mw.close()
                    return
                self.mw.closeAllStreams()
                self.mw.cameraPanel.lstCamera.clear()
                self.mw.cameraPanel.btnDiscoverClicked()
        else:
            self.manageAnalyzers(False)

    def radClientToggled(self, checked):
        self.lblRemote.setEnabled(checked)
        self.txtRemote.setEnabled(checked)
        self.chkListen.setEnabled(checked)
        if checked:
           self.setProxyType(ProxyType.CLIENT)

    def radStandAloneToggled(self, checked):
        if checked:
            self.setProxyType(ProxyType.STAND_ALONE)

    def radServerToggled(self, checked):
        self.lblConnect.setEnabled(checked)
        self.chkAutoDownload.setEnabled(checked)
        if checked:
            self.setProxyType(ProxyType.SERVER)
            self.btnServerManage.setEnabled(True)
        else:
            self.lblServer.setText("\n\n")
            self.btnServerManage.setEnabled(False)

    def setServersLabel(self):
        lbl = ""
        for if_addr in self.if_addrs:
            lbl += f'rtsp://{if_addr}:8554/\n'
        self.lblServer.setText(lbl)

    def txtRemoteEdited(self, arg):
        self.mw.settings.setValue(self.proxyRemoteKey, arg)
        comps = arg[:len(arg)-1][7:].split(":")
        ip_addr = comps[0]

        try:
            ip = ipaddress.ip_address(ip_addr)
            if self.txtRemote.text() != self.proxyRemote:
                self.btnUpdate.setEnabled(True)
        except ValueError:
            self.btnUpdate.setEnabled(False)

    def btnUpdateClicked(self):
        try:
            if len(self.mw.pm.players):
                QMessageBox.information(self.mw, "Closing Streams", "All current streams will be closed")
                self.mw.closeAllStreams()

            arg = self.txtRemote.text()
            comps = arg[:len(arg)-1][7:].split(":")
            ip_addr = comps[0]
            port = 8550
            remote = f'{ip_addr}:{port}'
            if not self.mw.client:
                self.mw.initializeClient(remote)
            else:
                self.mw.client.setEndpoint(remote)
            self.proxyRemote = self.txtRemote.text()

            if lstCamera := self.mw.cameraPanel.lstCamera:
                cameras = [lstCamera.item(x) for x in range(lstCamera.count())]
                for camera in cameras:
                    self.mw.addCameraProxy(camera)

            self.mw.startListener(self.getInterfaces())

            self.btnUpdate.setEnabled(False)
        except Exception as ex:
            QMessageBox.critical(self.mw, "Proxy Client Error", str(ex))

    def chkAutoDownloadChecked(self, state):
        self.mw.settings.setValue(self.autoDownloadKey, state)
        self.txtDirectoryMTX.setEnabled(not self.chkAutoDownload.isChecked())

    def chkListenChecked(self, state):
        self.mw.settings.setValue(self.listenKey, state)
        self.manageAnalyzers(state)

    def grpAlarmBroadcastChecked(self, state):
        self.mw.settings.setValue(self.broadcastEnableKey, state)
        self.mw.manageBroadcaster(self.getInterfaces())

    def chkBroadcastAllInterfacesChecked(self, state):
        self.mw.settings.setValue(self.broadcastAllInterfacesKey, state)
        self.mw.manageBroadcaster(self.getInterfaces())

    def generateAlarmsLocally(self):
        if self.proxyType == ProxyType.CLIENT and self.chkListen.isChecked():
            return False
        else:
            return True
        
    def manageAnalyzers(self, state):
        if state:
            self.mw.tab.removeTab(4)
            self.mw.tab.removeTab(3)
            self.mw.videoConfigure = None
            self.mw.audioConfigure = None
            self.mw.startListener(self.getInterfaces())
        else:
            self.mw.stopListener()
            self.mw.tab.addTab(self.mw.videoPanel, "Video")
            self.mw.tab.addTab(self.mw.audioPanel, "Audio")
            if not self.mw.videoConfigure:
                videoWorkerName = self.mw.videoPanel.cmbWorker.currentText()
                if len(videoWorkerName):
                    self.mw.loadVideoConfigure(videoWorkerName)
            if not self.mw.audioConfigure:
                audioWorkerName = self.mw.audioPanel.cmbWorker.currentText()
                if len(audioWorkerName):
                    self.mw.loadAudioConfigure(audioWorkerName)

    def getInterfaces(self):
        if_addrs = []
        match self.proxyType:
            case ProxyType.SERVER:
                if self.grpAlarmBroadcast.isChecked():
                    # only able to get this to work on one interface at a time needs research
                    '''
                    if self.chkBroadcastAllInterfaces.isChecked():
                        if_addrs = self.if_addrs
                    else:
                        if_addrs.append(self.cmbInterfaces.currentText())
                    '''
                    if_addrs.append(self.cmbInterfaces.currentText())
            case ProxyType.CLIENT:
                if_addrs = self.if_addrs
        return if_addrs
        
    def cmbInterfacesChanged(self, interface):
        print(interface)
        self.mw.settings.setValue(self.broadcastInterfaceKey, interface)
        self.setProxyType(self.proxyType)

    def setMediaMTXProxies(self):
        try:
            dir = None
            if self.chkAutoDownload.isChecked():
                dir = os.path.dirname(sys.executable)
            else:
                dir = self.txtDirectoryMTX.text()

            config_filename = f'{dir}/mediamtx.yml'
            if not os.path.isfile(config_filename):
                self.mw.signals.error.emit(f'Invalid path for MediaMTX configuration file: {config_filename}')
                return

            uri_map = {}
            for proxy_key in self.mw.proxies:
                username = None
                password = None
                if camera := self.mw.cameraPanel.getCamera(self.mw.proxies[proxy_key]):
                    if profile := camera.getDisplayProfile():
                        username = profile.username()
                        password = profile.password()
                mtx_key = self.mw.proxies[proxy_key][len("rtsp://"):]
                mtx_key = mtx_key[mtx_key.find('/')+1:]

                uri = None
                if username and password:
                    uri = f'rtsp://{username}:{password}@{proxy_key[len("rtsp://"):]}'
                else:
                    uri = proxy_key

                uri_map[mtx_key] = uri                

            mtx_uris = {}
            relevant = False
            key = None
            value = None
            with open(config_filename, 'r') as config:
                for readline in config:
                    line = readline.strip()

                    if relevant:
                        if not line.startswith("#"):
                            if not key:
                                key = line[:-1]
                            else:
                                value = line[len("source: "):]
                                mtx_uris[key] = value
                                key = None
                                value = None

                    if line.startswith("paths:"):
                        relevant = True

                    if line.startswith("all_others"):
                        relevant = False

            if mtx_uris == uri_map:
                return

            relevant = False
            lines = []
            with open(config_filename, 'r') as config:
                for readline in config:
                    line = readline.strip()

                    if not relevant or line.startswith("#"):
                        lines.append(readline)

                    if line.startswith("paths:"):
                        relevant = True

                        for key in uri_map:
                            lines.append(f'  {key}:\n')
                            lines.append(f'    source: {uri_map[key]}\n')

                    if line.startswith("all_others"):
                        lines.append(readline)
                        relevant = False

            config = open(config_filename, 'w')
            for line in lines:
                config.write(line)
            config.close()

        except Exception as ex:
            logger.error(f'Error setting Media MTX proxies: {ex}')
            self.mw.signals.error.emit(f'Error setting Media MTX proxies: {ex}')

    def btnTestClicked(self):
        print("test button clicked")
