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

from gui.enums import ProxyType
import libonvif as onvif
import ipaddress

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
        self.allInterfacesKey = "settings/allInterfaces"
        self.listenKey = "settings/alarmsFromServer"
        self.showBoxesKey = "settings/showBoxesFromServer"
        self.broadcastInterfaceKey = "settings/proxyBroadcastInterface"

        self.if_addrs = []
        session = onvif.Session()
        session.getActiveInterfaces()
        i = 0
        while len(session.active_interface(i)) > 0 and i < 16:
            self.if_addrs.append(session.active_interface(i).split(" - ")[0])
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

        self.chkAllInterfaces = QCheckBox("Use All Available Interfaces")
        self.chkAllInterfaces.setEnabled(False)
        self.chkAllInterfaces.setChecked(int(mw.settings.value(self.allInterfacesKey, 1)))
        self.chkAllInterfaces.stateChanged.connect(self.chkAllInterfacesChecked)
        self.cmbInterfaces = QComboBox()
        self.cmbInterfaces.setEnabled(False)
        self.lblInterfaces = QLabel("Network Interface")
        self.lblInterfaces.setEnabled(False)
        for if_addr in self.if_addrs:
            self.cmbInterfaces.addItem(if_addr)
        self.cmbInterfaces.setCurrentText(self.mw.settings.value(self.broadcastInterfaceKey, ""))
        self.cmbInterfaces.currentTextChanged.connect(self.cmbInterfacesChanged)

        self.proxyRemote = self.mw.settings.value(self.proxyRemoteKey)

        self.txtRemote = QLineEdit()
        self.txtRemote.setText(self.proxyRemote)
        self.txtRemote.textEdited.connect(self.txtRemoteEdited)
        self.txtRemote.setEnabled(False)
        self.lblRemote = QLabel("Enter connect url from server")
        self.lblRemote.setEnabled(False)

        self.btnUpdate = QPushButton("Update")
        self.btnUpdate.clicked.connect(self.btnUpdateClicked)
        self.btnUpdate.setEnabled(False)

        self.chkListen = QCheckBox("Get alarm events from server")
        self.chkListen.setChecked(int(mw.settings.value(self.listenKey, 1)))
        self.chkListen.stateChanged.connect(self.chkListenChecked)
        self.chkListen.setEnabled(False)

        self.chkShowBoxes = QCheckBox("Show detection boxes from server")
        self.chkShowBoxes.setChecked(int(mw.settings.value(self.showBoxesKey, 1)))
        self.chkShowBoxes.stateChanged.connect(self.chkShowBoxesChecked)
        self.chkShowBoxes.setEnabled(False)

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

        lytProxyType = QGridLayout(self.grpProxyType)
        lytProxyType.addWidget(self.radStandAlone,    0, 0, 1, 1)
        lytProxyType.addWidget(QLabel(),              1, 0, 1, 2)
        lytProxyType.addWidget(self.radClient,        2, 0, 1, 1)
        lytProxyType.addWidget(self.lblRemote,        2, 1, 1, 1, Qt.AlignmentFlag.AlignCenter)
        lytProxyType.addWidget(self.btnUpdate,        3, 0, 1, 1, Qt.AlignmentFlag.AlignCenter)
        lytProxyType.addWidget(self.txtRemote,        3, 1, 1, 1)
        lytProxyType.addWidget(self.chkListen,        4, 0, 1, 2)
        lytProxyType.addWidget(self.chkShowBoxes,     5, 0, 1, 2)
        lytProxyType.addWidget(QLabel(),              6, 0, 1, 1)
        lytProxyType.addWidget(self.radServer,        7, 0, 1, 1)
        lytProxyType.addWidget(self.chkAllInterfaces, 7, 1, 1, 1)
        lytProxyType.addWidget(self.lblInterfaces,    8, 0, 1, 1)
        lytProxyType.addWidget(self.cmbInterfaces,    8, 1, 1, 1)
        lytProxyType.addWidget(QLabel(),              9, 0, 1, 1)
        lytProxyType.addWidget(self.lblConnect,      10, 0, 1, 2)
        #lytProxyType.addWidget(self.btnServerManage, 11, 0, 1, 1, Qt.AlignmentFlag.AlignCenter)
        lytProxyType.addWidget(self.lblServer,       11, 1, 1, 1)

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

        if type == ProxyType.SERVER:
            self.mw.initializeBroadcaster(self.getInterfaces())
            if self.chkAllInterfaces.isChecked():
                self.mw.startProxyServer("")
                self.mw.startOnvifServer("")
            else:
                self.mw.startProxyServer(self.cmbInterfaces.currentText())
                self.mw.startOnvifServer(self.cmbInterfaces.currentText())
            self.setServersLabel()

        if not hasattr(self.mw, "cameraPanel"):
            return

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
        else:
            self.manageAnalyzers(False)

    def radClientToggled(self, checked):
        self.lblRemote.setEnabled(checked)
        self.txtRemote.setEnabled(checked)
        self.chkListen.setEnabled(checked)
        if self.chkListen.isChecked():
            self.chkShowBoxes.setEnabled(checked)
        else:
            self.chkShowBoxes.setEnabled(False)
        if checked:
           self.setProxyType(ProxyType.CLIENT)

    def radStandAloneToggled(self, checked):
        if checked:
            self.setProxyType(ProxyType.STAND_ALONE)

    def radServerToggled(self, checked):
        self.lblConnect.setEnabled(checked)
        self.chkAllInterfaces.setEnabled(checked)
        interface_enabled = checked and not self.chkAllInterfaces.isChecked()
        self.cmbInterfaces.setEnabled(interface_enabled)
        self.lblInterfaces.setEnabled(interface_enabled)
        if checked:
            self.setProxyType(ProxyType.SERVER)
            self.setServersLabel()
            self.btnServerManage.setEnabled(True)
        else:
            self.lblServer.setText("\n\n")
            self.btnServerManage.setEnabled(False)

    def setServersLabel(self):
        port = None
        root_uri = self.mw.proxy.getRootURI()
        tokens = root_uri[:len(root_uri)-1][7:].split(":")
        if len(tokens) > 1:
            port = tokens[1]

        lbl = ""
        if self.chkAllInterfaces.isChecked():
            uris = []
            for if_addr in self.if_addrs:
                if port:
                    uris.append(f'rtsp://{if_addr}:{port}/')
                else:
                    uris.append(f'rtsp://{if_addr}/')

            for uri in uris:
                lbl += uri
                lbl += "\n"
        else:
            if port:
                lbl = f'rtsp://{self.cmbInterfaces.currentText()}:{port}/\n'
            else:
                lbl = f'rtsp://{self.cmbInterfaces.currentText()}/\n'

        for _ in range(3 - len(lbl.split("\n"))):
            lbl += "\n"

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

    def chkAllInterfacesChecked(self, state):
        self.mw.settings.setValue(self.allInterfacesKey, state)
        self.setProxyType(self.proxyType)
        self.cmbInterfaces.setEnabled(not self.chkAllInterfaces.isChecked())
        self.lblInterfaces.setEnabled(not self.chkAllInterfaces.isChecked())
        #self.mw.initializeBroadcaster(self.getInterfaces())

    def chkListenChecked(self, state):
        self.mw.settings.setValue(self.listenKey, state)
        self.manageAnalyzers(state)
        if (state):
            self.chkShowBoxes.setEnabled(True)
        else:
            self.chkShowBoxes.setEnabled(False)

    def chkShowBoxesChecked(self, state):
        self.mw.settings.setValue(self.showBoxesKey, state)

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
        if self.chkAllInterfaces.isChecked():
            return self.if_addrs
        else:
            return [self.cmbInterfaces.currentText()]
        
    def cmbInterfacesChanged(self, interface):
        print(interface)
        self.mw.settings.setValue(self.broadcastInterfaceKey, interface)
        #self.mw.initializeBroadcaster(self.getInterfaces())
        self.setProxyType(self.proxyType)
