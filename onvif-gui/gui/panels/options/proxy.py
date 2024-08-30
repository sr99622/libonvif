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

from PyQt6.QtWidgets import QMessageBox, QLineEdit, \
    QGridLayout, QWidget, QLabel, QMessageBox, QRadioButton, \
    QGroupBox
from PyQt6.QtCore import Qt
from time import sleep
from gui.enums import ProxyType

class ProxyOptions(QWidget):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw

        self.proxyTypeKey = "settings/proxyType"
        self.proxyRemoteKey = "settings/proxyRemote"

        self.grpProxyType = QGroupBox("Select Proxy Type")
        self.radStandAlone = QRadioButton("Stand Alone", self.grpProxyType)
        self.radServer = QRadioButton("Server", self.grpProxyType)
        self.radClient = QRadioButton("Client", self.grpProxyType)

        self.lblServer = QLabel()
        self.lblConnect = QLabel("Connect url for clients")
        self.lblConnect.setEnabled(False)

        self.proxyRemote = self.mw.settings.value(self.proxyRemoteKey)

        self.txtRemote = QLineEdit()
        self.txtRemote.setText(self.proxyRemote)
        self.txtRemote.textEdited.connect(self.txtRemoteEdited)
        self.txtRemote.setEnabled(False)
        self.lblRemote = QLabel("Enter connect url from server")
        self.lblRemote.setEnabled(False)

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
        lytProxyType.addWidget(self.radStandAlone, 0, 0, 1, 1)
        lytProxyType.addWidget(self.radServer,     1, 0, 1, 1)
        lytProxyType.addWidget(self.lblConnect,    1, 1, 1, 1, Qt.AlignmentFlag.AlignRight)
        lytProxyType.addWidget(self.lblServer,     2, 0, 1, 2, Qt.AlignmentFlag.AlignRight)
        lytProxyType.addWidget(self.radClient,     3, 0, 1, 1)
        lytProxyType.addWidget(self.lblRemote,     3, 1, 1, 1, Qt.AlignmentFlag.AlignRight)
        lytProxyType.addWidget(self.txtRemote,     4, 0, 1, 2)
        lytProxyType.setColumnStretch(2, 10)

        lytMain = QGridLayout(self)
        lytMain.addWidget(self.grpProxyType,  0, 0, 1, 1)
        lytMain.addWidget(QLabel(),           1, 0, 1, 1)
        lytMain.setRowStretch(1, 10)

    def setProxyType(self, type):
        self.proxyType = type
        self.mw.settings.setValue(self.proxyTypeKey, type)

        if type == ProxyType.SERVER:
            self.mw.startProxyServer()

        if not hasattr(self.mw, "cameraPanel"):
            return

        if len(self.mw.pm.players):
            QMessageBox.information(self.mw, "Closing Streams", "All current streams will be closed")
            self.mw.closeAllStreams()

        getProxyURI = None
        if type != ProxyType.STAND_ALONE:
            getProxyURI = self.mw.getProxyURI

        if lstCamera := self.mw.cameraPanel.lstCamera:
            cameras = [lstCamera.item(x) for x in range(lstCamera.count())]
            for camera in cameras:
                self.mw.addCameraProxy(camera)
                camera.onvif_data.getProxyURI = getProxyURI
                for profile in camera.profiles:
                    profile.getProxyURI = getProxyURI

    def radStandAloneToggled(self, checked):
        if checked:
            self.setProxyType(ProxyType.STAND_ALONE)

    def radServerToggled(self, checked):
        self.lblConnect.setEnabled(checked)
        if checked:
            self.setProxyType(ProxyType.SERVER)
            self.lblServer.setText(self.mw.proxy.getRootURI())
        else:
            self.mw.stopProxyServer()
            self.lblServer.setText("")

    def radClientToggled(self, checked):
        self.lblRemote.setEnabled(checked)
        self.txtRemote.setEnabled(checked)
        if checked:
           self.setProxyType(ProxyType.CLIENT)

    def txtRemoteEdited(self, arg):
        self.mw.settings.setValue(self.proxyRemoteKey, arg)