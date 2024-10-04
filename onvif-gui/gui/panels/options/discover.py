#/********************************************************************
# libonvif/onvif-gui/gui/panels/options/discover.py 
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

from PyQt6.QtWidgets import QLineEdit, QGridLayout, QWidget, QCheckBox, \
    QLabel, QComboBox, QPushButton, QDialog, QDialogButtonBox, \
    QRadioButton, QGroupBox
from PyQt6.QtCore import Qt, QRegularExpression
from PyQt6.QtGui import QRegularExpressionValidator
from loguru import logger
import libonvif as onvif

class AddCameraDialog(QDialog):
    def __init__(self, mw):
        super().__init__(mw)
        self.mw = mw

        ipRange = "(?:[0-1]?[0-9]?[0-9]|2[0-4][0-9]|25[0-5])" 
        ipRegex = QRegularExpression("^" + ipRange + "\\." + ipRange + "\\." + ipRange + "\\." + ipRange + "$")
        ipValidator = QRegularExpressionValidator(ipRegex, self)           
    
        self.setWindowTitle("Add Camera")
        self.txtIPAddress = QLineEdit()
        self.txtIPAddress.setValidator(ipValidator)
        self.lblIPAddress = QLabel("IP Address")
        self.txtOnvifPort = QLineEdit()
        self.lblOnvifPort = QLabel("Onvif Port")

        buttonBox = QDialogButtonBox( \
            QDialogButtonBox.StandardButton.Ok | \
            QDialogButtonBox.StandardButton.Cancel)

        lytMain = QGridLayout(self)
        lytMain.addWidget(self.lblIPAddress,   1, 0, 1, 1)
        lytMain.addWidget(self.txtIPAddress,   1, 1, 1, 1)
        lytMain.addWidget(self.lblOnvifPort,   2, 0, 1, 1)
        lytMain.addWidget(self.txtOnvifPort,   2, 1, 1, 1)
        lytMain.addWidget(buttonBox,           5, 0, 1, 2)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

        self.txtIPAddress.setFocus()

class DiscoverOptions(QWidget):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw

        self.interfaceKey = "settings/interface"
        self.autoDiscoverKey = "settings/autoDiscover"
        self.discoveryTypeKey = "settings/discoveryType"
        self.autoStartKey = "settings/autoStart"
        self.scanAllKey = "settings/scanAll"
        self.cameraListKey = "settings/cameraList"


        self.grpDiscoverType = QGroupBox("Set Camera Discovery Method")
        self.radDiscover = QRadioButton("Discover Broadcast", self.grpDiscoverType )
        self.radDiscover.setChecked(int(self.mw.settings.value(self.discoveryTypeKey, 1)))
        self.radDiscover.toggled.connect(self.radDiscoverToggled)
        self.radCached = QRadioButton("Cached Addresses", self.grpDiscoverType )
        self.radCached.setChecked(not self.radDiscover.isChecked())
        lytDiscoverType = QGridLayout(self.grpDiscoverType )
        lytDiscoverType.addWidget(self.radDiscover,   0, 0, 1, 1)
        lytDiscoverType.addWidget(self.radCached,     0, 1, 1, 1)

        self.chkScanAllNetworks = QCheckBox("Scan All Networks During Discovery")
        self.chkScanAllNetworks.setChecked(int(mw.settings.value(self.scanAllKey, 1)))
        self.chkScanAllNetworks.stateChanged.connect(self.scanAllNetworksChecked)
        self.cmbInterfaces = QComboBox()
        intf = self.mw.settings.value(self.interfaceKey, "")
        self.lblInterfaces = QLabel("Network")
        session = onvif.Session()
        session.getActiveInterfaces()
        i = 0
        while len(session.active_interface(i)) > 0 and i < 16:
            self.cmbInterfaces.addItem(session.active_interface(i))
            i += 1
        if len(intf) > 0:
            self.cmbInterfaces.setCurrentText(intf)
        self.cmbInterfaces.currentTextChanged.connect(self.cmbInterfacesChanged)
        self.cmbInterfaces.setEnabled(not self.chkScanAllNetworks.isChecked())
        self.lblInterfaces.setEnabled(not self.chkScanAllNetworks.isChecked())

        self.btnAddCamera = QPushButton("Add Camera")
        self.btnAddCamera.clicked.connect(self.btnAddCameraClicked)

        self.chkAutoDiscover = QCheckBox("Auto Discovery")
        self.chkAutoDiscover.setChecked(bool(int(mw.settings.value(self.autoDiscoverKey, 0))))
        self.chkAutoDiscover.stateChanged.connect(self.autoDiscoverChecked)
        
        self.chkAutoStart = QCheckBox("Auto Start")
        self.chkAutoStart.setChecked(bool(int(mw.settings.value(self.autoStartKey, 0))))
        self.chkAutoStart.stateChanged.connect(self.autoStartChecked)

        pnlInterface = QGroupBox("Discovery Options")
        lytInterface = QGridLayout(pnlInterface)
        lytInterface.addWidget(self.grpDiscoverType,     0, 0, 1, 2)
        lytInterface.addWidget(self.chkScanAllNetworks,  2, 0, 1, 2)
        lytInterface.addWidget(self.lblInterfaces,       4, 0, 1, 1)
        lytInterface.addWidget(self.cmbInterfaces,       4, 1, 1, 1)
        lytInterface.addWidget(self.btnAddCamera,        5, 0, 1, 2, Qt.AlignmentFlag.AlignCenter)
        lytInterface.setColumnStretch(1, 10)
        lytInterface.setContentsMargins(10, 10, 10, 10)

        lytMain = QGridLayout(self)
        lytMain.addWidget(pnlInterface,          0, 0, 1, 2)
        lytMain.addWidget(QLabel(),              1, 0, 1, 2)
        lytMain.addWidget(self.chkAutoDiscover,  2, 0, 1, 1)
        lytMain.addWidget(self.chkAutoStart,     2, 1, 1, 1)
        lytMain.addWidget(QLabel(),              3, 0, 1, 2)
        lytMain.setRowStretch(3, 10)

        self.radDiscoverToggled(self.radDiscover.isChecked())

    def radDiscoverToggled(self, checked):
        self.chkScanAllNetworks.setEnabled(checked)
        if self.chkScanAllNetworks.isChecked():
            self.lblInterfaces.setEnabled(False)
            self.cmbInterfaces.setEnabled(False)
        else:
            self.lblInterfaces.setEnabled(checked)
            self.cmbInterfaces.setEnabled(checked)
        self.mw.settings.setValue(self.discoveryTypeKey, int(checked))

    def scanAllNetworksChecked(self, state):
        self.mw.settings.setValue(self.scanAllKey, state)
        self.cmbInterfaces.setEnabled(not self.chkScanAllNetworks.isChecked())
        self.lblInterfaces.setEnabled(not self.chkScanAllNetworks.isChecked())

    def cmbInterfacesChanged(self, network):
        self.mw.settings.setValue(self.interfaceKey, network)

    def btnAddCameraClicked(self):
        dlg = AddCameraDialog(self.mw)
        if dlg.exec():
            ip_address = dlg.txtIPAddress.text()
            onvif_port = dlg.txtOnvifPort.text()
            if not len(onvif_port):
                onvif_port = "80"
            xaddrs = f'http://{ip_address}:{onvif_port}/onvif/device_service'
            logger.debug(f'Attempting to add camera manually using xaddrs: {xaddrs}')
            data = onvif.Data()
            data.getData = self.mw.cameraPanel.getData
            data.getCredential = self.mw.cameraPanel.getCredential
            data.setXAddrs(xaddrs)
            data.setDeviceService("POST /onvif/device_service HTTP/1.1\n")
            data.manual_fill()

    def autoDiscoverChecked(self, state):
        self.mw.settings.setValue(self.autoDiscoverKey, state)

    def autoStartChecked(self, state):
        self.mw.settings.setValue(self.autoStartKey, state)

