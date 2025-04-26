#/********************************************************************
# libonvif/onvif-gui/onvif_gui/panels/cameras/networktab.py 
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

from PyQt6.QtWidgets import QCheckBox, QLineEdit, QGridLayout, QWidget, QLabel
from PyQt6.QtCore import QRegularExpression
from PyQt6.QtGui import QRegularExpressionValidator

class NetworkTab(QWidget):
    def __init__(self, cp):
        super().__init__()
        self.cp = cp

        ipRange = "(?:[0-1]?[0-9]?[0-9]|2[0-4][0-9]|25[0-5])" 
        ipRegex = QRegularExpression("^" + ipRange + "\\." + ipRange + "\\." + ipRange + "\\." + ipRange + "$")
        ipValidator = QRegularExpressionValidator(ipRegex, self)           

        self.chkDHCP = QCheckBox("DHCP enabled")
        self.chkDHCP.clicked.connect(cp.onEdit)
        self.chkDHCP.clicked.connect(self.onChkDHCPChecked)
        self.txtIPAddress = QLineEdit()
        self.txtIPAddress.setValidator(ipValidator)
        self.txtIPAddress.textEdited.connect(cp.onEdit)
        lblIPAddress = QLabel("Address")
        self.txtSubnetMask = QLineEdit()
        self.txtSubnetMask.setValidator(ipValidator)
        self.txtSubnetMask.textEdited.connect(cp.onEdit)
        lblSubnetMask = QLabel("Subnet")
        self.txtDefaultGateway = QLineEdit()
        self.txtDefaultGateway.setValidator(ipValidator)
        self.txtDefaultGateway.textEdited.connect(cp.onEdit)
        lblDefaultGateway = QLabel("Gateway")
        self.txtDNS = QLineEdit()
        self.txtDNS.setValidator(ipValidator)
        self.txtDNS.textEdited.connect(cp.onEdit)
        lblDNS = QLabel("DNS")

        lytMain = QGridLayout(self)
        lytMain.addWidget(self.chkDHCP,           0, 1, 1, 1)
        lytMain.addWidget(lblIPAddress,           1, 0, 1, 1)
        lytMain.addWidget(self.txtIPAddress,      1, 1, 1, 1)
        lytMain.addWidget(lblSubnetMask,          2, 0, 1, 1)
        lytMain.addWidget(self.txtSubnetMask,     2, 1, 1, 1)
        lytMain.addWidget(lblDefaultGateway,      3, 0, 1, 1)
        lytMain.addWidget(self.txtDefaultGateway, 3, 1, 1, 1)
        lytMain.addWidget(lblDNS,                 4, 0, 1, 1)
        lytMain.addWidget(self.txtDNS,            4, 1, 1, 1)

    def fill(self, onvif_data):
        self.chkDHCP.setChecked(onvif_data.dhcp_enabled())
        self.txtIPAddress.setText(onvif_data.ip_address_buf())
        self.txtDefaultGateway.setText(onvif_data.default_gateway_buf())
        self.txtDNS.setText(onvif_data.dns_buf())
        self.txtSubnetMask.setText(onvif_data.mask_buf())
        self.txtIPAddress.setEnabled(not onvif_data.dhcp_enabled())
        self.txtDefaultGateway.setEnabled(not onvif_data.dhcp_enabled())
        self.txtSubnetMask.setEnabled(not onvif_data.dhcp_enabled())
        self.txtDNS.setEnabled(not onvif_data.dhcp_enabled())
        self.setEnabled(len(onvif_data.ip_address_buf()))
        self.cp.onEdit()

    def edited(self, onvif_data):
        result = False
        if self.isEnabled():
            if not onvif_data.dhcp_enabled() == self.chkDHCP.isChecked():
                result = True
            if not onvif_data.ip_address_buf() == self.txtIPAddress.text():
                result = True
            if not onvif_data.default_gateway_buf() == self.txtDefaultGateway.text():
                result = True
            if not onvif_data.dns_buf() == self.txtDNS.text():
                result = True
            if not onvif_data.mask_buf() == self.txtSubnetMask.text():
                result = True

        return result
    
    def update(self, onvif_data):
        if self.edited(onvif_data):
            onvif_data.setDHCPEnabled(self.chkDHCP.isChecked())
            onvif_data.setIPAddressBuf(self.txtIPAddress.text())
            onvif_data.setDefaultGatewayBuf(self.txtDefaultGateway.text())
            onvif_data.setDNSBuf(self.txtDNS.text())
            onvif_data.setMaskBuf(self.txtSubnetMask.text())
            onvif_data.startUpdateNetwork()

    def onChkDHCPChecked(self):
        checked = self.chkDHCP.isChecked()
        self.txtIPAddress.setEnabled(False)
        self.txtDefaultGateway.setEnabled(False)
        self.txtSubnetMask.setEnabled(False)
        self.txtDNS.setEnabled(False)
