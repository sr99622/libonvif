import os
import sys
import numpy as np
import cv2
from time import sleep
from PyQt6.QtWidgets import QCheckBox, QLineEdit, QSpinBox, \
QGridLayout, QWidget, QSlider, QLabel, QMessageBox, QListWidget
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QRegularExpression
from PyQt6.QtGui import QRegularExpressionValidator

sys.path.append("../build/libonvif")
import onvif

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
        lblIPAddress = QLabel("IP Address")
        self.txtSubnetMask = QLineEdit()
        self.txtSubnetMask.setValidator(ipValidator)
        self.txtSubnetMask.textEdited.connect(cp.onEdit)
        lblSubnetMask = QLabel("Subnet Mask")
        self.txtDefaultGateway = QLineEdit()
        self.txtDefaultGateway.setValidator(ipValidator)
        self.txtDefaultGateway.textEdited.connect(cp.onEdit)
        lblDefaultGateway = QLabel("Default Gateway")
        self.txtDNS = QLineEdit()
        self.txtDNS.setValidator(ipValidator)
        self.txtDNS.textEdited.connect(cp.onEdit)
        lblDNS = QLabel("Primary DNS")

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
        self.setEnabled(True)
        self.onChkDHCPChecked()

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
            print("network tab update")
            onvif_data.setDHCPEnabled(self.chkDHCP.isChecked())
            onvif_data.setIPAddressBuf(self.txtIPAddress.text())
            onvif_data.setDefaultGatewayBuf(self.txtDefaultGateway.text())
            onvif_data.setDNSBuf(self.txtDNS.text())
            onvif_data.setMaskBuf(self.txtSubnetMask.text())
            self.cp.boss.onvif_data = onvif_data
            self.cp.boss.startPyUpdateNetwork()

    def onChkDHCPChecked(self):
        checked = self.chkDHCP.isChecked()
        self.txtIPAddress.setEnabled(not checked)
        self.txtDefaultGateway.setEnabled(not checked)
        self.txtSubnetMask.setEnabled(not checked)
        self.txtDNS.setEnabled(not checked)
