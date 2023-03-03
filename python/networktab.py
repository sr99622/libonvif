import os
import sys
import numpy as np
import cv2
from time import sleep
from PyQt6.QtWidgets import QCheckBox, QLineEdit, QSpinBox, \
QGridLayout, QWidget, QSlider, QLabel, QMessageBox, QListWidget
from PyQt6.QtCore import Qt, pyqtSignal, QObject

sys.path.append("../build/libonvif")
import onvif

class NetworkTab(QWidget):
    def __init__(self):
        super().__init__()

        self.chkDHCP = QCheckBox("DHCP enabled")
        self.txtIPAddress = QLineEdit()
        lblIPAddress = QLabel("IP Address")
        self.txtSubnetMask = QLineEdit()
        lblSubnetMask = QLabel("Subnet Mask")
        self.txtDefaultGateway = QLineEdit()
        lblDefaultGateway = QLabel("Default Gateway")
        self.txtDNS = QLineEdit()
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

    def fill(self, D):
        self.chkDHCP.setChecked(D.dhcp_enabled())
        self.txtIPAddress.setText(D.ip_address_buf())
        self.txtDefaultGateway.setText(D.default_gateway_buf())
        self.txtDNS.setText(D.dns_buf())