import os
import sys
import numpy as np
import cv2
from time import sleep
from PyQt6.QtWidgets import QDialogButtonBox, QLineEdit, QPushButton, \
QGridLayout, QWidget, QSlider, QLabel, QMessageBox, QListWidget, \
QTabWidget
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QSettings
from logindialog import LoginDialog
from videotab import VideoTab
from imagetab import ImageTab
from networktab import NetworkTab
from ptztab import PTZTab
from admintab import AdminTab

sys.path.append("../build/libonvif")
import onvif

class CameraPanelSignals(QObject):
    fill = pyqtSignal(onvif.Data)
    login = pyqtSignal(onvif.Data)


class CameraList(QListWidget):
    def __init__(self):
        super().__init__()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Return:
            self.itemDoubleClicked.emit(self.currentItem())
        return super().keyPressEvent(event)
    
class CameraPanel(QWidget):
    def __init__(self, mw):
        super().__init__()
        self.devices = []
        self.mw = mw
        self.dlgLogin = LoginDialog(self)
        self.settings = QSettings("onvif", "alias")
        self.icnDiscover = QIcon('image:discover.png')
        self.icnApply = QIcon("image:apply.png")
        self.icnMute = QIcon('image:mute.png')
        self.icnAudio = QIcon('image:audio.png')
        self.icnRecord = QIcon("image:record.png")
        self.icnRecording = QIcon("image:recording.png")

        self.boss = onvif.Manager()
        self.boss.discovered = lambda : self.discovered()
        self.boss.getCredential = lambda D : self.getCredential(D)
        self.boss.getData = lambda D : self.getData(D)
        self.boss.filled = lambda D : self.filled(D)

        self.btnRecord = QPushButton()
        self.btnRecord.setIcon(self.icnRecord)
        self.btnRecord.setToolTip("Record")
        self.btnRecord.setToolTipDuration(2000)
        self.btnRecord.setMinimumWidth(self.icnRecord.availableSizes()[0].width() * 2)
        self.btnRecord.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btnRecord.clicked.connect(self.btnRecordClicked)

        self.btnMute = QPushButton()
        if self.mw.mute:
            self.btnMute.setIcon(self.icnMute)
        else:
            self.btnMute.setIcon(self.icnAudio)
        self.btnMute.setToolTip("Mute")
        self.btnMute.setToolTipDuration(2000)
        self.btnMute.setMinimumWidth(self.icnMute.availableSizes()[0].width() * 2)
        self.btnMute.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btnMute.clicked.connect(self.btnMuteClicked)

        self.sldVolume = QSlider(Qt.Orientation.Horizontal)
        self.sldVolume.setValue(self.mw.volume)
        self.sldVolume.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.sldVolume.valueChanged.connect(self.sldVolumeChanged)

        self.btnDiscover = QPushButton()
        self.btnDiscover.setIcon(self.icnDiscover)
        self.btnDiscover.setToolTip("Discover")
        self.btnDiscover.setToolTipDuration(2000)
        self.btnDiscover.setMinimumWidth(self.icnDiscover.availableSizes()[0].width() * 2)
        self.btnDiscover.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btnDiscover.clicked.connect(self.btnDiscoverClicked)

        self.btnApply = QPushButton()
        self.btnApply.setIcon(self.icnApply)
        self.btnApply.setToolTip("Apply")
        self.btnApply.setToolTipDuration(2000)
        self.btnApply.setMinimumWidth(self.icnApply.availableSizes()[0].width() * 2)
        self.btnApply.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btnApply.clicked.connect(self.btnApplyClicked)
        self.btnApply.setEnabled(False)
        
        self.lstCamera = CameraList()
        self.lstCamera.currentRowChanged.connect(self.onCurrentRowChanged)
        self.lstCamera.itemDoubleClicked.connect(self.onItemDoubleClicked)

        self.tabOnvif = QTabWidget()
        self.tabVideo = VideoTab(self)
        self.tabImage = ImageTab(self)
        self.tabNetwork = NetworkTab(self)
        self.ptzTab = PTZTab(self)
        self.adminTab = AdminTab(self)
        self.tabOnvif.addTab(self.tabVideo, "Video")
        self.tabOnvif.addTab(self.tabImage, "Image")
        self.tabOnvif.addTab(self.tabNetwork, "Network")
        self.tabOnvif.addTab(self.ptzTab, "PTZ")
        self.tabOnvif.addTab(self.adminTab, "Admin")

        self.signals = CameraPanelSignals()
        self.signals.fill.connect(self.tabVideo.fill)
        self.signals.fill.connect(self.tabImage.fill)
        self.signals.fill.connect(self.tabNetwork.fill)
        self.signals.fill.connect(self.ptzTab.fill)
        self.signals.fill.connect(self.adminTab.fill)
        self.signals.login.connect(self.onShowLogin)

        lytMain = QGridLayout(self)
        lytMain.addWidget(self.lstCamera,   0, 0, 1, 5)
        lytMain.addWidget(self.tabOnvif,    1, 0, 1, 5)
        lytMain.addWidget(self.btnRecord,   2, 0, 1, 1, Qt.AlignmentFlag.AlignCenter)
        lytMain.addWidget(self.btnMute,     2, 1, 1, 1, Qt.AlignmentFlag.AlignCenter)
        lytMain.addWidget(self.sldVolume,   2, 2, 1, 1)
        lytMain.addWidget(self.btnDiscover, 2, 3, 1, 1, Qt.AlignmentFlag.AlignCenter)
        lytMain.addWidget(self.btnApply,    2, 4, 1, 1, Qt.AlignmentFlag.AlignCenter)
        lytMain.setRowStretch(0, 10)

    def btnDiscoverClicked(self):
        self.boss.startPyDiscover()
        self.btnDiscover.setEnabled(False)

    def filled(self, onvif_data):
        if len(onvif_data.last_error()) > 0:
            print("ERROR:", onvif_data.last_error())
        self.devices[self.lstCamera.currentRow()] = onvif_data
        self.signals.fill.emit(onvif_data)
        self.btnApply.setEnabled(False)
        if not self.mw.connecting:
            self.setEnabled(True)
            self.lstCamera.setFocus()

    def discovered(self):
        self.btnDiscover.setEnabled(True)

    def getCredential(self, onvif_data):
        for d in self.devices:
            if d == onvif_data:
                onvif_data.cancelled = True
                return onvif_data
            
        if len(self.mw.settingsPanel.txtPassword.text()) > 0 and \
                len(onvif_data.last_error()) == 0:
            onvif_data.setUsername(self.mw.settingsPanel.txtUsername.text())
            onvif_data.setPassword(self.mw.settingsPanel.txtPassword.text())
        else:
            onvif_data.clearLastError()
            self.dlgLogin.active = True
            self.signals.login.emit(onvif_data)
            while self.dlgLogin.active:
                sleep(0.01)

        return onvif_data
    
    def onShowLogin(self, onvif_data):
        self.dlgLogin.exec(onvif_data)
    
    def getData(self, onvif_data):
        onvif_data.alias = self.settings.value(onvif_data.serial_number(), onvif_data.camera_name())
        self.devices.append(onvif_data)
        self.lstCamera.addItem(onvif_data.alias)

    def onCurrentRowChanged(self, row):
        onvif_data = self.devices[self.lstCamera.currentRow()]
        if onvif_data.filled:
            self.setTabsEnabled(False)
            self.signals.fill.emit(onvif_data)
        else:
            self.boss.onvif_data = onvif_data
            self.setTabsEnabled(False)
            self.boss.startPyFill()

    def onItemDoubleClicked(self, item):
        onvif_data = self.devices[self.lstCamera.currentRow()]
        uri = onvif_data.stream_uri()[0 : 7] + onvif_data.username() + ":" \
            + onvif_data.password() + "@" + onvif_data.stream_uri()[7:]
        self.mw.connecting = True
        self.mw.playMedia(uri)

    def setTabsEnabled(self, enabled):
        self.tabVideo.setEnabled(enabled)
        self.tabImage.setEnabled(enabled)
        self.tabNetwork.setEnabled(enabled)
        self.ptzTab.setEnabled(enabled)
        self.adminTab.setEnabled(enabled)

    def btnApplyClicked(self):
        onvif_data = self.devices[self.lstCamera.currentRow()]
        self.tabVideo.update(onvif_data)
        self.tabImage.update(onvif_data)
        self.tabNetwork.update(onvif_data)
        self.adminTab.update(onvif_data)
        self.setEnabled(False)

    def onEdit(self):
        onvif_data = self.devices[self.lstCamera.currentRow()]
        if self.tabVideo.edited(onvif_data) or \
                self.tabImage.edited(onvif_data) or \
                self.tabNetwork.edited(onvif_data) or \
                self.adminTab.edited(onvif_data):
            self.btnApply.setEnabled(True)
        else:
            self.btnApply.setEnabled(False)

    def onMediaStarted(self, n):
        self.setEnabled(True)
        if self.mw.tab.currentIndex() == 0:
            self.lstCamera.setFocus()
            self.mw.setWindowTitle(self.lstCamera.currentItem().text())

    def sldVolumeChanged(self, value):
        self.mw.filePanel.control.sldVolume.setValue(value)
        self.mw.setVolume(value)

    def setBtnMute(self):
        if self.mw.mute:
            self.btnMute.setIcon(self.icnMute)
        else:
            self.btnMute.setIcon(self.icnAudio)
       
    def btnMuteClicked(self):
        self.mw.toggleMute()
        self.setBtnMute()
        self.mw.filePanel.control.setBtnMute()

    def btnRecordClicked(self):
        print("btnRecordClicked")