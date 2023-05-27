#/********************************************************************
# onvif-gui/gui/panels/camerapanel.py 
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
import datetime
from PyQt6.QtWidgets import QPushButton, QGridLayout, QWidget, QSlider, \
QListWidget, QTabWidget
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QSettings, QTimer
from gui.onvif import AdminTab, NetworkTab, ImageTab, VideoTab, PTZTab, LoginDialog

import libonvif as onvif

ICON_SIZE = 26

class CameraPanelSignals(QObject):
    fill = pyqtSignal(onvif.Data)
    login = pyqtSignal(onvif.Data)
    stopTimeout = pyqtSignal()
    remove = pyqtSignal()

class CameraList(QListWidget):
    def __init__(self):
        super().__init__()
        self.signals = CameraPanelSignals()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Return:
            self.itemDoubleClicked.emit(self.currentItem())
        if event.key() == Qt.Key.Key_Delete:
            self.signals.remove.emit()
        return super().keyPressEvent(event)
    
class CameraPanel(QWidget):
    def __init__(self, mw):
        super().__init__()
        self.devices = []
        self.mw = mw
        self.dlgLogin = LoginDialog(self)
        self.settings = QSettings("onvif", "alias")

        self.boss = onvif.Manager()
        self.boss.discovered = lambda : self.discovered()
        self.boss.getCredential = lambda D : self.getCredential(D)
        self.boss.getData = lambda D : self.getData(D)
        self.boss.filled = lambda D : self.filled(D)

        self.removing = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.applyTimeout)

        self.btnStop = QPushButton()
        self.btnStop.setToolTipDuration(2000)
        self.btnStop.setMinimumWidth(int(ICON_SIZE * 1.5))
        self.btnStop.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btnStop.clicked.connect(self.btnStopClicked)
        self.setBtnStop()

        self.btnRecord = QPushButton()
        self.btnRecord.setToolTip("Record")
        self.btnRecord.setToolTipDuration(2000)
        self.btnRecord.setMinimumWidth(int(ICON_SIZE * 1.5))
        self.btnRecord.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btnRecord.clicked.connect(self.btnRecordClicked)
        self.setBtnRecord()

        self.btnMute = QPushButton()
        self.btnMute.setToolTip("Mute")
        self.btnMute.setToolTipDuration(2000)
        self.btnMute.setMinimumWidth(int(ICON_SIZE * 1.5))
        self.btnMute.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btnMute.clicked.connect(self.btnMuteClicked)
        self.setBtnMute()

        self.sldVolume = QSlider(Qt.Orientation.Horizontal)
        self.sldVolume.setValue(int(self.mw.volume))
        self.sldVolume.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.sldVolume.valueChanged.connect(self.sldVolumeChanged)

        self.btnDiscover = QPushButton()
        self.btnDiscover.setStyleSheet(self.getButtonStyle("discover"))
        self.btnDiscover.setToolTip("Discover")
        self.btnDiscover.setToolTipDuration(2000)
        self.btnDiscover.setMinimumWidth(int(ICON_SIZE * 1.5))
        self.btnDiscover.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btnDiscover.clicked.connect(self.btnDiscoverClicked)

        self.btnApply = QPushButton()
        self.btnApply.setStyleSheet(self.getButtonStyle("apply"))
        self.btnApply.setToolTip("Apply")
        self.btnApply.setToolTipDuration(2000)
        self.btnApply.setMinimumWidth(int(ICON_SIZE * 1.5))
        self.btnApply.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btnApply.clicked.connect(self.btnApplyClicked)
        self.btnApply.setEnabled(False)
        
        self.lstCamera = CameraList()
        self.lstCamera.currentRowChanged.connect(self.onCurrentRowChanged)
        self.lstCamera.itemDoubleClicked.connect(self.onItemDoubleClicked)
        self.lstCamera.signals.remove.connect(self.removeCurrent)

        self.tabOnvif = QTabWidget()
        self.tabOnvif.setUsesScrollButtons(False)
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
        self.signals.stopTimeout.connect(self.timer.stop)

        lytMain = QGridLayout(self)
        lytMain.addWidget(self.lstCamera,   0, 0, 1, 6)
        lytMain.addWidget(self.tabOnvif,    1, 0, 1, 6)
        lytMain.addWidget(self.btnStop,     2, 0, 1, 1, Qt.AlignmentFlag.AlignCenter)
        lytMain.addWidget(self.btnRecord,   2, 1, 1, 1, Qt.AlignmentFlag.AlignCenter)
        lytMain.addWidget(self.btnMute,     2, 2, 1, 1, Qt.AlignmentFlag.AlignCenter)
        lytMain.addWidget(self.sldVolume,   2, 3, 1, 1)
        lytMain.addWidget(self.btnDiscover, 2, 4, 1, 1, Qt.AlignmentFlag.AlignCenter)
        lytMain.addWidget(self.btnApply,    2, 5, 1, 1, Qt.AlignmentFlag.AlignCenter)
        lytMain.setRowStretch(0, 10)

        self.setTabsEnabled(False)

    def btnDiscoverClicked(self):
        self.boss.interface = self.mw.settingsPanel.cmbInterfaces.currentText().split(" - ")[0]
        self.boss.startDiscover()
        self.btnDiscover.setEnabled(False)
        self.timer.start(5000)

    def discovered(self):
        self.setBtnStop()
        self.btnDiscover.setEnabled(True)
        self.setEnabled(True)

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

    def removeCurrent(self):
        self.removing = True
        del self.devices[self.lstCamera.currentRow()]
        self.lstCamera.clear()
        for data in self.devices:
            self.lstCamera.addItem(data.alias)

    def filled(self, onvif_data):
        if self.removing:
            self.removing = False
            self.setEnabled(True)
            onvif_data.clear(0)
            self.signals.fill.emit(onvif_data)
        else:
            self.devices[self.lstCamera.currentRow()] = onvif_data
            self.signals.fill.emit(onvif_data)
            if not self.mw.connecting:
                self.setEnabled(True)
                self.lstCamera.setFocus()
        self.signals.stopTimeout.emit()
        self.btnDiscover.setEnabled(True)
        self.btnApply.setEnabled(False)

    def onCurrentRowChanged(self, row):
        if row > -1:
            onvif_data = self.devices[row]
            if onvif_data.filled:
                self.setTabsEnabled(True)
                self.signals.fill.emit(onvif_data)
            else:
                self.boss.onvif_data = onvif_data
                self.setTabsEnabled(False)
                self.boss.startFill()

    def onItemDoubleClicked(self, item):
        onvif_data = self.devices[self.lstCamera.currentRow()]
        self.mw.connecting = True
        self.mw.playMedia(self.getStreamURI(onvif_data))

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
        self.timer.start(5000)

    def onEdit(self):
        if self.lstCamera.count() > 0:
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
            if not self.devices[self.lstCamera.currentRow()].filled:
                self.boss.onvif_data = self.devices[self.lstCamera.currentRow()]
                self.boss.startFill()
        self.setBtnStop()

    def sldVolumeChanged(self, value):
        self.mw.filePanel.control.sldVolume.setValue(value)
        self.mw.setVolume(value)

    def setBtnMute(self):
        if self.mw.mute:
            self.btnMute.setStyleSheet(self.getButtonStyle("mute"))
        else:
            self.btnMute.setStyleSheet(self.getButtonStyle("audio"))
       
    def btnMuteClicked(self):
        self.mw.toggleMute()
        self.setBtnMute()
        self.mw.filePanel.control.setBtnMute()

    def btnRecordClicked(self):
        filename = '{0:%Y%m%d%H%M%S.mp4}'.format(datetime.datetime.now())
        filename = self.mw.filePanel.dirSetter.txtDirectory.text() + "/" + filename

        if self.mw.player is not None:
            self.mw.player.toggleRecording(filename)
        self.setBtnRecord()
        self.mw.filePanel.control.setBtnRecord()

    def setBtnRecord(self):
        if self.mw.player is not None:
            if self.mw.player.isRecording():
                self.btnRecord.setStyleSheet(self.getButtonStyle("recording"))
            else:
                self.btnRecord.setStyleSheet(self.getButtonStyle("record"))
        else:
            self.btnRecord.setStyleSheet(self.getButtonStyle("record"))

    def setBtnStop(self):
        if self.mw.playing:
            self.btnStop.setStyleSheet(self.getButtonStyle("stop"))
            self.btnStop.setToolTip("Stop")
        else:
            self.btnStop.setStyleSheet(self.getButtonStyle("play"))
            self.btnStop.setToolTip("Play")

    def btnStopClicked(self):
        if self.mw.playing:
            self.mw.stopMedia()
        else:
            if self.lstCamera.count() > 0 and self.lstCamera.currentRow() > -1:
                onvif_data = self.devices[self.lstCamera.currentRow()]
                self.mw.connecting = True
                self.mw.playMedia(self.getStreamURI(onvif_data))
        self.setBtnStop()

    def onMediaStopped(self):
        self.setBtnStop()
        self.setBtnRecord()

    def getStreamURI(self, onvif_data):
        uri = onvif_data.stream_uri()[0 : 7] + onvif_data.username() + ":" \
            + onvif_data.password() + "@" + onvif_data.stream_uri()[7:]
        return uri

    def applyTimeout(self):
        if self.lstCamera.currentRow() < 0:
            onvif_data = onvif.Data()
            self.signals.fill.emit(onvif_data)
            self.setTabsEnabled(False)
        self.setEnabled(True)
        self.btnDiscover.setEnabled(True)

    def getButtonStyle(self, name):
        strStyle = "QPushButton { image : url(image:%1.png); } \
                    QPushButton:hover { image : url(image:%1_hi.png); } \
                    QPushButton:pressed { image : url(image:%1_lo.png); } \
                    QPushButton:disabled { image : url(image:%1_lo.png); }"
        strStyle = strStyle.replace("%1", name)
        return strStyle
