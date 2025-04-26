#/********************************************************************
# libonvif/onvif-gui/onvif_gui/panels/file/controlpanel.py 
#
# Copyright (c) 2025  Stephen Rhodes
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

import os
from PyQt6.QtWidgets import QPushButton, QGridLayout, \
    QWidget, QSlider, QCheckBox, QMessageBox
from PyQt6.QtCore import Qt
import sys
from time import sleep
from .searchdialog import FileSearchDialog

class FileControlPanel(QWidget):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw
        self.hideCameraKey = "filePanel/hideCameraPanel"

        self.dlgSearch = FileSearchDialog(self.mw)

        self.btnSearch = QPushButton()
        self.btnSearch.setStyleSheet(self.getButtonStyle("search"))
        self.btnSearch.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btnSearch.clicked.connect(self.btnSearchClicked)

        self.btnRefresh = QPushButton()
        self.btnRefresh.setStyleSheet(self.getButtonStyle("refresh"))
        self.btnRefresh.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btnRefresh.clicked.connect(self.btnRefreshClicked)

        self.chkHideCameras = QCheckBox("Hide Camera Panel")
        self.chkHideCameras.setChecked(bool(int(self.mw.settings.value(self.hideCameraKey, 0))))
        self.chkHideCameras.stateChanged.connect(self.chkHideCamerasChecked)

        self.btnPlay = QPushButton()
        self.btnPlay.setStyleSheet(self.getButtonStyle("play"))
        self.btnPlay.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btnPlay.clicked.connect(self.btnPlayClicked)
        
        self.btnStop = QPushButton()
        self.btnStop.setStyleSheet(self.getButtonStyle("stop"))
        self.btnStop.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btnStop.clicked.connect(self.btnStopClicked)

        self.btnPrevious = QPushButton()
        self.btnPrevious.setStyleSheet(self.getButtonStyle("previous"))
        self.btnPrevious.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btnPrevious.clicked.connect(self.btnPreviousClicked)

        self.btnNext = QPushButton()
        self.btnNext.setStyleSheet(self.getButtonStyle("next"))
        self.btnNext.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btnNext.clicked.connect(self.btnNextClicked)

        #spacer = QLabel()
        #spacer.setMinimumWidth(self.btnStop.minimumWidth())
        
        self.btnMute = QPushButton()
        self.btnMute.setStyleSheet(self.getButtonStyle("mute"))
        self.btnMute.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btnMute.clicked.connect(self.btnMuteClicked)

        self.sldVolume = QSlider(Qt.Orientation.Horizontal)
        self.sldVolume.setValue(80)
        self.sldVolume.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.sldVolume.valueChanged.connect(self.sldVolumeChanged)
        #self.sldVolume.setEnabled(False)

        lytMain =  QGridLayout(self)
        lytMain.addWidget(self.btnSearch,       0, 0, 1, 1)
        lytMain.addWidget(self.btnRefresh,      0, 1, 1, 1)
        lytMain.addWidget(self.chkHideCameras,  0, 3, 1, 3)
        lytMain.addWidget(self.btnPrevious,     1, 0, 1, 1)
        lytMain.addWidget(self.btnPlay,         1, 1, 1, 1)
        lytMain.addWidget(self.btnNext,         1, 2, 1, 1)
        lytMain.addWidget(self.btnStop,         1, 3, 1, 1)
        lytMain.addWidget(self.btnMute,         1, 4, 1, 1)
        lytMain.addWidget(self.sldVolume,       1, 5, 1, 1)
        lytMain.setColumnStretch(5, 10)
        lytMain.setContentsMargins(0, 0, 0, 0)

    def btnStopClicked(self):
        for player in self.mw.pm.players:
            if not player.isCameraStream():
                player.requestShutdown()
        self.setBtnPlay()

    def btnPlayClicked(self):
        self.startPlayer()

    def startPlayer(self, file_start_from_seek=-1.0):
        tree = self.mw.filePanel.tree
        tree.model().ref = tree.currentIndex()

        for player in self.mw.pm.players:
            if not player.isCameraStream():
                if player.uri != self.mw.filePanel.getCurrentFileURI():
                    if player.isPaused():
                        player.togglePaused()
                    player.requestShutdown()
                    while not player.stopped:
                        sleep(0.001)

        found = False
        for player in self.mw.pm.players:
            if player.uri == self.mw.filePanel.getCurrentFileURI():
                found = True
                player.togglePaused()
        if not found:
            if uri := self.mw.filePanel.getCurrentFileURI():
                self.mw.playMedia(uri, file_start_from_seek=file_start_from_seek)
                self.mw.glWidget.focused_uri = uri

        self.setBtnPlay()

    def setBtnPlay(self):
        self.btnPlay.setStyleSheet(self.getButtonStyle("play"))
        player = self.mw.pm.getPlayer(self.mw.filePanel.getCurrentFileURI())
        if player:
            if not player.isPaused():
                self.btnPlay.setStyleSheet(self.getButtonStyle("pause"))

    def btnMuteClicked(self):
        player = self.mw.pm.getPlayer(self.mw.filePanel.getCurrentFileURI())
        if player:
            player.setMute(not player.isMuted())
            self.mw.filePanel.setMute(player.isMuted())
        else:
            self.mw.filePanel.setMute(not self.mw.filePanel.getMute())
        self.setBtnMute()

    def setBtnMute(self):
        self.btnMute.setStyleSheet(self.getButtonStyle("mute"))
        #self.sldVolume.setEnabled(False)
        if not self.mw.filePanel.getMute():
            self.btnMute.setStyleSheet(self.getButtonStyle("audio"))
            #self.sldVolume.setEnabled(True)
        '''
        player = self.mw.pm.getPlayer(self.mw.filePanel.getCurrentFileURI())
        if player:
            if not player.isMuted():
                self.btnMute.setStyleSheet(self.getButtonStyle("audio"))
                self.sldVolume.setEnabled(True)
        else:
            if not self.mw.filePanel.getMute():
                self.btnMute.setStyleSheet(self.getButtonStyle("audio"))
                self.sldVolume.setEnabled(True)
        '''

    def btnPreviousClicked(self):
        tree = self.mw.filePanel.tree
        index = tree.currentIndex()
        if index.isValid():
            prevIndex = tree.indexAbove(index)
            if prevIndex.isValid():
                tree.setCurrentIndex(prevIndex)
                tree.scrollTo(prevIndex)

                for player in self.mw.pm.players:
                    if not player.isCameraStream():
                        self.mw.pm.playerShutdownWait(player.uri)
                
                if tree.model().fileInfo(prevIndex).isFile():
                    tree.model().ref = prevIndex
                    uri = tree.model().fileInfo(prevIndex).filePath()
                    self.mw.playMedia(uri)
                    self.mw.glWidget.focused_uri = uri

    def btnNextClicked(self):
        tree = self.mw.filePanel.tree
        index = tree.currentIndex()
        if index.isValid():
            fileInfo = tree.model().fileInfo(index)
            if fileInfo.isDir():
                if not tree.isExpanded(index):
                    tree.expand(index)
                    return
                
            nextIndex = tree.indexBelow(index)
            if nextIndex.isValid():
                tree.setCurrentIndex(nextIndex)
                tree.scrollTo(nextIndex)

                for player in self.mw.pm.players:
                    if not player.isCameraStream():
                        self.mw.pm.playerShutdownWait(player.uri)

                if tree.model().fileInfo(nextIndex).isFile():
                    tree.model().ref = nextIndex
                    uri = tree.model().fileInfo(nextIndex).filePath()
                    if uri:
                        self.mw.playMedia(uri)
                        self.mw.glWidget.focused_uri = uri

    def btnSearchClicked(self):
        camera_names = []
        path = self.mw.filePanel.dirSetter.txtDirectory.text()
        for name in os.listdir(path):
            if os.path.isdir(os.path.join(path, name)):
                valid = True
                if sys.platform == "win32" and name == "Captures":
                    valid = False
                if sys.platform == "darwin" and name == "TV":
                    valid = False
                if valid:
                    camera_names.append(name)
        camera_names = sorted(camera_names, key=lambda s: s.casefold())
        self.dlgSearch.cameras.clear()
        self.dlgSearch.cameras.addItems(camera_names)

        if not self.dlgSearch.positionInitialized:
            w = 240
            h = 320
            x = int(self.mw.x() + self.mw.width()/2 - w/2)
            y = int(self.mw.y() + self.mw.height()/2 - h/2)
            self.dlgSearch.move(x, y)
        self.dlgSearch.show()

    def btnRefreshClicked(self):
        self.mw.filePanel.refresh()

    def chkHideCamerasChecked(self, state):
        self.mw.settings.setValue(self.hideCameraKey, state)
        if state:
            self.mw.tab.removeTab(0)
        else:
            self.mw.tab.insertTab(0, self.mw.cameraPanel, "Cameras")

    def sldVolumeChanged(self, value):
        self.mw.filePanel.setVolume(value)
        player = self.mw.pm.getPlayer(self.mw.filePanel.getCurrentFileURI())
        if player:
            player.setVolume(value)

    def setSldVolume(self):
        volume = 80
        player = self.mw.pm.getPlayer(self.mw.filePanel.getCurrentFileURI())
        if player:
            volume = player.getVolume()
        else:
            volume = self.mw.filePanel.getVolume()
        self.sldVolume.setValue(volume)

    def getButtonStyle(self, name):
        strStyle = "QPushButton { image : url(image:%1.png); } QPushButton:hover { image : url(image:%1_hi.png); } QPushButton:pressed { image : url(image:%1_lo.png); }"
        strStyle = strStyle.replace("%1", name)
        return strStyle
