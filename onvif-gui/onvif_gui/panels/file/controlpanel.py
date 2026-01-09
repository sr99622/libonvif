#/********************************************************************
# onvif-gui/onvif_gui/panels/file/controlpanel.py 
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
    QWidget, QSlider, QFileDialog, QMessageBox
from PyQt6.QtCore import Qt, QStandardPaths
import sys
import platform
from datetime import datetime
from loguru import logger
from pathlib import Path
from time import sleep
from .searchdialog import FileSearchDialog

class FileControlPanel(QWidget):
    def __init__(self, mw, panel):
        super().__init__()
        self.mw = mw
        self.panel = panel
        self.hideCameraKey = "filePanel/hideCameraPanel"

        self.dlgSearch = FileSearchDialog(self.mw)

        self.btnSearch = QPushButton()
        self.btnSearch.setStyleSheet(self.getButtonStyle("search"))
        self.btnSearch.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btnSearch.clicked.connect(self.btnSearchClicked)

        self.btnSnapshot = QPushButton()
        self.btnSnapshot.setStyleSheet(self.getButtonStyle("snapshot"))
        self.btnSnapshot.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btnSnapshot.clicked.connect(self.btnSnapshotClicked)

        self.btnRefresh = QPushButton()
        self.btnRefresh.setStyleSheet(self.getButtonStyle("refresh"))
        self.btnRefresh.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btnRefresh.clicked.connect(self.btnRefreshClicked)

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

        self.btnMute = QPushButton()
        self.btnMute.setStyleSheet(self.getButtonStyle("mute"))
        self.btnMute.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btnMute.clicked.connect(self.btnMuteClicked)

        self.sldVolume = QSlider(Qt.Orientation.Horizontal)
        self.sldVolume.setValue(80)
        self.sldVolume.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.sldVolume.valueChanged.connect(self.sldVolumeChanged)

        lytMain =  QGridLayout(self)
        lytMain.addWidget(self.btnSearch,       0, 0, 1, 1)
        lytMain.addWidget(self.btnRefresh,      0, 1, 1, 1)
        lytMain.addWidget(self.btnSnapshot,     0, 2, 1, 1)
        lytMain.addWidget(self.btnPrevious,     1, 0, 1, 1)
        lytMain.addWidget(self.btnPlay,         1, 1, 1, 1)
        lytMain.addWidget(self.btnNext,         1, 2, 1, 1)
        lytMain.addWidget(self.btnStop,         1, 3, 1, 1)
        lytMain.addWidget(self.btnMute,         1, 4, 1, 1)
        lytMain.addWidget(self.sldVolume,       1, 5, 1, 1)
        lytMain.setColumnStretch(5, 10)
        lytMain.setContentsMargins(0, 0, 0, 0)

    def btnStopClicked(self):
        self.mw.closeAnyPlayingFiles()
        self.setBtnPlay()

    def btnPlayClicked(self):
        if type(self.panel).__name__ == "FilePanel":
            self.startPlayer()
        else:
            self.mw.picturePanel.playVideo()

    def startPlayer(self, file_start_from_seek=-1.0):
        uri = self.mw.filePanel.getCurrentFileURI()
        if player := self.mw.pm.getPlayer(uri):
            player.togglePaused()
        else:
            self.mw.closeAnyPlayingFiles()
            self.mw.playMedia(uri, file_start_from_seek=file_start_from_seek)
            if uri: self.mw.glWidget.focused_uri = uri

        self.setBtnPlay()

    def setBtnPlay(self):
        self.btnPlay.setStyleSheet(self.getButtonStyle("play"))
        if player := self.mw.pm.getPlayer(self.mw.filePanel.getCurrentFileURI()):
            if not player.isPaused():
                self.btnPlay.setStyleSheet(self.getButtonStyle("pause"))

    def btnMuteClicked(self):
        if player := self.mw.pm.getPlayer(self.mw.filePanel.getCurrentFileURI()):
            player.setMute(not player.isMuted())
            self.mw.filePanel.setMute(player.isMuted())
        else:
            self.mw.filePanel.setMute(not self.mw.filePanel.getMute())
        self.setBtnMute()

    def setBtnMute(self):
        self.btnMute.setStyleSheet(self.getButtonStyle("mute"))
        if not self.mw.filePanel.getMute():
            self.btnMute.setStyleSheet(self.getButtonStyle("audio"))

    def btnPreviousClicked(self):
        if type(self.panel).__name__ == "FilePanel":
            tree = self.mw.filePanel.tree
        else:
            tree = self.mw.picturePanel.tree
        index = tree.currentIndex()
        if index.isValid():
            prevIndex = tree.indexAbove(index)
            if prevIndex.isValid():
                tree.setCurrentIndex(prevIndex)
                tree.scrollTo(prevIndex)
                if type(self.panel).__name__ == "FilePanel":
                    self.mw.closeAnyPlayingFiles()                
                    if tree.model().fileInfo(prevIndex).isFile():
                        uri = tree.model().fileInfo(prevIndex).filePath()
                        self.mw.playMedia(uri)
                        self.mw.glWidget.focused_uri = uri

    def btnNextClicked(self):
        if type(self.panel).__name__ == "FilePanel":
            tree = self.mw.filePanel.tree
        else:
            tree = self.mw.picturePanel.tree
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
                if type(self.panel).__name__ == "FilePanel":
                    self.mw.closeAnyPlayingFiles()
                    if tree.model().fileInfo(nextIndex).isFile():
                        uri = tree.model().fileInfo(nextIndex).filePath()
                        if uri:
                            self.mw.playMedia(uri)
                            self.mw.glWidget.focused_uri = uri

    def btnSearchClicked(self):
        camera_names = []
        path = self.mw.filePanel.dirArchive.txtDirectory.text()
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
        self.panel.refresh()

    def btnSnapshotClicked(self):
        try:
            if player := self.mw.filePanel.getCurrentlyPlayingFile():
                picture_dir = QStandardPaths.standardLocations(QStandardPaths.StandardLocation.PicturesLocation)[0]
                filename = '{0:%Y%m%d%H%M%S.jpg}'.format(datetime.now())
                filepath = Path(picture_dir) / filename
                if platform.system() == "Linux":
                    filename = QFileDialog.getSaveFileName(self, "Save File As", str(filepath), options=QFileDialog.Option.DontUseNativeDialog)[0]
                else:
                    filename = QFileDialog.getSaveFileName(self, "Save File As", str(filepath))[0]

                answer = QMessageBox.StandardButton.Yes
                if not filename.endswith(".jpg"):
                    filename += ".jpg"
                    if Path(filename).is_file():
                        answer = QMessageBox.question(self.mw, "File Exists", "You are about to overwrite an existing file, are you sure you wnat to do this?")
                if answer == QMessageBox.StandardButton.Yes:
                    player.image.save(filename)
                    logger.debug(f'Snapshot saved as {filename}')
        except Exception as ex:
            logger.error(f"ControlPanel btnSnapshotClicked exception: {ex}")

    def sldVolumeChanged(self, value):
        self.mw.filePanel.setVolume(value)
        if player := self.mw.filePanel.getCurrentlyPlayingFile():
            player.setVolume(value)

    def setSldVolume(self):
        self.sldVolume.setValue(self.mw.filePanel.getVolume())

    def getButtonStyle(self, name):
        strStyle = "QPushButton { image : url(image:%1.png); } QPushButton:hover { image : url(image:%1_hi.png); } QPushButton:pressed { image : url(image:%1_lo.png); }"
        strStyle = strStyle.replace("%1", name)
        return strStyle
