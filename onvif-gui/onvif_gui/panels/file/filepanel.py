#/********************************************************************
# libonvif/onvif-gui/onvif_gui/panels/file/filepanel.py 
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

import os
import sys
from PyQt6.QtWidgets import QGridLayout, QWidget, QLabel, \
    QMessageBox, QMenu, QApplication
from PyQt6.QtGui import QAction, QFileSystemModel
from PyQt6.QtCore import Qt, QStandardPaths
from onvif_gui.components import Progress
from loguru import logger
import avio
from . import FileControlPanel
from onvif_gui.components.directoryselector import DirectorySelector
from . import TreeView

class FilePanel(QWidget):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw
        self.videoModelSettings = None
        self.audioModelSettings = None
        self.alarmSoundVolume = 80
        self.expandedPaths = []
        self.loadedCount = 0
        self.restorationPath = None
        self.verticalScrollBarPosition = 0

        if self.mw.parent_window:
            video_dir = self.mw.parent_window.settingsPanel.storage.dirArchive.text()
        else:
            video_dir = QStandardPaths.standardLocations(QStandardPaths.StandardLocation.MoviesLocation)[0]
        self.dirArchive = DirectorySelector(mw, self.mw.settingsPanel.storage.archiveKey, "", video_dir)
        self.dirArchive.signals.dirChanged.connect(self.mw.settingsPanel.storage.dirArchiveChanged)

        self.model = QFileSystemModel()
        self.model.fileRenamed.connect(self.onFileRenamed)
        self.model.directoryLoaded.connect(self.loaded)
        self.tree = TreeView(mw)
        self.tree.setModel(self.model)
        self.tree.doubleClicked.connect(self.treeDoubleClicked)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self.showContextMenu)

        self.progress = Progress(mw)
        self.control = FileControlPanel(mw)

        lytMain = QGridLayout(self)
        lytMain.addWidget(self.dirArchive,  0, 0, 1, 1)
        lytMain.addWidget(self.tree,       1, 0, 1, 1)
        lytMain.addWidget(self.progress,   2, 0, 1, 1)
        lytMain.addWidget(QLabel(),        3, 0, 1, 1)
        lytMain.addWidget(self.control,    4, 0, 1, 1)
        lytMain.setRowStretch(1, 10)

        self.dirChanged(self.dirArchive.text())

        self.menu = QMenu("Context Menu", self)
        self.remove = QAction("Delete", self)
        self.rename = QAction("Rename", self)
        self.info = QAction("Info", self)
        self.play = QAction("Play", self)
        self.stop = QAction("Stop", self)
        self.remove.triggered.connect(self.onMenuRemove)
        self.rename.triggered.connect(self.onMenuRename)
        self.info.triggered.connect(self.onMenuInfo)
        self.play.triggered.connect(self.onMenuPlay)
        self.stop.triggered.connect(self.onMenuStop)
        self.menu.addAction(self.remove)
        self.menu.addAction(self.rename)
        self.menu.addAction(self.info)
        self.menu.addAction(self.play)
        self.menu.addAction(self.stop)

    def loaded(self, path):
        self.loadedCount += 1
        self.model.sort(0)
        for i in range(self.model.rowCount(self.model.index(path))):
            idx = self.model.index(i, 0, self.model.index(path))
            if idx.isValid():
                if self.model.filePath(idx) in self.expandedPaths:
                    self.tree.setExpanded(idx, True)

        if len(self.expandedPaths):
            if self.loadedCount == len(self.expandedPaths) + 1:
                if self.verticalScrollBarPosition:
                    QApplication.processEvents()
                    self.tree.verticalScrollBar().setValue(self.verticalScrollBarPosition)
                self.expandedPaths.clear()
                self.restoreSelectedPath()
        else:
            self.restoreSelectedPath()

    def restoreSelectedPath(self):
        if self.restorationPath:
            idx = self.model.index(self.restorationPath)
            if idx.isValid():
                self.tree.setCurrentIndex(idx)
            self.restorationPath = None

    def refresh(self):
        self.loadedCount = 0
        self.expandedPaths = []
        self.restorationPath = self.model.filePath(self.tree.currentIndex())
        path = self.dirArchive.txtDirectory.text()
        self.model.sort(0)
        for i in range(self.model.rowCount(self.model.index(path))):
            idx = self.model.index(i, 0, self.model.index(path))
            if idx.isValid():
                if self.tree.isExpanded(idx):
                    self.expandedPaths.append(self.model.filePath(idx))
        self.verticalScrollBarPosition = self.tree.verticalScrollBar().value()
        self.model = QFileSystemModel()
        self.model.setRootPath(path)
        self.model.fileRenamed.connect(self.onFileRenamed)
        self.model.directoryLoaded.connect(self.loaded)
        self.tree.setModel(self.model)
        self.tree.setRootIndex(self.model.index(path))

    def dirChanged(self, path):
        if len(path) > 0:
            self.model.setRootPath(path)
            self.tree.setRootIndex(self.model.index(path))

    def treeDoubleClicked(self, index):
        if index.isValid():
            fileInfo = self.model.fileInfo(index)
            if fileInfo.isDir():
                self.tree.setExpanded(index, self.tree.isExpanded(index))
            else:
                for player in self.mw.pm.players:
                    if not player.isCameraStream():
                        self.mw.pm.playerShutdownWait(player.uri)
                uri = self.getCurrentFileURI()
                if uri:
                    self.mw.playMedia(uri)
                    self.mw.glWidget.focused_uri = uri

    def onMediaStarted(self, duration):
        if self.mw.tab.currentIndex() == 1:
            self.tree.setFocus()
        self.control.setBtnPlay()
        self.control.setBtnMute()
        self.control.setSldVolume()

    def onMediaStopped(self, uri):
        self.control.setBtnPlay()
        self.progress.updateProgress(0.0)

        another = None
        for player in self.mw.pm.players:
            if not player.isCameraStream():
                another = player

        if not another:
            self.progress.lblDuration.setText("0:00")

    def onMediaProgress(self, pct, uri):
        player = self.mw.pm.getPlayer(uri)
        if player is not None:
            player.file_progress = pct

        if pct >= 0.0 and pct <= 1.0:
            if uri == self.mw.glWidget.focused_uri:
                if player is not None:
                    self.progress.updateDuration(player.duration)
                self.progress.updateProgress(pct)

    def showContextMenu(self, pos):
        player = self.mw.pm.getPlayer(self.getCurrentFileURI())
        self.remove.setDisabled(bool(player))
        self.rename.setDisabled(bool(player))
        index = self.tree.indexAt(pos)
        if index.isValid():
            fileInfo = self.model.fileInfo(index)
            if fileInfo.isFile():
                self.menu.exec(self.mapToGlobal(pos))

    def onMenuRemove(self):
        index = self.tree.currentIndex()
        if index.isValid():
            if self.mw.pm.getPlayer(self.model.filePath(index)):
                QMessageBox.warning(self, "Warning",
                                        "Camera is currently playing. Please stop before deleting.",
                                        QMessageBox.StandardButton.Ok)
                return
            
            ret = QMessageBox.warning(self, "onvif-gui",
                                        "You are about to ** PERMANENTLY ** delete this file.\n"
                                        "Are you sure you want to continue?",
                                        QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)

            if ret == QMessageBox.StandardButton.Ok:
                try:
                    idxAbove = self.tree.indexAbove(index)
                    idxBelow = self.tree.indexBelow(index)

                    self.model.remove(index)
                    
                    resolved = False
                    if idxAbove.isValid():
                        if os.path.isfile(self.model.filePath(idxAbove)):
                            self.tree.setCurrentIndex(idxAbove)
                            resolved = True
                    if not resolved:
                        if idxBelow.isValid():
                            if os.path.isfile(self.model.filePath(idxBelow)):
                                self.tree.setCurrentIndex(idxBelow)
                
                except Exception as e:
                    logger.error(f'File delete error: {e}')

    def onMenuRename(self):
        player = self.mw.pm.getPlayer(self.mw.filePanel.getCurrentFileURI())
        if player:
            self.mw.onError("Please stop the file playing in order to rename")
            return
        index = self.tree.currentIndex()
        if index.isValid():
            self.model.setReadOnly(False)
            self.tree.edit(index)

    def onFileRenamed(self, path, oldName, newName):
        self.model.setReadOnly(True)

    def onMenuInfo(self):
        strInfo = ""
        try:
            index = self.tree.currentIndex()
            if (index.isValid()):
                info = self.model.fileInfo(index)
                strInfo += "Filename: " + info.fileName()
                strInfo += "\nCreated: " + info.birthTime().toString()
                strInfo += "\nModified: " + info.lastModified().toString()

                reader = avio.Reader(info.absoluteFilePath(), None)
                duration = reader.duration()
                time_in_seconds = int(duration / 1000)
                hours = int(time_in_seconds / 3600)
                minutes = int((time_in_seconds - (hours * 3600)) / 60)
                seconds = int((time_in_seconds - (hours * 3600) - (minutes * 60)))
                strInfo += "\nDuration: " + str(minutes) + ":" + "{:02d}".format(seconds)
                title = reader.metadata("title")
                if len(title):
                    strInfo += "\nTitle: " + reader.metadata("title")

                if (reader.has_video()):
                    strInfo += "\n\nVideo Stream:"
                    strInfo += "\n    Resolution:  " + str(reader.width()) + " x " + str(reader.height())
                    strInfo += "\n    Frame Rate:  " + f'{reader.frame_rate().num / reader.frame_rate().den:.2f}'
                    strInfo += "  (" + str(reader.frame_rate().num) + " / " + str(reader.frame_rate().den) +")"
                    strInfo += "\n    Time Base:  " + str(reader.video_time_base().num) + " / " + str(reader.video_time_base().den)
                    strInfo += "\n    Video Codec:  " + reader.str_video_codec()
                    strInfo += "\n    Pixel Format:  " + reader.str_pix_fmt()
                    strInfo += "\n    Bitrate:  " + f'{reader.video_bit_rate():,}'
                
                if (reader.has_audio()):
                    strInfo += "\n\nAudio Stream:"
                    strInfo += "\n    Channel Layout:  " + reader.str_channel_layout()
                    strInfo += "\n    Audio Codec:  " + reader.str_audio_codec()
                    strInfo += "\n    Sample Rate:  " + str(reader.sample_rate())
                    strInfo += "\n    Sample Size:  " + str(reader.frame_size())
                    strInfo += "\n    Time Base:  " + str(reader.audio_time_base().num) + " / " + str(reader.audio_time_base().den)
                    strInfo += "\n    Sample Format:  " + reader.str_sample_format()
                    strInfo += "\n    Bitrate:  " + f'{reader.audio_bit_rate():,}'
                
            else:
                strInfo = "Invalid Index"
        except Exception as ex:
            strInfo = f'Unable to read file info: {ex}'

        msgBox = QMessageBox(self)
        msgBox.setWindowTitle("File Info")
        msgBox.setText(strInfo)
        msgBox.exec()

    def onMenuPlay(self):
        self.mw.filePanel.control.btnPlayClicked()

    def onMenuStop(self):
        self.mw.filePanel.control.btnStopClicked()

    def fastForward(self):
        pct = self.progress.sldProgress.value() / 1000
        if duration := self.progress.duration:
            interval = 10000 / duration
            tgt = pct + interval
            if tgt < 1.0:
                if player := self.getCurrentlyPlayingFile():
                    player.seek(tgt)

    def rewind(self):
        pct = self.progress.sldProgress.value() / 1000
        if duration := self.progress.duration:
            interval = 10000 / duration
            tgt = max(pct - interval, 0.0)
            if player := self.getCurrentlyPlayingFile():
                player.seek(tgt)

    def getCurrentFileURI(self):
        result = None
        index = self.tree.currentIndex()
        if index.isValid():
            info = self.model.fileInfo(index)
            if info.isFile():
                result = info.absoluteFilePath()
        return result
    
    def getCurrentlyPlayingFile(self):
        result = None
        for player in self.mw.pm.players:
            if not player.isCameraStream():
                result = player
                break
        return result
            
    def setCurrentFile(self, uri):
        index = self.model.index(uri)
        self.tree.setCurrentIndex(index)
        self.control.setBtnPlay()
        self.control.setBtnMute()
        self.control.setSldVolume()
        player = self.mw.pm.getPlayer(uri)
        if player:
            self.onMediaProgress(player.file_progress, uri)

    def showEvent(self, event):
        self.restoreHeader()

    def headerChanged(self, a, b, c):
        key = f'File/Header'
        self.mw.settings.setValue(key, self.tree.header().saveState())

    def restoreHeader(self):
        key = f'File/Header'
        data = self.mw.settings.value(key)
        if data:
            self.tree.header().restoreState(data)
        self.tree.update()
        self.tree.header().sectionResized.connect(self.headerChanged)
        self.tree.header().sectionMoved.connect(self.headerChanged)

    def getDirectory(self):
        try:
            if sys.platform == "win32":
                path = os.environ["HOMEPATH"]
            else:
                path = os.environ["HOME"]
            key = f'File/Directory'
            dirs = QStandardPaths.standardLocations(QStandardPaths.StandardLocation.MoviesLocation)
            path = self.mw.settings.value(key, dirs[0])
            os.makedirs(path, exist_ok=True)
        except Exception as ex:
            logger.error(f'Unable to find Videos directory: {ex}')
        return path

    def setDirectory(self, path):
        key = f'File/Directory'
        self.mw.settings.setValue(key, path)

    def getMute(self):
        key = f'File/Mute'
        return bool(int(self.mw.settings.value(key, 0)))
    
    def setMute(self, state):
        key = f'File/Mute'
        self.mw.settings.setValue(key, int(state))

    def getVolume(self):
        key = f'File/Volume'
        return int(self.mw.settings.value(key, 80))
    
    def setVolume(self, volume):
        key = f'File/Volume'
        self.mw.settings.setValue(key, volume)
