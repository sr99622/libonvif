#/********************************************************************
# libonvif/onvif-gui/gui/panels/filepanel.py 
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
import platform
from PyQt6.QtWidgets import QLineEdit, QPushButton, \
    QGridLayout, QWidget, QSlider, QLabel, QMessageBox, \
    QTreeView, QFileDialog, QMenu, QAbstractItemView
from PyQt6.QtGui import QFileSystemModel, QAction, QIcon
from PyQt6.QtCore import Qt, QStandardPaths, QObject, pyqtSignal
from gui.components import Progress
from gui.onvif import MediaSource
from loguru import logger
import avio

class TreeViewSignals(QObject):
    selectionChanged = pyqtSignal(str)

class TreeModel(QFileSystemModel):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw
        self.ref = None

    def data(self, index, role):
        if index.isValid():

            player = None
            if self.ref:
                if self.ref.isValid():
                    info = self.fileInfo(self.ref)
                    if info.isFile():
                        uri = info.filePath()
                        player = self.mw.pm.getPlayer(uri)

            condition = role == Qt.ItemDataRole.DecorationRole and \
                index.column() == 0 and \
                self.ref == index and \
                player
            
            if condition:
                return QIcon("image:play.png")

        return super().data(index, role)

class TreeView(QTreeView):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw
        self.signals = TreeViewSignals()
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

    def keyPressEvent(self, event):

        pass_along = True

        match event.key():

            case Qt.Key.Key_Return:
                index = self.currentIndex()
                if index.isValid():
                    fileInfo = self.model().fileInfo(index)
                    if fileInfo.isFile():
                        if self.model().isReadOnly():
                            for player in self.mw.pm.players:
                                if not player.isCameraStream():
                                    self.mw.pm.playerShutdownWait(player.uri)
                            self.mw.filePanel.control.btnPlayClicked()
                    else:
                        if self.isExpanded(index):
                            self.collapse(index)
                        else:
                            self.expand(index)

            case Qt.Key.Key_Space:
                index = self.currentIndex()
                if index.isValid():
                    fileInfo = self.model().fileInfo(index)
                    if fileInfo.isFile():
                        if self.model().isReadOnly():
                            self.mw.filePanel.control.btnPlayClicked()

            case Qt.Key.Key_Escape:
                if self.model().isReadOnly():
                    self.mw.filePanel.control.btnStopClicked()
                else:
                    self.model().setReadOnly(True)
        
            case Qt.Key.Key_F1:
                self.mw.filePanel.onMenuInfo()

            case Qt.Key.Key_F2:
                self.mw.filePanel.onMenuRename()

            case Qt.Key.Key_Delete:
                self.mw.filePanel.onMenuRemove()

            case Qt.Key.Key_Left:
                pct = self.mw.filePanel.progress.sldProgress.value() / 1000
                duration = self.mw.filePanel.progress.duration
                interval = 10000 / duration
                tgt = max(pct - interval, 0.0)
                player = self.mw.pm.getPlayer(self.mw.filePanel.getCurrentFileURI())
                if player:
                    player.seek(tgt)
                pass_along = False

            case Qt.Key.Key_Right:
                pct = self.mw.filePanel.progress.sldProgress.value() / 1000
                duration = self.mw.filePanel.progress.duration
                interval = 10000 / duration
                tgt = pct + interval
                if tgt < 1.0:
                    player = self.mw.pm.getPlayer(self.mw.filePanel.getCurrentFileURI())
                    if player:
                        player.seek(tgt)
                pass_along = False
        
        if pass_along:
            return super().keyPressEvent(event)

    def currentChanged(self, newItem, oldItem):
        if newItem.data():
            fullPath = os.path.join(self.model().rootPath(), newItem.data())
            if os.path.isfile(fullPath):
                player = self.mw.pm.getPlayer(str(fullPath))
                if player:
                    self.mw.glWidget.focused_uri = player.uri
            self.signals.selectionChanged.emit(fullPath)
            self.scrollTo(self.currentIndex())

class DirectorySetter(QWidget):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw
        self.txtDirectory = QLineEdit()
        self.btnSelect = QPushButton("...")
        self.btnSelect.clicked.connect(self.btnSelectClicked)
        self.dlgFile = QFileDialog()
        lytMain = QGridLayout(self)
        lytMain.setContentsMargins(0, 0, 0, 0)
        lytMain.addWidget(self.txtDirectory,   0, 0, 1, 1)
        lytMain.addWidget(self.btnSelect,      0, 1, 1, 1)
        lytMain.setColumnStretch(0, 10)
        self.setContentsMargins(0, 0, 0, 0)

    def btnSelectClicked(self):
        path = None
        if platform.system() == "Linux":
            path = QFileDialog.getExistingDirectory(self, "Select Directory", self.txtDirectory.text(), QFileDialog.Option.DontUseNativeDialog)
        else:
            path = QFileDialog.getExistingDirectory(self, "Select Directory", self.txtDirectory.text())
        if path:
            self.txtDirectory.setText(path)
            self.mw.filePanel.dirChanged(path)

class FileControlPanel(QWidget):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw

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

        spacer = QLabel()
        spacer.setMinimumWidth(self.btnStop.minimumWidth())
        
        self.btnMute = QPushButton()
        self.btnMute.setStyleSheet(self.getButtonStyle("mute"))
        self.btnMute.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btnMute.clicked.connect(self.btnMuteClicked)

        self.sldVolume = QSlider(Qt.Orientation.Horizontal)
        self.sldVolume.setValue(80)
        self.sldVolume.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.sldVolume.valueChanged.connect(self.sldVolumeChanged)
        self.sldVolume.setEnabled(False)

        lytMain =  QGridLayout(self)
        lytMain.addWidget(self.btnPrevious, 0, 0, 1, 1)
        lytMain.addWidget(self.btnPlay,     0, 1, 1, 1)
        lytMain.addWidget(self.btnNext,     0, 2, 1, 1)
        lytMain.addWidget(self.btnStop,     0, 3, 1, 1)
        lytMain.addWidget(self.btnMute,     0, 4, 1, 1)
        lytMain.addWidget(self.sldVolume,   0, 5, 1, 1)
        lytMain.setColumnStretch(5, 10)
        lytMain.setContentsMargins(0, 0, 0, 0)

    def btnStopClicked(self):
        for player in self.mw.pm.players:
            if not player.isCameraStream():
                player.requestShutdown()
        self.setBtnPlay()

    def btnPlayClicked(self):
        tree = self.mw.filePanel.tree
        tree.model().ref = tree.currentIndex()

        players = []
        for player in self.mw.pm.players:
            if not player.isCameraStream():
                players.append(player)

        if len(players):
            players[0].togglePaused()
        else:
            uri = self.mw.filePanel.getCurrentFileURI()
            if uri:
                self.mw.playMedia(uri)
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
            self.mw.filePanel.setMute(not player.isMuted())
        else:
            self.mw.filePanel.setMute(not self.mw.filePanel.getMute())
        self.setBtnMute()

    def setBtnMute(self):
        self.btnMute.setStyleSheet(self.getButtonStyle("mute"))
        self.sldVolume.setEnabled(False)
        player = self.mw.pm.getPlayer(self.mw.filePanel.getCurrentFileURI())
        if player:
            if not player.isMuted():
                self.btnMute.setStyleSheet(self.getButtonStyle("audio"))
                self.sldVolume.setEnabled(True)
        else:
            if not self.mw.filePanel.getMute():
                self.btnMute.setStyleSheet(self.getButtonStyle("audio"))
                self.sldVolume.setEnabled(True)

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

class FilePanelSignals(QObject):
    removeFile = pyqtSignal(str)
    renameFile = pyqtSignal(str, str)

class FilePanel(QWidget):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw
        self.videoModelSettings = None
        self.audioModelSettings = None
        self.alarmSoundVolume = 80

        self.signals = FilePanelSignals()
        self.signals.removeFile.connect(self.removeFile)

        self.dirSetter = DirectorySetter(mw)
        self.dirSetter.txtDirectory.setText(self.getDirectory())

        self.model = TreeModel(mw)
        self.model.fileRenamed.connect(self.onFileRenamed)
        self.tree = TreeView(mw)
        self.tree.setModel(self.model)
        self.tree.clicked.connect(self.treeClicked)
        self.tree.doubleClicked.connect(self.treeDoubleClicked)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self.showContextMenu)

        self.progress = Progress(mw)
        self.control = FileControlPanel(mw)

        lytMain = QGridLayout(self)
        lytMain.addWidget(self.dirSetter,  0, 0, 1, 1)
        lytMain.addWidget(self.tree,       1, 0, 1, 1)
        lytMain.addWidget(self.progress,   2, 0, 1, 1)
        lytMain.addWidget(QLabel(),        3, 0, 1, 1)
        lytMain.addWidget(self.control,    4, 0, 1, 1)
        lytMain.setRowStretch(1, 10)

        self.dirSetter.txtDirectory.textEdited.connect(self.dirChanged)
        self.dirChanged(self.dirSetter.txtDirectory.text())

        self.menu = QMenu("Context Menu", self)
        self.remove = QAction("Delete", self)
        self.rename = QAction("Rename", self)
        self.info = QAction("Info", self)
        self.play = QAction("Play", self)
        self.remove.triggered.connect(self.onMenuRemove)
        self.rename.triggered.connect(self.onMenuRename)
        self.info.triggered.connect(self.onMenuInfo)
        self.play.triggered.connect(self.onMenuPlay)
        self.menu.addAction(self.remove)
        self.menu.addAction(self.rename)
        self.menu.addAction(self.info)
        self.menu.addAction(self.play)

    def dirChanged(self, path):
        if len(path) > 0:
            self.model.setRootPath(path)
            self.tree.setRootIndex(self.model.index(path))
            self.setDirectory(path)

    def treeClicked(self, index):
        if index.isValid():
            fileInfo = self.model.fileInfo(index)
            self.mw.videoConfigure.setFile(fileInfo.canonicalFilePath())

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
                    self.tree.model().ref = self.tree.currentIndex()
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

        if self.mw.glWidget.focused_uri == uri:
            if self.mw.videoPanel.chkEnableFile.isChecked():
                if self.mw.videoWorker:
                    self.mw.videoWorker(None, None)
            if self.mw.audioPanel.chkEnableFile.isChecked():
                if self.mw.audioWorker:
                    self.mw.audioWorker(None, None)

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
        ret = QMessageBox.warning(self, "onvif-gui",
                                    "You are about to delete this file.\n"
                                    "Are you sure you want to continue?",
                                    QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)

        if ret == QMessageBox.StandardButton.Ok:

            indexes = [index for index in self.tree.selectedIndexes() if index.column() == 0]
            for i, index in enumerate(indexes):
                if index.isValid():
                    if i == 0:
                        idxAbove = self.tree.indexAbove(index)
                        idxBelow = self.tree.indexBelow(index)
                        resolved = False
                        if idxAbove.isValid():
                            if os.path.isfile(self.model.filePath(idxAbove)) or len(indexes) > 1:
                                self.tree.setCurrentIndex(idxAbove)
                                resolved = True
                        if not resolved:
                            if idxBelow.isValid():
                                self.tree.setCurrentIndex(idxBelow)
                    filename = self.model.filePath(index)
                    player = self.mw.pm.getPlayer(filename)
                    if player:
                        self.mw.pm.playerShutdownWait(player.uri)
                    self.signals.removeFile.emit(filename)

    def removeFile(self, filename):
        try:
            os.remove(filename)
        except Exception as e:
            msg = f'File delete exception {str(e)}'
            logger.debug(msg)
            self.mw.onError(msg)

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
        index = self.tree.currentIndex()
        if (index.isValid()):
            info = self.model.fileInfo(index)
            strInfo = ""
            strInfo += "Filename: " + info.fileName()
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

        msgBox = QMessageBox(self)
        msgBox.setWindowTitle("")
        msgBox.setText(strInfo)
        msgBox.exec()

    def onMenuPlay(self):
        index = self.tree.currentIndex()
        if (index.isValid()):
            info = self.model.fileInfo(index)
            self.mw.playMedia(info.absoluteFilePath())

    def getCurrentFileURI(self):
        result = None
        index = self.tree.currentIndex()
        if index.isValid():
            info = self.model.fileInfo(index)
            if info.isFile():
                result = info.filePath()
        return result
            
    def setCurrentFile(self, uri):
        index = self.model.index(uri)
        self.tree.setCurrentIndex(index)
        self.control.setBtnPlay()
        self.control.setBtnMute()
        self.control.setSldVolume()
        if self.mw.videoConfigure.source != MediaSource.FILE:
            if uri:
                self.mw.videoConfigure.setFile(uri)
        if self.mw.audioConfigure.source != MediaSource.FILE:
            if uri:
                self.mw.audioConfigure.setFile(uri)
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
        key = f'File/Directory'
        dirs = QStandardPaths.standardLocations(QStandardPaths.StandardLocation.MoviesLocation)
        return self.mw.settings.value(key, dirs[0])
    
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

    def getAnalyzeVideo(self):
        key = f'File/AnalyzeVideo'
        return bool(int(self.mw.settings.value(key, 0)))
        
    def setAnalyzeVideo(self, state):
        key = f'File/AnalyzeVideo'
        self.mw.settings.setValue(key, int(state))

    def getAnalyzeAudio(self):
        key = f'File/AnalyzeAudio'
        return bool(int(self.mw.settings.value(key, 0)))
        
    def setAnalyzeAudio(self, state):
        key = f'File/AnalyzeAudio'
        self.mw.settings.setValue(key, int(state))
