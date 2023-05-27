#/********************************************************************
# onvif-gui/gui/panels/filepanel.py 
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

import sys
import datetime
from PyQt6.QtWidgets import QLineEdit, QPushButton, \
QGridLayout, QWidget, QSlider, QLabel, QMessageBox, \
QTreeView, QFileDialog, QMenu
from PyQt6.QtGui import QFileSystemModel, QAction
from PyQt6.QtCore import Qt, QStandardPaths, QFile
from gui.components import Progress

import avio

ICON_SIZE = 26

class TreeView(QTreeView):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Return:
            if self.model().isReadOnly():
                self.doubleClicked.emit(self.currentIndex())
        if event.key() == Qt.Key.Key_F2:
            self.mw.filePanel.onMenuRename()
        if event.key() == Qt.Key.Key_Delete:
            self.mw.filePanel.onMenuRemove()
        return super().keyPressEvent(event)
    
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
        path = QFileDialog.getExistingDirectory(self, "Open Directory", self.txtDirectory.text())
        self.txtDirectory.setText(path)
        self.mw.filePanel.dirChanged(path)

class FileControlPanel(QWidget):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw

        self.btnPlay = QPushButton()
        self.btnPlay.setToolTipDuration(2000)
        self.btnPlay.setMinimumWidth(ICON_SIZE * 2)
        self.btnPlay.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btnPlay.clicked.connect(self.btnPlayClicked)
        self.setBtnPlay()
        
        self.btnStop = QPushButton()
        self.btnStop.setStyleSheet(self.getButtonStyle("stop"))
        self.btnStop.setToolTip("Stop")
        self.btnStop.setToolTipDuration(2000)
        self.btnStop.setMinimumWidth(ICON_SIZE * 2)
        self.btnStop.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btnStop.clicked.connect(self.btnStopClicked)

        self.btnRecord = QPushButton()
        self.btnRecord.setToolTip("Record")
        self.btnRecord.setToolTipDuration(2000)
        self.btnRecord.setMinimumWidth(ICON_SIZE * 2)
        self.btnRecord.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btnRecord.clicked.connect(self.btnRecordClicked)
        self.setBtnRecord()

        spacer = QLabel()
        spacer.setMinimumWidth(self.btnStop.minimumWidth())
        
        self.btnMute = QPushButton()
        self.setBtnMute()
        self.btnMute.setToolTip("Mute")
        self.btnMute.setToolTipDuration(2000)
        self.btnMute.setMinimumWidth(ICON_SIZE * 2)
        self.btnMute.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btnMute.clicked.connect(self.btnMuteClicked)

        self.sldVolume = QSlider(Qt.Orientation.Horizontal)
        self.sldVolume.setValue(int(self.mw.volume))
        self.sldVolume.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.sldVolume.valueChanged.connect(self.sldVolumeChanged)
        
        lytMain =  QGridLayout(self)
        lytMain.setContentsMargins(0, 0, 0, 0)
        lytMain.addWidget(self.btnPlay,   0, 0, 1, 1, Qt.AlignmentFlag.AlignCenter)
        lytMain.addWidget(self.btnStop,   0, 1, 1, 1, Qt.AlignmentFlag.AlignCenter)
        lytMain.addWidget(self.btnRecord, 0, 2, 1, 1, Qt.AlignmentFlag.AlignCenter)
        lytMain.addWidget(self.btnMute,   0, 3, 1, 1, Qt.AlignmentFlag.AlignCenter)
        lytMain.addWidget(self.sldVolume, 0, 4, 1, 1)

    def btnPlayClicked(self):
        if self.mw.playing:
            self.mw.player.togglePaused()
        else:
            index = self.mw.filePanel.tree.currentIndex()
            if index.isValid():
                fileInfo = self.mw.filePanel.model.fileInfo(index)
                self.mw.playMedia(fileInfo.filePath())
        self.setBtnPlay()

    def btnStopClicked(self):
        self.mw.stopMedia()
        self.btnPlay.setStyleSheet(self.getButtonStyle("play"))
        self.mw.filePanel.tree.setFocus()

    def setBtnRecord(self):
        if self.mw.player is not None:
            if self.mw.player.isRecording():
                self.btnRecord.setStyleSheet(self.getButtonStyle("recording"))
            else:
                self.btnRecord.setStyleSheet(self.getButtonStyle("record"))
        else:
            self.btnRecord.setStyleSheet(self.getButtonStyle("record"))

    def setBtnPlay(self):
        if self.mw.playing:
            if self.mw.player.isPaused():
                self.btnPlay.setStyleSheet(self.getButtonStyle("play"))
                self.btnPlay.setToolTip("Play")
            else:
                self.btnPlay.setStyleSheet(self.getButtonStyle("pause"))
                self.btnPlay.setToolTip("Pause")
        else:
            self.btnPlay.setStyleSheet(self.getButtonStyle("play"))
            self.btnPlay.setToolTip("Play")

    def setBtnMute(self):
        if self.mw.mute:
            self.btnMute.setStyleSheet(self.getButtonStyle("mute"))
        else:
            self.btnMute.setStyleSheet(self.getButtonStyle("audio"))

    def btnRecordClicked(self):
        filename = '{0:%Y%m%d%H%M%S.mp4}'.format(datetime.datetime.now())
        filename = self.mw.filePanel.dirSetter.txtDirectory.text() + "/" + filename

        if self.mw.player is not None:
            self.mw.player.toggleRecording(filename)
        self.setBtnRecord()
        self.mw.cameraPanel.setBtnRecord()

    def btnMuteClicked(self):
        self.mw.toggleMute()
        self.setBtnMute()
        self.mw.cameraPanel.setBtnMute()

    def sldVolumeChanged(self, value):
        self.mw.cameraPanel.sldVolume.setValue(value)
        self.mw.setVolume(value)

    def getButtonStyle(self, name):
        strStyle = "QPushButton { image : url(image:%1.png); } QPushButton:hover { image : url(image:%1_hi.png); } QPushButton:pressed { image : url(image:%1_lo.png); }"
        strStyle = strStyle.replace("%1", name)
        return strStyle

class FilePanel(QWidget):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw

        self.dirKey = "FilePanel/dir"
        self.headerKey = "FilePanel/header"

        dirs = QStandardPaths.standardLocations(QStandardPaths.StandardLocation.MoviesLocation)
        self.dirSetter = DirectorySetter(mw)
        self.dirSetter.txtDirectory.setText(mw.settings.value(self.dirKey, dirs[0]))

        self.model = QFileSystemModel()
        self.model.fileRenamed.connect(self.onFileRenamed)
        self.tree = TreeView(mw)
        self.tree.setModel(self.model)
        self.tree.doubleClicked.connect(self.treeDoubleClicked)
        self.tree.header().sectionResized.connect(self.headerChanged)
        self.tree.header().sectionMoved.connect(self.headerChanged)
        headerState = self.mw.settings.value(self.headerKey)
        if not headerState is None:
            self.tree.header().restoreState(headerState)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self.showContextMenu)

        self.progress = Progress(mw)
        self.control = FileControlPanel(mw)

        lytMain = QGridLayout(self)
        lytMain.addWidget(self.dirSetter,  0, 0, 1, 1)
        lytMain.addWidget(self.tree,       1, 0, 1, 1)
        lytMain.addWidget(self.progress,   2, 0, 1, 1)
        lytMain.addWidget(self.control,    3, 0, 1, 1)
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
            self.mw.settings.setValue(self.dirKey, path)

    def treeDoubleClicked(self, index):
        if index.isValid():
            fileInfo = self.model.fileInfo(index)
            if fileInfo.isDir():
                self.tree.setExpanded(index, self.tree.isExpanded(index))
            else:
                self.mw.playMedia(self.model.filePath(index))
                self.control.setBtnPlay()

    def headerChanged(self, a, b, c):
        self.mw.settings.setValue(self.headerKey, self.tree.header().saveState())

    def onMediaStarted(self, duration):
        if self.mw.tab.currentIndex() == 1:
            self.progress.updateDuration(duration)
            self.tree.setFocus()
            index = self.tree.currentIndex()
            if index.isValid():
                fileInfo = self.model.fileInfo(index)
                self.mw.setWindowTitle(fileInfo.fileName())
        self.control.setBtnPlay()

    def onMediaStopped(self):
        self.progress.updateProgress(0.0)
        self.progress.updateDuration(0)
        self.control.setBtnRecord()
        self.control.setBtnPlay()

    def onMediaProgress(self, pct):
        if pct >= 0.0 and pct <= 1.0:
            self.progress.updateProgress(pct)

    def showContextMenu(self, pos):
        self.remove.setDisabled(self.mw.playing)
        self.rename.setDisabled(self.mw.playing)
        index = self.tree.indexAt(pos)
        if index.isValid():
            self.menu.exec(self.mapToGlobal(pos))

    def onMenuRemove(self):
        index = self.tree.currentIndex()
        if index.isValid():
            ret = QMessageBox.warning(self, "onvif-gui",
                                        "You are about to delete this file.\n"
                                        "Are you sure you want to continue?",
                                        QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)

            if ret == QMessageBox.StandardButton.Ok:
                QFile.remove(self.model.filePath(self.tree.currentIndex()))

    def onMenuRename(self):
        if self.mw.playing:
            self.mw.infoCallback("Please stop playing file to rename")
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
            strInfo += "Filename: " + info.absoluteFilePath()
            strInfo += "\nLast Modified: " + info.lastModified().toString()

            reader = avio.Reader(info.absoluteFilePath())
            duration = reader.duration()
            time_in_seconds = int(duration / 1000)
            hours = int(time_in_seconds / 3600)
            minutes = int((time_in_seconds - (hours * 3600)) / 60)
            seconds = int((time_in_seconds - (hours * 3600) - (minutes * 60)))
            strInfo += "\nDuration: " + str(minutes) + ":" + str(seconds)

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
                strInfo += "\n    Time Base:  " + str(reader.audio_time_base().num) + " / " + str(reader.audio_time_base().den)
                strInfo += "\n    Sample Format:  " + reader.str_sample_format()
                strInfo += "\n    Bitrate:  " + f'{reader.audio_bit_rate():,}'
            
        else:
            strInfo = "Invalid Index"

        msgBox = QMessageBox(self)
        msgBox.setWindowTitle("File Info")
        msgBox.setText(strInfo)
        msgBox.exec()

    def onMenuPlay(self):
        index = self.tree.currentIndex()
        if (index.isValid()):
            info = self.model.fileInfo(index)
            self.mw.playMedia(info.absoluteFilePath())
