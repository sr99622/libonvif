import os
import sys
import numpy as np
import cv2
from time import sleep
from PyQt6.QtWidgets import QDialogButtonBox, QLineEdit, QPushButton, \
QGridLayout, QWidget, QSlider, QLabel, QMessageBox, QListWidget, \
QTabWidget, QTreeView, QFileDialog, QMenu
from PyQt6.QtGui import QFileSystemModel, QIcon, QAction
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QSettings, QStandardPaths
from progress import Progress

class TreeView(QTreeView):
    def __init__(self):
        super().__init__()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Return:
            self.doubleClicked.emit(self.currentIndex())
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
        self.icnPlay = QIcon('image:play.png')
        self.icnPause = QIcon('image:pause.png')
        self.icnStop = QIcon('image:stop.png')
        self.icnMute = QIcon('image:mute.png')
        self.icnAudio = QIcon('image:audio.png')

        self.btnPlay = QPushButton()
        self.btnPlay.setIcon(self.icnPlay)
        self.btnPlay.setToolTip("Play")
        self.btnPlay.setToolTipDuration(2000)
        self.btnPlay.setMinimumWidth(self.icnPlay.availableSizes()[0].width() * 2)
        self.btnPlay.clicked.connect(self.btnPlayClicked)
        
        self.btnStop = QPushButton()
        self.btnStop.setIcon(self.icnStop)
        self.btnStop.setToolTip("Stop")
        self.btnStop.setToolTipDuration(2000)
        self.btnStop.setMinimumWidth(self.icnStop.availableSizes()[0].width() * 2)
        self.btnStop.clicked.connect(self.btnStopClicked)

        spacer = QLabel()
        spacer.setMinimumWidth(self.btnStop.minimumWidth())
        
        self.btnMute = QPushButton()
        if self.mw.mute:
            self.btnMute.setIcon(self.icnMute)
        else:
            self.btnMute.setIcon(self.icnAudio)
        self.btnMute.setToolTip("Mute")
        self.btnMute.setToolTipDuration(2000)
        self.btnMute.setMinimumWidth(self.icnMute.availableSizes()[0].width() * 2)
        self.btnMute.clicked.connect(self.btnMuteClicked)

        self.sldVolume = QSlider(Qt.Orientation.Horizontal)
        self.sldVolume.setValue(self.mw.volume)
        self.sldVolume.valueChanged.connect(self.sldVolumeChanged)
        
        lytMain =  QGridLayout(self)
        lytMain.setContentsMargins(0, 0, 0, 0)
        lytMain.addWidget(self.btnPlay,   0, 0, 1, 1, Qt.AlignmentFlag.AlignCenter)
        lytMain.addWidget(self.btnStop,   0, 1, 1, 1, Qt.AlignmentFlag.AlignCenter)
        lytMain.addWidget(spacer,         0, 2, 1, 1)
        lytMain.addWidget(self.btnMute,   0, 3, 1, 1, Qt.AlignmentFlag.AlignCenter)
        lytMain.addWidget(self.sldVolume, 0, 4, 1, 1)
        self.setContentsMargins(0, 0, 0, 0)

    def btnPlayClicked(self):
        if self.mw.playing:
            self.mw.player.togglePaused()
            if self.mw.player.isPaused():
                self.btnPlay.setIcon(self.icnPlay)
            else:
                self.btnPlay.setIcon(self.icnPause)
        else:
            index = self.mw.filePanel.tree.currentIndex()
            if index.isValid():
                fileInfo = self.mw.filePanel.model.fileInfo(index)
                self.mw.playMedia(fileInfo.filePath())
                self.btnPlay.setIcon(self.icnPause)

    def btnStopClicked(self):
        self.mw.stopMedia()
        self.btnPlay.setIcon(self.icnPlay)
        self.mw.filePanel.tree.setFocus()

    def setBtnMute(self):
        if self.mw.mute:
            self.btnMute.setIcon(self.icnMute)
        else:
            self.btnMute.setIcon(self.icnAudio)

    def btnMuteClicked(self):
        self.mw.toggleMute()
        self.setBtnMute()
        self.mw.cameraPanel.setBtnMute()

    def sldVolumeChanged(self, value):
        self.mw.cameraPanel.sldVolume.setValue(value)
        self.mw.setVolume(value)

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
        self.tree = TreeView()
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
        remove = QAction("Delete", self)
        rename = QAction("Rename", self)
        info = QAction("Info", self)
        play = QAction("Play", self)
        remove.triggered.connect(self.onMenuRemove)
        rename.triggered.connect(self.onMenuRename)
        info.triggered.connect(self.onMenuInfo)
        play.triggered.connect(self.onMenuPlay)
        self.menu.addAction(remove)
        self.menu.addAction(rename)
        self.menu.addAction(info)
        self.menu.addAction(play)

    def dirChanged(self, path):
        self.model.setRootPath(path)
        self.tree.setRootIndex(self.model.index(path))
        self.mw.settings.setValue(self.dirKey, path)

    def treeDoubleClicked(self, index):
        if index.isValid():
            fileInfo = self.model.fileInfo(index)
            if fileInfo.isDir():
                self.tree.setExpanded(index, self.tree.isExpanded(index))
            else:
                self.control.btnPlay.setIcon(self.control.icnPause)
                self.mw.playMedia(self.model.filePath(index))

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

    def onMediaStopped(self):
        self.progress.updateProgress(0.0)
        self.progress.updateDuration(0)

    def onMediaProgress(self, pct):
        if pct >= 0.0 and pct <= 1.0:
            self.progress.updateProgress(pct)

    def showContextMenu(self, pos):
        index = self.tree.indexAt(pos)
        if index.isValid():
            self.menu.exec(self.mapToGlobal(pos))

    def onMenuRemove(self):
        print("onMenuRemove")

    def onMenuRename(self):
        print("onMenuRename")

    def onMenuInfo(self):
        print("onMenuInfo")

    def onMenuPlay(self):
        print("onMenuPlay")

