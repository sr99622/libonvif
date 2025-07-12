#/********************************************************************
# libonvif/onvif-gui/onvif_gui/panels/file/filesearchdialog.py 
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
import pathlib
from loguru import logger
from PyQt6.QtWidgets import QGridLayout, QWidget, \
    QLabel, QAbstractItemView, QDialog, QPushButton, \
    QAbstractItemView, QApplication, QSplitter, QTreeView
from PyQt6.QtGui import QPixmap, QFileSystemModel, \
    QKeySequence
from PyQt6.QtCore import Qt, QSize, QFileInfo, QDateTime, \
    QDate, QTime, QStandardPaths
from onvif_gui.components.directoryselector import DirectorySelector

class View(QLabel):
    def __init__(self):
        super().__init__()
        self.pm = None
        self.uri = None

    def sizeHint(self):
        return QSize(640, 480)
    
    def setPixmap(self, uri):
        self.uri = uri
        self.pm = QPixmap(uri)
        scaledPixmap = self.pm.scaled(self.width(), self.height(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        super().setPixmap(scaledPixmap)

    def resizeEvent(self, event):
        if self.pm:
            self.setPixmap(self.pm)
        return super().resizeEvent(event)   

class PicTreeView(QTreeView):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw

    def keyPressEvent(self, event):
        pass_through = True

        match event.key():
            case Qt.Key.Key_Return:
                pass_through = False
                index = self.currentIndex()
                if index.isValid():
                    fileInfo = self.model().fileInfo(index)
                    if fileInfo.isFile():
                        self.mw.filePanel.control.dlgPicture.btnViewClicked()
                    else:
                        if self.isExpanded(index):
                            self.collapse(index)
                        else:
                            self.expand(index)
            case Qt.Key.Key_Escape:
                if player := self.mw.filePanel.getCurrentlyPlayingFile():
                    self.mw.filePanel.control.btnStopClicked()
                    pass_through = False
            case Qt.Key.Key_Space:
                if player := self.mw.filePanel.getCurrentlyPlayingFile():
                    player.togglePaused()
            case Qt.Key.Key_Left:
                self.mw.filePanel.rewind()
                pass_through = False
            case Qt.Key.Key_Right:
                self.mw.filePanel.fastForward()
                pass_through = False

        if pass_through:
            return super().keyPressEvent(event)
        
    def showCurrentFile(self):
        index = self.currentIndex()
        if index.isValid():
            info = self.model().fileInfo(index)
            if info.isFile() and self.model().isReadOnly():
                self.mw.filePanel.control.dlgPicture.view.setPixmap(info.absoluteFilePath())

    def currentChanged(self, newIndex, oldIndex):
        if newIndex.isValid():
            self.showCurrentFile()
            self.scrollTo(newIndex)

class PictureDialog(QDialog):
    def __init__(self, mw):
        super().__init__(mw)
        self.mw = mw
        self.geometryKey = "PictureBrowseDialog/geometry"
        self.splitKey = "PictureBrowseDialog/splitSettings"
        self.headerKey = "PictureBrowseDialog/header"
        self.positionInitialized = False
        self.setWindowTitle("Event Browser")
        self.expandedPaths = []
        self.loadedCount = 0
        self.restorationPath = None
        self.verticalScrollBarPosition = 0

        if self.mw.parent_window:
            picture_dirs = self.mw.parent_window.settingsPanel.storage.dirPictures.text()
        else:
            picture_dirs = QStandardPaths.standardLocations(QStandardPaths.StandardLocation.PicturesLocation)[0]
        self.dirPictures = DirectorySelector(mw, self.mw.settingsPanel.storage.pictureKey, "", picture_dirs)
        self.dirPictures.signals.dirChanged.connect(self.mw.settingsPanel.storage.dirPicturesChanged)

        self.model = QFileSystemModel(mw)
        self.model.directoryLoaded.connect(self.loaded)
        self.tree = PicTreeView(mw)
        self.tree.setModel(self.model)
        self.tree.doubleClicked.connect(self.treeDoubleClicked)
        self.tree.setColumnHidden(1, True)
        self.tree.setColumnHidden(2, True)
        if data := self.mw.settings.value(self.headerKey):
            self.tree.header().restoreState(data)
        self.tree.update()
        self.tree.header().sectionResized.connect(self.headerChanged)
        self.tree.header().sectionMoved.connect(self.headerChanged)

        self.btnView = QPushButton("View")
        self.btnView.clicked.connect(self.btnViewClicked)
        self.btnView.setShortcut(QKeySequence("Ctrl+V"))
        self.btnRefresh = QPushButton("Refresh")
        self.btnRefresh.clicked.connect(self.btnRefreshClicked)
        self.btnRefresh.setShortcut(QKeySequence("Ctrl+R"))
        pnlButton = QWidget()
        lytButton = QGridLayout(pnlButton)
        lytButton.addWidget(self.btnRefresh,  0, 0, 1, 1, Qt.AlignmentFlag.AlignCenter)
        lytButton.addWidget(self.btnView,     0, 1, 1, 1, Qt.AlignmentFlag.AlignCenter)

        pnlTree = QWidget()
        lytTree = QGridLayout(pnlTree)
        lytTree.addWidget(self.dirPictures,   0, 0, 1, 1)
        lytTree.addWidget(self.tree,          1, 0, 1, 1)
        lytTree.addWidget(pnlButton,          2, 0, 1, 1)
        lytTree.setContentsMargins(0, 0, 0, 0)

        self.view = View()
        self.split = QSplitter()
        self.split.addWidget(self.view)
        self.split.addWidget(pnlTree)
        self.split.splitterMoved.connect(self.splitterMoved)
        if splitterState := self.mw.settings.value(self.splitKey):
            self.split.restoreState(splitterState)

        lytMain = QGridLayout(self)
        lytMain.addWidget(self.split)
        lytMain.setContentsMargins(0, 0, 0, 0)

        self.dirChanged(self.mw.settingsPanel.storage.dirPictures.txtDirectory.text())

    def headerChanged(self, a, b, c):
        self.mw.settings.setValue(self.headerKey, self.tree.header().saveState())

    def closeEvent(self, event):
        self.mw.settings.setValue(self.geometryKey, self.geometry())
        return super().closeEvent(event)

    def showEvent(self, event):
        if rect := self.mw.settings.value(self.geometryKey):
            if rect.width() and rect.height():
                self.setGeometry(rect)
                self.positionInitialized = True
        return super().showEvent(event)
    
    def splitterMoved(self, pos, index):
        self.mw.settings.setValue(self.splitKey, self.split.saveState())

    def filenameToQDateTime(self, filename):
        # file start times may be inaccurate on samba, use filename instead
        filename = pathlib.Path(filename).stem
        year = int(filename[0:4])
        month = int(filename[4:6])
        date = int(filename[6:8])
        hour = int(filename[8:10])
        minute = int(filename[10:12])
        second = int(filename[12:14])
        qdate = QDate(year, month, date)
        qtime = (QTime(hour, minute, second))
        return QDateTime(qdate, qtime)

    def btnViewClicked(self):
        try:
            alarm_buffer_size = self.mw.settingsPanel.alarm.spnBufferSize.value()
            index = self.tree.currentIndex()
            if index.isValid():
                pic_info = self.model.fileInfo(index)
                if pic_info.isFile():
                    dir = os.path.join(self.mw.filePanel.dirArchive.txtDirectory.text(), pic_info.absoluteDir().dirName())
                    files = os.listdir(dir)
                    for file in files:
                        vid_info = QFileInfo(os.path.join(dir, file))
                        target = self.filenameToQDateTime(pic_info.fileName())
                        start = self.filenameToQDateTime(file).addSecs(-alarm_buffer_size)
                        finish = vid_info.lastModified()
                        if (target >= start and target <= finish):
                            self.selectFileInTree(dir, file)
                            numerator = start.secsTo(target)
                            denominator = start.secsTo(finish)
                            if denominator:
                                pct = float(numerator / denominator)
                                if pct > 0.98:
                                    pct = 0.95
                                if player := self.mw.pm.getPlayer(vid_info.absoluteFilePath()):
                                    player.seek(pct)
                                else:
                                    self.mw.filePanel.control.startPlayer(file_start_from_seek=pct)
                            break
        except Exception as ex:
            logger.debug(f'Event browser exception: {ex}')

    def selectFileInTree(self, path, filename):
        tree = self.mw.filePanel.tree
        model = tree.model()
        if camera_idx := model.index(path):
            if camera_idx.isValid():
                if not tree.isExpanded(camera_idx):
                    tree.setExpanded(camera_idx, True)
                if file_idx := model.index(os.path.join(path, filename)):
                    if file_idx.isValid():
                        tree.setCurrentIndex(file_idx)
                        tree.scrollTo(file_idx, QAbstractItemView.ScrollHint.PositionAtCenter)

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

    def btnRefreshClicked(self):
        self.loadedCount = 0
        self.expandedPaths = []
        self.restorationPath = self.model.filePath(self.tree.currentIndex())
        path = self.dirPictures.txtDirectory.text()
        self.model.sort(0)
        for i in range(self.model.rowCount(self.model.index(path))):
            idx = self.model.index(i, 0, self.model.index(path))
            if idx.isValid():
                if self.tree.isExpanded(idx):
                    self.expandedPaths.append(self.model.filePath(idx))
        self.verticalScrollBarPosition = self.tree.verticalScrollBar().value()
        self.model = QFileSystemModel()
        self.model.setRootPath(path)
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
                self.btnViewClicked()
