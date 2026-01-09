import os
from loguru import logger
from PyQt6.QtWidgets import QGridLayout, QWidget, \
    QLabel, QAbstractItemView, QAbstractItemView, \
    QApplication, QTreeView, QMenu, QMessageBox
from PyQt6.QtGui import QFileSystemModel, QAction
from PyQt6.QtCore import Qt, QFileInfo, QDateTime, \
    QDate, QTime, QStandardPaths
from onvif_gui.components.directoryselector import DirectorySelector
from onvif_gui.panels.file import FileControlPanel
from onvif_gui.components import Progress, InfoDialog
from pathlib import Path
import traceback

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
                        self.mw.picturePanel.playVideo()
                    else:
                        if self.isExpanded(index):
                            self.collapse(index)
                        else:
                            self.expand(index)
            case Qt.Key.Key_Escape:
                if player := self.mw.filePanel.getCurrentlyPlayingFile():
                    #self.mw.filePanel.control.btnStopClicked()
                    self.mw.closeAllStreams()
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
        try:
            if self.mw.settings_profile != "Reader":
                return
            index = self.currentIndex()
            if index.isValid():
                info = self.model().fileInfo(index)
                if info.isFile() and self.model().isReadOnly():
                    self.mw.glWidget.drawFile(info.absoluteFilePath())
        except Exception as ex:
            logger.error(f"PicturePanel showCurrentFile exception: {ex}")

    def currentChanged(self, newIndex, oldIndex):
        if newIndex.isValid():
            self.showCurrentFile()
            self.scrollTo(newIndex)

class PicturePanel(QWidget):                
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
        self.control = FileControlPanel(mw, self)
        self.progress = Progress(mw)
        self.dlgInfo = InfoDialog(mw)

        if self.mw.parent_window:
            picture_dirs = self.mw.parent_window.settingsPanel.storage.dirPictures.text()
        else:
            picture_dirs = QStandardPaths.standardLocations(QStandardPaths.StandardLocation.PicturesLocation)[0]
        self.dirPictures = DirectorySelector(mw, self.mw.settingsPanel.storage.pictureKey, "", picture_dirs)
        #self.dirPictures.signals.dirChanged.connect(self.mw.settingsPanel.storage.dirPicturesChanged)
        self.dirPictures.signals.dirChanged.connect(self.dirChanged)

        self.model = QFileSystemModel(mw)
        self.model.directoryLoaded.connect(self.loaded)
        self.tree = PicTreeView(mw)
        self.tree.setModel(self.model)
        #self.tree.setSortingEnabled(True)
        #self.tree.sortByColumn(0, Qt.SortOrder.DescendingOrder)
        self.tree.doubleClicked.connect(self.treeDoubleClicked)
        self.tree.setColumnHidden(1, True)
        self.tree.setColumnHidden(2, True)
        if data := self.mw.settings.value(self.headerKey):
            self.tree.header().restoreState(data)
        self.tree.update()
        self.tree.header().sectionResized.connect(self.headerChanged)
        self.tree.header().sectionMoved.connect(self.headerChanged)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self.showContextMenu)

        pnlTree = QWidget()
        lytTree = QGridLayout(pnlTree)
        lytTree.addWidget(self.dirPictures,       0, 0, 1, 1)
        lytTree.addWidget(self.tree,              1, 0, 1, 1)
        lytTree.addWidget(self.progress,          2, 0, 1, 1)
        lytTree.addWidget(QLabel(),               3, 0, 1, 1)
        lytTree.addWidget(self.control,  4, 0, 1, 1)

        lytMain = QGridLayout(self)
        lytMain.addWidget(pnlTree)
        lytMain.setContentsMargins(0, 0, 0, 0)

        self.dirChanged(self.dirPictures.text())

        self.menu = QMenu("Context Menu", self)
        self.remove = QAction("Delete", self)
        self.rename = QAction("Rename", self)
        self.info = QAction("Info", self)
        self.remove.triggered.connect(self.onMenuRemove)
        self.rename.triggered.connect(self.onMenuRename)
        self.info.triggered.connect(self.onMenuInfo)
        self.menu.addAction(self.remove)
        self.menu.addAction(self.rename)
        self.menu.addAction(self.info)

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
        filename = Path(filename).stem
        year = int(filename[0:4])
        month = int(filename[4:6])
        date = int(filename[6:8])
        hour = int(filename[8:10])
        minute = int(filename[10:12])
        second = int(filename[12:14])
        qdate = QDate(year, month, date)
        qtime = (QTime(hour, minute, second))
        return QDateTime(qdate, qtime)
    
    def searchFiles(self, target, dir):
        try:
            alarm_buffer_size = self.mw.settingsPanel.alarm.spnBufferSize.value()
            files = os.listdir(dir)
            for file in files:
                file_info = QFileInfo(str(Path(dir) / file))
                start = self.filenameToQDateTime(file).addSecs(-alarm_buffer_size)
                finish = file_info.lastModified()
                if (target >= start and target <= finish):
                    return file
            return None
        except Exception as ex:
            logger.error(f"PicturePanel searchFile exception: {ex}")

    def calculateTargetPct(self, target, file, dir):
        alarm_buffer_size = self.mw.settingsPanel.alarm.spnBufferSize.value()
        start = self.filenameToQDateTime(file).addSecs(-alarm_buffer_size)
        finish = QFileInfo(str(Path(dir) / file)).lastModified()
        numerator = start.secsTo(target)
        denominator = start.secsTo(finish)
        if denominator > 0:
            pct = float(numerator / denominator)
            if pct > 0.98:
                pct = 0.95
        else:
            pct = 0.95
        return pct

    def playVideo(self):
        try:
            alarm_buffer_size = self.mw.settingsPanel.alarm.spnBufferSize.value()
            index = self.tree.currentIndex()
            if index.isValid():
                pic_info = self.model.fileInfo(index)
                if pic_info.isFile():
                    target = self.filenameToQDateTime(pic_info.fileName())
                    dir = Path(self.mw.filePanel.dirArchive.txtDirectory.text()) / str(pic_info.absoluteDir().dirName())

                    # everything is approximate, so give the algorithm a couple chances to find near miss
                    found = False
                    if file := self.searchFiles(target, dir):
                        found = True
                    else:
                        fuzz = target.addSecs(-alarm_buffer_size)
                        if file := self.searchFiles(fuzz, dir):
                            target = fuzz
                            found = True
                        else:
                            fuzz = target.addSecs(alarm_buffer_size)
                            if file := self.searchFiles(fuzz, dir):
                                target = fuzz
                                found = True

                    if not found:
                        return
                    
                    self.selectFileInFilePanelTree(str(dir), file)
                    pct = self.calculateTargetPct(target, file, dir)
                    vid_info = QFileInfo(os.path.join(dir, file))
                    if player := self.mw.pm.getPlayer(vid_info.absoluteFilePath()):
                        player.seek(pct)
                    else:
                        self.mw.closeAllStreams()
                        self.mw.playMedia(vid_info.absoluteFilePath(), file_start_from_seek=pct)
                        self.mw.glWidget.focused_uri = vid_info.absoluteFilePath()

        except Exception as ex:
            logger.error(f'Event browser exception: {ex}')
            logger.debug(traceback.format_exc())

    def selectFileInFilePanelTree(self, path, filename):
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
        #self.model.sort(0, Qt.SortOrder.DescendingOrder)
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
        try:
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
        except Exception as ex:
            logger.error(f"PicturePanel refresh exception: {ex}")

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
                self.playVideo()

    # some weird bug makes this not visible to calling signals in some cases
    def onMediaStarted(self, duration):
        if self.mw.tab.currentIndex() == 1:
            self.tree.setFocus()
        self.control.setBtnPlay()
        self.control.setBtnMute()
        self.control.setSldVolume()

    def onMediaStopped(self, uri):
        self.control.setBtnPlay()
        self.progress.updateProgress(0.0)
        self.progress.lblDuration.setText("0:00")
        # not relavent to camera playback
        self.tree.showCurrentFile()

    def onMediaProgress(self, pct, uri):
        if player := self.mw.pm.getPlayer(uri):
            player.file_progress = pct
            self.progress.updateDuration(player.duration())

        #if pct >= 0.0 and pct <= 1.0 and uri == self.mw.glWidget.focused_uri:
        if pct >= 0.0 and pct <= 1.0:
            self.progress.updateProgress(pct)

    def showContextMenu(self, pos):
        index = self.tree.indexAt(pos)
        if index.isValid():
            fileInfo = self.model.fileInfo(index)
            if fileInfo.isFile():
                self.menu.exec(self.mapToGlobal(pos))

    def onMenuRemove(self):
        index = self.tree.currentIndex()
        if index.isValid():
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
                strInfo += f"Filename:  {info.fileName()}"
                strInfo += f"\nCreated:  {info.birthTime().toString()}"
                width = self.mw.glWidget.pixmap.width()
                height = self.mw.glWidget.pixmap.height()
                strInfo += f"\nImage Dims: {width} x {height}"           
            else:
                strInfo = "Invalid Index"
        except Exception as ex:
            strInfo = f'Unable to read file info: {ex}'

        #msgBox = QMessageBox(self)
        #msgBox.setWindowTitle("File Info")
        #msgBox.setText(strInfo)
        #msgBox.exec()
        self.dlgInfo.lblMessage.setText(strInfo)
        self.dlgInfo.exec()
