#/********************************************************************
# onvif-gui/onvif_gui/panels/settings/storage.py 
#
# Copyright (c) 2024  Stephen Rhodes
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

from PyQt6.QtWidgets import QMessageBox, QSpinBox, \
    QGridLayout, QWidget, QCheckBox, QLabel, QMessageBox, QGroupBox
from PyQt6.QtCore import QStandardPaths
from loguru import logger
from pathlib import Path
import shutil
from onvif_gui.components import DirectorySelector
from PyQt6.QtCore import pyqtSignal, QObject
import threading
import os
import tempfile

class StorageSignals(QObject):
    updateDiskUsage = pyqtSignal()
    showWaitDialog = pyqtSignal(str)

class StorageOptions(QWidget):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw
        self.max_size = 100

        self.archiveKey = "settings/archive"
        self.pictureKey = "settings/picture"
        self.diskLimitKey = "settings/diskLimit"
        self.maxFileDurationKey = "settings/maxFileDuration"
        self.mangageDiskUsagekey = "settings/manageDiskUsage"
        self.writeBufferSizeKey = "setting/writeBufferSize"

        self.signals = StorageSignals()
        self.signals.updateDiskUsage.connect(self.updateDiskUsage)
        self.signals.showWaitDialog.connect(self.mw.dlgWait.show)

        video_dirs = QStandardPaths.standardLocations(QStandardPaths.StandardLocation.MoviesLocation)
        self.dirArchive = DirectorySelector(mw, self.archiveKey, "Archive Dir", video_dirs[0])
        self.dirArchive.signals.dirChanged.connect(self.dirArchiveChanged)

        picture_dirs = QStandardPaths.standardLocations(QStandardPaths.StandardLocation.PicturesLocation)
        self.dirPictures = DirectorySelector(mw, self.pictureKey, "Picture Dir", picture_dirs[0])
        self.dirPictures.signals.dirChanged.connect(self.dirPicturesChanged)

        self.grpDiskUsage = QGroupBox()

        self.chkManageDiskUsage = QCheckBox()
        self.chkManageDiskUsage.setChecked(bool(int(self.mw.settings.value(self.mangageDiskUsagekey, 0))))
        self.chkManageDiskUsage.clicked.connect(self.chkManageDiskUsageChanged)

        self.spnDiskLimit = QSpinBox()
        self.spnDiskLimit.valueChanged.connect(self.spnDiskLimitChanged)

        self.spnMaxFileDuration = QSpinBox()
        self.spnMaxFileDuration.setMaximum(60)
        self.spnMaxFileDuration.setValue(int(self.mw.settings.value(self.maxFileDurationKey, 15)))
        self.spnMaxFileDuration.valueChanged.connect(self.spnMaxFileDurationChanged)
        lblMaxFileDuration = QLabel("Max File Duration (minutes)")

        self.spnWriteBufferSize = QSpinBox()
        self.spnWriteBufferSize.setValue(int(self.mw.settings.value(self.writeBufferSizeKey, 10)))
        self.spnWriteBufferSize.valueChanged.connect(self.spnWriteBufferSizeChanged)
        lblWriteBuffferSize = QLabel("File Write Buffer Size (GB)")

        self.updateDiskUsage()
        #disk_limit = min(int(self.mw.settings.value(self.diskLimitKey, 100)), self.max_size)
        disk_limit = int(self.mw.settings.value(self.diskLimitKey, 100))
        self.spnDiskLimit.setValue(disk_limit)

        lytDiskUsage = QGridLayout(self.grpDiskUsage)
        lytDiskUsage.addWidget(self.chkManageDiskUsage, 0, 0, 1, 1)
        lytDiskUsage.addWidget(self.spnDiskLimit,       0, 2, 1, 1)
        lytDiskUsage.addWidget(QLabel("GB"),            0, 3, 1, 1)
        lytDiskUsage.addWidget(self.dirArchive,         1, 0, 1, 4)
        lytDiskUsage.addWidget(self.dirPictures,        2, 0, 1, 4)
        lytDiskUsage.addWidget(lblMaxFileDuration,      3, 0, 1, 2)
        lytDiskUsage.addWidget(self.spnMaxFileDuration, 3, 2, 1, 1)
        lytDiskUsage.addWidget(lblWriteBuffferSize,     4, 0, 1, 2)
        lytDiskUsage.addWidget(self.spnWriteBufferSize, 4, 2, 1, 1)
        lytDiskUsage.setColumnStretch(2, 10)

        lytMain = QGridLayout(self)
        lytMain.addWidget(self.grpDiskUsage, 0, 0, 1, 1)
        lytMain.addWidget(QLabel(),          1, 0, 1, 1)
        lytMain.addWidget(QLabel(),          2, 0, 1, 1)
        lytMain.setRowStretch(2, 10)


    def spnDiskLimitChanged(self, value):
        self.mw.settings.setValue(self.diskLimitKey, value)

    def spnMaxFileDurationChanged(self, value):
        self.mw.settings.setValue(self.maxFileDurationKey, value)
        self.mw.STD_FILE_DURATION = value * 60

    def chkManageDiskUsageChanged(self):
        if self.chkManageDiskUsage.isChecked():
            ret = QMessageBox.warning(self, "** WARNING **",
                                        "You are giving full control of both the video and picture archive directories to this program.  "
                                        "Any files contained within those directories or their sub-directories are subject to deletion.  "
                                        "You should only enable this feature if you are sure that this is ok.\n\n"
                                        "Are you sure you want to continue?",
                                        QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
            if ret == QMessageBox.StandardButton.Cancel:
                self.chkManageDiskUsage.setChecked(False)
            else:
                self.signals.showWaitDialog.emit("Please wait for disk check to complete")
                thread = threading.Thread(target=self.calculateDiskUsage)
                thread.start()
                #self.updateDiskUsage()
        else:
            self.updateDiskUsage()
        self.mw.settings.setValue(self.mangageDiskUsagekey, int(self.chkManageDiskUsage.isChecked()))

    def dirArchiveChanged(self, path):
        logger.debug(f'Video archive directory changed to {path}')
        self.dirArchive.txtDirectory.setText(path)
        self.mw.settings.setValue(self.archiveKey, path)
        self.updateDiskUsage()
        #self.mw.filePanel.dirArchive.txtDirectory.setText(path)
        #self.mw.filePanel.dirChanged(path)

    def dirPicturesChanged(self, path):
        logger.debug(f'Picture directory changed to {path}')
        self.dirPictures.txtDirectory.setText(path)
        self.mw.settings.setValue(self.pictureKey, path)
        #self.mw.filePanel.control.dlgPicture.dirPictures.txtDirectory.setText(path)
        #self.mw.filePanel.control.dlgPicture.dirChanged(path)

    def spnWriteBufferSizeChanged(self, value):
        self.mw.settings.setValue(self.writeBufferSizeKey, value)
        self.updateDiskUsage()

    def calculateDiskUsage(self):
        if not self.mw.settings_profile == "gui":
            return
        dir = self.dirArchive.txtDirectory.text()
        size = sum(f.stat().st_size for f in Path(dir).rglob('*') if f.is_file())
        _, _, free = shutil.disk_usage(dir)
        buffer_size = self.spnWriteBufferSize.value() * 1_000_000_000
        self.max_size = int((free + size - buffer_size)/1_000_000_000)
        self.lblManage = f'Auto Manage (max {self.max_size} GB)'
        self.chkManageDiskUsage.setText(self.lblManage)
        self.spnDiskLimit.setMaximum(self.max_size)
        if self.spnDiskLimit.value() > self.max_size:
            self.spnDiskLimit.setValue(self.max_size)
        str_size_in_GB = "{:.2f}".format(size / 1_000_000_000)
        self.grpDiskUsage.setTitle(f'Disk Usage (currently {str_size_in_GB} GB)')
        self.mw.signals.hideWaitDialog.emit()

    def updateDiskUsage(self):
        if self.chkManageDiskUsage.isChecked():
            self.calculateDiskUsage()
        else:
            self.spnDiskLimit.setMaximum(self.max_size)
            self.grpDiskUsage.setTitle("Disk Usage Auto Manage")
            self.chkManageDiskUsage.setText("Auto Manage")