#/********************************************************************
# libonvif/onvif-gui/onvif_gui/panels/settings/storage.py 
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

import os
from PyQt6.QtWidgets import QMessageBox, QSpinBox, \
    QGridLayout, QWidget, QCheckBox, QLabel, QMessageBox, QGroupBox
from PyQt6.QtCore import QStandardPaths
from loguru import logger
from onvif_gui.components import DirectorySelector
from PyQt6.QtCore import pyqtSignal, QObject
import shutil

class StorageSignals(QObject):
    updateDiskUsage =pyqtSignal()

class StorageOptions(QWidget):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw

        self.archiveKey = "settings/archive"
        self.pictureKey = "settings/picture"
        self.diskLimitKey = "settings/diskLimit"
        self.maxFileDurationKey = "settings/maxFileDuration"
        self.mangageDiskUsagekey = "settings/manageDiskUsage"

        self.signals = StorageSignals()
        self.signals.updateDiskUsage.connect(self.updateDiskUsage)

        video_dirs = QStandardPaths.standardLocations(QStandardPaths.StandardLocation.MoviesLocation)
        self.dirArchive = DirectorySelector(mw, self.archiveKey, "Archive Dir", video_dirs[0])
        self.dirArchive.signals.dirChanged.connect(self.dirArchiveChanged)

        picture_dirs = QStandardPaths.standardLocations(QStandardPaths.StandardLocation.PicturesLocation)
        self.dirPictures = DirectorySelector(mw, self.pictureKey, "Picture Dir", picture_dirs[0])
        self.dirPictures.signals.dirChanged.connect(self.dirPicturesChanged)

        #dir_size = "{:.2f}".format(self.getDirectorySizeLocally(self.dirArchive.text()) / 1000000000)
        self.grpDiskUsage = QGroupBox()
        self.spnDiskLimit = QSpinBox()
        max_size = int(self.mw.diskManager.getMaximumAvailableForDirectory(self.dirArchive.txtDirectory.text())/1_000_000_000)
        self.spnDiskLimit.setMaximum(max_size)
        disk_limit = min(int(self.mw.settings.value(self.diskLimitKey, 100)), max_size)
        self.spnDiskLimit.setValue(disk_limit)
        self.spnDiskLimit.valueChanged.connect(self.spnDiskLimitChanged)
        self.updateDiskUsage()

        lbl = f'Auto Manage (max {max_size} GB)'
        self.chkManageDiskUsage = QCheckBox(lbl)
        self.chkManageDiskUsage.setChecked(bool(int(self.mw.settings.value(self.mangageDiskUsagekey, 0))))
        self.chkManageDiskUsage.clicked.connect(self.chkManageDiskUsageChanged)

        self.spnMaxFileDuration = QSpinBox()
        self.spnMaxFileDuration.setMaximum(60)
        self.spnMaxFileDuration.setValue(int(self.mw.settings.value(self.maxFileDurationKey, 15)))
        self.spnMaxFileDuration.valueChanged.connect(self.spnMaxFileDurationChanged)
        lblMaxFileDuration = QLabel("Max File Duration (minutes)")


        lytDiskUsage = QGridLayout(self.grpDiskUsage)
        lytDiskUsage.addWidget(self.chkManageDiskUsage, 0, 0, 1, 1)
        lytDiskUsage.addWidget(self.spnDiskLimit,       0, 2, 1, 1)
        lytDiskUsage.addWidget(QLabel("GB"),            0, 3, 1, 1)
        lytDiskUsage.addWidget(self.dirArchive,         1, 0, 1, 4)
        lytDiskUsage.addWidget(self.dirPictures,        2, 0, 1, 4)
        lytDiskUsage.addWidget(lblMaxFileDuration,      3, 0, 1, 2)
        lytDiskUsage.addWidget(self.spnMaxFileDuration, 3, 1, 1, 2)
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
        self.mw.settings.setValue(self.mangageDiskUsagekey, int(self.chkManageDiskUsage.isChecked()))

    def dirArchiveChanged(self, path):
        logger.debug(f'Video archive directory changed to {path}')
        self.dirArchive.txtDirectory.setText(path)
        self.mw.settings.setValue(self.archiveKey, path)
        max_size = int(self.mw.diskManager.getMaximumAvailableForDirectory(path)/1_000_000_000)
        self.spnDiskLimit.setMaximum(max_size)
        lbl = f'Auto Manage (max {max_size} GB)'
        self.chkManageDiskUsage.setText(lbl)
        disk_limit = min(int(self.mw.settings.value(self.diskLimitKey, 100)), max_size)
        self.spnDiskLimit.setValue(disk_limit)
        self.chkManageDiskUsageChanged()
        self.mw.filePanel.dirArchive.txtDirectory.setText(path)
        self.mw.filePanel.dirChanged(path)

    def dirPicturesChanged(self, path):
        logger.debug(f'Picture directory changed to {path}')
        self.dirPictures.txtDirectory.setText(path)
        self.mw.settings.setValue(self.pictureKey, path)
        self.mw.filePanel.control.dlgPicture.dirPictures.txtDirectory.setText(path)
        self.mw.filePanel.control.dlgPicture.dirChanged(path)

    def updateDiskUsage(self):
        _, size = self.mw.diskManager.list_files(self.dirArchive.text())
        str_size_in_GB = "{:.2f}".format(size / 1_000_000_000)
        self.grpDiskUsage.setTitle(f'Disk Usage (currently {str_size_in_GB} GB)')
        # ADD A CHECK FOR DIR SIZE EXCEEDING AVAILABLE SPACE AND REDUCE LIMIT IF NECESSARY