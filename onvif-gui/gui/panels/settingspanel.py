#/********************************************************************
# libonvif/onvif-gui/gui/panels/settingspanel.py 
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
from PyQt6.QtWidgets import QMessageBox, QLineEdit, QSpinBox, \
    QGridLayout, QWidget, QCheckBox, QLabel, QComboBox, QPushButton, \
    QDialog, QTextEdit, QMessageBox, QDialogButtonBox, QRadioButton, \
    QGroupBox, QSlider, QFileDialog
from PyQt6.QtCore import Qt, QStandardPaths, QFile, QRegularExpression, QRect
from PyQt6.QtGui import QRegularExpressionValidator, QTextCursor, QTextOption
from loguru import logger
from gui.components import DirectorySelector
import libonvif as onvif
import avio
import shutil
from time import sleep
import webbrowser
import platform

class AddCameraDialog(QDialog):
    def __init__(self, mw):
        super().__init__(mw)
        self.mw = mw

        ipRange = "(?:[0-1]?[0-9]?[0-9]|2[0-4][0-9]|25[0-5])" 
        ipRegex = QRegularExpression("^" + ipRange + "\\." + ipRange + "\\." + ipRange + "\\." + ipRange + "$")
        ipValidator = QRegularExpressionValidator(ipRegex, self)           
    
        self.setWindowTitle("Add Camera")
        self.txtIPAddress = QLineEdit()
        self.txtIPAddress.setValidator(ipValidator)
        self.lblIPAddress = QLabel("IP Address")
        self.txtOnvifPort = QLineEdit()
        self.lblOnvifPort = QLabel("Onvif Port")

        buttonBox = QDialogButtonBox( \
            QDialogButtonBox.StandardButton.Ok | \
            QDialogButtonBox.StandardButton.Cancel)

        lytMain = QGridLayout(self)
        lytMain.addWidget(self.lblIPAddress,   1, 0, 1, 1)
        lytMain.addWidget(self.txtIPAddress,   1, 1, 1, 1)
        lytMain.addWidget(self.lblOnvifPort,   2, 0, 1, 1)
        lytMain.addWidget(self.txtOnvifPort,   2, 1, 1, 1)
        lytMain.addWidget(buttonBox,           5, 0, 1, 2)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

        self.txtIPAddress.setFocus()

class LogText(QTextEdit):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWordWrapMode(QTextOption.WrapMode.NoWrap)

    def scrollToBottom(self):
        self.moveCursor(QTextCursor.MoveOperation.End, QTextCursor.MoveMode.MoveAnchor)
        self.ensureCursorVisible()

class LogDialog(QDialog):
    def __init__(self, mw):
        super().__init__(mw)
        self.mw = mw
        self.geometryKey = "LogDialog/geometry"
        rect = self.mw.settings.value(self.geometryKey, QRect(0, 0, 480, 640))
        if rect is not None:
            if rect.width() and rect.height():
                self.setGeometry(rect)

        self.editor = LogText(self)
        self.editor.setReadOnly(True)
        self.editor.setFontFamily("courier")

        self.lblSize = QLabel("Log File Size:")
        self.btnArchive = QPushButton("Archive")
        self.btnArchive.clicked.connect(self.btnArchiveClicked)
        self.btnClear = QPushButton("  Clear  ")
        self.btnClear.clicked.connect(self.btnClearClicked)

        pnlButton = QWidget()
        lytButton = QGridLayout(pnlButton)
        lytButton.addWidget(self.btnArchive,   0, 1, 1, 1)
        lytButton.addWidget(self.btnClear,     0, 2, 1, 1)
        lytButton.setContentsMargins(0, 0, 0, 0)

        panel = QWidget()
        lytPanel = QGridLayout(panel)
        lytPanel.addWidget(self.lblSize,    0, 0, 1, 1)
        lytPanel.addWidget(pnlButton,       0, 1, 1, 1)
        lytPanel.addWidget(QLabel(),        0, 2, 1, 1)
        lytPanel.setColumnStretch(2, 10)

        lyt = QGridLayout(self)
        lyt.addWidget(self.editor, 0, 0, 1, 1)
        lyt.addWidget(panel,       1, 0, 1, 1)
        lyt.setRowStretch(0, 10)

    def closeEvent(self, e):
        self.mw.settings.setValue(self.geometryKey, self.geometry())

    def readLogFile(self, path):
        data = ""
        if os.path.isfile(path):
            with open(path, 'r') as file:
                data = file.read()
        self.setWindowTitle(path)
        self.editor.setText(data)
        y = "{:.2f}".format(os.stat(path).st_size/1000000)
        self.lblSize.setText(f'Log File Size: {y}  MB    ')
        self.editor.scrollToBottom()

    def btnArchiveClicked(self):
        path = None
        if platform.system() == "Linux":
            path = QFileDialog.getOpenFileName(self, "Select File", self.windowTitle(), options=QFileDialog.Option.DontUseNativeDialog)[0]
        else:
            path = QFileDialog.getOpenFileName(self, "Select File", self.windowTitle())[0]
        if path:
            if len(path) > 0:
                self.readLogFile(path)

    def btnClearClicked(self):
        filename = self.windowTitle()
        ret = QMessageBox.warning(self, "Deleting File",
                                    f'\n{filename}\n\n'
                                    "You are about to delete this file.\n"
                                    "Are you sure you want to continue?",
                                    QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)

        if ret == QMessageBox.StandardButton.Ok:
            if filename == self.mw.settingsPanel.getLogFilename():
                ret = QMessageBox.warning(self, "Deleting Current Log",
                                          "You are about to delete the current log.\n"
                                          "Are you sure you want to continue?",
                                          QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
                if ret == QMessageBox.StandardButton.Ok:
                    QFile.remove(filename)
                    logger.add(filename)
                    logger.debug("Created new log file")
                    self.readLogFile(filename)
            else:
                QFile.remove(filename)
                self.readLogFile(self.mw.settingsPanel.getLogFilename())
                QMessageBox.information(self, "Current Log Displayed", "The current log has been loaded into the display", QMessageBox.StandardButton.Ok)

class SettingsPanel(QWidget):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw
        self.usernameKey = "settings/username"
        self.passwordKey = "settings/password"
        self.decoderKey = "settings/decoder"
        self.bufferSizeKey = "settings/bufferSize"
        self.lagTimeKey = "settings/lagTime"
        self.interfaceKey = "settings/interface"
        self.autoDiscoverKey = "settings/autoDiscover"
        self.startFullScreenKey = "settings/startFullScreen"
        self.autoTimeSyncKey = "settings/autoTimeSync"
        self.autoStartKey = "settings/autoStart"
        self.archiveKey = "settings/archive"
        self.pictureKey = "settings/picture"
        self.scanAllKey = "settings/scanAll"
        self.cameraListKey = "settings/cameraList"
        self.discoveryTypeKey = "settings/discoveryType"
        self.alarmSoundFileKey = "settings/alarmSoundFile"
        self.alarmSoundVolumeKey = "settings/alarmSoundVolume"
        self.diskLimitKey = "settings/diskLimit"
        self.mangageDiskUsagekey = "settings/manageDiskUsage"
        self.displayRefreshKey = "settings/displayRefresh"
        self.cacheMaxSizeKey = "settings/cacheMaxSize"

        decoders = ["NONE"]
        if sys.platform == "win32":
            decoders += ["CUDA", "DXVA2", "D3D11VA"]
        if sys.platform == "linux":
            decoders += ["CUDA", "VAAPI", "VDPAU"]

        self.dlgLog = None

        self.txtUsername = QLineEdit()
        self.txtUsername.setText(mw.settings.value(self.usernameKey, ""))
        self.txtUsername.textChanged.connect(self.usernameChanged)
        lblUsername = QLabel("Common Username")
        
        self.txtPassword = QLineEdit()
        self.txtPassword.setText(mw.settings.value(self.passwordKey, ""))
        self.txtPassword.textChanged.connect(self.passwordChanged)
        lblPassword = QLabel("Common Password")
        
        self.cmbDecoder = QComboBox()
        self.cmbDecoder.addItems(decoders)
        self.cmbDecoder.setCurrentText(mw.settings.value(self.decoderKey, "NONE"))
        self.cmbDecoder.currentTextChanged.connect(self.cmbDecoderChanged)
        lblDecoders = QLabel("Hardware Decoder")

        self.chkStartFullScreen = QCheckBox("Start Full Screen")
        self.chkStartFullScreen.setChecked(bool(int(mw.settings.value(self.startFullScreenKey, 0))))
        self.chkStartFullScreen.stateChanged.connect(self.startFullScreenChecked)

        self.chkAutoDiscover = QCheckBox("Auto Discovery")
        self.chkAutoDiscover.setChecked(bool(int(mw.settings.value(self.autoDiscoverKey, 0))))
        self.chkAutoDiscover.stateChanged.connect(self.autoDiscoverChecked)
        
        self.chkAutoTimeSync = QCheckBox("Auto Time Sync")
        self.chkAutoTimeSync.setChecked(bool(int(mw.settings.value(self.autoTimeSyncKey, 0))))
        self.chkAutoTimeSync.stateChanged.connect(self.autoTimeSyncChecked)

        self.chkAutoStart = QCheckBox("Auto Start")
        self.chkAutoStart.setChecked(bool(int(mw.settings.value(self.autoStartKey, 0))))
        self.chkAutoStart.stateChanged.connect(self.autoStartChecked)

        pnlChecks = QWidget()
        lytChecks = QGridLayout(pnlChecks)
        lytChecks.addWidget(self.chkStartFullScreen,  0, 0, 1, 1)
        lytChecks.addWidget(self.chkAutoDiscover,     0, 1, 1, 1)
        lytChecks.addWidget(self.chkAutoStart,        1, 0, 1, 1)
        lytChecks.addWidget(self.chkAutoTimeSync,     1, 1, 1, 1)

        self.spnBufferSize = QSpinBox()
        self.spnBufferSize.setMinimum(1)
        self.spnBufferSize.setMaximum(60)
        self.spnBufferSize.setMaximumWidth(80)
        self.spnBufferSize.setValue(int(self.mw.settings.value(self.bufferSizeKey, 10)))
        self.spnBufferSize.valueChanged.connect(self.spnBufferSizeChanged)
        lblBufferSize = QLabel("Pre-Alarm Buffer Size (in seconds)")

        self.spnLagTime = QSpinBox()
        self.spnLagTime.setMinimum(1)
        self.spnLagTime.setMaximum(60)
        self.spnLagTime.setMaximumWidth(80)
        self.spnLagTime.setValue(int(self.mw.settings.value(self.lagTimeKey, 5)))
        self.spnLagTime.valueChanged.connect(self.spnLagTimeChanged)
        lblLagTime = QLabel("Post-Alarm Lag Time (in seconds)")

        self.spnDisplayRefresh = QSpinBox()
        self.spnDisplayRefresh.setMinimum(1)
        self.spnDisplayRefresh.setMaximum(1000)
        self.spnDisplayRefresh.setMaximumWidth(80)
        refresh = 10
        if sys.platform == "win32":
            refresh = 20
        self.spnDisplayRefresh.setValue(int(self.mw.settings.value(self.displayRefreshKey, refresh)))
        self.spnDisplayRefresh.valueChanged.connect(self.spnDisplayRefreshChanged)
        lblDisplayRefresh = QLabel("Display Refresh Interval (in milliseconds)")

        self.spnCacheMax = QSpinBox()
        self.spnCacheMax.setMaximum(200)
        self.spnCacheMax.setValue(100)
        self.spnCacheMax.setMaximumWidth(80)
        self.spnCacheMax.setValue(int(self.mw.settings.value(self.cacheMaxSizeKey, 100)))
        self.spnCacheMax.valueChanged.connect(self.spnCacheMaxChanged)
        lblCacheMax = QLabel("Maximum Input Stream Cache Size")

        self.cmbSoundFiles = QComboBox()
        d = f'{self.mw.getLocation()}/gui/resources'
        sounds = [f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f)) and f.endswith(".mp3")]
        self.cmbSoundFiles.addItems(sounds)
        self.cmbSoundFiles.currentTextChanged.connect(self.cmbSoundFilesChanged)
        self.cmbSoundFiles.setCurrentText(self.mw.settings.value(self.alarmSoundFileKey, "drops.mp3"))
        lblSoundFiles = QLabel("Alarm Sounds")
        self.sldAlarmVolume = QSlider(Qt.Orientation.Horizontal)
        self.sldAlarmVolume.setValue(int(self.mw.settings.value(self.alarmSoundVolumeKey, 80)))
        self.sldAlarmVolume.valueChanged.connect(self.sldAlarmVolumeChanged)

        pnlSoundFile = QWidget()
        lytSoundFile =  QGridLayout(pnlSoundFile)
        lytSoundFile.addWidget(lblSoundFiles,        0, 0, 1, 1)
        lytSoundFile.addWidget(self.cmbSoundFiles,   0, 1, 1, 1)
        lytSoundFile.addWidget(self.sldAlarmVolume,  0, 2, 1, 1)
        lytSoundFile.setColumnStretch(1, 10)

        pnlBuffer = QWidget()
        lytBuffer = QGridLayout(pnlBuffer)
        lytBuffer.addWidget(lblBufferSize,          1, 0, 1, 3)
        lytBuffer.addWidget(self.spnBufferSize,     1, 3, 1, 1)
        lytBuffer.addWidget(lblLagTime,             2, 0, 1, 3)
        lytBuffer.addWidget(self.spnLagTime,        2, 3, 1, 1)
        lytBuffer.addWidget(pnlSoundFile,           3, 0, 1, 4)
        lytBuffer.addWidget(lblDisplayRefresh,      4, 0, 1, 3)
        lytBuffer.addWidget(self.spnDisplayRefresh, 4, 3, 1, 1)
        lytBuffer.addWidget(lblCacheMax,            5, 0, 1, 3)
        lytBuffer.addWidget(self.spnCacheMax,       5, 3, 1, 1)
        lytBuffer.setContentsMargins(0, 0, 0, 0)

        self.grpDiscoverType = QGroupBox("Set Camera Discovery Method")
        self.radDiscover = QRadioButton("Discover Broadcast", self.grpDiscoverType )
        self.radDiscover.setChecked(int(self.mw.settings.value(self.discoveryTypeKey, 1)))
        self.radDiscover.toggled.connect(self.radDiscoverToggled)
        self.radCached = QRadioButton("Cached Addresses", self.grpDiscoverType )
        self.radCached.setChecked(not self.radDiscover.isChecked())
        lytDiscoverType = QGridLayout(self.grpDiscoverType )
        lytDiscoverType.addWidget(self.radDiscover,   0, 0, 1, 1)
        lytDiscoverType.addWidget(self.radCached,     0, 1, 1, 1)

        self.chkScanAllNetworks = QCheckBox("Scan All Networks During Discovery")
        self.chkScanAllNetworks.setChecked(int(mw.settings.value(self.scanAllKey, 1)))
        self.chkScanAllNetworks.stateChanged.connect(self.scanAllNetworksChecked)
        self.cmbInterfaces = QComboBox()
        intf = self.mw.settings.value(self.interfaceKey, "")
        self.lblInterfaces = QLabel("Network")
        session = onvif.Session()
        session.getActiveInterfaces()
        i = 0
        while len(session.active_interface(i)) > 0 and i < 16:
            self.cmbInterfaces.addItem(session.active_interface(i))
            i += 1
        if len(intf) > 0:
            self.cmbInterfaces.setCurrentText(intf)
        self.cmbInterfaces.currentTextChanged.connect(self.cmbInterfacesChanged)
        self.cmbInterfaces.setEnabled(not self.chkScanAllNetworks.isChecked())
        self.lblInterfaces.setEnabled(not self.chkScanAllNetworks.isChecked())

        self.btnAddCamera = QPushButton("Add Camera")
        self.btnAddCamera.clicked.connect(self.btnAddCameraClicked)

        pnlInterface = QGroupBox("Discovery Options")
        lytInterface = QGridLayout(pnlInterface)
        lytInterface.addWidget(self.grpDiscoverType,     0, 0, 1, 2)
        lytInterface.addWidget(self.chkScanAllNetworks,  2, 0, 1, 2)
        lytInterface.addWidget(self.lblInterfaces,       4, 0, 1, 1)
        lytInterface.addWidget(self.cmbInterfaces,       4, 1, 1, 1)
        lytInterface.addWidget(self.btnAddCamera,        5, 0, 1, 2, Qt.AlignmentFlag.AlignCenter)
        lytInterface.setColumnStretch(1, 10)
        lytInterface.setContentsMargins(10, 10, 10, 10)

        self.radDiscoverToggled(self.radDiscover.isChecked())

        video_dirs = QStandardPaths.standardLocations(QStandardPaths.StandardLocation.MoviesLocation)
        self.dirArchive = DirectorySelector(mw, self.archiveKey, "Archive Dir", video_dirs[0])
        self.dirArchive.signals.dirChanged.connect(self.dirArchiveChanged)

        picture_dirs = QStandardPaths.standardLocations(QStandardPaths.StandardLocation.PicturesLocation)
        self.dirPictures = DirectorySelector(mw, self.pictureKey, "Picture Dir", picture_dirs[0])
        self.dirPictures.signals.dirChanged.connect(self.dirPicturesChanged)

        dir_size = "{:.2f}".format(self.getDirectorySize(self.dirArchive.text()) / 1000000000)
        self.grpDiskUsage = QGroupBox(f'Disk Usage (currently {dir_size} GB)')
        self.spnDiskLimit = QSpinBox()
        max_size = int(self.getMaximumDirectorySize())
        self.spnDiskLimit.setMaximum(max_size)
        disk_limit = min(int(self.mw.settings.value(self.diskLimitKey, 100)), max_size)
        self.spnDiskLimit.setValue(disk_limit)
        self.spnDiskLimit.valueChanged.connect(self.spnDiskLimitChanged)

        lbl = f'Auto Manage (max {max_size} GB)'
        self.chkManageDiskUsage = QCheckBox(lbl)
        self.chkManageDiskUsage.setChecked(bool(int(self.mw.settings.value(self.mangageDiskUsagekey, 0))))
        self.chkManageDiskUsage.clicked.connect(self.chkManageDiskUsageChanged)

        lytDiskUsage = QGridLayout(self.grpDiskUsage)
        lytDiskUsage.addWidget(self.chkManageDiskUsage, 0, 0, 1, 1)
        lytDiskUsage.addWidget(self.spnDiskLimit,       0, 2, 1, 1)
        lytDiskUsage.addWidget(QLabel("GB"),            0, 3, 1, 1)
        lytDiskUsage.addWidget(self.dirArchive,         1, 0, 1, 4)
        lytDiskUsage.addWidget(self.dirPictures,        2, 0, 1, 4)

        self.btnCloseAll = QPushButton("Start All Cameras")
        self.btnCloseAll.clicked.connect(self.btnCloseAllClicked)

        self.btnShowLogs = QPushButton("Show Logs")
        self.btnShowLogs.clicked.connect(self.btnShowLogsClicked)

        self.btnHelp = QPushButton("Help")
        self.btnHelp.clicked.connect(self.btnHelpClicked)

        pnlButtons = QWidget()
        lytButtons = QGridLayout(pnlButtons)
        lytButtons.addWidget(self.btnCloseAll,   0, 0, 1, 1)
        lytButtons.addWidget(self.btnShowLogs,   0, 1, 1, 1)
        lytButtons.addWidget(self.btnHelp,       0, 2, 1, 1)

        self.lblSpacer = QLabel("")

        lytMain = QGridLayout(self)
        lytMain.addWidget(lblUsername,         1, 0, 1, 1)
        lytMain.addWidget(self.txtUsername,    1, 1, 1, 1)
        lytMain.addWidget(lblPassword,         2, 0, 1, 1)
        lytMain.addWidget(self.txtPassword,    2, 1, 1, 1)
        lytMain.addWidget(lblDecoders,         3, 0, 1, 1)
        lytMain.addWidget(self.cmbDecoder,     3, 1, 1, 1)
        lytMain.addWidget(pnlChecks,           4, 0, 1, 3)
        lytMain.addWidget(pnlBuffer,           5, 0, 1, 3)
        lytMain.addWidget(pnlInterface,        6, 0, 1, 3)
        lytMain.addWidget(self.grpDiskUsage,   7, 0, 1, 3)
        lytMain.addWidget(pnlButtons,          8, 0, 1, 3)
        lytMain.addWidget(self.lblSpacer,      9, 0, 1, 3)
        lytMain.setRowStretch(9, 10)

    def showEvent(self, event):
        self.lblSpacer.setFocus()
        return super().showEvent(event)

    def usernameChanged(self, username):
        self.mw.settings.setValue(self.usernameKey, username)

    def passwordChanged(self, password):
        self.mw.settings.setValue(self.passwordKey, password)

    def cmbDecoderChanged(self, decoder):
        self.mw.settings.setValue(self.decoderKey, decoder)

    def autoDiscoverChecked(self, state):
        self.mw.settings.setValue(self.autoDiscoverKey, state)

    def startFullScreenChecked(self, state):
        self.mw.settings.setValue(self.startFullScreenKey, state)

    def autoTimeSyncChecked(self, state):
        self.mw.settings.setValue(self.autoTimeSyncKey, state)
        self.mw.cameraPanel.enableAutoTimeSync(state)

    def autoStartChecked(self, state):
        self.mw.settings.setValue(self.autoStartKey, state)

    def spnDisplayRefreshChanged(self, i):
        self.mw.settings.setValue(self.displayRefreshKey, i)
        self.mw.glWidget.timer.setInterval(i)

    def spnCacheMaxChanged(self, i):
        self.mw.settings.setValue(self.cacheMaxSizeKey, i)

    def spnBufferSizeChanged(self, i):
        self.mw.settings.setValue(self.bufferSizeKey, i)

    def spnLagTimeChanged(self, i):
        self.mw.settings.setValue(self.lagTimeKey, i)

    def cmbInterfacesChanged(self, network):
        self.mw.settings.setValue(self.interfaceKey, network)

    def onMediaStarted(self):
        if len(self.mw.pm.players):
            self.btnCloseAll.setText("Close All Streams")

    def onMediaStopped(self):
        if not len(self.mw.pm.players):
            self.btnCloseAll.setText("Start All Cameras")


    def btnCloseAllClicked(self):
        try:
            if self.btnCloseAll.text() == "Close All Streams":
                for player in self.mw.pm.players:
                    player.requestShutdown()
                for timer in self.mw.timers.values():
                    timer.stop()
                self.mw.pm.auto_start_mode = False
                lstCamera = self.mw.cameraPanel.lstCamera
                if lstCamera:
                    cameras = [lstCamera.item(x) for x in range(lstCamera.count())]
                    for camera in cameras:
                        camera.setIconIdle()

                count = 0
                while len(self.mw.pm.players):
                    sleep(0.1)
                    count += 1
                    if count > 200:
                        logger.debug("not all players closed within the allotted time, flushing player manager")
                        self.mw.pm.players.clear()
                        break

                self.mw.pm.ordinals.clear()
                self.mw.pm.sizes.clear()
                self.mw.cameraPanel.syncGUI()
            else:
                lstCamera = self.mw.cameraPanel.lstCamera
                if lstCamera:
                    cameras = [lstCamera.item(x) for x in range(lstCamera.count())]
                    for camera in cameras:
                        self.mw.cameraPanel.setCurrentCamera(camera.uri())
                        self.mw.cameraPanel.onItemDoubleClicked(camera)
        except Exception as ex:
            logger.error(ex)

    def scanAllNetworksChecked(self, state):
        self.mw.settings.setValue(self.scanAllKey, state)
        self.cmbInterfaces.setEnabled(not self.chkScanAllNetworks.isChecked())
        self.lblInterfaces.setEnabled(not self.chkScanAllNetworks.isChecked())

    def getLogFilename(self):
        filename = ""
        if sys.platform == "win32":
            filename = os.environ['HOMEPATH'] + "/.cache/onvif-gui/logs.txt"
        else:
            filename = os.environ['HOME'] + "/.cache/onvif-gui/logs.txt"
        return filename

    def btnShowLogsClicked(self):
        filename = self.getLogFilename()
        if not self.dlgLog:
            self.dlgLog = LogDialog(self.mw)
        self.dlgLog.readLogFile(filename)
        self.dlgLog.exec()

    def btnHelpClicked(self):
        result = webbrowser.get().open("https://github.com/sr99622/libonvif#readme-ov-file")
        if not result:
            webbrowser.get().open("https://github.com/sr99622/libonvif")

    def radDiscoverToggled(self, checked):
        self.chkScanAllNetworks.setEnabled(checked)
        if self.chkScanAllNetworks.isChecked():
            self.lblInterfaces.setEnabled(False)
            self.cmbInterfaces.setEnabled(False)
        else:
            self.lblInterfaces.setEnabled(checked)
            self.cmbInterfaces.setEnabled(checked)
        self.mw.settings.setValue(self.discoveryTypeKey, int(checked))

    def radEntireDiskToggled(self, checked):
        self.spnDiskLimit.setEnabled(not checked)
        self.mw.settings.setValue(self.entireDiskKey, int(checked))

    def spnDiskLimitChanged(self, value):
        self.mw.settings.setValue(self.diskLimitKey, value)

    def cmbSoundFilesChanged(self, value):
        self.mw.settings.setValue(self.alarmSoundFileKey, value)

    def sldAlarmVolumeChanged(self, value):
        self.mw.settings.setValue(self.alarmSoundVolumeKey, value)

    def getDecoder(self):
        result = avio.AV_HWDEVICE_TYPE_NONE
        if self.cmbDecoder.currentText() == "CUDA":
            result = avio.AV_HWDEVICE_TYPE_CUDA
        if self.cmbDecoder.currentText() == "VAAPI":
            result = avio.AV_HWDEVICE_TYPE_VAAPI
        if self.cmbDecoder.currentText() == "VDPAU":
            result = avio.AV_HWDEVICE_TYPE_VDPAU
        if self.cmbDecoder.currentText() == "DXVA2":
            result = avio.AV_HWDEVICE_TYPE_DXVA2
        if self.cmbDecoder.currentText() == "D3D11VA":
            result = avio.AV_HWDEVICE_TYPE_D3D11VA
        return result
     
    def btnAddCameraClicked(self):
        dlg = AddCameraDialog(self.mw)
        if dlg.exec():
            ip_address = dlg.txtIPAddress.text()
            onvif_port = dlg.txtOnvifPort.text()
            if not len(onvif_port):
                onvif_port = "80"
            xaddrs = f'http://{ip_address}:{onvif_port}/onvif/device_service'
            logger.debug(f'Attempting to add camera manually using xaddrs: {xaddrs}')
            data = onvif.Data()
            data.getData = self.mw.cameraPanel.getData
            data.getCredential = self.mw.cameraPanel.getCredential
            data.setXAddrs(xaddrs)
            data.setDeviceService(xaddrs)
            data.manual_fill()

    def chkManageDiskUsageChanged(self):
        if self.chkManageDiskUsage.isChecked():
            ret = QMessageBox.warning(self, "** WARNING **",
                                        "You are giving full control of the archive directory to this program.  "
                                        "Any files contained within this directory or its sub-directories are subject to deletion.  "
                                        "You should only enable this feature if you are sure that this is ok.\n\n"
                                        "Are you sure you want to continue?",
                                        QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
            if ret == QMessageBox.StandardButton.Cancel:
                self.chkManageDiskUsage.setChecked(False)
        self.mw.settings.setValue(self.mangageDiskUsagekey, int(self.chkManageDiskUsage.isChecked()))

    def dirArchiveChanged(self, path):
        logger.debug(f'Video archive directory changed to {path}')
        self.mw.settings.setValue(self.archiveKey, path)
        max_size = int(self.getMaximumDirectorySize())
        self.spnDiskLimit.setMaximum(max_size)
        lbl = f'Auto Manage (max {max_size} GB)'
        self.chkManageDiskUsage.setText(lbl)
        disk_limit = min(int(self.mw.settings.value(self.diskLimitKey, 100)), max_size)
        self.spnDiskLimit.setValue(disk_limit)
        self.chkManageDiskUsageChanged()

    def dirPicturesChanged(self, path):
        logger.debug(f'Picture directory changed to {path}')
        self.mw.settings.setValue(self.pictureKey, path)

    def getMaximumDirectorySize(self):
        # compute disk space available for archive directory in GB
        d = self.dirArchive.txtDirectory.text()
        d_size = 0
        for dirpath, dirnames, filenames in os.walk(d):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    d_size += os.path.getsize(fp)
        total, used, free = shutil.disk_usage(d)
        max_available = (free + d_size - 10000000000) / 1000000000
        return max_available

    def getDirectorySize(self, d):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(d):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
        
        return total_size
    
