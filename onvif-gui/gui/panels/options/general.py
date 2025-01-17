#/********************************************************************
# libonvif/onvif-gui/gui/panels/options/general.py 
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
import sys
from PyQt6.QtWidgets import QMessageBox, QLineEdit, QSpinBox, \
    QGridLayout, QWidget, QCheckBox, QLabel, QComboBox, QPushButton, \
    QDialog, QTextEdit, QMessageBox, QFileDialog, QGroupBox, \
    QListWidget, QDialogButtonBox, QListWidgetItem
from PyQt6.QtCore import QFile, QRect, Qt, QSettings
from PyQt6.QtGui import QTextCursor, QTextOption
from loguru import logger
import avio
import webbrowser
import platform
from gui.player import Player
from gui.enums import Style, ProxyType

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


class ProfileItem(QListWidgetItem):
    def __init__(self, name):
        super().__init__(name)
        self.original = name
        if name != "Focus":
            self.setFlags(self.flags() | Qt.ItemFlag.ItemIsEditable)

class ProfileDialog(QDialog):
    def __init__(self, mw):
        super().__init__(mw)
        self.mw = mw
        self.editingItem = None
        self.deleted = []

        self.setModal(True)
        self.setWindowTitle("Manage Profiles")
        self.list = QListWidget()

        self.list.itemDoubleClicked.connect(self.onItemDoubleClicked)
        self.list.itemSelectionChanged.connect(self.onItemSelectionChanged)

        self.buttonBox = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttonBox.button(QDialogButtonBox.StandardButton.Cancel).setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.buttonBox.rejected.connect(self.reject)
        self.buttonBox.accepted.connect(self.accept)

        self.btnAdd = QPushButton("+")
        self.btnAdd.clicked.connect(self.btnAddClicked)
        self.btnDelete = QPushButton(" - ")
        self.btnDelete.clicked.connect(self.btnDeleteClicked)
        self.btnUp = QPushButton("^")
        self.btnUp.clicked.connect(self.btnUpClicked)
        self.btnDown = QPushButton("v")
        self.btnDown.clicked.connect(self.btnDownClicked)

        lytMain = QGridLayout(self)
        lytMain.addWidget(QLabel(),         0, 0, 1, 1)
        lytMain.addWidget(self.btnUp,      1, 0, 1, 1, Qt.AlignmentFlag.AlignCenter)
        lytMain.addWidget(self.btnDown,    2, 0, 1, 1, Qt.AlignmentFlag.AlignCenter)
        lytMain.addWidget(QLabel(),        3, 0, 1, 1)
        lytMain.addWidget(self.list,       0, 1, 4, 5)
        lytMain.addWidget(self.btnAdd,     4, 1, 1, 1, Qt.AlignmentFlag.AlignRight)
        lytMain.addWidget(self.btnDelete,  4, 2, 1, 1, Qt.AlignmentFlag.AlignLeft)
        lytMain.addWidget(self.buttonBox,  4, 3, 1, 3)
        lytMain.setRowStretch(0, 10)
        lytMain.setRowStretch(3, 10)

    def reject(self):
        self.hide()

    def accept(self):
        self.clearProfileNames()
        self.mw.settingsPanel.general.cmbViewerProfile.clear()
        for i in range(self.list.count()):
            name = self.list.item(i).text() 
            key = f'SettingsProfile_{i}'
            self.mw.settings.setValue(key, name)
            self.mw.settingsPanel.general.cmbViewerProfile.addItem(name)
        for name in self.deleted:
            settings = QSettings("onvif", name.text())
            settings.clear()
        self.hide()

    def onItemDoubleClicked(self, item):
        self.editingItem = item
        self.list.editItem(item)
        
    def onItemSelectionChanged(self):
        self.syncGui()

    def btnAddClicked(self):
        row = self.list.currentRow()+1
        item = ProfileItem("New Profile")
        self.list.insertItem(row, item)
        self.list.setCurrentRow(row)
        self.list.itemDoubleClicked.emit(item)
        self.syncGui()

    def btnDeleteClicked(self):
        self.deleted.append(self.list.takeItem(self.list.currentRow()))
        self.syncGui()

    def btnUpClicked(self):
        row = self.list.currentRow()
        item = self.list.takeItem(row)
        self.list.insertItem(row-1, item)
        self.list.setCurrentRow(row-1)
        self.syncGui()

    def btnDownClicked(self):
        row = self.list.currentRow()
        item = self.list.takeItem(row)
        self.list.insertItem(row+1, item)
        self.list.setCurrentRow(row+1)
        self.syncGui()

    def syncGui(self):
        self.btnDelete.setEnabled(self.list.count() > 1)
        if self.list.count():
            self.btnUp.setEnabled(self.list.currentRow() > 0)
            self.btnDown.setEnabled(self.list.currentRow() < self.list.count()-1)
        if item := self.list.currentItem():
            if item.text() == "Focus":
                self.btnDelete.setEnabled(False)

    def exec(self):
        self.list.clear()
        self.deleted.clear()
        cmb = self.mw.settingsPanel.general.cmbViewerProfile
        for i in range(cmb.count()):
            self.list.addItem(ProfileItem(cmb.itemText(i)))
        self.list.setCurrentRow(0)
        self.list.setFocus()
        self.syncGui()
        super().exec()

    def clearProfileNames(self):
        i = 0
        while self.mw.settings.contains(f'SettingsProfile_{i}'):
            self.mw.settings.remove(f'SettingsProfile_{i}')
            i += 1

class GeneralOptions(QWidget):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw
        self.process = None

        self.usernameKey = "settings/username"
        self.passwordKey = "settings/password"
        self.decoderKey = "settings/decoder"
        self.bufferSizeKey = "settings/bufferSize"
        self.lagTimeKey = "settings/lagTime"
        self.startFullScreenKey = "settings/startFullScreen"
        self.autoTimeSyncKey = "settings/autoTimeSync"
        self.alarmSoundFileKey = "settings/alarmSoundFile"
        self.alarmSoundVolumeKey = "settings/alarmSoundVolume"
        self.displayRefreshKey = "settings/displayRefresh"
        self.cacheMaxSizeKey = "settings/cacheMaxSize"
        self.audioDriverIndexKey = "settings/audioDriverIndex"
        self.appearanceKey = "settings/appearance"

        decoders = ["NONE"]
        if sys.platform == "win32":
            decoders += ["CUDA", "DXVA2", "D3D11VA"]
        if sys.platform == "linux":
            decoders += ["CUDA", "VAAPI", "VDPAU"]

        p = Player("", self)
        audioDrivers = p.getAudioDrivers()

        self.mediamtx_process = None

        self.dlgLog = None
        self.dlgProfiles = None

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

        self.cmbAudioDriver = QComboBox()
        self.cmbAudioDriver.addItems(audioDrivers)
        self.cmbAudioDriver.setCurrentIndex(int(mw.settings.value(self.audioDriverIndexKey, 0)))
        self.cmbAudioDriver.currentIndexChanged.connect(self.cmbAudioDriverChanged)
        lblAudioDrivers = QLabel("Audio Driver")

        self.cmbAppearance = QComboBox()
        self.cmbAppearance.addItems(["Dark", "Light"])
        self.cmbAppearance.setCurrentText(mw.settings.value(self.appearanceKey, "Dark"))
        self.cmbAppearance.currentTextChanged.connect(self.cmbAppearanceChanged)
        lblAppearance = QLabel("Appearance")

        self.chkStartFullScreen = QCheckBox("Start Full Screen")
        self.chkStartFullScreen.setChecked(bool(int(mw.settings.value(self.startFullScreenKey, 0))))
        self.chkStartFullScreen.stateChanged.connect(self.startFullScreenChecked)

        self.chkAutoTimeSync = QCheckBox("Auto Time Sync")
        self.chkAutoTimeSync.setChecked(bool(int(mw.settings.value(self.autoTimeSyncKey, 0))))
        self.chkAutoTimeSync.stateChanged.connect(self.autoTimeSyncChecked)

        pnlChecks = QWidget()
        lytChecks = QGridLayout(pnlChecks)
        lytChecks.addWidget(self.chkStartFullScreen,  0, 0, 1, 1)
        lytChecks.addWidget(self.chkAutoTimeSync,     0, 1, 1, 1)

        self.spnDisplayRefresh = QSpinBox()
        self.spnDisplayRefresh.setMinimum(1)
        self.spnDisplayRefresh.setMaximum(1000)
        self.spnDisplayRefresh.setMaximumWidth(80)
        refresh = 10
        if sys.platform == "win32":
            refresh = 20
        self.spnDisplayRefresh.setValue(int(self.mw.settings.value(self.displayRefreshKey, refresh)))
        self.spnDisplayRefresh.valueChanged.connect(self.spnDisplayRefreshChanged)
        lblDisplayRefresh = QLabel("Display Refresh Interval (ms)")

        self.spnCacheMax = QSpinBox()
        self.spnCacheMax.setMaximum(200)
        self.spnCacheMax.setValue(100)
        self.spnCacheMax.setMaximumWidth(80)
        self.spnCacheMax.setValue(int(self.mw.settings.value(self.cacheMaxSizeKey, 100)))
        self.spnCacheMax.valueChanged.connect(self.spnCacheMaxChanged)
        lblCacheMax = QLabel("Maximum Input Stream Cache Size")

        self.btnCloseAll = QPushButton("Start All")
        self.btnCloseAll.clicked.connect(self.btnCloseAllClicked)

        self.btnShowLogs = QPushButton("Show Logs")
        self.btnShowLogs.clicked.connect(self.btnShowLogsClicked)

        self.btnHelp = QPushButton("Help")
        self.btnHelp.clicked.connect(self.btnHelpClicked)

        self.btnTest = QPushButton("Test")
        self.btnTest.clicked.connect(self.btnTestClicked)

        grpNewView = QGroupBox("Open New Window")
        self.btnNewView = QPushButton("Open")
        self.btnNewView.clicked.connect(self.btnNewViewClicked)
        self.cmbViewerProfile = QComboBox()

        if self.mw.settings_profile == "gui":
            names = self.getProfileNames()
            for name in names:
                self.cmbViewerProfile.addItem(name)
        else:
            grpNewView.setEnabled(False)
        
        self.btnManageProfiles = QPushButton("...")
        self.btnManageProfiles.clicked.connect(self.btnManageProfilesClicked)
        lytNewView = QGridLayout(grpNewView)
        lytNewView.addWidget(self.btnNewView,        0, 0, 1, 1, Qt.AlignmentFlag.AlignLeft)
        lytNewView.addWidget(QLabel("Profile"),      0, 1, 1, 1)
        lytNewView.addWidget(self.cmbViewerProfile,  0, 2, 1, 1)
        lytNewView.addWidget(self.btnManageProfiles, 0, 3, 1, 1, Qt.AlignmentFlag.AlignRight)
        lytNewView.setColumnStretch(2, 10)

        pnlBuffer = QWidget()
        lytBuffer = QGridLayout(pnlBuffer)
        lytBuffer.addWidget(lblDisplayRefresh,      4, 0, 1, 3)
        lytBuffer.addWidget(self.spnDisplayRefresh, 4, 3, 1, 1)
        lytBuffer.addWidget(lblCacheMax,            5, 0, 1, 3)
        lytBuffer.addWidget(self.spnCacheMax,       5, 3, 1, 1)
        lytBuffer.setContentsMargins(0, 0, 0, 0)

        pnlButtons = QWidget()
        lytButtons = QGridLayout(pnlButtons)
        lytButtons.addWidget(self.btnCloseAll,   0, 0, 1, 1)
        lytButtons.addWidget(self.btnShowLogs,   0, 1, 1, 1)
        lytButtons.addWidget(self.btnHelp,       0, 2, 1, 1)
        #lytButtons.addWidget(self.btnTest,   1, 1, 1, 1)

        self.lblMemory = QLabel()

        lytMain = QGridLayout(self)
        lytMain.addWidget(lblUsername,         1, 0, 1, 1)
        lytMain.addWidget(self.txtUsername,    1, 1, 1, 1)
        lytMain.addWidget(lblPassword,         2, 0, 1, 1)
        lytMain.addWidget(self.txtPassword,    2, 1, 1, 1)
        lytMain.addWidget(lblDecoders,         3, 0, 1, 1)
        lytMain.addWidget(self.cmbDecoder,     3, 1, 1, 1)
        lytMain.addWidget(lblAudioDrivers,     4, 0, 1, 1)
        lytMain.addWidget(self.cmbAudioDriver, 4, 1, 1, 1)
        lytMain.addWidget(lblAppearance,       5, 0, 1, 1)
        lytMain.addWidget(self.cmbAppearance,  5, 1, 1, 1)
        lytMain.addWidget(pnlChecks,           6, 0, 1, 3)
        lytMain.addWidget(pnlBuffer,           7, 0, 1, 3)
        lytMain.addWidget(grpNewView,          8, 0, 1, 3)
        lytMain.addWidget(pnlButtons,          9, 0, 1, 3)
        lytMain.addWidget(self.lblMemory,     10, 0, 1, 3)
        lytMain.setRowStretch(10, 10)

    def usernameChanged(self, username):
        self.mw.settings.setValue(self.usernameKey, username)

    def passwordChanged(self, password):
        self.mw.settings.setValue(self.passwordKey, password)

    def cmbDecoderChanged(self, decoder):
        self.mw.settings.setValue(self.decoderKey, decoder)

    def cmbAudioDriverChanged(self, index):
        self.mw.settings.setValue(self.audioDriverIndexKey, index)
        if self.mw.audioStatus != avio.AudioStatus.UNINITIALIZED:
            QMessageBox.warning(self.mw, "Application Restart Required", "It is necessary to re-start Onvif GUI in order to enable this change")

    def cmbAppearanceChanged(self, text):
        self.mw.settings.setValue(self.appearanceKey, text)
        if text == "Dark":
            self.mw.setStyleSheet(self.mw.style(Style.DARK))
        if text == "Light":
            self.mw.setStyleSheet(self.mw.style(Style.LIGHT))

    def getProfileNames(self):
        result = []
        i = 0
        while self.mw.settings.contains(f'SettingsProfile_{i}'):
            result.append(self.mw.settings.value(f'SettingsProfile_{i}'))
            i += 1
        if not len(result):
            result.append("Focus")
        return result

    def autoDiscoverChecked(self, state):
        self.mw.settings.setValue(self.autoDiscoverKey, state)

    def startFullScreenChecked(self, state):
        self.mw.settings.setValue(self.startFullScreenKey, state)

    def autoTimeSyncChecked(self, state):
        self.mw.settings.setValue(self.autoTimeSyncKey, state)
        self.mw.cameraPanel.enableAutoTimeSync(state)

    def spnDisplayRefreshChanged(self, i):
        self.mw.settings.setValue(self.displayRefreshKey, i)
        self.mw.glWidget.timer.setInterval(i)

    def spnCacheMaxChanged(self, i):
        self.mw.settings.setValue(self.cacheMaxSizeKey, i)

    def cmbInterfacesChanged(self, network):
        self.mw.settings.setValue(self.interfaceKey, network)

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
     
    def btnCloseAllClicked(self):
        if self.btnCloseAll.text() == "Close All":
            self.mw.closeAllStreams()
        else:
            self.mw.startAllCameras()

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

    def btnNewViewClicked(self):
        from gui.main import MainWindow
        print(self.cmbViewerProfile.currentText())
        if self.cmbViewerProfile.currentText() == "Focus":
            if self.mw.settingsPanel.proxy.proxyType == ProxyType.STAND_ALONE:
                QMessageBox.warning(self.mw, "Feature Unavailable", "Focus Window is not available in Stand Alone Configuration, please use Proxy Server mode to enable this feature", QMessageBox.StandardButton.Ok)
                return
            if not self.mw.focus_window:
                self.mw.focus_window = MainWindow(settings_profile = self.cmbViewerProfile.currentText())
                self.mw.focus_window.audioStatus = self.mw.audioStatus
                self.mw.initializeFocusWindowSettings()
            else:
                self.mw.focus_window.show()
        else:
            window = MainWindow(settings_profile = self.cmbViewerProfile.currentText())
            window.audioStatus = self.mw.audioStatus
            self.mw.external_windows.append(window)
            window.show()

    def btnManageProfilesClicked(self):
        if not self.dlgProfiles:
            self.dlgProfiles = ProfileDialog(self.mw)
        self.dlgProfiles.exec()

    def btnTestClicked(self):
        print("test button clicked")
