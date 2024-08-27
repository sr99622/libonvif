#/********************************************************************
# libonvif/onvif-gui/gui/onvif/systemtab.py 
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

from PyQt6.QtWidgets import QGridLayout, QWidget, QPushButton, QGroupBox, \
    QMessageBox, QRadioButton, QComboBox, QLabel
from PyQt6.QtCore import Qt
import datetime
import pathlib
import webbrowser

class SystemTabSettings():
    def __init__(self, mw, camera):
        self.camera = camera
        self.mw = mw

        self.record_enable = self.getRecordAlarmEnabled()
        self.record_always = self.getRecordAlways()
        self.record_alarm = self.getRecordOnAlarm()
        self.sound_alarm_enable = self.getSoundAlarmEnabled()
        self.sound_alarm_once = self.getSoundAlarmOnce()
        self.sound_alarm_loop = self.getSoundAlarmLoop()
        self.record_profile = self.getRecordProfile()

    def managePlayers(self):
        record = False
        if self.record_enable:
            if self.record_always or (self.record_alarm and self.camera.isAlarming()):
                record = True
        if record:
            profile = self.camera.getRecordProfile()
            if profile:
                player = self.mw.pm.getPlayer(profile.uri())
                if player:
                    if not player.isRecording():
                        d = self.mw.settingsPanel.storage.dirArchive.txtDirectory.text()
                        filename = player.getPipeOutFilename(d)
                        if filename:
                            player.toggleRecording(filename)
        else:
            players = self.mw.pm.getStreamPairPlayers(self.camera.uri())
            for player in players:
                if player.isRecording():
                    player.toggleRecording("")
                
        self.mw.cameraPanel.syncGUI()

    def getRecordProfile(self):
        key = f'{self.camera.serial_number()}/RecordProfile'
        return int(self.mw.settings.value(key, 0))
    
    def setRecordProfile(self, ordinal):
        self.record_profile = ordinal
        key = f'{self.camera.serial_number()}/RecordProfile'
        self.mw.settings.setValue(key, ordinal)

    def getRecordAlarmEnabled(self):
        key = f'{self.camera.serial_number()}/RecordAlarmEnabled'
        return bool(int(self.mw.settings.value(key, 0)))
    
    def setRecordAlarmEnabled(self, state):
        self.record_enable = bool(state)
        key = f'{self.camera.serial_number()}/RecordAlarmEnabled'
        self.managePlayers()
        self.mw.settings.setValue(key, int(state))

    def getRecordAlways(self):
        key = f'{self.camera.serial_number()}/RecordAlways'
        return bool(int(self.mw.settings.value(key, 0)))
    
    def setRecordAlways(self, state):
        self.record_always = bool(state)
        key = f'{self.camera.serial_number()}/RecordAlways'
        self.mw.settings.setValue(key, int(state))
        self.managePlayers()

    def getRecordOnAlarm(self):
        key = f'{self.camera.serial_number()}/RecordOnAlarm'
        return bool(int(self.mw.settings.value(key, 1)))
    
    def setRecordOnAlarm(self, state):
        self.record_alarm = bool(state)
        key = f'{self.camera.serial_number()}/RecordOnAlarm'
        self.mw.settings.setValue(key, int(state))

    def getSoundAlarmEnabled(self):
        key = f'{self.camera.serial_number()}/SoundAlarmEnabled'
        return bool(int(self.mw.settings.value(key, 0)))

    def setSoundAlarmEnabled(self, state):
        self.sound_alarm_enable = bool(state)
        key = f'{self.camera.serial_number()}/SoundAlarmEnabled'
        self.mw.settings.setValue(key, int(state))

    def getSoundAlarmOnce(self):
        key = f'{self.camera.serial_number()}/SoundAlarmOnce'
        return bool(int(self.mw.settings.value(key, 0)))

    def setSoundAlarmOnce(self, state):
        self.sound_alarm_once = bool(state)
        key = f'{self.camera.serial_number()}/SoundAlarmOnce'
        self.mw.settings.setValue(key, int(state))

    def getSoundAlarmLoop(self):
        key = f'{self.camera.serial_number()}/SoundAlarmLoop'
        return bool(int(self.mw.settings.value(key, 1)))

    def setSoundAlarmLoop(self, state):
        self.sound_alarm_loop = bool(state)
        key = f'{self.camera.serial_number()}/SoundAlarmLoop'
        self.mw.settings.setValue(key, int(state))

class SystemTab(QWidget):
    def __init__(self, cp):
        super().__init__()
        self.cp = cp

        self.radRecordAlways = QRadioButton("Always")
        self.radRecordAlways.clicked.connect(self.radRecordAlwaysClicked)
        self.radRecordAlways.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.radRecordOnAlarm = QRadioButton("Alarms")
        self.radRecordOnAlarm.clicked.connect(self.radRecordOnAlarmClicked)
        self.radRecordOnAlarm.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.grpRecord = QGroupBox("Record")
        self.grpRecord.setCheckable(True)
        self.grpRecord.clicked.connect(self.grpRecordClicked)
        self.grpRecord.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        lytGroup = QGridLayout(self.grpRecord)
        lytGroup.addWidget(self.radRecordAlways,    0, 0, 1, 1)
        lytGroup.addWidget(self.radRecordOnAlarm,   1, 0, 1, 1)

        self.radSoundOnce = QRadioButton("Once")
        self.radSoundOnce.clicked.connect(self.radSoundOnceClicked)
        self.radSoundOnce.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.radSoundLoop = QRadioButton("Loop")
        self.radSoundLoop.clicked.connect(self.radSoundLoopClicked)
        self.radSoundLoop.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.grpSounds = QGroupBox("Sounds")
        self.grpSounds.setCheckable(True)
        self.grpSounds.clicked.connect(self.grpSoundsClicked)
        self.grpSounds.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        lytGroupSounds = QGridLayout(self.grpSounds)
        lytGroupSounds.addWidget(self.radSoundOnce,   0, 0, 1, 1)
        lytGroupSounds.addWidget(self.radSoundLoop,   1, 0, 1, 1)

        self.cmbRecordProfile = QComboBox()
        self.cmbRecordProfile.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.cmbRecordProfile.currentIndexChanged.connect(self.cmbRecordProfileChanged)
        self.lblRecordProfile = QLabel("    Record    ")
        pnlRecordProfile = QWidget()
        lytRecordProfile = QGridLayout(pnlRecordProfile)
        lytRecordProfile.addWidget(self.lblRecordProfile, 0, 0, 1, 1)
        lytRecordProfile.addWidget(self.cmbRecordProfile, 0, 1, 1, 1)
        lytRecordProfile.setColumnStretch(1, 5)
        lytRecordProfile.setContentsMargins(0, 0, 0, 0)

        self.btnReboot = QPushButton("Reboot")
        self.btnReboot.clicked.connect(self.btnRebootClicked)
        self.btnReboot.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btnSyncTime = QPushButton("Sync Time")
        self.btnSyncTime.clicked.connect(self.btnSyncTimeClicked)
        self.btnSyncTime.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btnBrowser = QPushButton("Browser")
        self.btnBrowser.clicked.connect(self.btnBrowserClicked)
        self.btnBrowser.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.btnSnapshot = QPushButton("JPEG")
        self.btnSnapshot.clicked.connect(self.btnSnapshotClicked)
        self.btnSnapshot.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        pnlButton = QWidget()
        lytButton = QGridLayout(pnlButton)
        lytButton.addWidget(self.btnReboot,     0, 0, 1, 1)
        lytButton.addWidget(self.btnSyncTime,   1, 0, 1, 1)
        lytButton.addWidget(self.btnBrowser,    2, 0, 1, 1)
        lytButton.addWidget(self.btnSnapshot,   3, 0, 1, 1)
        lytButton.setContentsMargins(6, 0, 6, 0)

        lytMain = QGridLayout(self)
        lytMain.addWidget(self.grpRecord,         0, 0, 1, 1)
        lytMain.addWidget(self.grpSounds,         0, 1, 1, 1)
        lytMain.addWidget(pnlRecordProfile,       1, 0, 1, 2)
        lytMain.addWidget(pnlButton,              0, 2, 2, 1)

    def grpRecordClicked(self, state):
        camera = self.cp.getCurrentCamera()
        if camera:
            camera.systemTabSettings.setRecordAlarmEnabled(state)

    def radRecordAlwaysClicked(self, state):
        camera = self.cp.getCurrentCamera()
        if camera:
            camera.systemTabSettings.setRecordAlways(state)
            camera.systemTabSettings.setRecordOnAlarm(not state)

    def radRecordOnAlarmClicked(self, state):
        camera = self.cp.getCurrentCamera()
        if camera:
            camera.systemTabSettings.setRecordOnAlarm(state)
            camera.systemTabSettings.setRecordAlways(not state)

    def grpSoundsClicked(self, state):
        camera = self.cp.getCurrentCamera()
        if camera:
            camera.systemTabSettings.setSoundAlarmEnabled(state)

    def radSoundOnceClicked(self, state):
        camera = self.cp.getCurrentCamera()
        if camera:
            camera.systemTabSettings.setSoundAlarmOnce(state)
            camera.systemTabSettings.setSoundAlarmLoop(not state)

    def radSoundLoopClicked(self, state):
        camera = self.cp.getCurrentCamera()
        if camera:
            camera.systemTabSettings.setSoundAlarmLoop(state)
            camera.systemTabSettings.setSoundAlarmOnce(not state)

    def cmbRecordProfileChanged(self, index):
        camera = self.cp.getCurrentCamera()
        if camera:
            players = self.cp.mw.pm.getStreamPairPlayers(camera.uri())
            camera.systemTabSettings.setRecordProfile(index)
            if len(players):
                for player in players:
                    self.cp.mw.pm.playerShutdownWait(player.uri)
                self.cp.onItemDoubleClicked(camera)

    def fill(self, onvif_data):
        self.cmbRecordProfile.disconnect()
        self.cmbRecordProfile.clear()
        for profile in onvif_data.profiles:
            self.cmbRecordProfile.addItem(profile.profile())
        
        camera = self.cp.getCurrentCamera()
        if camera:
            self.cmbRecordProfile.setCurrentIndex(camera.systemTabSettings.getRecordProfile())

        self.cmbRecordProfile.currentIndexChanged.connect(self.cmbRecordProfileChanged)
        self.syncGUI()
        self.setEnabled(True)

    def syncGUI(self):
        camera = self.cp.getCurrentCamera()
        if camera:
            self.grpRecord.setChecked(camera.systemTabSettings.record_enable)
            if camera.systemTabSettings.record_always:
                self.radRecordAlways.setChecked(True)
                self.radRecordOnAlarm.setChecked(False)
            if camera.systemTabSettings.record_alarm:
                self.radRecordOnAlarm.setChecked(True)
                self.radRecordAlways.setChecked(False)
            self.grpSounds.setChecked(camera.systemTabSettings.sound_alarm_enable)
            if camera.systemTabSettings.sound_alarm_once:
                self.radSoundOnce.setChecked(True)
                self.radSoundLoop.setChecked(False)
            if camera.systemTabSettings.sound_alarm_loop:
                self.radSoundLoop.setChecked(True)
                self.radSoundOnce.setChecked(False)

            self.cp.btnRecord.setEnabled(not (self.grpRecord.isChecked() and self.radRecordAlways.isChecked()))
            if camera.isRecording():
                self.cp.btnRecord.setStyleSheet(self.cp.getButtonStyle("recording"))
            else:
                self.cp.btnRecord.setStyleSheet(self.cp.getButtonStyle("record"))

    def btnRebootClicked(self):
        camera = self.cp.getCurrentCamera()
        if camera:
            result = QMessageBox.question(self, "Warning", f'{camera.name()}: Please confirm reboot')
            if result == QMessageBox.StandardButton.Yes:
                camera.onvif_data.startReboot()

    def btnSyncTimeClicked(self):
        camera = self.cp.getCurrentCamera()
        if camera:
            camera.onvif_data.startUpdateTime()

    def btnBrowserClicked(self):
        camera = self.cp.lstCamera.currentItem()
        if camera:
            host = "http://" + camera.onvif_data.host()
            webbrowser.get().open(host)

    def btnSnapshotClicked(self):
        if player := self.cp.getCurrentPlayer():
            root = self.cp.mw.settingsPanel.storage.dirPictures.txtDirectory.text() + "/" + self.cp.getCamera(player.uri).text()
            pathlib.Path(root).mkdir(parents=True, exist_ok=True)
            filename = '{0:%Y%m%d%H%M%S.jpg}'.format(datetime.datetime.now())
            filename = root + "/" + filename
            player.save_image_filename = filename
