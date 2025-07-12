#/********************************************************************
# libonvif/onvif-gui/onvif_gui/panels/cameras/systemtab.py 
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
    QMessageBox, QRadioButton, QComboBox, QLabel, QDialog, QDialogButtonBox, \
    QCheckBox, QLineEdit
from PyQt6.QtCore import Qt, QTimer
from datetime import datetime, timedelta
import webbrowser
import time
from onvif_gui.enums import ProxyType

class TimeDialog(QDialog):
    def __init__(self, mw):
        super().__init__(mw)
        self.mw = mw
        self.onvif_data = None
        self.timer = QTimer()
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.timeout)

        self.setWindowTitle("Time Settings")
        self.setMinimumWidth(640)

        self.lblComputerDate = QLabel()
        self.lblComputerTime = QLabel()
        self.lblComputerDST = QLabel()
        self.lblComputerTZ = QLabel()
        self.lblComputerUTCDiff = QLabel()
        self.updateComputerTime()        
        grpComputer = QGroupBox("Computer Time")
        lytComputer = QGridLayout(grpComputer)
        lytComputer.addWidget(QLabel("Date"),          0, 0, 1, 1)
        lytComputer.addWidget(self.lblComputerDate,    0, 1, 1, 1)
        lytComputer.addWidget(QLabel("Time"),          1, 0, 1, 1)
        lytComputer.addWidget(self.lblComputerTime,    1, 1, 1, 1)
        lytComputer.addWidget(QLabel("DST"),           2, 0, 1, 1)
        lytComputer.addWidget(self.lblComputerDST,     2, 1, 1, 1)
        lytComputer.addWidget(QLabel("Time Zone"),     3, 0, 1, 1)
        lytComputer.addWidget(self.lblComputerTZ,      3, 1, 1, 1)
        lytComputer.addWidget(QLabel("UTC Offset"),    4, 0, 1, 1)
        lytComputer.addWidget(self.lblComputerUTCDiff, 4, 1, 1, 1)

        self.lblCameraDate = QLabel()
        self.lblCameraTime = QLabel()
        self.lblCameraDST = QLabel()
        self.lblCameraTZ = QLabel()
        self.lblCameraUTCDiff = QLabel()
        self.lblCameraDiffLbl = QLabel("Diff (sec)")
        self.updateCameraTime()        
        grpCamera = QGroupBox("Camera Time")
        lytCamera = QGridLayout(grpCamera)
        lytCamera.addWidget(QLabel("Date"),        0, 0, 1, 1)
        lytCamera.addWidget(self.lblCameraDate,    0, 1, 1, 1)
        lytCamera.addWidget(QLabel("Time"),        1, 0, 1, 1)
        lytCamera.addWidget(self.lblCameraTime,    1, 1, 1, 1)
        lytCamera.addWidget(QLabel("DST"),         2, 0, 1, 1)
        lytCamera.addWidget(self.lblCameraDST,     2, 1, 1, 1)
        lytCamera.addWidget(QLabel("Time Zone"),   3, 0, 1, 1)
        lytCamera.addWidget(self.lblCameraTZ,      3, 1, 1, 1)
        lytCamera.addWidget(self.lblCameraDiffLbl, 4, 0, 1, 1)
        lytCamera.addWidget(self.lblCameraUTCDiff, 4, 1, 1, 1)

        self.txtTimezone = QLineEdit()
        self.lblTimezone = QLabel("Time Zone")
        self.txtNTP = QLineEdit()
        self.lblNTP = QLabel("NTP Server")
        self.chkDST = QCheckBox("Daylight Savings Time")
        self.cmbDateTimeType = QComboBox()
        self.cmbDateTimeType.addItem("Manual")
        self.cmbDateTimeType.addItem("NTP")

        self.radNTP = QRadioButton("NTP")
        self.radNTP.toggled.connect(self.radNTPToggled)
        self.radManual = QRadioButton("Manual")
        self.radManual.toggled.connect(self.radManualToggled)
        self.radUTCasLocal = QRadioButton("UTC as Local")
        self.radUTCasLocal.toggled.connect(self.radUTCasLocalToggled)

        grpSettings = QGroupBox("Time Sync Method")
        lytSettings = QGridLayout(grpSettings)
        lytSettings.addWidget(self.radNTP,             0, 0, 1, 1, Qt.AlignmentFlag.AlignCenter)
        lytSettings.addWidget(self.radManual,          0, 1, 1, 1, Qt.AlignmentFlag.AlignCenter)
        lytSettings.addWidget(self.radUTCasLocal,      0, 2, 1, 1, Qt.AlignmentFlag.AlignCenter)
        
        self.radFromDHCP = QRadioButton("From DHCP")
        self.radFromDHCP.toggled.connect(self.radFromDHCPToggled)
        self.radIPv4 = QRadioButton("IPv4 Address")
        self.radDNS = QRadioButton("Domain Name")
        
        self.grpNTPServer = QGroupBox("NTP Server")
        lytNTPServer = QGridLayout(self.grpNTPServer)
        lytNTPServer.addWidget(QLabel("NTP Server Address Type:"),   0, 1, 1, 1)
        lytNTPServer.addWidget(self.radFromDHCP,                     0, 2, 1, 1)
        lytNTPServer.addWidget(self.radIPv4,                         1, 2, 1, 1)
        lytNTPServer.addWidget(self.radDNS,                          2, 2, 1, 1)
        lytNTPServer.addWidget(self.lblNTP,                          4, 0, 1, 1, Qt.AlignmentFlag.AlignRight)
        lytNTPServer.addWidget(self.txtNTP,                          4, 1, 1, 3)

        buttonBox = QDialogButtonBox( \
            QDialogButtonBox.StandardButton.Ok | \
            QDialogButtonBox.StandardButton.Cancel)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

        lytMain = QGridLayout(self)
        lytMain.addWidget(grpComputer,             0, 0, 1, 2)
        lytMain.addWidget(grpCamera,               0, 2, 1, 2)
        lytMain.addWidget(QLabel(),                1, 0, 1, 4)
        lytMain.addWidget(self.lblTimezone,        2, 0, 1, 1)
        lytMain.addWidget(self.txtTimezone,        2, 1, 1, 3)
        lytMain.addWidget(self.chkDST,             3, 0, 1, 4)
        lytMain.addWidget(grpSettings,             4, 1, 1, 2)
        lytMain.addWidget(self.grpNTPServer,       6, 0, 1, 4)
        lytMain.addWidget(buttonBox,               7, 0, 1, 4)

    def radFromDHCPToggled(self, arg):
        self.txtNTP.setEnabled(not arg)
        self.lblNTP.setEnabled(not arg)

    def radNTPToggled(self, arg):
        self.grpNTPServer.setEnabled(arg)
        if arg and self.onvif_data:
            self.onvif_data.setDateTimeType('N')

    def radManualToggled(self, arg):
        if arg and self.onvif_data:
            self.onvif_data.setDateTimeType('M')

    def radUTCasLocalToggled(self, arg):
        if arg and self.onvif_data:
            self.onvif_data.setDateTimeType('U')
        self.txtTimezone.setEnabled(not arg)
        self.lblTimezone.setEnabled(not arg)
        self.chkDST.setEnabled(not arg)

    def exec(self, onvif_data):
        self.onvif_data = onvif_data

        if onvif_data.datetimetype() == 'U':
            self.radUTCasLocal.setChecked(True)
            self.radNTPToggled(False)
            self.radManualToggled(False)
        if onvif_data.datetimetype() == 'M':
            self.radManual.setChecked(True)
            self.radNTPToggled(False)
        if onvif_data.datetimetype() == 'N':
            self.radNTP.setChecked(True)
            self.radManualToggled(False)

        if onvif_data.ntp_dhcp():
            self.radFromDHCP.setChecked(True)
        else:
            if onvif_data.ntp_type() == "IPv4":
                self.radIPv4.setChecked(True)
            if onvif_data.ntp_type() == "DNS":
                self.radDNS.setChecked(True)
        self.txtNTP.setText(onvif_data.ntp_addr())

        tz = onvif_data.timezone()
        self.lblCameraTZ.setText(tz)
        self.txtTimezone.setText(tz)
        self.lblCameraDST.setText(str(bool(onvif_data.dst())))
        self.txtNTP.setText(onvif_data.ntp_addr())
        self.chkDST.setChecked(onvif_data.dst())
        self.lblCameraUTCDiff.setText(str(onvif_data.time_offset()))

        self.updateComputerTime()
        self.updateCameraTime()

        self.timer.start()
        super().exec()

    def accept(self):
        self.onvif_data.setNTPDHCP(self.radFromDHCP.isChecked())
        if self.radIPv4.isChecked():
            self.onvif_data.setNTPType("IPv4")
        if self.radDNS.isChecked():
            self.onvif_data.setNTPType("DNS")
        self.onvif_data.setNTPAddr(self.txtNTP.text())
        self.onvif_data.setDST(self.chkDST.isChecked())
        self.onvif_data.setTimezone(self.txtTimezone.text())
        self.onvif_data.startUpdateTime()
        self.close()

    def reject(self):
        self.close()

    def closeEvent(self, e):
        self.timer.stop()

    def timeout(self):
        self.updateComputerTime()
        self.updateCameraTime()

    def updateComputerTime(self):
        now = datetime.now()
        self.lblComputerDate.setText(now.strftime("%b-%d-%Y"))
        self.lblComputerTime.setText(now.strftime('%H:%M:%S'))
        dst = bool(time.localtime().tm_isdst)
        self.lblComputerDST.setText(str(dst))
        local_now = now.astimezone()
        local_tz = local_now.tzinfo
        self.lblComputerTZ.setText(local_tz.tzname(local_now))
        self.lblComputerUTCDiff.setText(str(local_now)[-6:])

    def updateCameraTime(self):
        now = datetime.now()
        if self.onvif_data:
            offset = self.onvif_data.time_offset()
            if self.radUTCasLocal.isChecked():
                local_now = now.astimezone()
                utc_diff = local_now.utcoffset()
                seconds_diff = utc_diff.days * 24 * 60 * 60 + utc_diff.seconds
                offset -= seconds_diff
                self.lblCameraDiffLbl.setText("Diff (sec) *adj")
                self.lblCameraUTCDiff.setText(str(offset))
            else:
                self.lblCameraDiffLbl.setText("Diff (sec)")
                self.lblCameraUTCDiff.setText(str(self.onvif_data.time_offset()))

            delta = timedelta(seconds=offset)
            now += delta

        self.lblCameraDate.setText(now.strftime("%b-%d-%Y"))
        self.lblCameraTime.setText(now.strftime('%H:%M:%S'))

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
            if profile := self.camera.getRecordProfile():
                if player := self.mw.pm.getPlayer(profile.uri()):
                    if not player.isRecording():
                        if filename := player.getPipeOutFilename():
                            player.toggleRecording(filename)
        else:
            players = self.mw.pm.getStreamPairPlayers(self.camera.uri())
            for player in players:
                if player.isRecording():
                    player.toggleRecording("")
                    self.mw.diskManager.getDirectorySize(self.mw.settingsPanel.storage.dirArchive.txtDirectory.text())
                
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
        return bool(int(self.mw.settings.value(key, 1)))
    
    def setRecordAlways(self, state):
        self.record_always = bool(state)
        key = f'{self.camera.serial_number()}/RecordAlways'
        self.mw.settings.setValue(key, int(state))
        self.managePlayers()

    def getRecordOnAlarm(self):
        key = f'{self.camera.serial_number()}/RecordOnAlarm'
        return bool(int(self.mw.settings.value(key, 0)))
    
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

        self.dlgTimeDialog = TimeDialog(self.cp.mw)

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
        self.lblRecordResolution = QLabel("      ")
        self.lblRecordProfile = QLabel("  Record Profile   ")
        self.chkRecordAudio = QCheckBox("   Record Audio     ")
        self.chkRecordAudio.stateChanged.connect(self.chkRecordAudioClicked)
        pnlRecordProfile = QWidget()
        lytRecordProfile = QGridLayout(pnlRecordProfile)
        lytRecordProfile.addWidget(self.lblRecordProfile,     0, 0, 1, 1)
        lytRecordProfile.addWidget(self.cmbRecordProfile,     0, 1, 1, 1)
        lytRecordProfile.addWidget(QLabel("               "), 0, 2, 1, 1)
        lytRecordProfile.addWidget(self.lblRecordResolution,  1, 0, 1, 1)
        #lytRecordProfile.addWidget(self.chkRecordAudio,       1, 0, 1, 3, Qt.AlignmentFlag.AlignCenter)
        lytRecordProfile.addWidget(self.chkRecordAudio,       1, 1, 1, 1)
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

        pnlButton = QWidget()
        lytButton = QGridLayout(pnlButton)
        lytButton.addWidget(self.btnReboot,     0, 0, 1, 1)
        lytButton.addWidget(self.btnSyncTime,   1, 0, 1, 1)
        lytButton.addWidget(self.btnBrowser,    2, 0, 1, 1)
        lytButton.setContentsMargins(6, 2, 6, 2)

        lytMain = QGridLayout(self)
        lytMain.addWidget(self.grpRecord,         0, 0, 1, 1)
        lytMain.addWidget(self.grpSounds,         0, 1, 1, 1)
        lytMain.addWidget(pnlRecordProfile,       1, 0, 1, 3)
        lytMain.addWidget(pnlButton,              0, 2, 1, 1)

    def grpRecordClicked(self, state):
        if camera := self.cp.getCurrentCamera():
            camera.systemTabSettings.setRecordAlarmEnabled(state)

    def radRecordAlwaysClicked(self, state):
        if camera := self.cp.getCurrentCamera():
            camera.systemTabSettings.setRecordAlways(state)
            camera.systemTabSettings.setRecordOnAlarm(not state)

    def radRecordOnAlarmClicked(self, state):
        if camera := self.cp.getCurrentCamera():
            camera.systemTabSettings.setRecordOnAlarm(state)
            camera.systemTabSettings.setRecordAlways(not state)

    def grpSoundsClicked(self, state):
        if camera := self.cp.getCurrentCamera():
            camera.systemTabSettings.setSoundAlarmEnabled(state)

    def radSoundOnceClicked(self, state):
        if camera := self.cp.getCurrentCamera():
            camera.systemTabSettings.setSoundAlarmOnce(state)
            camera.systemTabSettings.setSoundAlarmLoop(not state)

    def radSoundLoopClicked(self, state):
        if camera := self.cp.getCurrentCamera():
            camera.systemTabSettings.setSoundAlarmLoop(state)
            camera.systemTabSettings.setSoundAlarmOnce(not state)

    def cmbRecordProfileChanged(self, index):
        if camera := self.cp.getCurrentCamera():
            players = self.cp.mw.pm.getStreamPairPlayers(camera.uri())
            camera.systemTabSettings.setRecordProfile(index)
            if len(players):
                for player in players:
                    self.cp.mw.pm.playerShutdownWait(player.uri)
                self.cp.onItemDoubleClicked(camera)
        self.syncGUI()

    def chkRecordAudioClicked(self, state):
        if camera := self.cp.getCurrentCamera():
            if profile := camera.getRecordProfile():
                profile.setDisableAudio(not state)
                if player := self.cp.mw.pm.getPlayer(profile.uri()):
                    player.disable_audio = bool(not state)
                    self.cp.mw.pm.playerShutdownWait(player.uri)
                    self.cp.mw.playMedia(player.uri)

    def fill(self, onvif_data):
        self.cmbRecordProfile.currentIndexChanged.disconnect()
        self.cmbRecordProfile.clear()
        for profile in onvif_data.profiles:
            self.cmbRecordProfile.addItem(profile.profile())
        
        if camera := self.cp.getCurrentCamera():
            self.cmbRecordProfile.setCurrentIndex(camera.systemTabSettings.getRecordProfile())

        self.cmbRecordProfile.currentIndexChanged.connect(self.cmbRecordProfileChanged)
        self.syncGUI()
        self.setEnabled(True)

    def syncGUI(self):
        if camera := self.cp.getCurrentCamera():
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

            '''
            # attempted to put record resolution on the panel
            # Unfortunately I wasn't able to get consistent data for this display
            if recordProfile := camera.getRecordProfile():
                self.lblRecordResolution.setText(f"( {recordProfile.width()} x {recordProfile.height()} )")
            else:
                self.lblRecordResolution.setText("      ")
            '''

            if not camera.hasAudio():
                self.chkRecordAudio.setChecked(False)
                self.chkRecordAudio.setEnabled(False)
                return

            matchedProfiles = True
            if recordProfile := camera.getRecordProfile():
                self.chkRecordAudio.stateChanged.disconnect()
                self.chkRecordAudio.setChecked(not recordProfile.getDisableAudio())
                self.chkRecordAudio.stateChanged.connect(self.chkRecordAudioClicked)
                if displayProfile := camera.getDisplayProfile():
                    if recordProfile.uri() != displayProfile.uri():
                        matchedProfiles = False

            self.chkRecordAudio.setEnabled(not matchedProfiles)
            self.chkRecordAudio.setChecked(not recordProfile.getDisableAudio())

    def btnRebootClicked(self):
        if camera := self.cp.getCurrentCamera():
            result = QMessageBox.question(self, "Warning", f'{camera.name()}: Please confirm reboot')
            if result == QMessageBox.StandardButton.Yes:
                if self.cp.mw.settingsPanel.proxy.proxyType == ProxyType.CLIENT:
                    arg = "REBOOT\n\n" + camera.onvif_data.toJSON() + "\r\n"
                    self.cp.mw.client.transmit(arg)
                else:
                    camera.onvif_data.startReboot()

    def btnSyncTimeClicked(self):
        if camera := self.cp.getCurrentCamera():
            self.dlgTimeDialog.exec(camera.onvif_data)

    def btnBrowserClicked(self):
        if camera := self.cp.lstCamera.currentItem():
            host = "http://" + camera.onvif_data.host()
            webbrowser.get().open(host)
