#/********************************************************************
# libonvif/onvif-gui/onvif_gui/panels/cameras/datastructures.py 
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

from PyQt6.QtWidgets import QListWidgetItem
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QTimer
from PyQt6.QtGui import QIcon, QColor
import libonvif as onvif
from onvif_gui.panels.camera.systemtab import SystemTabSettings
from onvif_gui.enums import ProxyType, StreamState
from loguru import logger

class SessionSignals(QObject):
    finished = pyqtSignal()

class Session(onvif.Session):
    def __init__(self, cp, interface):
        super().__init__()
        self.cp = cp
        self.interface = interface
        self.signals = SessionSignals()
        self.discovered = lambda : self.finish()
        self.getCredential = lambda D : self.cp.getCredential(D)
        self.getData = lambda D : self.cp.getData(D)
        self.infoCallback = lambda msg : self.info(msg)
        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.timeout)
        self.signals.finished.connect(self.timer.stop)
        self.active = False

    def start(self):
        self.active = True
        self.startDiscover()
        self.timer.start(10000)

    def finish(self):
        self.active = False
        self.signals.finished.emit()
        self.cp.discovered()

    def timeout(self):
        self.cp.discoveryTimeout()

    def info(self, msg):
        try:
            logger.debug(msg)
        except Exception as ex:
            logger.error("session info callback exception: {ex}")

class Camera(QListWidgetItem):
    def __init__(self, onvif_data, mw):
        super().__init__(onvif_data.alias)
        self.mw = mw
        self.icnIdle = QIcon("image:idle_lo.png")
        self.icnOn = QIcon("image:record.png")
        self.icnRecord = QIcon("image:recording_hi.png")
        self.defaultForeground = self.foreground()
        self.filled = False
        self.manual_fill = False
        self.last_msg = ""

        self.assignData(onvif_data)

        self.videoModelSettings = None
        self.audioModelSettings = None
        self.systemTabSettings = SystemTabSettings(self.mw, self)
        self.manual_recording = False
        self.ordinalKey = f'{self.serial_number()}/Ordinal'
        self.ordinal = self.getOrdinal()
        self.volumeKey = f'{self.serial_number()}/Volume'
        self.volume = self.getVolume()
        self.muteKey = f'{self.serial_number()}/Mute'
        self.mute = self.getMute()

    def assignData(self, data):
        self.onvif_data = data
        data.setSetting = self.mw.settings.setValue
        data.getSetting = self.mw.settings.value
        if self.mw.settingsPanel.proxy.proxyType != ProxyType.STAND_ALONE:
            data.getProxyURI = self.mw.getProxyURI

        for profile in data.profiles:
            profile.setSetting = self.mw.settings.setValue
            profile.getSetting = self.mw.settings.value
            if self.mw.settingsPanel.proxy.proxyType != ProxyType.STAND_ALONE:
                profile.getProxyURI = self.mw.getProxyURI
        
        self.profiles = data.profiles

    def uri(self):
        return self.onvif_data.uri()
    
    def serial_number(self):
        return self.onvif_data.serial_number()
    
    def name(self):
        return self.onvif_data.alias
    
    def xaddrs(self):
        return self.onvif_data.xaddrs()

    def hasAudio(self):
        return bool(self.onvif_data.audio_bitrate())
    
    def setOrdinal(self, value):
        self.ordinal = value
        self.mw.settings.setValue(self.ordinalKey, value)

    def getOrdinal(self):
        return int(self.mw.settings.value(self.ordinalKey, -1))
    
    def getMute(self):
        return bool(int(self.mw.settings.value(self.muteKey, 0)))
    
    def setMute(self, state):
        self.mute = bool(state)
        self.mw.settings.setValue(self.muteKey, int(state))

    def getVolume(self):
        return int(self.mw.settings.value(self.volumeKey, 80))
    
    def setVolume(self, volume):
        self.volume = volume
        self.mw.settings.setValue(self.volumeKey, volume)

    def isRunning(self):
        result = False
        players = self.mw.pm.getStreamPairPlayers(self.uri())
        if len(players):
            result = True
        return result
    
    def isRecording(self):
        result = False
        players = self.mw.pm.getStreamPairPlayers(self.uri())
        for player in players:
            if player.isRecording():
                result = True
        return result
    
    def isAlarming(self):
        result = False
        players = self.mw.pm.getStreamPairPlayers(self.uri())
        for player in players:
            if player.alarm_state:
                result = True
        return result
    
    def isCurrent(self):
        result = False
        for profile in self.profiles:
            #if profile.uri() == self.mw.glWidget.focused_uri:
                camera = self.mw.cameraPanel.getCurrentCamera()
                #if player.uri == self.mw.glWidget.focused_uri:
                if camera.getProfile(profile.uri()):
                    result = True
        return result

    def editing(self):
        return self.flags() & Qt.ItemFlag.ItemIsEditable
    
    def setIconIdle(self):
        if not self.flags() & Qt.ItemFlag.ItemIsEditable:
            self.setIcon(self.icnIdle)

    def setIconOn(self):
        if not self.flags() & Qt.ItemFlag.ItemIsEditable:
            self.setIcon(self.icnOn)

    def setIconRecord(self):
        if not self.flags() & Qt.ItemFlag.ItemIsEditable:
            self.setIcon(self.icnRecord)
        
    def dimForeground(self):
        self.setForeground(QColor("#808D9E"))

    def restoreForeground(self):
        self.setForeground(self.defaultForeground)

    def isCurrent(self):
        result = False
        current_camera = self.mw.cameraPanel.getCurrentCamera()
        if current_camera:
            if current_camera.serial_number() == self.serial_number():
                result = True
        return result
    
    def getStreamState(self, index):
        result = StreamState.INVALID
        if len(self.profiles) <= index:
            return result
        profile = self.profiles[index]
        if profile:
            player = self.mw.pm.getPlayer(profile.uri())
            if player:
                if player.image:
                    result = StreamState.CONNECTED
                else:
                    result = StreamState.CONNECTING
            else:
                result = StreamState.IDLE

            timer = self.mw.timers.get(profile.uri(), None)
            if timer:
                if timer.isActive():
                    result = StreamState.CONNECTING
        return result
    
    def profileName(self, uri):
        result = ""
        for profile in self.profiles:
            if profile.uri() == uri:
                result = profile.profile()
        return result
    
    def recordProfileIndex(self):
        return self.systemTabSettings.record_profile

    def displayProfileIndex(self):
        return self.onvif_data.displayProfile
    
    def setDisplayProfile(self, index):
        self.onvif_data.setProfile(index)
        self.mw.settings.setValue(f'{self.serial_number()}/DisplayProfile', index)
        for i, profile in enumerate(self.profiles):
            if i == index:
                profile.setHidden(False)
            else:
                profile.setHidden(True)

    def getDisplayProfileSetting(self):
        return int(self.mw.settings.value(f'{self.serial_number()}/DisplayProfile', self.displayProfileIndex()))

    def getProfile(self, uri):
        result = None
        for profile in self.profiles:
            if  profile.uri() == uri:
                result = profile
                break
        return result
    
    def getRecordProfile(self):
        result = None
        if len(self.profiles) > self.recordProfileIndex():
            result = self.profiles[self.recordProfileIndex()]
        return result
    
    def isRecordProfile(self, uri):
        result = False
        recordProfile = self.getRecordProfile()
        if recordProfile:
            if uri == recordProfile.uri():
                result = True
        return result
    
    def getDisplayProfile(self):
        result = None
        if len(self.profiles) > self.displayProfileIndex():
            result = self.profiles[self.displayProfileIndex()]
        return result
    
    def isDisplayProfile(self, uri):
        result = False
        if displayProfile := self.getDisplayProfile():
            if uri == displayProfile.uri():
                result = True
        return result


    def companionURI(self, uri):
        result = None
        recordProfile = self.getRecordProfile()
        displayProfile = self.getDisplayProfile()
        if recordProfile and displayProfile:
            if uri == recordProfile.uri():
                result = displayProfile.uri()
            if uri == displayProfile.uri():
                result = recordProfile.uri()
        return result

    def syncData(self, data):
        for profile in self.profiles:
            if profile.profile() == data.profile():
                profile.syncData(data)
        data.profiles = self.profiles
        if self.onvif_data.profile() == data.profile():
            self.onvif_data.syncData(data)
            if self.isCurrent():
                self.mw.cameraPanel.signals.fill.emit(data)