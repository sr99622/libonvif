#********************************************************************
# onvif-gui/onvif_gui/player.py
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

from datetime import datetime
from pathlib import Path
from PyQt6.QtGui import QPainter, QColorConstants
from PyQt6.QtCore import pyqtSignal, QObject, QTimer, QRectF
from collections import deque
import avio
import time
import os
import pathlib
from loguru import logger
from datetime import datetime
import threading
from onvif_gui.panels.camera import Snapshot

class PlayerSignals(QObject):
    start = pyqtSignal()
    stop = pyqtSignal()
    play_alarm_sound = pyqtSignal(str)

class Player(avio.Player):
    def __init__(self, uri, mw):
        super().__init__(uri)
        self.mw = mw
        self.signals = PlayerSignals()
        self.ary = None
        self.image = None
        self.desired_aspect = 0
        self.analyze_video = False
        self.analyze_audio = False
        self.videoModelSettings = None
        self.audioModelSettings = None
        self.detection_count = deque()
        self.last_image = None
        self.last_render = None
        self.needs_render = False
        self.timer = None
        self.thread_lock = False

        self.boxes = None
        self.labels = []
        self.scores = []
        self.last_alarm_sound = datetime.now()

        self.save_image_filename = None
        self.output_file_start_time = None
        self.estimated_file_size = 0
        self.packet_drop_frame_counter = 0
        self.last_msg = ""

        self.alarm_state = 0
        self.last_alarm_state = 0
        self.file_progress = 0.0

        self.snapshot = Snapshot(mw)

        if (len(uri)):
            self.timer = QTimer()
            self.timer.setInterval(self.mw.settingsPanel.alarm.spnLagTime.value() * 1000)
            self.timer.setSingleShot(True)
            self.timer.timeout.connect(self.timeout)
            self.signals.start.connect(self.timer.start)
            self.signals.stop.connect(self.timer.stop)
            self.signals.play_alarm_sound.connect(self.soundAlarm)

    def lock(self):
        # the lock protects the image
        if self.thread_lock:
            time.sleep(0.001)
        self.thread_lock = True

    def unlock(self):
        self.thread_lock = False

    def requestShutdown(self, reconnect=False):
        if self.isCameraStream():
            self.setAlarmState(0)
        else:
            if self.isPaused():
                self.togglePaused()
        self.analyze_video = False
        self.analyze_audio = False
        self.request_reconnect = reconnect
        self.terminate()

    def systemTabSettings(self):
        result = None
        if camera := self.mw.cameraPanel.getCamera(self.uri):
            result = camera.systemTabSettings
        return result

    def setAlarmState(self, state):
         
        self.alarm_state = int(state)

        record_enable = self.systemTabSettings().record_enable if self.systemTabSettings() else False
        record_alarm = self.systemTabSettings().record_alarm if self.systemTabSettings() else False
        if camera := self.mw.cameraPanel.getCamera(self.uri):
            manual_recording = camera.manual_recording if camera else False
            profile = camera.getRecordProfile() if camera else None
            player = self.mw.pm.getPlayer(profile.uri()) if profile else None

            if state:
                self.signals.start.emit()
                if record_enable and record_alarm:
                    if player:
                        if not player.isRecording():
                            if filename := player.getOutputFilename():
                                player.toggleRecording(filename)
                                if current_camera := self.mw.cameraPanel.getCurrentCamera():
                                    if camera.serial_number() == current_camera.serial_number():
                                        self.mw.cameraPanel.syncGUI()
            else:
                self.signals.stop.emit()
                if record_alarm and not manual_recording:
                    if player:
                        if player.isRecording():
                            player.toggleRecording("")
                            self.mw.settingsPanel.storage.signals.updateDiskUsage.emit()
                            if current_camera := self.mw.cameraPanel.getCurrentCamera():
                                if camera.serial_number() == current_camera.serial_number():
                                    self.mw.cameraPanel.syncGUI()

    def timeout(self):
        self.setAlarmState(0)

    def getOutputFilename(self):
        filename = None
        if camera := self.mw.cameraPanel.getCamera(self.uri):
            d = self.mw.settingsPanel.storage.dirArchive.txtDirectory.text()
            root = os.path.join(d, camera.text())
            Path(root).mkdir(parents=True, exist_ok=True)
            self.output_file_start_time = datetime.now()
            filename = '{0:%Y%m%d%H%M%S}'.format(self.output_file_start_time)
            filename = os.path.join(root, filename)
            self.setMetaData("title", camera.text())
        return filename

    def handleAlarm(self, state):
        if self.analyze_video or self.analyze_audio:
            if state:
                self.setAlarmState(1)

                if self.systemTabSettings():
                    save_picture = self.alarm_state != self.last_alarm_state \
                                and self.mw.settingsPanel.alarm.chkSavePicture.isChecked() \
                                and self.systemTabSettings().record_enable

                    if save_picture and self.image:
                        try:
                            saveLocal= not self.systemTabSettings().remote_snapshot
                            if saveLocal:
                                self.lock()
                                img = self.image.copy()
                                self.unlock()
                                root = os.path.join(self.mw.settingsPanel.storage.dirPictures.txtDirectory.text(), self.mw.cameraPanel.getCamera(self.uri).text())
                                pathlib.Path(root).mkdir(parents=True, exist_ok=True)
                                filename = '{0:%Y%m%d%H%M%S.jpg}'.format(datetime.now())
                                filename = os.path.join(root, filename)

                                painter_img = QPainter(img)
                                painter_img.setPen(QColorConstants.Red)
                                if self.boxes:
                                    for box in self.boxes:
                                        p = (box[0])
                                        q = (box[1])
                                        r = (box[2] - box[0])
                                        s = (box[3] - box[1])
                                        painter_img.drawRect(QRectF(p, q, r, s))
                                painter_img.end()
                                img.save(filename)
                            else:
                                root = self.mw.settingsPanel.storage.dirPictures.txtDirectory.text() + "/" + self.mw.cameraPanel.getCamera(self.uri).text()
                                Path(root).mkdir(parents=True, exist_ok=True)
                                filename = '{0:%Y%m%d%H%M%S.jpg}'.format(datetime.now())
                                filename = str(root + "/" + filename)

                                if camera := self.mw.cameraPanel.getCamera(self.uri):
                                    profile = camera.getRecordProfile()
                                    if not profile:
                                        profile = camera.getProfile(self.uri)

                                    thread = threading.Thread(target=self.snapshot, args=(profile, filename, camera, self))
                                    thread.start()

                        except Exception as ex:
                            logger.error(f'player handle alarm write file exception: {ex}')

                    if self.systemTabSettings().sound_alarm_enable:
                        filename = os.path.join(self.mw.getLocation(), "onvif_gui", "resources", self.mw.settingsPanel.alarm.cmbSoundFiles.currentText())
                        if self.systemTabSettings().sound_alarm_once:
                            if self.alarm_state != self.last_alarm_state:
                                self.signals.play_alarm_sound.emit(filename)
                        if self.systemTabSettings().sound_alarm_loop:
                            p = self.mw.pm.getPlayer(filename)
                            if not p:
                                self.signals.play_alarm_sound.emit(filename)

            self.last_alarm_state = self.alarm_state

        else:
            self.setAlarmState(0)

    def soundAlarm(self, filename):
        player = Player(filename, self.mw)
        player.live_stream = False
        player.mediaPlayingStopped = self.mw.pm.removePlayer
        player.audio_driver_index = self.mw.settingsPanel.general.cmbAudioDriver.currentIndex()
        player.request_reconnect = False
        player.disable_video = True
        player.setVolume(self.mw.settingsPanel.alarm.sldAlarmVolume.value())
        self.mw.pm.startPlayer(player)

    def processModelOutput(self):
        sum = 0
        if self.boxes is not None:
            if camera := self.mw.cameraPanel.getCamera(self.uri):
                if len(self.detection_count) > camera.videoModelSettings.sampleSize - 1 and len(self.detection_count):
                    self.detection_count.popleft()
                if len(self.boxes):
                    self.detection_count.append(1)
                else:
                    self.detection_count.append(0)

                for count in self.detection_count:
                    sum += count
        return sum

    def loadRemoteDetections(self):
        for idx, alarm in enumerate(self.mw.alarm_states):
            serial_number = self.mw.alarm_ordinals.get(idx, None)
            if camera := self.mw.cameraPanel.getCamera(self.uri):
                if camera.serial_number() == serial_number:
                    self.handleAlarm(int(alarm))
