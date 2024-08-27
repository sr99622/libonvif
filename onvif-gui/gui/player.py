import os
from time import sleep
from datetime import datetime
from pathlib import Path
from PyQt6.QtCore import pyqtSignal, QObject, QTimer, QFile
from collections import deque
import shutil
import avio
from loguru import logger

class PlayerSignals(QObject):
    start = pyqtSignal()
    stop = pyqtSignal()

class Player(avio.Player):
    def __init__(self, uri, mw):
        super().__init__(uri)
        self.mw = mw
        self.signals = PlayerSignals()
        self.image = None
        self.rendering = False
        self.desired_aspect = 0
        self.systemTabSettings = None
        self.analyze_video = False
        self.analyze_audio = False
        self.videoModelSettings = None
        self.audioModelSettings = None
        self.detection_count = deque()
        self.last_image = None
        self.last_render = None
        self.timer = None

        self.boxes = []
        self.labels = []
        self.scores = []

        self.save_image_filename = None
        self.pipe_output_start_time = None
        self.estimated_file_size = 0
        self.packet_drop_frame_counter = 0
        self.last_msg = ""

        self.alarm_state = 0
        self.last_alarm_state = 0
        self.file_progress = 0.0

        if (len(uri)):
            self.timer = QTimer()
            self.timer.setInterval(self.mw.settingsPanel.alarm.spnLagTime.value() * 1000)
            self.timer.setSingleShot(True)
            self.timer.timeout.connect(self.timeout)
            self.signals.start.connect(self.timer.start)
            self.signals.stop.connect(self.timer.stop)

    def requestShutdown(self):
        self.setAlarmState(0)
        self.analyze_video = False
        self.analyze_audio = False
        self.request_reconnect = False
        self.running = False

    def setAlarmState(self, state):
        self.alarm_state = int(state)

        record_enable = self.systemTabSettings.record_enable if self.systemTabSettings else False
        record_alarm = self.systemTabSettings.record_alarm if self.systemTabSettings else False
        if camera := self.mw.cameraPanel.getCamera(self.uri):
            manual_recording = camera.manual_recording if camera else False
            profile = camera.getRecordProfile() if camera else None
            player = self.mw.pm.getPlayer(profile.uri()) if profile else None

            if state:
                self.signals.start.emit()
                if record_enable and record_alarm:
                    if player:
                        if not player.isRecording():
                            d = self.mw.settingsPanel.storage.dirArchive.txtDirectory.text()
                            if self.mw.settingsPanel.storage.chkManageDiskUsage.isChecked():
                                player.manageDirectory(d)
                            filename = player.getPipeOutFilename(d)
                            if filename:
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
                            if current_camera := self.mw.cameraPanel.getCurrentCamera():
                                if camera.serial_number() == current_camera.serial_number():
                                    self.mw.cameraPanel.syncGUI()
    
    def timeout(self):
        self.setAlarmState(0)

    def getPipeOutFilename(self, d):
        filename = None
        camera = self.mw.cameraPanel.getCamera(self.uri)
        if camera:
            d = self.mw.settingsPanel.storage.dirArchive.txtDirectory.text()
            root = d + "/" + camera.text()
            Path(root).mkdir(parents=True, exist_ok=True)
            self.pipe_output_start_time = datetime.now()
            filename = '{0:%Y%m%d%H%M%S}'.format(self.pipe_output_start_time)
            filename = root + "/" + filename + ".mp4"
            self.setMetaData("title", camera.text())
        return filename

    def estimateFileSize(self):
        # duration is in seconds, cameras report bitrate in kbps, result in bytes
        result = 0
        bitrate = 0
        profile = self.mw.cameraPanel.getProfile(self.uri)
        if profile:
            audio_bitrate = min(profile.audio_bitrate(), 128)
            video_bitrate = min(profile.bitrate(), 16384)
            bitrate = video_bitrate + audio_bitrate
        result = (bitrate * 1000 / 8) * self.mw.STD_FILE_DURATION
        self.estimated_file_size = result
        return result

    def getCommittedSize(self):
        committed = 0
        for player in self.mw.pm.players:
            if player.isRecording():
                committed += player.estimateFileSize() - player.pipeBytesWritten()
        return committed

    def getDirectorySize(self, d):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(d):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    try:
                        total_size += os.path.getsize(fp)
                    except FileNotFoundError:
                        pass

        dir_size = "{:.2f}".format(total_size / 1000000000)
        self.mw.settingsPanel.storage.grpDiskUsage.setTitle(f'Disk Usage (currently {dir_size} GB)')
        return total_size
    
    def getOldestFile(self, d):
        oldest_file = None
        oldest_time = None
        for dirpath, dirnames, filenames in os.walk(d):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):

                    stem = Path(fp).stem
                    if len(stem) == 14 and stem.isnumeric():
                        try:
                            if oldest_file is None:
                                oldest_file = fp
                                oldest_time = os.path.getmtime(fp)
                            else:
                                file_time = os.path.getmtime(fp)
                                if file_time < oldest_time:
                                    oldest_file = fp
                                    oldest_time = file_time
                        except FileNotFoundError:
                            pass
        return oldest_file
    
    def getMaximumDirectorySize(self, d):
        estimated_file_size = self.estimateFileSize()
        space_committed = self.getCommittedSize()
        allowed_space = min(self.mw.settingsPanel.storage.spnDiskLimit.value() * 1000000000, shutil.disk_usage(d)[2])
        return allowed_space - (space_committed + estimated_file_size)
    
    def manageDirectory(self, d):
        while self.getDirectorySize(d) > self.getMaximumDirectorySize(d):
            oldest_file = self.getOldestFile(d)
            if oldest_file:
                QFile.remove(oldest_file)
                #logger.debug(f'File has been deleted by auto process: {oldest_file}')
            else:
                logger.debug("Unable to find the oldest file for deletion during disk management")
                break

    def handleAlarm(self, state):
        if self.analyze_video or self.analyze_audio:
            if state:
                self.setAlarmState(1)
            if self.alarm_state:
                if self.isCameraStream():
                    if self.systemTabSettings.sound_alarm_enable:
                        filename = f'{self.mw.getLocation()}/gui/resources/{self.mw.settingsPanel.alarm.cmbSoundFiles.currentText()}'
                        if self.systemTabSettings.sound_alarm_once:
                            if self.alarm_state != self.last_alarm_state:
                                self.mw.playMedia(filename, True)
                        if self.systemTabSettings.sound_alarm_loop:
                            p = self.mw.pm.getPlayer(filename)
                            if not p:
                                self.mw.playMedia(filename, True)
            self.last_alarm_state = self.alarm_state
        else:
            self.setAlarmState(0)

    def getFrameRate(self):
        frame_rate = self.getVideoFrameRate()
        if frame_rate <= 0:
            profile = self.mw.cameraPanel.getProfile(self.uri)
            if profile:
                frame_rate = profile.frame_rate()
        return frame_rate

    def processModelOutput(self):

        while self.rendering:
            sleep(0.001)

        sum = 0

        if len(self.detection_count) > self.videoModelSettings.sampleSize - 1 and len(self.detection_count):
            self.detection_count.popleft()

        if len(self.boxes):
            self.detection_count.append(1)
        else:
            self.detection_count.append(0)

        for count in self.detection_count:
            sum += count

        return sum
