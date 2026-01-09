#********************************************************************
# onvif-gui/onvif_gui/glwidget.py
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

from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtGui import QPainter, QImage, QColorConstants, QPen, QMovie, QIcon, QPixmap
from PyQt6.QtCore import QSize, QPointF, QRectF, QTimer, QMargins, Qt, QRect, QPoint
from PyQt6.QtWidgets import QMessageBox
import numpy as np
from datetime import datetime
import time
from onvif_gui.enums import StreamState, ProxyType
from loguru import logger
import threading
import traceback

class GLWidget(QOpenGLWidget):
    def __init__(self, mw):
        super().__init__()

        self.mw = mw
        self.focused_uri = None
        self.model_loading = False
        self.spinner = QMovie("image:spinner.gif")
        self.spinner.start()
        self.plain_recording = QMovie("image:plain_recording.gif")
        self.plain_recording.start()
        self.alarm_recording = QMovie("image:alarm_recording.gif")
        self.alarm_recording.start()

        self.buffer = QImage(self.size(), QImage.Format.Format_RGB888)
        self.file_draw = False
        self.pixmap = None
        self.force_clear = False

        self.timer = QTimer()
        self.timer.timeout.connect(self.timerCallback)
        refreshInterval = self.mw.settingsPanel.general.spnDisplayRefresh.value()
        self.timer.start(refreshInterval)

        self.last_alarm_check = time.time()
        self.alarms = {}
        self.last_player_count = 1
        self.last_manage_directory = datetime.now()
    
    def renderCallback(self, F, uri):
        try :
            if F is None:
                logger.error(f"a null value was returned for frame {uri}")
                return

            player = self.mw.pm.getPlayer(uri)
            if not player:
                logger.error(f"a null value was returned for player {uri}")
                return
            
            camera = self.mw.cameraPanel.getCamera(player.uri)
            if player.isCameraStream() and not camera:
                logger.error(f'a null value was returned for camera {uri}')
                return

            player.needs_render = True

            self.mw.pm.lock()
            player.lock()

            player.ary = np.array(F, copy=True)

            if len(player.ary.shape) < 3:
                return
            h = player.ary.shape[0]
            w = player.ary.shape[1]
            d = player.ary.shape[2]
            player.image = QImage(player.ary.data, w, h, d * w, QImage.Format.Format_RGB888)

            if self.mw.settingsPanel.proxy.generateAlarmsLocally():
                if self.mw.videoConfigure:
                    if player.analyze_video and self.mw.videoConfigure.initialized:
                        if camera.videoModelSettings.skipCounter < camera.videoModelSettings.skipFrames:
                            camera.videoModelSettings.skipCounter += 1
                        else:
                            camera.videoModelSettings.skipCounter = 0
                            player.boxes = self.mw.pyVideoCallback(player.ary, uri)
                            result = player.processModelOutput()
                            alarmState = result >= camera.videoModelSettings.limit if result else False
                            player.handleAlarm(alarmState)
                            if camera.isCurrent():
                                level = 0
                                if camera.videoModelSettings.limit:
                                    level = result / camera.videoModelSettings.limit
                                else:
                                    if result:
                                        level = 1.0

                                if camera.videoModelSettings.module_name == "yolox":
                                    self.mw.videoConfigure.selTargets.barLevel.setLevel(level)
                                    if alarmState:
                                        self.mw.videoConfigure.selTargets.indAlarm.setState(1)
                    else:
                        # clear the video panel alarm display
                        if player.uri == self.focused_uri:
                            if self.mw.videoWorker:
                                self.mw.videoWorker(None, None)
            else:
                player.loadRemoteDetections()


            w_s = w
            h_s = h
            # the aspect ratios are multiplied by 100 and converted to int for comparison purpose
            actual_ratio = int(100.0 * float(w) / float(h))
            if actual_ratio and player.desired_aspect:
                if actual_ratio != player.desired_aspect:
                    if player.desired_aspect > 1:
                        w_s = int(h * player.desired_aspect / 100)
                    else:
                        h_s = int(w * 100 / player.desired_aspect)

            self.mw.pm.sizes[player.uri] = QSize(w_s, h_s)


            if player.packet_drop_frame_counter > 0:
                player.packet_drop_frame_counter -= 1
            else:
                player.packet_drop_frame_counter = 0

        except BaseException as ex:
            logger.error(f'GLWidget render callback exception: {str(ex)}')
            logger.debug(traceback.format_exc())

        player.unlock()
        self.mw.pm.unlock()
        self.alarmBroadcast()

    def reconnectCycle(self):
        self.mw.pm.lock()
        try:
            for player in self.mw.pm.players.values():
                if not player.last_render:
                    player.last_render = datetime.now()
                if player.isCameraStream() and player.last_render:
                    interval = datetime.now() - player.last_render
                    if interval.total_seconds() > 30:
                        player.requestShutdown(reconnect=True)
                        player.last_render = datetime.now()
                        continue
        except Exception as ex:
            logger.error(f'GLWidget reconnect exception: {str(ex)}')
        self.mw.pm.unlock()

    def timerCallback(self):

        if self.force_clear:
            self.force_clear = False
            if self.mw.settings_profile != "Reader":
                self.buffer.fill(QColorConstants.Black)


        if (not self.mw.pm.countPlayers() and not self.mw.getActiveTimerCount() and not self.last_player_count):
            return

        # limit calls to manage directory to once per cycle to reduce load
        md_interval = datetime.now() - self.last_manage_directory
        if md_interval.total_seconds() > self.mw.STD_FILE_DURATION:
            d = self.mw.settingsPanel.storage.dirArchive.txtDirectory.text()
            if self.mw.settingsPanel.storage.chkManageDiskUsage.isChecked() and self.mw.enable_disk_mgmt:
                thread = threading.Thread(target=self.mw.diskManager.manageDirectory, args=(d,))
                thread.start()
                self.last_manage_directory = datetime.now()
        
        # stress testing
        if self.mw.settingsPanel.general.chkStressTest.isChecked():
            self.reconnectCycle()
        
        if self.mw.last_alarm:
            interval = datetime.now() - self.mw.last_alarm
            if interval.total_seconds() > 10:
                self.mw.alarm_states = []

        if self.mw.split.sizes()[0]:
            if self.mw.pm.countPlayers() or len(self.mw.timers):
                self.file_draw = False
                self.buildImage()
            else:
                if self.buffer and not self.file_draw:
                    self.buffer.fill(QColorConstants.Black)

            self.update()

        self.last_player_count = self.mw.pm.countPlayers() + self.mw.getActiveTimerCount()

    def paintGL(self):
        try:
            if self.buffer:
                if not self.buffer.isNull():
                    painter = QPainter(self)
                    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                    painter.drawImage(self.rect(), self.buffer)
        except Exception as ex:
            logger.error(f'GLWidget paintGL exception: {str(ex)}')
            logger.debug(traceback.format_exc())


    def alarmBroadcast(self):
        proxyPanel = self.mw.settingsPanel.proxy
        if proxyPanel.proxyType == ProxyType.SERVER and proxyPanel.grpAlarmBroadcast.isChecked():
            interval = time.time() - self.last_alarm_check
            if interval > 1:
                self.last_alarm_check = time.time()
                camera_alarms = {}
                alarm_states = []
                if self.mw.cameraPanel:
                    if self.mw.cameraPanel.lstCamera:
                        lstCamera = self.mw.cameraPanel.lstCamera
                        cameras = [lstCamera.item(x) for x in range(lstCamera.count())]
                        for camera in cameras:
                            camera_alarms[camera.serial_number()] = camera.isAlarming()
                            alarm_states.append(camera.isAlarming())
                
                udp_msg = str(datetime.now()) + "\n\nALARMS"
                tmp = []
                self.alarms = camera_alarms
                for alarm_state in alarm_states:
                    tmp.append(alarm_state)
                    udp_msg += "\n\n" + str(int(alarm_state))

                self.mw.broadcaster.send(udp_msg)

    def sizeHint(self):
        return QSize(640, 480)

    def resizeEvent(self, event):
        if self.buffer.isNull():
            self.buffer = QImage(self.size(), QImage.Format.Format_RGB888)
        self.buffer = self.buffer.scaled(event.size())
        if self.file_draw:
            self.drawFile("")
        return super().resizeEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self.mw.settingsPanel.proxy.proxyType == ProxyType.STAND_ALONE:
            ret = QMessageBox.warning(self, "Incompatible Configuration",
                                            "Focus Window is not available in stand alone configuration, please use Settings -> Proxy Server mode to enable",
                                            QMessageBox.StandardButton.Ok)
            return
        
        if self.mw.settings_profile == "Focus":
            return super().mouseDoubleClickEvent(event)

        if camera := self.mw.cameraPanel.getCurrentCamera():
            if profile := camera.getRecordProfile():
                self.mw.openFocusWindow()
                count = 0
                while not self.mw.focus_window.cameraPanel.getCamera(profile.uri()):
                    time.sleep(0.01)
                    count += 1
                    if count > 200:
                        logger.error("timeout error opening focus window")
                        break
                if camera := self.mw.focus_window.cameraPanel.getCamera(profile.uri()):
                    self.mw.focus_window.cameraPanel.onItemDoubleClicked(camera)

        return super().mouseDoubleClickEvent(event)
    
    def mousePressEvent(self, event):
        resolved = False
        self.mw.pm.lock()
        for player in self.mw.pm.players.values():
            if self.mw.pm.displayRect(player.uri, self.buffer.size()).contains(event.position()):
                if not player.hidden:
                    self.focused_uri = player.uri
                    if self.mw.isCameraStreamURI(player.uri):
                        self.mw.cameraPanel.setCurrentCamera(player.uri)
                        resolved = True
                        break
                    else:
                        if self.mw.isSplitterCollapsed():
                            self.mw.restoreSplitter()
                        self.mw.filePanel.setCurrentFile(self.focused_uri)
                        resolved = True
                        break
        self.mw.pm.unlock()
        if not resolved:
            for timer in self.mw.timers.values():
                if timer.isActive():
                    if self.mw.pm.displayRect(timer.uri, self.buffer.size()).contains(event.position()):
                        self.focused_uri = timer.uri
                        if self.mw.isSplitterCollapsed():
                            self.mw.restoreSplitter()
                        self.mw.cameraPanel.setCurrentCamera(self.focused_uri)
                        break
        return super().mousePressEvent(event)
    
    def isFocusedURI(self, uri):
        result = False
        if uri == self.focused_uri:
            result = True
        else:
            uris = self.mw.pm.getStreamPairURIs(uri)
            for u in uris:
                if u == self.focused_uri:
                    result = True
        return result

    def drawFile(self, filename):
        if self.buffer.isNull():
            return
        
        painter = QPainter(self.buffer)
        if not painter.isActive():
            return

        if self.mw.pm.countPlayers():
            return

        if len(filename):
            self.pixmap = QPixmap(filename)
        scaled_size = self.pixmap.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        scaled_rect = QRect(QPoint(0, 0), scaled_size)
        painter.fillRect(self.rect(), QColorConstants.Black)
        scaled_rect.moveCenter(self.rect().center())
        painter.drawPixmap(scaled_rect, self.pixmap)
        self.file_draw = True
        self.update()
    
    def buildImage(self):
        try:
            if self.buffer.isNull():
                return
            
            painter = QPainter(self.buffer)
            if not painter.isActive():
                return
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            if self.model_loading:
                rectSpinner = QRectF(0, 0, 40, 40)
                rectSpinner.moveCenter(QPointF(self.rect().center()))
                painter.drawImage(rectSpinner, self.spinner.currentImage())
                return
            
            if not len(self.mw.pm.players):
                painter.fillRect(self.rect(), QColorConstants.Black)

            self.mw.pm.lock()
            for player in self.mw.pm.players.values():

                if player.isRecording():
                    self.mw.enable_disk_mgmt = True

                if player.output_file_start_time:
                    interval = datetime.now() - player.output_file_start_time
                    if interval.total_seconds() > self.mw.STD_FILE_DURATION:
                        if filename := player.getOutputFilename():
                            player.startFileBreak(filename)

                if player.disable_video or player.hidden:
                    continue

                if not player.image and player.isCameraStream():
                    rect = self.mw.pm.displayRect(player.uri, self.buffer.size())
                    if rect.isValid():
                        painter.fillRect(rect, QColorConstants.Black)
                        rectSpinner = QRectF(0, 0, 40, 40)
                        rectSpinner.moveCenter(rect.center())
                        painter.drawImage(rectSpinner, self.spinner.currentImage())
                        if self.isFocusedURI(player.uri):
                            painter.setPen(QColorConstants.White)
                            painter.drawRect(rect.adjusted(1, 1, -2, -2))
                    if camera := self.mw.cameraPanel.getCamera(player.uri):
                        camera.setIcon(QIcon(self.spinner.currentPixmap()))
                        if camera.isCurrent():
                            self.mw.cameraPanel.setTabsEnabled(False)
                    continue

                if camera := self.mw.cameraPanel.getCamera(player.uri):
                    camera.setIconOn()
                    if camera.isCurrent():
                        self.mw.cameraPanel.setTabsEnabled(True)

                player.lock()

                if player.needs_render:
                    player.needs_render = False
                else:
                    continue

                rect = self.mw.pm.displayRect(player.uri, self.buffer.size())
                if player.image:
                    painter.drawImage(rect, player.image)

                x = rect.x()
                y = rect.y()
                w = rect.width()
                h = rect.height()

                if not (player.analyze_video or player.analyze_audio) and player.alarm_state:
                    player.setAlarmState(0)

                # display blinking alarm signal
                if self.mw.settingsPanel.alarm.chkShowDisplay.isChecked():
                    blinker_size = 20
                    rectBlinker = QRectF(x+w-1.25*blinker_size, y+h-1.25*blinker_size, blinker_size, blinker_size)
                    if camera := self.mw.cameraPanel.getCamera(player.uri):
                        if camera.isRecording():
                            if camera.isAlarming():
                                painter.drawImage(rectBlinker, self.alarm_recording.currentImage())
                            else:
                                if not player.systemTabSettings().record_always or not player.systemTabSettings().record_enable:
                                    painter.drawImage(rectBlinker, self.plain_recording.currentImage())
                        else:
                            if camera.isAlarming():
                                painter.drawImage(rectBlinker, QImage("image:alarm_plain.png"))

                # show model detection boxes
                if player.boxes is not None:
                    show = False
                    if self.mw.settingsPanel.proxy.generateAlarmsLocally() and camera.videoModelSettings:
                            show = camera.videoModelSettings.show
                    if show and len(player.boxes) and player.analyze_video:
                        scalex = w / player.image.rect().width()
                        scaley = h / player.image.rect().height()
                        painter.setPen(QColorConstants.Red)
                        for box in player.boxes:
                            p = (box[0] * scalex + x)
                            q = (box[1] * scaley + y)
                            r = (box[2] - box[0]) * scalex
                            s = (box[3] - box[1]) * scaley
                            painter.drawRect(QRectF(p, q, r, s).intersected(rect))

                if player.save_image_filename:
                    player.image.save(player.save_image_filename)
                    player.save_image_filename = None

                if self.isFocusedURI(player.uri) and player.isCameraStream():
                    painter.setPen(QColorConstants.White)
                    painter.drawRect(rect.adjusted(1, 1, -2, -2))

                if player.packet_drop_frame_counter > 0:
                    pen = QPen(QColorConstants.Yellow)
                    pen.setWidth(3)
                    painter.setPen(pen)
                    painter.drawRect(rect.adjusted(3, 3, -5, -5))

                player.unlock()

                blanks = self.mw.pm.getBlankSpace(self.buffer.size())
                for blank in blanks:
                    painter.fillRect(blank, QColorConstants.Black)

            self.mw.pm.unlock()

            for timer in self.mw.timers.values():

                if timer.attempting_reconnect:
                    if camera := self.mw.cameraPanel.getCamera(timer.uri):
                        camera.setIcon(QIcon(self.spinner.currentPixmap()))
                        if camera.isCurrent():
                            self.mw.cameraPanel.setTabsEnabled(False)
                        if camera.displayProfileIndex() != camera.recordProfileIndex():
                            mainState = camera.getStreamState(camera.recordProfileIndex())
                            subState = camera.getStreamState(camera.displayProfileIndex())
                            if subState == StreamState.CONNECTING and mainState == StreamState.CONNECTING:
                                if recordProfile := camera.getRecordProfile():
                                    if timer.uri == recordProfile.uri():
                                        continue
                    
                    timer.lock()
                    rect = self.mw.pm.displayRect(timer.uri, self.buffer.size())
                    painter.fillRect(rect, QColorConstants.Black)
                    painter.setPen(QColorConstants.LightGray)
                    rectSpinner = QRectF(0, 0, 40, 40)
                    rectSpinner.moveCenter(rect.center())
                    painter.drawImage(rectSpinner, self.spinner.currentImage())

                    day = timer.disconnected_time.strftime("%m/%d/%Y")
                    if day == datetime.now().strftime("%m/%d/%Y"):
                        day = "today"

                    text = f'{self.mw.getCameraName(timer.uri)}'
                    rectText = self.getTextRect(painter, text)
                    rectText.moveCenter(QPointF(rect.center().x(), rectSpinner.top()-3*rectText.height()))
                    painter.drawText(rectText, text)

                    text = f'Disconnected {day} at {timer.disconnected_time.strftime("%H:%M:%S")} ({self.interval2string(datetime.now() - timer.disconnected_time)})'
                    rectText = self.getTextRect(painter, text)
                    rectText.moveCenter(QPointF(rect.center().x(), rectSpinner.top()-2*rectText.height()))
                    painter.drawText(rectText, text)

                    text = f'Next reconnect attempt in {int(timer.remainingTime()/1000)} seconds'
                    rectText = self.getTextRect(painter, text)
                    rectText.moveCenter(QPointF(rectSpinner.center().x(), rectSpinner.top()-rectText.height()))
                    painter.drawText(rectText, text)

                    if self.isFocusedURI(timer.uri) and rect.isValid():
                        painter.setPen(QColorConstants.White)
                        painter.drawRect(rect.adjusted(1, 1, -2, -2))

                    timer.unlock()

        except Exception as ex:
            logger.error(f'GLWidget buildImage exception: {str(ex)}')
            logger.debug(traceback.format_exc())

    def getTextRect(self, painter, text):
        # to get exact bounding box, first estimate
        estimate = painter.fontMetrics().boundingRect(text)
        return painter.fontMetrics().boundingRect(estimate, 0, text).toRectF()

    def interval2string(self, interval):
        time_interval = int(interval.total_seconds())
        hours = int(time_interval / 3600)
        minutes = int ((time_interval - (hours * 3600)) / 60)
        seconds = int ((time_interval - (hours * 3600) - (minutes * 60)))
        buf = ""
        if hours > 0:
            buf += f'{hours} hours ' 
        if minutes > 0:
            buf += f'{minutes} minutes '
        buf += f'{seconds} seconds'
        return buf
