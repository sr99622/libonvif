#********************************************************************
# libonvif/onvif-gui/onvif_gui/glwidget.py
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
from PyQt6.QtGui import QPainter, QImage, QColorConstants, QPen, QMovie, QIcon
from PyQt6.QtCore import QSize, QPointF, QRectF, QTimer, QObject, pyqtSignal
from PyQt6.QtWidgets import QMessageBox
import numpy as np
from datetime import datetime
import time
from onvif_gui.enums import StreamState, ProxyType
from loguru import logger
#import psutil

class GLWidgetSignals(QObject):
    mouseClick = pyqtSignal(QPointF)

class GLWidget(QOpenGLWidget):
    def __init__(self, mw):
        super().__init__()

        #self.process = psutil.Process(os.getpid())

        self.mw = mw
        self.focused_uri = None
        self.model_loading = False
        self.spinner = QMovie("image:spinner.gif")
        self.spinner.start()
        self.plain_recording = QMovie("image:plain_recording.gif")
        self.plain_recording.start()
        self.alarm_recording = QMovie("image:alarm_recording.gif")
        self.alarm_recording.start()

        self.buffer = QImage(self.size(), QImage.Format.Format_ARGB32)

        self.signals = GLWidgetSignals()
        self.signals.mouseClick.connect(self.handleMouseClick)

        self.timer = QTimer()
        self.timer.timeout.connect(self.timerCallback)
        refreshInterval = self.mw.settingsPanel.general.spnDisplayRefresh.value()
        self.timer.start(refreshInterval)

        self.last_alarm_check = time.time()
        self.alarms = {}
        self.count = 0
    
    def renderCallback(self, F, player):
        try :
            self.mw.pm.lock()

            if self.mw.settingsPanel.proxy.generateAlarmsLocally():
                if self.mw.videoConfigure:
                    if player.analyze_video and self.mw.videoConfigure.initialized:
                        F = self.mw.pyVideoCallback(F, player)
                    else:
                        # clear the video panel alarm display
                        if player.uri == self.focused_uri:
                            if self.mw.videoWorker:
                                self.mw.videoWorker(None, None)
            else:
                player.loadRemoteDetections()

            ary = np.array(F, copy = False) 

            if len(ary.shape) < 2:
                return
            h = ary.shape[0]
            w = ary.shape[1]
            d = 1
            if ary.ndim > 2:
                d = ary.shape[2]

            w_s = w
            h_s = h
            # the aspect ratios are multiplied by 100 and converted to int for comparison purpose
            actual_ratio = int(100.0 * float(w) / float(h))
            if actual_ratio and player.desired_aspect:
                if actual_ratio != player.desired_aspect:
                    if player.desired_aspect > 1:
                        w_s = int(h * player.desired_aspect / 100)
                    else:
                        h_s = int(w *100 / player.desired_aspect)

            self.mw.pm.sizes[player.uri] = QSize(w_s, h_s)

            player.lock()

            if d > 1:
                player.image = QImage(ary.data, w, h, d * w, QImage.Format.Format_RGB888)
            else:
                player.image = QImage(ary.data, w, h, w, QImage.Format.Format_Grayscale8)

            #if player.save_image_filename:
            #    player.image.save(player.save_image_filename)
            #    player.save_image_filename = None

            if player.packet_drop_frame_counter > 0:
                player.packet_drop_frame_counter -= 1
            else:
                player.packet_drop_frame_counter = 0

            player.unlock()
            self.mw.pm.unlock()

        except BaseException as ex:
            logger.error(f'GLWidget render callback exception: {str(ex)}')

        self.alarmBroadcast()

    def chores(self):
        try:

            #self.mw.settingsPanel.general.lblMemory.setText(f'{self.process.memory_info().rss:_}')
 
            '''
            #
            # RECONNECT CYCLING stress testing for streams
            #

            if not player.last_render:
                player.last_render = datetime.now()
            uri = player.uri
            if player.isCameraStream() and player.running and player.last_render:
                interval = datetime.now() - player.last_render
                if interval.total_seconds() > 30:
                    player.last_render = datetime.now()
                    player.requestShutdown(reconnect=True)
                    continue
            #'''

            for player in self.mw.pm.players:
                if player.pipe_output_start_time:
                    interval = datetime.now() - player.pipe_output_start_time
                    if interval.total_seconds() > self.mw.STD_FILE_DURATION:
                        d = self.mw.settingsPanel.storage.dirArchive.txtDirectory.text()
                        if self.mw.settingsPanel.storage.chkManageDiskUsage.isChecked():
                            self.mw.diskManager.manageDirectory(d, player.uri)
                        #else:
                        #    self.mw.diskManager.getDirectorySize(d)

                        if filename := player.getPipeOutFilename():
                            player.startFileBreakPipe(filename)

                if player.disable_video or player.hidden:
                    continue

                if camera := self.mw.cameraPanel.getCamera(player.uri):
                    if player.image:
                        camera.setIconOn()
                        if camera.isCurrent():
                            self.mw.cameraPanel.setTabsEnabled(True)
                    else:
                        camera.setIcon(QIcon(self.spinner.currentPixmap()))
                        if camera.isCurrent():
                            self.mw.cameraPanel.setTabsEnabled(False)

            for timer in self.mw.timers.values():
                if timer.attempting_reconnect:
                    if camera := self.mw.cameraPanel.getCamera(timer.uri):
                        camera.setIcon(QIcon(self.spinner.currentPixmap()))
                        if camera.isCurrent():
                            self.mw.cameraPanel.setTabsEnabled(False)

        except Exception as ex:
            logger.error(f'GLWidget timerCallback exception: {str(ex)}')

    def timerCallback(self):

        self.chores()

        if self.mw.last_alarm:
            interval = datetime.now() - self.mw.last_alarm
            if interval.total_seconds() > 10:
                self.mw.alarm_states = []

        if self.mw.split.sizes()[0]:
            if len(self.mw.pm.players) or len(self.mw.timers):
                self.buildImage()
            else:
                if self.buffer:
                    self.buffer.fill(QColorConstants.Black)

            self.update()

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

                existing = False
                if self.mw.focus_window:
                    if self.mw.focus_window.isVisible():
                        existing = True

                if not existing:
                    self.mw.initializeFocusWindowSettings()

                if camera := self.mw.focus_window.cameraPanel.getCamera(profile.uri()):
                    self.mw.focus_window.cameraPanel.onItemDoubleClicked(camera)
                else:
                    self.mw.focus_window.playMedia(profile.uri())

        return super().mouseDoubleClickEvent(event)
    
    def handleMouseClick(self, pos):
        resolved = False
        for player in self.mw.pm.players:
            if self.mw.pm.displayRect(player.uri, self.size()).contains(pos):
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
        if not resolved:
            for timer in self.mw.timers.values():
                if timer.isActive():
                    if self.mw.pm.displayRect(timer.uri, self.size()).contains(pos):
                        self.focused_uri = timer.uri
                        if self.mw.isSplitterCollapsed():
                            self.mw.restoreSplitter()
                        self.mw.cameraPanel.setCurrentCamera(self.focused_uri)
                        break
    
    def mouseReleaseEvent(self, event):
        self.handleMouseClick(event.position())
        return super().mouseReleaseEvent(event)
    
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
    
    def paintGL(self):
        try:
            if self.buffer:
                if not self.buffer.isNull():
                    painter = QPainter(self)
                    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                    painter.fillRect(self.rect(), QColorConstants.Black)
                    painter.drawImage(self.rect(), self.buffer)
        except Exception as ex:
            logger.error(f'GLWidget paintGL exception: {str(ex)}')

    def buildImage(self):
        try:
            self.buffer = QImage(self.size(), QImage.Format.Format_ARGB32)
            if self.buffer.isNull():
                return
            painter = QPainter(self.buffer)
            if not painter.isActive():
                return
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.fillRect(self.rect(), QColorConstants.Black)

            if self.model_loading:
                rectSpinner = QRectF(0, 0, 40, 40)
                rectSpinner.moveCenter(QPointF(self.rect().center()))
                painter.drawImage(rectSpinner, self.spinner.currentImage())
                return

            self.mw.pm.lock()
            for player in self.mw.pm.players:

                if player.disable_video or player.hidden:
                    continue

                camera = self.mw.cameraPanel.getCamera(player.uri)

                if player.image is None:
                    rect = self.mw.pm.displayRect(player.uri, self.size())
                    rectSpinner = QRectF(0, 0, 40, 40)
                    rectSpinner.moveCenter(rect.center())
                    painter.drawImage(rectSpinner, self.spinner.currentImage())

                    if self.isFocusedURI(player.uri):
                        if rect.isValid():
                            painter.setPen(QColorConstants.White)
                            painter.drawRect(rect.adjusted(1, 1, -2, -2))

                    continue

                player.lock()

                rect = self.mw.pm.displayRect(player.uri, self.size())
                x = rect.x()
                y = rect.y()
                w = rect.width()
                h = rect.height()

                painter.drawImage(rect, player.image)
                if not (player.analyze_video or player.analyze_audio) and player.alarm_state:
                    player.setAlarmState(0)

                b = 26
                rectBlinker = QRectF(0, 0, b, b)
                rectBlinker.moveCenter(QPointF(x+w-b, y+h-b))
                enabled = self.mw.settingsPanel.alarm.chkShowDisplay.isChecked()
                if camera:
                    if camera.isRecording():
                        if camera.isAlarming():
                            if enabled: painter.drawImage(rectBlinker, self.alarm_recording.currentImage())
                        else:
                            if not player.systemTabSettings().record_always or not player.systemTabSettings().record_enable:
                                if enabled: painter.drawImage(rectBlinker, self.plain_recording.currentImage())
                    else:
                        if camera.isAlarming():
                            if enabled: painter.drawImage(rectBlinker, QImage("image:alarm_plain.png"))

                if not player.isCameraStream() and player.alarm_state:
                    if enabled: painter.drawImage(rectBlinker, QImage("image:alarm_plain.png"))

                if self.isFocusedURI(player.uri):
                    painter.setPen(QColorConstants.White)
                    painter.drawRect(rect.adjusted(1, 1, -2, -2))

                if player.packet_drop_frame_counter > 0:
                    pen = QPen(QColorConstants.Yellow)
                    pen.setWidth(3)
                    painter.setPen(pen)
                    painter.drawRect(rect.adjusted(3, 3, -5, -5))

                show = False
                if self.mw.settingsPanel.proxy.generateAlarmsLocally():
                    if player.videoModelSettings:
                        show = player.videoModelSettings.show

                if show and len(player.boxes) and player.analyze_video:
                    if player.remote_width:
                        scalex = w / player.remote_width
                    else:
                        scalex = w / player.image.rect().width()

                    if player.remote_height:
                        scaley = h / player.remote_height
                    else:
                        scaley = h / player.image.rect().height()

                    painter.setPen(QColorConstants.Red)
                    for box in player.boxes:
                        p = (box[0] * scalex + x)
                        q = (box[1] * scaley + y)
                        r = (box[2] - box[0]) * scalex
                        s = (box[3] - box[1]) * scaley
                        painter.drawRect(QRectF(p, q, r, s))

                if player.save_image_filename:
                    #if show and player.analyze_video:
                    #    img = player.image.copy()
                    #    painter_img = QPainter(img)
                    #    painter_img.setPen(QColorConstants.Red)
                    #    for box in player.boxes:
                    #        p = (box[0])
                    #        q = (box[1])
                    #        r = (box[2] - box[0])
                    #        s = (box[3] - box[1])
                    #        painter_img.drawRect(QRectF(p, q, r, s))
                    #img.save(player.save_image_filename)
                    player.image.save(player.save_image_filename)
                    player.save_image_filename = None


                player.unlock()

            self.mw.pm.unlock()

            for timer in self.mw.timers.values():

                if timer.attempting_reconnect:
                    if camera := self.mw.cameraPanel.getCamera(timer.uri):

                        if camera.displayProfileIndex() != camera.recordProfileIndex():
                            mainState = camera.getStreamState(camera.recordProfileIndex())
                            subState = camera.getStreamState(camera.displayProfileIndex())
                            if subState == StreamState.CONNECTING and mainState == StreamState.CONNECTING:
                                if recordProfile := camera.getRecordProfile():
                                    if timer.uri == recordProfile.uri():
                                        continue
                    
                    timer.lock()
                    rect = self.mw.pm.displayRect(timer.uri, self.size())
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

        except BaseException as ex:
            logger.error(f'GLWidget onPaint exception: {str(ex)}')

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
