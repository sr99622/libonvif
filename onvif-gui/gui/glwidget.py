#********************************************************************
# libavio/onvif-gui/gui/glwidget.py
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
import numpy as np
from datetime import datetime
import time
from gui.enums import StreamState
from loguru import logger

class GLWidgetSignals(QObject):
    mouseClick = pyqtSignal(QPointF)

class GLWidget(QOpenGLWidget):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw
        self.focused_uri = None
        self.image_loading = False
        self.model_loading = False
        self.spinner = QMovie("image:spinner.gif")
        self.spinner.start()
        self.plain_recording = QMovie("image:plain_recording.gif")
        self.plain_recording.start()
        self.alarm_recording = QMovie("image:alarm_recording.gif")
        self.alarm_recording.start()

        self.signals = GLWidgetSignals()
        self.signals.mouseClick.connect(self.handleMouseClick)

        self.timer = QTimer()
        self.timer.timeout.connect(self.timerCallback)
        refreshInterval = self.mw.settingsPanel.general.spnDisplayRefresh.value()
        self.timer.start(refreshInterval)
    
    def renderCallback(self, F, player):
        try :
            player.last_render = datetime.now()
            if player.analyze_video and self.mw.videoConfigure.initialized:
                F = self.mw.pyVideoCallback(F, player)
            else:
                if player.uri == self.focused_uri:
                    if self.mw.videoWorker:
                        self.mw.videoWorker(None, None)

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

            while player.rendering:
                time.sleep(0.001)

            self.image_loading = True

            if d > 1:
                player.image = QImage(ary.data, w, h, d * w, QImage.Format.Format_RGB888)
            else:
                player.image = QImage(ary.data, w, h, w, QImage.Format.Format_Grayscale8)

            if player.save_image_filename:
                player.image.save(player.save_image_filename)
                player.save_image_filename = None

            if player.packet_drop_frame_counter > 0:
                player.packet_drop_frame_counter -= 1
            else:
                player.packet_drop_frame_counter = 0

            self.image_loading = False

            current = self.mw.cameraPanel.getCurrentPlayer()
            if current:
                if player.uri == current.uri:
                    self.mw.cameraPanel.tabVideo.updateCacheSize(player.getCacheSize())

        except BaseException as ex:
            logger.error(f'GLWidget render callback exception: {str(ex)}')

    def timerCallback(self):
        #self.repaint()
        self.update()

    def sizeHint(self):
        return QSize(640, 480)

    def mouseDoubleClickEvent(self, event):
        if self.mw.isFullScreen():
            self.mw.showNormal()
        else:
            self.mw.showFullScreen()

        return super().mouseDoubleClickEvent(event)
    
    def handleMouseClick(self, pos):
        resolved = False
        for player in self.mw.pm.players:
            if self.mw.pm.displayRect(player.uri, self.size()).contains(pos):
                if not player.hidden:
                    self.focused_uri = player.uri
                    if self.mw.isCameraStreamURI(player.uri):
                        if self.mw.isSplitterCollapsed():
                            self.mw.restoreSplitter()
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
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.fillRect(self.rect(), QColorConstants.Black)
 
            if self.model_loading:
                rectSpinner = QRectF(0, 0, 40, 40)
                rectSpinner.moveCenter(QPointF(self.rect().center()))
                painter.drawImage(rectSpinner, self.spinner.currentImage())
                return

            for player in self.mw.pm.players:

                if player.isCameraStream() and player.running and player.last_render:
                    interval = datetime.now() - player.last_render
                    if interval.total_seconds() > 5:
                        logger.debug(f'Lost signal for {self.mw.getCameraName(player.uri)}')
                        player.requestShutdown()
                        self.mw.playMedia(player.uri)

                if player.pipe_output_start_time:
                    interval = datetime.now() - player.pipe_output_start_time
                    if interval.total_seconds() > self.mw.STD_FILE_DURATION:
                        d = self.mw.settingsPanel.storage.dirArchive.txtDirectory.text()
                        if self.mw.settingsPanel.storage.chkManageDiskUsage.isChecked():
                            player.manageDirectory(d)

                        filename = player.getPipeOutFilename(d)
                        if filename:
                            player.startFileBreakPipe(filename)

                if player.disable_video or player.hidden:
                    continue

                camera = self.mw.cameraPanel.getCamera(player.uri)

                if player.image is None:
                    rect = self.mw.pm.displayRect(player.uri, self.size())
                    rectSpinner = QRectF(0, 0, 40, 40)
                    rectSpinner.moveCenter(rect.center())
                    painter.drawImage(rectSpinner, self.spinner.currentImage())
                    if camera:
                        camera.setIcon(QIcon(self.spinner.currentPixmap()))
                        if camera.isCurrent():
                            self.mw.cameraPanel.setTabsEnabled(False)

                    if self.isFocusedURI(player.uri):
                        if rect.isValid():
                            painter.setPen(QColorConstants.White)
                            painter.drawRect(rect.adjusted(1, 1, -2, -2))

                    continue

                while self.image_loading:
                    time.sleep(0.001)

                player.rendering = True

                rect = self.mw.pm.displayRect(player.uri, self.size())
                x = rect.x()
                y = rect.y()
                w = rect.width()
                h = rect.height()

                painter.drawImage(rect, player.image)

                b = 26
                rectBlinker = QRectF(0, 0, b, b)
                rectBlinker.moveCenter(QPointF(x+w-b, y+h-b))
                if camera:
                    if camera.isRecording():
                        if camera.isAlarming():
                            painter.drawImage(rectBlinker, self.alarm_recording.currentImage())
                        else:
                            if not player.systemTabSettings.record_always:
                                painter.drawImage(rectBlinker, self.plain_recording.currentImage())
                    else:
                        if camera.isAlarming():
                            painter.drawImage(rectBlinker, QImage("image:alarm_plain.png"))

                if not player.isCameraStream() and player.alarm_state:
                    painter.drawImage(rectBlinker, QImage("image:alarm_plain.png"))

                if self.isFocusedURI(player.uri):
                    painter.setPen(QColorConstants.White)
                    painter.drawRect(rect.adjusted(1, 1, -2, -2))

                if player.packet_drop_frame_counter > 0:
                    pen = QPen(QColorConstants.Yellow)
                    pen.setWidth(3)
                    painter.setPen(pen)
                    painter.drawRect(rect.adjusted(3, 3, -5, -5))

                show = False
                if player.videoModelSettings:
                    show = player.videoModelSettings.show

                if show and player.boxes is not None and player.analyze_video:
                    scalex = w / player.image.rect().width()
                    scaley = h / player.image.rect().height()

                    painter.setPen(QColorConstants.Red)
                    for box in player.boxes:
                        p = (box[0] * scalex + x)
                        q = (box[1] * scaley + y)
                        r = (box[2] - box[0]) * scalex
                        s = (box[3] - box[1]) * scaley
                        painter.drawRect(QRectF(p, q, r, s))

                player.rendering = False

                if camera:
                    if camera.isCurrent():
                        self.mw.cameraPanel.setTabsEnabled(True)

                    if player.image:
                        camera.setIconOn()

            for timer in self.mw.timers.values():

                if timer.attempting_reconnect:
                    camera = self.mw.cameraPanel.getCamera(timer.uri)
                    if camera:
                        camera.setIcon(QIcon(self.spinner.currentPixmap()))
                        if camera.isCurrent():
                            self.mw.cameraPanel.setTabsEnabled(False)

                        if camera.displayProfileIndex() != camera.recordProfileIndex():
                            mainState = camera.getStreamState(camera.recordProfileIndex())
                            subState = camera.getStreamState(camera.displayProfileIndex())
                            if subState == StreamState.CONNECTING and mainState == StreamState.CONNECTING:
                                recordProfile = camera.getRecordProfile()
                                if recordProfile:
                                    if timer.uri == recordProfile.uri():
                                        continue
                    
                    timer.rendering = True
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

                    timer.rendering = False
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
