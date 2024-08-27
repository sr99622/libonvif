#/********************************************************************
# libonvif/onvif-gui/modules/audio/sample.py 
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

import numpy as np
from numpy.fft import fft
import math
import os
from loguru import logger
from PyQt6.QtWidgets import QGridLayout, QWidget, QSlider, QLabel, QWidget, QCheckBox, \
    QGroupBox, QRadioButton
from PyQt6.QtGui import QPainter, QColorConstants, QColor
from PyQt6.QtCore import QPointF, Qt, QRectF
from gui.components import WarningBar, Indicator
from gui.enums import MediaSource

MODULE_NAME = "sample"

class SampleSettings():
    def __init__(self, mw, camera=None):
        self.camera = camera
        self.mw = mw
        self.id = "File"
        if camera:
            self.id = camera.serial_number()

        self.amplitudeEnabled = self.getAudioAmplitudeEnabled()
        self.amplitudeGain = self.getAudioAmplitudeGain()
        self.frequencyEnabled = self.getAudioFrequencyEnabled()
        self.frequencyGain = self.getAudioFrequencyGain()
        self.frequencyPctHighPass = self.getAudioFrequencyPctHighPass()
        self.frequencyPctLowPass = self.getAudioFrequencyPctLowPass()
        self.frequencyPctCoverage = self.getAudioFrequencyPctCoverage()
        self.alarmLimitOver = self.getAudioAlarmLimitOver()
        
    def getAudioAmplitudeEnabled(self):
        key = f'{self.id}/{MODULE_NAME}AudioAmplitudeEnabled'
        return bool(int(self.mw.settings.value(key, 1)))
    
    def setAudioAmplitudeEnabled(self, state):
        key = f'{self.id}/{MODULE_NAME}AudioAmplitudeEnabled'
        self.amplitudeEnabled = bool(state)
        self.mw.settings.setValue(key, int(state))

    def getAudioAmplitudeGain(self):
        key = f'{self.id}/{MODULE_NAME}AudioAmplitudeGain'
        return int(self.mw.settings.value(key, 50))
    
    def setAudioAmplitudeGain(self, value):
        key = f'{self.id}/{MODULE_NAME}AudioAmplitudeGain'
        self.amplitudeGain = value
        self.mw.settings.setValue(key, value)

    def getAudioFrequencyEnabled(self):
        key = f'{self.id}/{MODULE_NAME}AudioFrequencyEnabled'
        return bool(int(self.mw.settings.value(key, 1)))
    
    def setAudioFrequencyEnabled(self, state):
        key = f'{self.id}/{MODULE_NAME}AudioFrequencyEnabled'
        self.frequencyEnabled = bool(state)
        self.mw.settings.setValue(key, int(state))

    def getAudioFrequencyGain(self):
        key = f'{self.id}/{MODULE_NAME}AudioFrequencyGain'
        return int(self.mw.settings.value(key, 50))
    
    def setAudioFrequencyGain(self, value):
        key = f'{self.id}/{MODULE_NAME}AudioFrequencyGain'
        self.frequencyGain = value
        self.mw.settings.setValue(key, value)
    
    def getAudioFrequencyPctHighPass(self):
        key = f'{self.id}/{MODULE_NAME}AudioFrequencyPctHighPass'
        return float(self.mw.settings.value(key, 0))
    
    def setAudioFrequencyPctHighPass(self, value):
        key = f'{self.id}/{MODULE_NAME}AudioFrequencyPctHighPass'
        self.frequencyPctHighPass = value
        self.mw.settings.setValue(key, value)

    def getAudioFrequencyPctLowPass(self):
        key = f'{self.id}/{MODULE_NAME}AudioFrequencyPctLowPass'
        return float(self.mw.settings.value(key, 1))
    
    def setAudioFrequencyPctLowPass(self, value):
        key = f'{self.id}/{MODULE_NAME}AudioFrequencyPctLowPass'
        self.frequencyPctLowPass = value
        self.mw.settings.setValue(key, value)

    def getAudioFrequencyPctCoverage(self):
        key = f'{self.id}/{MODULE_NAME}AudioFrequencyPctCoverage'
        return float(self.mw.settings.value(key, 1))
    
    def setAudioFrequencyPctCoverage(self, value):
        key = f'{self.id}/{MODULE_NAME}AudioFrequencyPctCoverage'
        self.frequencyPctCoverage = value
        self.mw.settings.setValue(key, value)

    def getAudioAlarmLimitOver(self):
        key = f'{self.id}/{MODULE_NAME}AudioAlarmLimitOver'
        return bool(int(self.mw.settings.value(key, 1)))
    
    def setAudioAlarmLimitOver(self, state):
        key = f'{self.id}/{MODULE_NAME}AudioAlarmLimitOver'
        self.alarmLimitOver = bool(state)
        self.mw.settings.setValue(key, int(state))

class AmplitudeDisplay(QLabel):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw
        self.data = None
        self.lock = False
        self.setStyleSheet("QLabel { background : black; }")
    
    def paintEvent(self, event):
        last_point = None
        width = self.size().width()
        height = self.size().height()
        painter = QPainter(self)
        painter.setPen(QColorConstants.DarkGreen)

        if self.data is not None:
            x_unit = width / self.data.size
            for i, value in enumerate(self.data):
                x = i * x_unit
                y = height*(1 - (1 + value)/2)
                current_point = QPointF(x, y)
                if last_point and current_point:
                    painter.drawLine(last_point, current_point)
                last_point = current_point

        if not self.mw.audioConfigure.chkAmplitude.isChecked() or not self.isEnabled() or self.data is None:
            painter.eraseRect(0, 0, width, height)

    def setData(self, data):
        self.data = data
        self.update()

class FrequencyDisplay(QLabel):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw
        self.data = None
        self.lock = False
        self.setStyleSheet("QLabel { background : black; }")
        self.setMouseTracking(True)
        self.draggingLowPass = False
        self.draggingHighPass = False
        self.pctCoverage = 1.0
        self.pctHighPass = 0
        self.pctLowPass = 1.0
        self.handleLowPass = QRectF(0, 0, 8, 8)
        self.handleHighPass = QRectF(0, 0, 8, 8)

    def paintEvent(self, event):
        width = self.size().width()
        height = self.size().height()
        painter = QPainter(self)
        painter.setPen(QColorConstants.DarkGreen)
        lowPass = int(self.pctLowPass * width)
        highPass = int(self.pctHighPass * width)

        if self.data is not None:
            x_unit = width / self.data.size
            # the y_scale is arbitrary based on esthetics
            y_scale = height/4 
            for i, value in enumerate(self.data):
                x = i * x_unit
                y = height - (value * y_scale)
                painter.setPen(QColorConstants.DarkGreen)
                if highPass < lowPass:
                    if x < highPass or x > lowPass:
                        painter.setPen(QColor("#303030"))
                else:
                    if x < highPass and x > lowPass:
                        painter.setPen(QColor("#303030"))
                painter.drawLine(QPointF(x, height), QPointF(x, y))

        if not self.mw.audioConfigure.chkFrequency.isChecked() or not self.isEnabled() or self.data is None:
            painter.eraseRect(0, 0, width, height)

        self.handleLowPass.moveCenter(QPointF(lowPass, height/2))
        self.handleHighPass.moveCenter(QPointF(highPass, height/2))
        painter.setPen(QColorConstants.DarkYellow)
        painter.drawRect(self.handleLowPass)
        painter.drawRect(self.handleHighPass)
        painter.drawLine(QPointF(lowPass, 0), QPointF(lowPass, self.handleLowPass.top()))
        painter.drawLine(QPointF(lowPass, self.handleLowPass.bottom()), QPointF(lowPass, height-1))
        painter.drawLine(QPointF(highPass, 0), QPointF(highPass, self.handleHighPass.top()))
        painter.drawLine(QPointF(highPass, self.handleHighPass.bottom()), QPointF(highPass, height-1))

    def mouseMoveEvent(self, event):
        if self.handleLowPass.contains(QPointF(event.pos())) or self.handleHighPass.contains(QPointF(event.pos())):
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

        lowPass = int(self.pctLowPass * self.width())
        if self.draggingLowPass:
            x = min(max(event.pos().x(), 0), self.size().width()-1)
            lowPass = x
            self.pctLowPass = lowPass / self.width()
            match self.mw.audioConfigure.source:
                case MediaSource.CAMERA:
                    camera = self.mw.cameraPanel.getCurrentCamera()
                    if camera:
                        camera.audioModelSettings.setAudioFrequencyPctLowPass(self.pctLowPass)
                case MediaSource.FILE:
                    self.mw.filePanel.audioModelSettings.setAudioFrequencyPctLowPass(self.pctLowPass)
            self.update()

        highPass = int(self.pctHighPass * self.width())
        if self.draggingHighPass:
            x = min(max(event.pos().x(), 0), self.size().width())
            highPass = x
            self.pctHighPass = highPass / self.width()
            match self.mw.audioConfigure.source:
                case MediaSource.CAMERA:
                    camera = self.mw.cameraPanel.getCurrentCamera()
                    if camera:
                        camera.audioModelSettings.setAudioFrequencyPctHighPass(self.pctHighPass)
                case MediaSource.FILE:
                    self.mw.filePanel.audioModelSettings.setAudioFrequencyPctHighPass(self.pctHighPass)
            self.update()

        if self.draggingHighPass or self.draggingLowPass:
            if highPass < lowPass:
                self.pctCoverage = 1 - (self.width() - lowPass + highPass - 1) / self.width()
            else:
                self.pctCoverage = (highPass - lowPass - 1) / self.width()
            self.pctCoverage = max(self.pctCoverage, 0.01)
            match self.mw.audioConfigure.source:
                case MediaSource.CAMERA:
                    camera = self.mw.cameraPanel.getCurrentCamera()
                    if camera:
                        camera.audioModelSettings.setAudioFrequencyPctCoverage(self.pctCoverage)
                case MediaSource.FILE:
                    self.mw.filePanel.audioModelSettings.setAudioFrequencyPctCoverage(self.pctCoverage)

        return super().mouseMoveEvent(event)
    
    def mousePressEvent(self, event):
        if self.handleLowPass.contains(QPointF(event.pos())):
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self.draggingLowPass = True

        if self.handleHighPass.contains(QPointF(event.pos())):
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self.draggingHighPass = True

        return super().mousePressEvent(event)
    
    def mouseReleaseEvent(self, event):
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.draggingLowPass = False
        self.draggingHighPass = False
        return super().mouseReleaseEvent(event)
    
    def setData(self, data):
        self.data = data
        self.update()

    def setPctHighPass(self, value):
        self.pctHighPass = value
        self.highPass = max(int(value * self.width()), 0)

    def setPctLowPass(self, value):
        self.pctLowPass = value
        self.lowPass = min(int(value * self.width()), self.width() - 1)

class AudioConfigure(QWidget):
    def __init__(self, mw):
        try:
            super().__init__()
            self.mw = mw
            self.name = MODULE_NAME
            self.source = MediaSource.CAMERA
            self.amplitudeGain = 1.0
            self.frequencyGain = 1.0

            self.dspAmplitude = AmplitudeDisplay(self.mw)
            self.chkAmplitude = QCheckBox("Amplitude")
            self.chkAmplitude.setChecked(True)
            self.chkAmplitude.stateChanged.connect(self.chkAmpStateChanged)
            
            self.sldAmplitudeGain = QSlider(Qt.Orientation.Vertical)
            self.sldAmplitudeGain.setValue(50)
            self.sldAmplitudeGain.valueChanged.connect(self.sldAmpGainValueChanged)

            self.barAmplitude = WarningBar()
            self.indAmplitude = Indicator(self.mw)
            self.lblAmplitudeGain = QLabel("Gain")
            self.dspFrequency = FrequencyDisplay(self.mw)
            self.chkFrequency = QCheckBox("Frequency")
            self.chkFrequency.setChecked(True)
            self.chkFrequency.stateChanged.connect(self.chkFreqStateChanged)

            self.sldFrequencyGain = QSlider(Qt.Orientation.Vertical)
            self.sldFrequencyGain.setValue(50)
            self.sldFrequencyGain.valueChanged.connect(self.sldFreqGainValueChanged)
            self.lblFrequencyGain = QLabel("Gain")
            self.barFrequency = WarningBar()
            self.indFrequency = Indicator(self.mw)

            self.grpAlarmCondition = QGroupBox("Trigger Alarm if Limit Reached")
            self.radOver = QRadioButton("Over", self.grpAlarmCondition)
            self.radOver.setChecked(True)
            self.radOver.toggled.connect(self.radOverToggled)
            self.radUnder = QRadioButton("Under", self.grpAlarmCondition)
            self.radUnder.setChecked(not self.radOver.isChecked())
            lytAlarmCondition = QGridLayout(self.grpAlarmCondition)
            lytAlarmCondition.addWidget(self.radOver,   0, 0, 1, 1)
            lytAlarmCondition.addWidget(self.radUnder,  0, 1, 1, 1)

            self.pnlAmplitude = QWidget()
            lytAmplitude = QGridLayout(self.pnlAmplitude)
            lytAmplitude.addWidget(self.chkAmplitude,       0, 0, 1, 1, Qt.AlignmentFlag.AlignCenter)
            lytAmplitude.addWidget(self.dspAmplitude,       1, 0, 4, 1)
            lytAmplitude.addWidget(self.sldAmplitudeGain,   2, 3, 1, 1, Qt.AlignmentFlag.AlignHCenter)
            lytAmplitude.addWidget(self.lblAmplitudeGain,   3, 3, 1, 1)
            lytAmplitude.addWidget(self.indAmplitude,       1, 4, 1, 1)
            lytAmplitude.addWidget(self.barAmplitude,       2, 4, 2, 1)
            lytAmplitude.setColumnStretch(0, 10)
            lytAmplitude.setColumnStretch(4, 2)
            lytAmplitude.setContentsMargins(0, 0, 0, 0)

            self.pnlFrequency = QWidget()
            lytFrequency = QGridLayout(self.pnlFrequency)
            lytFrequency.addWidget(self.chkFrequency,       0, 0, 1, 1, Qt.AlignmentFlag.AlignCenter)
            lytFrequency.addWidget(self.dspFrequency,       1, 0, 4, 1)
            lytFrequency.addWidget(self.sldFrequencyGain,   2, 3, 1, 1, Qt.AlignmentFlag.AlignHCenter)
            lytFrequency.addWidget(self.lblFrequencyGain,   3, 3, 1, 1)
            lytFrequency.addWidget(self.indFrequency,       1, 4, 1, 1)
            lytFrequency.addWidget(self.barFrequency,       2, 4, 2, 1)
            lytFrequency.setColumnStretch(0, 10)
            lytFrequency.setColumnStretch(4, 2)
            lytFrequency.setContentsMargins(0, 0, 0, 0)

            self.pnlAmplitude.setMinimumHeight(200)
            self.pnlFrequency.setMinimumHeight(200)

            lytMain = QGridLayout(self)
            lytMain.addWidget(self.pnlAmplitude,      0, 0, 1, 4)
            lytMain.addWidget(self.pnlFrequency,      1, 0, 1, 4)
            lytMain.addWidget(self.grpAlarmCondition, 2, 0, 1, 3)
            lytMain.addWidget(QLabel(),               3, 0, 1, 4)
            lytMain.setRowStretch(3, 3)

            self.enableControls(False)

        except:
            logger.exception("sample configuration failed to load")

    def radOverToggled(self):
        self.barAmplitude.inverted = not self.radOver.isChecked()
        self.barAmplitude.update()
        self.barFrequency.inverted = not self.radOver.isChecked()
        self.barFrequency.update()
        match self.source:
            case MediaSource.CAMERA:
                camera = self.mw.cameraPanel.getCurrentCamera()
                if camera:
                    camera.audioModelSettings.setAudioAlarmLimitOver(self.radOver.isChecked())
            case MediaSource.FILE:
                self.mw.filePanel.audioModelSettings.setAudioAlarmLimitOver(self.radOver.isChecked())

    def sldAmpGainValueChanged(self, value):
        self.amplitudeGain = math.exp(0.1 * (value - 50))
        match self.source:
            case MediaSource.CAMERA:
                camera = self.mw.cameraPanel.getCurrentCamera()
                if camera:
                    camera.audioModelSettings.setAudioAmplitudeGain(value)
            case MediaSource.FILE:
                self.mw.filePanel.audioModelSettings.setAudioAmplitudeGain(value)

    def sldFreqGainValueChanged(self, value):
        self.frequencyGain = math.exp(0.05 * (value - 50))
        match self.source:
            case MediaSource.CAMERA:
                camera = self.mw.cameraPanel.getCurrentCamera()
                if camera:
                    camera.audioModelSettings.setAudioFrequencyGain(value)
            case MediaSource.FILE:
                self.mw.filePanel.audioModelSettings.setAudioFrequencyGain(value)

    def chkAmpStateChanged(self, state):
        self.barAmplitude.setLevel(0.0)
        self.indAmplitude.setState(0)
        self.dspAmplitude.setData(None)
        match self.source:
            case MediaSource.CAMERA:
                camera = self.mw.cameraPanel.getCurrentCamera()
                if camera:
                    camera.audioModelSettings.setAudioAmplitudeEnabled(state)
            case MediaSource.FILE:
                self.mw.filePanel.audioModelSettings.setAudioAmplitudeEnabled(state)

    def chkFreqStateChanged(self, state):
        self.barFrequency.setLevel(0.0)
        self.indFrequency.setState(0)
        self.dspFrequency.setData(None)
        match self.source:
            case MediaSource.CAMERA:
                camera = self.mw.cameraPanel.getCurrentCamera()
                if camera:
                    camera.audioModelSettings.setAudioFrequencyEnabled(state)
            case MediaSource.FILE:
                self.mw.filePanel.audioModelSettings.setAudioFrequencyEnabled(state)

    def clearIndicators(self):
        self.barAmplitude.setLevel(0)
        self.indAmplitude.setState(0)
        self.barFrequency.setLevel(0)
        self.indFrequency.setState(0)

    def enableControls(self, state):
        self.clearIndicators()
        self.pnlAmplitude.setEnabled(bool(state))
        self.pnlFrequency.setEnabled(bool(state))
        self.grpAlarmCondition.setEnabled(bool(state))

    def setCamera(self, camera):
        self.source = MediaSource.CAMERA
        if camera:

            if not self.isModelSettings(camera.audioModelSettings):
                camera.audioModelSettings = SampleSettings(self.mw, camera)

            self.mw.audioPanel.lblCamera.setText(f'Camera - {camera.name()}')
            self.chkAmplitude.setChecked(camera.audioModelSettings.amplitudeEnabled)
            self.sldAmplitudeGain.setValue(camera.audioModelSettings.amplitudeGain)
            self.chkFrequency.setChecked(camera.audioModelSettings.frequencyEnabled)
            self.sldFrequencyGain.setValue(camera.audioModelSettings.frequencyGain)
            self.radUnder.setChecked(not camera.audioModelSettings.alarmLimitOver)
            self.dspFrequency.setPctHighPass(camera.audioModelSettings.frequencyPctHighPass)
            self.dspFrequency.setPctLowPass(camera.audioModelSettings.frequencyPctLowPass)
            self.barAmplitude.setLevel(0)
            self.indAmplitude.setState(0)
            self.barFrequency.setLevel(0)
            self.indFrequency.setState(0)

            if not camera.hasAudio():
                self.chkAmplitude.setChecked(False)
                self.chkFrequency.setChecked(False)

            enable = True
            profile = self.mw.cameraPanel.getProfile(camera.uri())
            if profile:
                if profile.getDisableAudio() or not profile.getAnalyzeAudio():
                    enable = False
            if not camera.hasAudio():
                enable = False
            self.enableControls(enable)

            self.update()

    def setFile(self, file):
        self.source = MediaSource.FILE
        if file:

            if not self.isModelSettings(self.mw.filePanel.audioModelSettings):
                self.mw.filePanel.audioModelSettings = SampleSettings(self.mw)

            self.mw.audioPanel.lblCamera.setText(f'File - {os.path.split(file)[1]}')
            self.chkAmplitude.setChecked(self.mw.filePanel.audioModelSettings.amplitudeEnabled)
            self.sldAmplitudeGain.setValue(self.mw.filePanel.audioModelSettings.amplitudeGain)
            self.chkFrequency.setChecked(self.mw.filePanel.audioModelSettings.frequencyEnabled)
            self.sldFrequencyGain.setValue(self.mw.filePanel.audioModelSettings.frequencyGain)
            self.radUnder.setChecked(not self.mw.filePanel.audioModelSettings.alarmLimitOver)
            self.dspFrequency.setPctHighPass(self.mw.filePanel.audioModelSettings.frequencyPctHighPass)
            self.dspFrequency.setPctLowPass(self.mw.filePanel.audioModelSettings.frequencyPctLowPass)
            self.enableControls(self.mw.audioPanel.chkEnableFile.isChecked())
            self.barAmplitude.setLevel(0)
            self.indAmplitude.setState(0)
            self.barFrequency.setLevel(0)
            self.indFrequency.setState(0)
            self.update()

    def isModelSettings(self, arg):
        return type(arg) == SampleSettings

class AudioWorker:
    def __init__(self, mw):
        try:
            self.mw = mw
            self.last_error = ""
        except:
            logger.exception("sample worker failed to load")

    def __call__(self, F, player):
        try:
            if not F or not player:
                self.mw.audioConfigure.dspAmplitude.setData(None)
                self.mw.audioConfigure.dspFrequency.setData(None)
                self.mw.audioConfigure.clearIndicators()
                return

            samples = np.array(F, copy=True)
            if F.channels() > 1:
                samples = samples[::F.channels()]

            if self.mw.audioConfigure.name != MODULE_NAME:
                return
            
            camera = self.mw.cameraPanel.getCamera(player.uri)
            if not self.mw.audioConfigure.isModelSettings(player.audioModelSettings):
                if player.isCameraStream():
                    if camera:
                        camera.audioModelSettings = SampleSettings(self.mw, camera)
                    player.audioModelSettings = camera.audioModelSettings
                else:
                    if not self.mw.audioConfigure.isModelSettings(self.mw.filePanel.audioModelSettings):
                        self.mw.filePanel.audioModelSettings = SampleSettings(self.mw)
                    player.audioModelSettings = self.mw.filePanel.audioModelSettings

            if not player.audioModelSettings:
                raise Exception("Unable to set audio model parameters for player")

            spectrum = None
            if player.audioModelSettings.frequencyEnabled and samples.size:
                spectrum = fft(samples)

            if player.audioModelSettings.amplitudeEnabled:
                samples *= math.exp(0.2 * (player.audioModelSettings.amplitudeGain - 50))
                rms = 0
                if samples.size:
                    for s in samples:
                        rms += s * s
                    rms = math.sqrt(rms/samples.size)

                alarmState = False
                if player.audioModelSettings.alarmLimitOver:
                    if rms > 1:
                        alarmState = True
                else:
                    if rms < 0.005:
                        alarmState = True

                '''
                focused = False
                if player.isCameraStream():
                    if camera:
                        if camera.isFocus():
                            focused = True
                else:
                    if player.uri == self.mw.glWidget.focused_uri:
                        focused = True
                if focused:
                '''
                if player.uri == self.mw.glWidget.focused_uri:
                    self.mw.audioConfigure.barAmplitude.setLevel(rms)
                    self.mw.audioConfigure.dspAmplitude.setData(samples)
                    if alarmState:
                        self.mw.audioConfigure.indAmplitude.setState(1)

                player.handleAlarm(alarmState)

            if player.audioModelSettings.frequencyEnabled and spectrum is not None:
                half_length = int(len(spectrum)/2)
                frequencies = np.abs(spectrum[:half_length])
                frequencies *= math.exp(0.05 * (player.audioModelSettings.frequencyGain - 50))

                highPass = player.audioModelSettings.frequencyPctHighPass * frequencies.size
                lowPass = player.audioModelSettings.frequencyPctLowPass * frequencies.size
                sph = 0
                for i, f in enumerate(frequencies):
                    if highPass < lowPass:
                        if i > highPass and i < lowPass:
                            sph += f
                    else:
                        if i > highPass or i < lowPass:
                            sph += f
                sph /= (frequencies.size * player.audioModelSettings.frequencyPctCoverage)

                alarmState = False
                if player.audioModelSettings.alarmLimitOver:
                    if sph > 1:
                        alarmState = True
                else:
                    if sph < 0.005:
                        alarmState = True
                '''
                focused = False
                if player.isCameraStream():
                    if camera:
                        if camera.isFocus():
                            focused = True
                else:
                    if player.uri == self.mw.glWidget.focused_uri:
                        focused = True
                if focused:
                '''
                if player.uri == self.mw.glWidget.focused_uri:
                    self.mw.audioConfigure.barFrequency.setLevel(sph)
                    self.mw.audioConfigure.dspFrequency.setData(frequencies)
                    if alarmState:
                        self.mw.audioConfigure.indFrequency.setState(1)

                player.handleAlarm(alarmState)

        except BaseException as err:
            if str(err) != self.last_error:
                logger.exception("pyAudioCallback exception")
            else:
                print("Error during audio analysis:", str(err))
            self.last_error = str(err)

