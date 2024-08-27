#/********************************************************************
# libonvif/onvif-gui/modules/video/motion.py 
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

import math
import numpy as np
import cv2
import os
from PyQt6.QtWidgets import QGridLayout, QWidget, QSlider, QCheckBox, QGroupBox, QLabel
from PyQt6.QtCore import Qt
from loguru import logger
from gui.components import WarningBar, Indicator
from gui.enums import MediaSource

MODULE_NAME = "motion"

class MotionSettings():
    def __init__(self, mw, camera=None):
        self.camera = camera
        self.mw = mw
        self.id = "File"
        if camera:
            self.id = camera.serial_number()
        self.show = False
        self.gain = self.getModelOutputGain()

    def getModelOutputGain(self):
        key = f'{self.id}/{MODULE_NAME}/MotionGain'
        return int(self.mw.settings.value(key, 50))
    
    def setModelOutputGain(self, value):
        key = f'{self.id}/{MODULE_NAME}/MotionGain'
        self.gain = value
        self.mw.settings.setValue(key, value)

class VideoConfigure(QWidget):
    def __init__(self, mw):
        try:
            super().__init__()
            self.mw = mw
            self.name = MODULE_NAME
            self.source = None
            self.media = None
            self.initialized = False
            
            self.chkShow = QCheckBox("Show Diff Image")
            self.barLevel = WarningBar()
            self.indAlarm = Indicator(self.mw)
            self.sldGain = QSlider(Qt.Orientation.Vertical)
            self.sldGain.setValue(50)
            self.sldGain.valueChanged.connect(self.sldGainValueChanged)
            grpSlide = QGroupBox()
            grpSlide.setMaximumHeight(260)
        
            lytSlide = QGridLayout(grpSlide)
            lytSlide.addWidget(self.sldGain,   1, 0, 1, 1, Qt.AlignmentFlag.AlignHCenter)
            lytSlide.addWidget(QLabel("Gain"), 2, 0, 1, 1, Qt.AlignmentFlag.AlignCenter)
            lytSlide.addWidget(self.indAlarm,  0, 1, 1, 1)
            lytSlide.addWidget(self.barLevel,  1, 1, 2, 1)
            lytSlide.addWidget(QLabel(),       0, 2, 1, 1)
            lytSlide.setRowStretch(1, 10)
            
            lytMain = QGridLayout(self)
            lytMain.addWidget(self.chkShow,  0, 0, 1, 1)
            lytMain.addWidget(grpSlide,      0, 1, 1, 1)
            lytMain.addWidget(QLabel(),      1, 0, 1, 2)

            self.enableControls(False)
            if camera := self.mw.cameraPanel.getCurrentCamera():
                self.setCamera(camera)
            else:
                if file := self.mw.filePanel.getCurrentFileURI():
                    self.setFile(file)
            self.initialized = True

        except:
            logger.exception("sample configuration failed to load")

    def sldGainValueChanged(self, value):
        match self.source:
            case MediaSource.CAMERA:
                camera = self.mw.cameraPanel.getCurrentCamera()
                if camera:
                    camera.videoModelSettings.setModelOutputGain(value)
            case MediaSource.FILE:
                self.mw.filePanel.videoModelSettings.setModelOutputGain(value)

    def setCamera(self, camera):
        self.source = MediaSource.CAMERA
        self.media = camera
        if camera:
            if not self.isModelSettings(camera.videoModelSettings):
                camera.videoModelSettings = MotionSettings(self.mw, camera)
            self.mw.videoPanel.lblCamera.setText(f'Camera - {camera.name()}')
            self.sldGain.setValue(camera.videoModelSettings.gain)
            self.barLevel.setLevel(0)
            self.indAlarm.setState(0)
            profile = self.mw.cameraPanel.getProfile(camera.uri())
            if profile:
                self.enableControls(profile.getAnalyzeVideo())

    def setFile(self, file):
        self.source = MediaSource.FILE
        self.media = file
        if file:
            if not self.isModelSettings(self.mw.filePanel.videoModelSettings):
                self.mw.filePanel.videoModelSettings = MotionSettings(self.mw)
            self.mw.videoPanel.lblCamera.setText(f'File - {os.path.split(file)[1]}')
            self.sldGain.setValue(self.mw.filePanel.videoModelSettings.gain)
            self.barLevel.setLevel(0)
            self.indAlarm.setState(0)
            self.enableControls(self.mw.videoPanel.chkEnableFile.isChecked())

    def isModelSettings(self, arg):
        return type(arg) == MotionSettings
    
    def enableControls(self, state):
        self.setEnabled(bool(state))

class VideoWorker:
    def __init__(self, mw):
        try:
            self.mw = mw
            self.last_ex = ""
            self.last_img = None
            self.first_pass = True
            self.kernel = np.array((9, 9), dtype=np.uint8)
            self.last_alarm_state = False

        except:
            logger.exception("sample worker failed to load")

    def __call__(self, F, player):
        try:

            if not F or not player or self.mw.videoConfigure.name != MODULE_NAME:
                self.mw.videoConfigure.barLevel.setLevel(0)
                self.mw.videoConfigure.indAlarm.setState(0)
                return

            img = np.array(F, copy = False)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            if not self.mw.videoConfigure.isModelSettings(player.videoModelSettings):
                if player.isCameraStream():
                    camera = self.mw.cameraPanel.getCamera(player.uri)
                    if camera:
                        if not self.mw.videoConfigure.isModelSettings(camera.videoModelSettings):
                            camera.videoModelSettings = MotionSettings(self.mw, camera)
                        player.videoModelSettings = camera.videoModelSettings
                else:
                    if not self.mw.videoConfigure.isModelSettings(self.mw.filePanel.videoModelSettings):
                        self.mw.filePanel.videoModelSettings = MotionSettings(self.mw)
                    player.videoModelSettings = self.mw.filePanel.videoModelSettings

            if not player.videoModelSettings:
                raise Exception("Unable to set video model parameters for player")

            level = 0
            diff = img
            if player.last_image is not None:
                diff = cv2.subtract(img, player.last_image)
                diff = cv2.medianBlur(diff, 3)
                diff = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3)
                diff = cv2.medianBlur(diff, 3)
                diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, self.kernel, iterations=1)

                motion = diff.sum() / (diff.shape[0] * diff.shape[1])
                level = math.exp(0.2 * (player.videoModelSettings.gain - 50)) * motion

            player.last_image = img

            alarmState = level > 1.0

            if player.uri == self.mw.glWidget.focused_uri:
                self.mw.videoConfigure.barLevel.setLevel(level)
                if alarmState:
                    self.mw.videoConfigure.indAlarm.setState(1)

                if self.mw.videoConfigure.chkShow.isChecked():
                    return diff

            player.handleAlarm(alarmState)

        except Exception as ex:
            if self.last_ex != str(ex) and self.mw.videoConfigure.name == MODULE_NAME:
                logger.exception("sample worker call error")
            self.last_ex = str(ex)