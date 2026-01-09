#/********************************************************************
# onvif-gui/panels/video/modules/motion.py 
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
from PyQt6.QtWidgets import QGridLayout, QWidget, QSlider, QGroupBox, QLabel
from PyQt6.QtCore import Qt
from loguru import logger
from onvif_gui.components import WarningBar, Indicator
from onvif_gui.panels.video.modules.settings import VideoModelSettings

MODULE_NAME = "motion"

class VideoConfigure(QWidget):
    def __init__(self, mw):
        try:
            super().__init__()
            self.mw = mw
            self.name = MODULE_NAME
            self.source = None
            self.camera = None
            self.initialized = False
            
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
            lytMain.addWidget(grpSlide,      0, 0, 1, 1)
            lytMain.addWidget(QLabel(),      1, 0, 1, 1)

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
        if camera := self.mw.cameraPanel.getCurrentCamera():
            camera.videoModelSettings.setModelOutputGain(value)

    def setCamera(self, camera):
        if not camera:
            return
        
        self.camera = camera
        if camera.videoModelSettings.module_name != MODULE_NAME:
            camera.videoModelSettings = VideoModelSettings(self.mw, camera, "motion")
        self.mw.videoPanel.lblCamera.setText(f'Camera - {camera.name()}')
        self.sldGain.setValue(camera.videoModelSettings.gain)
        self.barLevel.setLevel(0)
        self.indAlarm.setState(0)
        profile = self.mw.cameraPanel.getProfile(camera.uri())
        if profile:
            self.enableControls(profile.getAnalyzeVideo())

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

    def __call__(self, F, uri):
        try:
            if not self.mw.videoConfigure:
                return

            player = self.mw.pm.getPlayer(uri)
            if not player:
                return
            
            camera = self.mw.cameraPanel.getCamera(uri)
            if not camera:
                return

            if not len(F) or not player or self.mw.videoConfigure.name != MODULE_NAME:
                self.mw.videoConfigure.barLevel.setLevel(0)
                self.mw.videoConfigure.indAlarm.setState(0)
                return
            
            if camera.videoModelSettings.module_name != MODULE_NAME:
                camera.videoModelSettings = VideoModelSettings(self.mw, camera, MODULE_NAME)

            img = np.array(F, copy = False)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            level = 0
            diff = img
            if player.last_image is not None:
                diff = cv2.subtract(img, player.last_image)
                diff = cv2.medianBlur(diff, 3)
                diff = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3)
                diff = cv2.medianBlur(diff, 3)
                diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, self.kernel, iterations=1)

                motion = diff.sum() / (diff.shape[0] * diff.shape[1])
                level = math.exp(0.2 * (camera.videoModelSettings.gain - 50)) * motion

            player.last_image = img

            alarmState = level > 1.0

            if player.uri == self.mw.glWidget.focused_uri:
                self.mw.videoConfigure.barLevel.setLevel(level)
                if alarmState:
                    self.mw.videoConfigure.indAlarm.setState(1)

            player.handleAlarm(alarmState)

        except Exception as ex:
            if self.last_ex != str(ex) and self.mw.videoConfigure.name == MODULE_NAME:
                logger.exception(f'Video worker motion detector call error {ex}')
            self.last_ex = str(ex)