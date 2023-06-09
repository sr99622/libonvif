#/********************************************************************
# onvif-gui/modules/video/sample.py 
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

import os
import sys
import numpy as np
import cv2
from PyQt6.QtWidgets import QGridLayout, QWidget, QCheckBox
from loguru import logger
if sys.platform == "win32":
    filename = os.environ['HOMEPATH'] + "/.cache/onvif-gui/errors.txt"
else:
    filename = os.environ['HOME'] + "/.cache/onvif-gui/errors.txt"
logger.add(filename, retention="10 days")

MODULE_NAME = "sample"

class VideoConfigure(QWidget):
    def __init__(self, mw):
        try:
            super().__init__()
            self.mw = mw
            self.name = MODULE_NAME
            self.showBorderKey = "Module/" + MODULE_NAME + "/showBorder"
            self.chkShowBorder = QCheckBox("Show Border")
            self.chkShowBorder.setChecked(int(self.mw.settings.value(self.showBorderKey, 0)))
            self.chkShowBorder.stateChanged.connect(self.chkShowBorderClicked)
            lytMain = QGridLayout(self)
            lytMain.addWidget(self.chkShowBorder, 0, 0, 1, 1)
        except:
            logger.exception("sample configuration failed to load")

    def chkShowBorderClicked(self, state):
        self.mw.settings.setValue(self.showBorderKey, state)

class VideoWorker:
    def __init__(self, mw):
        try:
            self.mw = mw
            self.last_ex = ""

        except:
            logger.exception("sample worker failed to load")

    def __call__(self, F):
        try:
            img = np.array(F, copy = False)
            milliseconds = F.m_rts
            seconds, milliseconds = divmod(milliseconds, 1000)
            minutes, seconds = divmod(seconds, 60)
            timestamp = f'{int(minutes):02d}:{int(seconds):02d}.{int(milliseconds/100):01d}'

            imgWidth = img.shape[1]
            imgHeight = img.shape[0]

            if self.mw.configure.chkShowBorder.isChecked():
                cv2.rectangle(img, (0, 0), (imgWidth, imgHeight), (0, 255, 0), 20)

            textSize, _ = cv2.getTextSize(timestamp, cv2.FONT_HERSHEY_PLAIN, 12, 12)
            textWidth, textHeight = textSize
            textX = max((imgWidth / 2) - (textWidth / 2), 0)
            textY = max((imgHeight / 2) + (textHeight / 2), 0)

            color = (255, 255, 255)
            if self.mw.player.isRecording():
                color = (255, 0, 0)

            cv2.putText(img, timestamp, (int(textX), int(textY)), cv2.FONT_HERSHEY_PLAIN, 12, color, 12)

        except Exception as ex:
            if self.last_ex != str(ex) and self.mw.configure.name == MODULE_NAME:
                logger.exception("sample worker call error")
            self.last_ex = str(ex)