#/********************************************************************
# libonvif/python/modues/sample.py 
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
import cv2

class Sample:
    def __init__(self, mw):
        self.mw = mw

    def __call__(self, F):
        img = np.array(F, copy = False)
        milliseconds = F.m_rts
        seconds, milliseconds = divmod(milliseconds, 1000)
        minutes, seconds = divmod(seconds, 60)
        timestamp = f'{int(minutes):02d}:{int(seconds):02d}.{int(milliseconds/100):01d}'

        imgWidth = img.shape[1]
        imgHeight = img.shape[0]

        cv2.rectangle(img, (0, 0), (imgWidth, imgHeight), (0, 255, 0), 20)
        textSize, _ = cv2.getTextSize(timestamp, cv2.FONT_HERSHEY_PLAIN, 12, 12)
        textWidth, textHeight = textSize

        textX = max((imgWidth / 2) - (textWidth / 2), 0)
        textY = max((imgHeight / 2) + (textHeight / 2), 0)

        color = (255, 255, 255)
        if self.mw.player.isRecording():
            color = (255, 0, 0)

        cv2.putText(img, timestamp, (int(textX), int(textY)), cv2.FONT_HERSHEY_PLAIN, 12, color, 12)

