#/********************************************************************
# onvif-gui/modules/audio/sample.py 
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
from loguru import logger
from PyQt6.QtWidgets import QGridLayout, QWidget, QCheckBox, QLabel

MODULE_NAME = "sample"

class AudioConfigure(QWidget):
    def __init__(self, mw):
        try:
            super().__init__()
            self.mw = mw
            self.name = MODULE_NAME
            self.muteLeftKey = "AudioModule/" + MODULE_NAME + "/muteLeft"
            self.muteRightKey = "AudioModule/" + MODULE_NAME + "/muteRight"
            self.chkMuteLeft = QCheckBox("Mute Left Channel")
            self.chkMuteLeft.setChecked(int(self.mw.settings.value(self.muteLeftKey, 0)))
            self.chkMuteLeft.stateChanged.connect(self.chkMuteLeftClicked)
            self.chkMuteRight = QCheckBox("Mute Right Channel")
            self.chkMuteRight.setChecked(int(self.mw.settings.value(self.muteRightKey, 0)))
            self.chkMuteRight.stateChanged.connect(self.chkMuteRightClicked)
            self.lblStatus = QLabel()
            lytMain = QGridLayout(self)
            lytMain.addWidget(self.chkMuteLeft,  0, 0, 1, 1)
            lytMain.addWidget(self.chkMuteRight, 1, 0, 1, 1)
            lytMain.addWidget(self.lblStatus,    2, 0, 1, 1)
        except:
            logger.exception("sample configuration failed to load")

    def chkMuteLeftClicked(self, state):
        self.mw.settings.setValue(self.muteLeftKey, state)

    def chkMuteRightClicked(self, state):
        self.mw.settings.setValue(self.muteRightKey, state)

class AudioWorker:
    def __init__(self, mw):
        try:
            self.mw = mw
        except:
            logger.exception("sample worker failed to load")

    def __call__(self, F):
        try:
            if F.isValid():
                sample = np.array(F, copy=False)

                #print(F.m_rts)
                #print(sample.shape)
                #print(F.nb_samples())
                #print(F.sample_rate())
                #print(F.channels())
                #print(np.sum(sample))

                if F.channels() == 2:
                    self.mw.audioConfigure.lblStatus.setText("Processing 2 channel audio")
                    
                    left = sample[::2]
                    if self.mw.audioConfigure.chkMuteLeft.isChecked():
                        left *= 0
                    
                    right = sample[1::2]
                    if self.mw.audioConfigure.chkMuteRight.isChecked():
                        right *= 0

                else:
                    self.mw.audioConfigure.lblStatus.setText("Error: only 2 channel streams are supported")

        except:
            logger.exception("pyAudioCallback exception")
