#/********************************************************************
# libonvif/onvif-gui/onvif_gui/panels/settings/alarm.py 
#
# Copyright (c) 2024  Stephen Rhodes
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
from PyQt6.QtWidgets import QSpinBox, QGridLayout, QWidget, \
    QLabel, QComboBox, QSlider, QCheckBox, QLabel
from PyQt6.QtCore import Qt

class AlarmOptions(QWidget):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw

        self.bufferSizeKey = "settings/bufferSize"
        self.lagTimeKey = "settings/lagTime"
        self.alarmSoundFileKey = "settings/alarmSoundFile"
        self.alarmSoundVolumeKey = "settings/alarmSoundVolume"
        self.showDisplayAlarmKey = "settings/showDisplayAlarm"
        self.savePictureKey = "settings/savePictureOnAlarm"

        self.spnBufferSize = QSpinBox()
        self.spnBufferSize.setMinimum(1)
        self.spnBufferSize.setMaximum(60)
        self.spnBufferSize.setMaximumWidth(80)
        self.spnBufferSize.setValue(int(self.mw.settings.value(self.bufferSizeKey, 10)))
        self.spnBufferSize.valueChanged.connect(self.spnBufferSizeChanged)
        lblBufferSize = QLabel("Pre-Alarm Buffer Size (in seconds)")

        self.spnLagTime = QSpinBox()
        self.spnLagTime.setMinimum(1)
        self.spnLagTime.setMaximum(60)
        self.spnLagTime.setMaximumWidth(80)
        self.spnLagTime.setValue(int(self.mw.settings.value(self.lagTimeKey, 5)))
        self.spnLagTime.valueChanged.connect(self.spnLagTimeChanged)
        lblLagTime = QLabel("Post-Alarm Lag Time (in seconds)")

        self.cmbSoundFiles = QComboBox()
        d = f'{self.mw.getLocation()}/onvif_gui/resources'
        sounds = [f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f)) and f.endswith(".mp3")]
        self.cmbSoundFiles.addItems(sounds)
        self.cmbSoundFiles.currentTextChanged.connect(self.cmbSoundFilesChanged)
        self.cmbSoundFiles.setCurrentText(self.mw.settings.value(self.alarmSoundFileKey, "drops.mp3"))
        lblSoundFiles = QLabel("Alarm Sounds")
        self.sldAlarmVolume = QSlider(Qt.Orientation.Horizontal)
        self.sldAlarmVolume.setValue(int(self.mw.settings.value(self.alarmSoundVolumeKey, 80)))
        self.sldAlarmVolume.valueChanged.connect(self.sldAlarmVolumeChanged)

        self.chkShowDisplay = QCheckBox("Show Alarms on Display")
        self.chkShowDisplay.setChecked(int(self.mw.settings.value(self.showDisplayAlarmKey, 1)))
        self.chkShowDisplay.stateChanged.connect(self.chkShowDisplayClicked)

        self.chkSavePicture = QCheckBox("Save Picture for Alarms")
        self.chkSavePicture.setChecked(int(self.mw.settings.value(self.savePictureKey, 1)))
        self.chkSavePicture.stateChanged.connect(self.chkSavePictureClicked)

        pnlSoundFile = QWidget()
        lytSoundFile =  QGridLayout(pnlSoundFile)
        lytSoundFile.addWidget(lblSoundFiles,        0, 0, 1, 1)
        lytSoundFile.addWidget(self.cmbSoundFiles,   0, 1, 1, 1)
        lytSoundFile.addWidget(self.sldAlarmVolume,  0, 2, 1, 1)
        lytSoundFile.addWidget(QLabel(),             1, 0, 1, 3)
        lytSoundFile.addWidget(self.chkShowDisplay,  2, 0, 1, 2)
        lytSoundFile.addWidget(self.chkSavePicture,  3, 0, 1, 2)
        lytSoundFile.setColumnStretch(1, 10)

        pnlBuffer = QWidget()
        lytBuffer = QGridLayout(pnlBuffer)
        lytBuffer.addWidget(lblBufferSize,          1, 0, 1, 3)
        lytBuffer.addWidget(self.spnBufferSize,     1, 3, 1, 1)
        lytBuffer.addWidget(lblLagTime,             2, 0, 1, 3)
        lytBuffer.addWidget(self.spnLagTime,        2, 3, 1, 1)
        lytBuffer.addWidget(pnlSoundFile,           3, 0, 1, 4)
        lytBuffer.setContentsMargins(0, 0, 0, 0)

        lytMain = QGridLayout(self)
        lytMain.addWidget(pnlBuffer,  0, 0, 1, 1)
        lytMain.addWidget(QLabel(),   1, 0, 1, 1)
        lytMain.setRowStretch(1, 10)

    def spnBufferSizeChanged(self, i):
        self.mw.settings.setValue(self.bufferSizeKey, i)

    def spnLagTimeChanged(self, i):
        self.mw.settings.setValue(self.lagTimeKey, i)

    def cmbSoundFilesChanged(self, value):
        self.mw.settings.setValue(self.alarmSoundFileKey, value)

    def sldAlarmVolumeChanged(self, value):
        self.mw.settings.setValue(self.alarmSoundVolumeKey, value)

    def chkShowDisplayClicked(self, state):
        self.mw.settings.setValue(self.showDisplayAlarmKey, state)

    def chkSavePictureClicked(self, state):
        self.mw.settings.setValue(self.savePictureKey, state)