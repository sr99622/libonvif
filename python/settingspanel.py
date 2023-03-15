#/********************************************************************
# libonvif/python/settingspanel.py 
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
from PyQt6.QtWidgets import QMessageBox, QLineEdit, QGroupBox, \
QGridLayout, QWidget, QCheckBox, QLabel, QRadioButton, QComboBox

sys.path.append("../build/libonvif")
import onvif
sys.path.append("../build/libavio")
import avio

class SettingsPanel(QWidget):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw
        self.autoDiscoverKey = "settings/autoDiscover"
        self.usernameKey = "settings/username"
        self.passwordKey = "settings/password"
        self.decoderKey = "settings/decoder"
        self.latencyKey = "settings/latency"
        self.generateKey = "settings/generate"
        self.renderKey = "settings/render"
        self.convertKey = "settings/convert"
        self.disableAudioKey = "settings/disableAudio"
        self.disableVideoKey = "settings/disableVideo"
        self.postEncodeKey = "settings/postEncode"
        self.hardwareEncodeKey = "settings/hardwareEncode"
        self.processFrameKey = "settings/processFrame"

        decoders = ["NONE", "CUDA", "VAAPI", "VDPAU", "DXVA2", "D3D11VA"]

        self.chkAutoDiscover = QCheckBox("Enable Auto Discovery")
        self.chkAutoDiscover.setChecked(int(mw.settings.value(self.autoDiscoverKey, 0)))
        self.chkAutoDiscover.stateChanged.connect(self.autoDiscoverChecked)
        
        self.txtUsername = QLineEdit()
        self.txtUsername.setText(mw.settings.value(self.usernameKey, ""))
        self.txtUsername.textChanged.connect(self.usernameChanged)
        lblUsername = QLabel("Common Username")
        
        self.txtPassword = QLineEdit()
        self.txtPassword.setText(mw.settings.value(self.passwordKey, ""))
        self.txtPassword.textChanged.connect(self.passwordChanged)
        lblPassword = QLabel("Common Password")
        
        self.cmbDecoder = QComboBox()
        self.cmbDecoder.addItems(decoders)
        self.cmbDecoder.setCurrentText(mw.settings.value(self.decoderKey, "NONE"))
        self.cmbDecoder.currentTextChanged.connect(self.cmbDecodersChanged)
        lblDecoders = QLabel("Hardware Decoder")
        
        self.chkDirectRender = QCheckBox("Direct Rendering")
        self.render = int(mw.settings.value(self.renderKey, 0))
        self.chkDirectRender.setChecked(self.render)
        self.chkDirectRender.clicked.connect(self.directRenderChecked)

        if sys.platform == "win32":
            self.chkDirectRender.setEnabled(True)
        else:
            self.chkDirectRender.setEnabled(False)

        self.chkConvert2RGB = QCheckBox("Convert to RGB")
        self.chkConvert2RGB.setChecked(int(mw.settings.value(self.convertKey, 1)))
        self.chkConvert2RGB.stateChanged.connect(self.convert2RGBChecked)
        self.chkConvert2RGB.setEnabled(self.chkDirectRender.isChecked())

        self.chkDisableAudio = QCheckBox("Disable Audio")
        self.chkDisableAudio.setChecked(int(mw.settings.value(self.disableAudioKey, 0)))
        self.chkDisableAudio.stateChanged.connect(self.disableAudioChecked)

        self.chkDisableVideo = QCheckBox("Disable Video")
        self.chkDisableVideo.setChecked(int(mw.settings.value(self.disableVideoKey, 0)))
        self.chkDisableVideo.stateChanged.connect(self.disableVideoChecked)

        self.chkPostEncode = QCheckBox("Post Process Record")
        self.chkPostEncode.setChecked(int(mw.settings.value(self.postEncodeKey, 0)))
        self.chkPostEncode.stateChanged.connect(self.postEncodeChecked)

        self.chkHardwareEncode = QCheckBox("Hardware Encode")
        self.chkHardwareEncode.setChecked(int(mw.settings.value(self.hardwareEncodeKey, 0)))
        self.chkHardwareEncode.setEnabled(self.chkPostEncode.isChecked())
        self.chkHardwareEncode.stateChanged.connect(self.hardwareEncodeChecked)

        self.chkProcessFrame = QCheckBox("Process Frame")
        self.chkProcessFrame.setChecked(int(mw.settings.value(self.processFrameKey, 0)))
        self.chkProcessFrame.stateChanged.connect(self.processFrameChecked)

        pnlChecks = QWidget()
        lytChecks = QGridLayout(pnlChecks)
        lytChecks.addWidget(self.chkDirectRender,   0, 0, 1, 1)
        lytChecks.addWidget(self.chkConvert2RGB,    0, 1, 1, 1)
        lytChecks.addWidget(self.chkDisableAudio,   1, 0, 1, 1)
        lytChecks.addWidget(self.chkDisableVideo,   1, 1, 1, 1)
        lytChecks.addWidget(self.chkPostEncode,     2, 0, 1, 1)
        lytChecks.addWidget(self.chkHardwareEncode, 2, 1, 1, 1)
        lytChecks.addWidget(self.chkProcessFrame,   3, 0, 1, 1)

        self.txtVideoFilter = QLineEdit()
        lblVideoFilter = QLabel("Video Filter")

        self.radGenerateFilename = QRadioButton("Generate Unique")
        self.radGenerateFilename.clicked.connect(self.radioFilenameChecked)
        self.radDefaultFilename = QRadioButton("Use Default")
        self.radDefaultFilename.clicked.connect(self.radioFilenameChecked)
        self.grpRecordFilename = QGroupBox("Record Filename")
        lytRecordFilename = QGridLayout(self.grpRecordFilename)
        lytRecordFilename.addWidget(self.radGenerateFilename, 0, 0, 1, 1)
        lytRecordFilename.addWidget(self.radDefaultFilename,  0, 1, 1, 1)
        if self.mw.settings.value(self.generateKey, 1) == 1:
            self.radGenerateFilename.setChecked(True)
        else:
            self.radDefaultFilename.setChecked(True)

        pnlFilter = QWidget()
        lytFilter = QGridLayout(pnlFilter)
        lytFilter.addWidget(lblVideoFilter,      0, 0, 1, 1)
        lytFilter.addWidget(self.txtVideoFilter, 0, 1, 1, 1)

        lytMain = QGridLayout(self)
        lytMain.addWidget(self.chkAutoDiscover,   0, 0, 1, 2)
        lytMain.addWidget(lblUsername,            1, 0, 1, 1)
        lytMain.addWidget(self.txtUsername,       1, 1, 1, 1)
        lytMain.addWidget(lblPassword,            2, 0, 1, 1)
        lytMain.addWidget(self.txtPassword,       2, 1, 1, 1)
        lytMain.addWidget(lblDecoders,            3, 0, 1, 1)
        lytMain.addWidget(self.cmbDecoder,        3, 1, 1, 1)
        lytMain.addWidget(pnlFilter,              5, 0, 1, 4)
        lytMain.addWidget(pnlChecks,              6, 0, 1, 4)
        lytMain.addWidget(self.grpRecordFilename, 8, 0, 1, 4)
        lytMain.addWidget(QLabel(),               9, 0, 1, 4)
        lytMain.setRowStretch(9, 10)

    def autoDiscoverChecked(self, state):
        self.mw.settings.setValue(self.autoDiscoverKey, state)

    def usernameChanged(self, username):
        self.mw.settings.setValue(self.usernameKey, username)

    def passwordChanged(self, password):
        self.mw.settings.setValue(self.passwordKey, password)

    def cmbDecodersChanged(self, decoder):
        self.mw.settings.setValue(self.decoderKey, decoder)

    def directRenderChecked(self):
        ret = QMessageBox.warning(self, "onvif-gui",
                                    "Application must  to enact change.\n"
                                    "Are you sure you want to continue?",
                                    QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
        if ret == QMessageBox.StandardButton.Ok:
            if self.render == 1:
                self.mw.settings.setValue(self.renderKey, 0)
            else:
                self.mw.settings.setValue(self.renderKey, 1)
            self.chkConvert2RGB.setEnabled(self.chkDirectRender.isChecked())
            if not self.chkConvert2RGB.isEnabled():
                self.chkConvert2RGB.setChecked(True)
            self.mw.stopMedia()
            os.execv(sys.executable, ['python'] + sys.argv)
        else:
            self.chkDirectRender.setChecked(not self.chkDirectRender.isChecked())
            self.chkConvert2RGB.setEnabled(self.chkDirectRender.isChecked())
            if not self.chkConvert2RGB.isEnabled():
                self.chkConvert2RGB.setChecked(True)

    def convert2RGBChecked(self, state):
        self.mw.settings.setValue(self.convertKey, state)

    def disableAudioChecked(self, state):
        self.mw.settings.setValue(self.disableAudioKey, state)

    def disableVideoChecked(self, state):
        self.mw.settings.setValue(self.disableVideoKey, state)

    def postEncodeChecked(self, state):
        self.mw.settings.setValue(self.postEncodeKey, state)
        if state == 0:
            self.chkHardwareEncode.setChecked(False)
        self.chkHardwareEncode.setEnabled(state)

    def hardwareEncodeChecked(self, state):
        self.mw.settings.setValue(self.hardwareEncodeKey, state)

    def processFrameChecked(self, state):
        self.mw.settings.setValue(self.processFrameKey, state)

    def radioFilenameChecked(self):
        if self.radGenerateFilename.isChecked():
            self.mw.settings.setValue(self.generateKey, 1)
        else:
            self.mw.settings.setValue(self.generateKey, 0)

    def getDecoder(self):
        result = avio.AV_HWDEVICE_TYPE_NONE
        if self.cmbDecoder.currentText() == "CUDA":
            result = avio.AV_HWDEVICE_TYPE_CUDA
        if self.cmbDecoder.currentText() == "VAAPI":
            result = avio.AV_HWDEVICE_TYPE_VAAPI
        if self.cmbDecoder.currentText() == "VDPAU":
            result = avio.AV_HWDEVICE_TYPE_VDPAU
        if self.cmbDecoder.currentText() == "DXVA2":
            result = avio.AV_HWDEVICE_TYPE_DXVA2
        if self.cmbDecoder.currentText() == "D3D11VA":
            result = avio.AV_HWDEVICE_TYPE_D3D11VA
        return result
     
