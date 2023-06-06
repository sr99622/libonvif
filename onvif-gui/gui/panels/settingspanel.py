#/********************************************************************
# onvif-gui/gui/panels/settingspanel.py 
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
from PyQt6.QtWidgets import QMessageBox, QLineEdit, QSpinBox, \
QGridLayout, QWidget, QCheckBox, QLabel, QComboBox
from PyQt6.QtCore import Qt

import libonvif as onvif
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
        self.renderKey = "settings/render"
        self.convertKey = "settings/convert"
        self.workerKey = "settings/worker"
        self.disableAudioKey = "settings/disableAudio"
        self.disableVideoKey = "settings/disableVideo"
        self.postEncodeKey = "settings/postEncode"
        self.hardwareEncodeKey = "settings/hardwareEncode"
        self.processPauseKey = "settings/processPause"
        self.processFrameKey = "settings/processFrame"
        self.cacheSizeKey = "settings/cacheSize"
        self.interfaceKey = "settings/interface"
        self.videoFilterKey = "settings/videoFilter"
        self.audioFilterKey = "settings/audioFilter"
        self.autoReconnectKey = "settings/autoReconnect"

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
        self.cmbDecoder.currentTextChanged.connect(self.cmbDecoderChanged)
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

        if self.chkDisableAudio.isChecked():
            self.chkDisableVideo.setEnabled(False)
        else:
            if self.chkDisableVideo.isChecked():
                self.chkDisableAudio.setEnabled(False)

        self.chkPostEncode = QCheckBox("Post Process Record")
        self.chkPostEncode.setChecked(int(mw.settings.value(self.postEncodeKey, 0)))
        self.chkPostEncode.stateChanged.connect(self.postEncodeChecked)

        self.chkHardwareEncode = QCheckBox("Hardware Encode")
        if sys.platform == "win32":
            self.chkHardwareEncode.setEnabled(False)
        else:
            self.chkHardwareEncode.setChecked(int(mw.settings.value(self.hardwareEncodeKey, 0)))
            self.chkHardwareEncode.setEnabled(self.chkPostEncode.isChecked())
            self.chkHardwareEncode.stateChanged.connect(self.hardwareEncodeChecked)

        self.chkProcessPause = QCheckBox("Process Pause")
        self.chkProcessPause.setChecked(int(mw.settings.value(self.processPauseKey, 0)))
        self.chkProcessPause.stateChanged.connect(self.processPauseChecked)

        self.chkLowLatency = QCheckBox("Low Latency")
        self.chkLowLatency.setChecked(int(mw.settings.value(self.latencyKey, 0)))
        self.chkLowLatency.stateChanged.connect(self.lowLatencyChecked)

        self.chkAutoReconnect = QCheckBox("Auto Reconnect")
        self.chkAutoReconnect.setChecked(int(mw.settings.value(self.autoReconnectKey, 0)))
        self.chkAutoReconnect.stateChanged.connect(self.autoReconnectChecked)

        pnlChecks = QWidget()
        lytChecks = QGridLayout(pnlChecks)
        lytChecks.addWidget(self.chkDirectRender,   0, 0, 1, 1)
        lytChecks.addWidget(self.chkConvert2RGB,    0, 1, 1, 1)
        lytChecks.addWidget(self.chkDisableAudio,   1, 0, 1, 1)
        lytChecks.addWidget(self.chkDisableVideo,   1, 1, 1, 1)
        lytChecks.addWidget(self.chkPostEncode,     2, 0, 1, 1)
        lytChecks.addWidget(self.chkHardwareEncode, 2, 1, 1, 1)
        lytChecks.addWidget(self.chkProcessPause,   3, 0, 1, 1)
        lytChecks.addWidget(self.chkLowLatency,     3, 1, 1, 1)
        lytChecks.addWidget(self.chkAutoReconnect,  4, 0, 1, 1)

        self.spnCacheSize = QSpinBox()
        self.spnCacheSize.setMinimum(1)
        self.spnCacheSize.setMaximum(10)
        self.spnCacheSize.setMaximumWidth(80)
        self.spnCacheSize.setValue(int(self.mw.settings.value(self.cacheSizeKey, 1)))
        self.spnCacheSize.valueChanged.connect(self.spnCacheSizeChanged)
        lblCacheSize = QLabel("Pre-Record Cache Size")

        self.cmbInterfaces = QComboBox()
        intf = self.mw.settings.value(self.interfaceKey, "")
        lblInterfaces = QLabel("Network")
        session = onvif.Session()
        session.getActiveInterfaces()
        i = 0
        while len(session.active_interface(i)) > 0 and i < 16:
            self.cmbInterfaces.addItem(session.active_interface(i))
            i += 1
        if len(intf) > 0:
            self.cmbInterfaces.setCurrentText(intf)
        self.cmbInterfaces.currentTextChanged.connect(self.cmbInterfacesChanged)

        pnlInterface = QWidget()
        lytInterface = QGridLayout(pnlInterface)
        lytInterface.addWidget(lblInterfaces,      0, 0, 1, 1)
        lytInterface.addWidget(self.cmbInterfaces, 0, 1, 1, 1)
        lytInterface.setColumnStretch(1, 10)
        lytInterface.setContentsMargins(0, 0, 0, 0)

        self.txtVideoFilter = QLineEdit()
        self.txtVideoFilter.setText(mw.settings.value(self.videoFilterKey, ""))
        self.txtVideoFilter.textChanged.connect(self.videoFilterChanged)
        lblVideoFilter = QLabel("Video Filter")

        self.txtAudioFilter = QLineEdit()
        self.txtAudioFilter.setText(mw.settings.value(self.audioFilterKey, ""))
        self.txtAudioFilter.textChanged.connect(self.audioFilterChanged)
        lblAudioFilter = QLabel("Audio Filter")

        pnlFilter = QWidget()
        lytFilter = QGridLayout(pnlFilter)
        lytFilter.addWidget(lblVideoFilter,      0, 0, 1, 1)
        lytFilter.addWidget(self.txtVideoFilter, 0, 1, 1, 1)
        lytFilter.addWidget(lblAudioFilter,      1, 0, 1, 1)
        lytFilter.addWidget(self.txtAudioFilter, 1, 1, 1, 1)
        lytFilter.setColumnStretch(1, 10)
        lytFilter.setContentsMargins(0, 0, 0, 0)

        self.lblSpacer = QLabel("")

        lytMain = QGridLayout(self)
        lytMain.addWidget(self.chkAutoDiscover,   0, 0, 1, 2)
        lytMain.addWidget(lblUsername,            1, 0, 1, 1)
        lytMain.addWidget(self.txtUsername,       1, 1, 1, 1)
        lytMain.addWidget(lblPassword,            2, 0, 1, 1)
        lytMain.addWidget(self.txtPassword,       2, 1, 1, 1)
        lytMain.addWidget(lblDecoders,            3, 0, 1, 1)
        lytMain.addWidget(self.cmbDecoder,        3, 1, 1, 1)
        lytMain.addWidget(pnlFilter,              6, 0, 1, 3)
        lytMain.addWidget(pnlChecks,              7, 0, 1, 3)
        lytMain.addWidget(lblCacheSize,           8, 0, 1, 1)
        lytMain.addWidget(self.spnCacheSize,      8, 1, 1, 1)
        lytMain.addWidget(pnlInterface,           9, 0, 1, 3)
        lytMain.addWidget(self.lblSpacer,        11, 0, 1, 3, Qt.AlignmentFlag.AlignCenter)
        lytMain.setRowStretch(11, 10)

    def autoDiscoverChecked(self, state):
        self.mw.settings.setValue(self.autoDiscoverKey, state)

    def usernameChanged(self, username):
        self.mw.settings.setValue(self.usernameKey, username)

    def passwordChanged(self, password):
        self.mw.settings.setValue(self.passwordKey, password)

    def cmbDecoderChanged(self, decoder):
        self.mw.settings.setValue(self.decoderKey, decoder)

    def directRenderChecked(self):
        ret = QMessageBox.warning(self, "onvif-gui",
                                    "Application must restart to enact change.\n"
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
        if state == 0:
            self.chkDisableVideo.setEnabled(True)
        else:
            self.chkDisableVideo.setEnabled(False)

    def disableVideoChecked(self, state):
        self.mw.settings.setValue(self.disableVideoKey, state)
        if state == 0:
            self.chkDisableAudio.setEnabled(True)
        else:
            self.chkDisableAudio.setEnabled(False)

    def postEncodeChecked(self, state):
        self.mw.settings.setValue(self.postEncodeKey, state)
        if sys.platform != "win32":
            if state == 0:
                self.chkHardwareEncode.setChecked(False)
            self.chkHardwareEncode.setEnabled(state)

    def hardwareEncodeChecked(self, state):
        self.mw.settings.setValue(self.hardwareEncodeKey, state)

    def processPauseChecked(self, state):
        self.mw.settings.setValue(self.processPauseKey, state)

    def lowLatencyChecked(self, state):
        self.mw.settings.setValue(self.latencyKey, state)

    def autoReconnectChecked(self, state):
        self.mw.settings.setValue(self.autoReconnectKey, state)

    def radioFilenameChecked(self):
        if self.radGenerateFilename.isChecked():
            self.mw.settings.setValue(self.generateKey, 1)
        else:
            self.mw.settings.setValue(self.generateKey, 0)

    def spnCacheSizeChanged(self, i):
        self.mw.settings.setValue(self.cacheSizeKey, i)

    def cmbInterfacesChanged(self, network):
        self.mw.settings.setValue(self.interfaceKey, network)

    def videoFilterChanged(self, filter):
        self.mw.settings.setValue(self.videoFilterKey, filter)

    def audioFilterChanged(self, filter):
        self.mw.settings.setValue(self.audioFilterKey, filter)

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
     
    def onMediaStarted(self, duration):
        self.lblSpacer.setText("Settings Tab is disabled during media processing")
        self.setEnabled(False)

    def onMediaStopped(self):
        self.lblSpacer.setText("")
        self.setEnabled(True)
