#/********************************************************************
# libonvif/onvif-gui/gui/panels/videopanel.py 
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
from PyQt6.QtWidgets import QGridLayout, QWidget, QCheckBox, \
    QLabel, QComboBox, QVBoxLayout
from PyQt6.QtCore import Qt
from gui.enums import MediaSource

class VideoPanel(QWidget):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw
        self.panel = None
        self.layout = QVBoxLayout(self)
        self.workerKey = "VideoPanel/worker"
        self.enableFileKey = "VideoPanel/enableFile"
        self.stdLocation = mw.getLocation() + "/modules/video"

        self.lblCamera = QLabel("Please select a camera to enable this panel")

        self.cmbWorker = QComboBox()
        self.fillModules()
        self.cmbWorker.setCurrentText(mw.settings.value(self.workerKey, "motion.py"))
        self.cmbWorker.currentTextChanged.connect(self.cmbWorkerChanged)
        lblWorkers = QLabel("Python Worker")

        self.chkEnableFile = QCheckBox("Enable File")
        self.chkEnableFile.setChecked(self.mw.filePanel.getAnalyzeVideo())
        self.chkEnableFile.stateChanged.connect(self.chkEnableFileChanged)

        fixedPanel = QWidget()
        lytFixed = QGridLayout(fixedPanel)
        lytFixed.addWidget(lblWorkers,         2, 0, 1, 1)
        lytFixed.addWidget(self.cmbWorker,     2, 1, 1, 1)
        lytFixed.addWidget(self.chkEnableFile, 3, 0, 1, 1)
        lytFixed.addWidget(self.lblCamera,     4, 0, 1, 2, Qt.AlignmentFlag.AlignCenter)
        lytFixed.setColumnStretch(1, 10)
        self.layout.addWidget(fixedPanel)

    def showEvent(self, event):
        self.lblCamera.setFocus()
        super().showEvent(event)

    def fillModules(self):
        d = self.stdLocation
        workers = [f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]
        for worker in workers:
            if not worker.endswith(".py") or worker == "__init__.py":
                workers.remove(worker)
        workers.sort()
        self.cmbWorker.clear()
        self.cmbWorker.addItems(workers)

    def setPanel(self, panel):
        if self.panel is not None:
            self.layout.removeWidget(self.panel)
        self.panel = panel
        self.panel.setMaximumWidth(self.mw.tab.width())
        self.layout.addWidget(panel)
        self.layout.setStretch(1, 10)

    def cmbWorkerChanged(self, worker):
        source = None
        media = None
        if self.panel:
            source = self.panel.source
            media = self.panel.media

        self.mw.loadVideoConfigure(worker)

        if source:
            self.mw.videoConfigure.source = source
            match source:
                case MediaSource.CAMERA:
                    self.mw.videoConfigure.setCamera(media)
                case MediaSource.FILE:
                    self.mw.videoConfigure.setFile(media)

        self.mw.videoWorkerHook = None
        self.mw.videoWorker = None

        player = self.mw.pm.getCurrentPlayer()
        if player:
            if player.isCameraStream():
                camera = self.mw.cameraPanel.getCamera(player.uri)
                if camera:
                    self.mw.videoConfigure.setCamera(camera)
            else:
                self.mw.videoConfigure.setFile(player.uri)

        self.mw.settings.setValue(self.workerKey, worker)
        
    def chkEnableFileChanged(self, state):
        self.mw.filePanel.setAnalyzeVideo(state)
        for player in self.mw.pm.players:
            if not player.isCameraStream():
                player.analyze_video = bool(state)
        if self.mw.videoConfigure.source == MediaSource.FILE:
            self.mw.videoConfigure.enableControls(state)
