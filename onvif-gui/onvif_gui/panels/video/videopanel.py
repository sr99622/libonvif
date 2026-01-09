#/********************************************************************
# onvif-gui/onvif_gui/panels/video/videopanel.py 
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
from PyQt6.QtWidgets import QGridLayout, QWidget, \
    QLabel, QComboBox, QVBoxLayout
from PyQt6.QtCore import Qt

class VideoPanel(QWidget):
    def __init__(self, mw):
        super().__init__()

        self.mw = mw
        self.panel = None
        self.layout = QVBoxLayout(self)
        self.workerKey = "VideoPanel/worker"
        self.stdLocation = mw.getLocation() + "/onvif_gui/panels/video/modules"

        self.lblCamera = QLabel("Please select a camera to enable this panel")

        self.cmbWorker = QComboBox()
        self.fillModules()
        self.cmbWorker.setCurrentText(mw.settings.value(self.workerKey, "motion.py"))
        self.cmbWorker.currentTextChanged.connect(self.cmbWorkerChanged)
        lblWorkers = QLabel("Video Analyzer")

        fixedPanel = QWidget()
        lytFixed = QGridLayout(fixedPanel)
        lytFixed.addWidget(lblWorkers,         2, 0, 1, 1)
        lytFixed.addWidget(self.cmbWorker,     2, 1, 1, 1)
        lytFixed.addWidget(QLabel(),           3, 0, 1, 1)
        lytFixed.addWidget(self.lblCamera,     4, 0, 1, 2, Qt.AlignmentFlag.AlignCenter)
        lytFixed.setColumnStretch(1, 10)
        self.layout.addWidget(fixedPanel)

    def showEvent(self, event):
        self.lblCamera.setFocus()
        super().showEvent(event)

    def fillModules(self):
        d = self.stdLocation
        file_list = [f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]
        workers = []
        for file in file_list:
            if file != "__init__.py" and file != "settings.py":
                workers.append(file)

        workers = sorted(workers, key=lambda s: s.casefold())
        self.cmbWorker.clear()
        self.cmbWorker.addItems(workers)

    def setPanel(self, panel):
        if self.panel is not None:
            self.layout.removeWidget(self.panel)
        self.panel = panel
        self.layout.addWidget(panel)
        self.layout.setStretch(1, 10)

    def cmbWorkerChanged(self, worker):
        print("cmbWorkerChanged")
        camera = None
        if self.panel:
            camera = self.panel.camera

        self.mw.loadVideoConfigure(worker)
        self.mw.videoConfigure.setCamera(camera)
        self.mw.videoWorkerHook = None
        self.mw.videoWorker = None

        if lstCamera := self.mw.cameraPanel.lstCamera:
            cameras = [lstCamera.item(x) for x in range(lstCamera.count())]
            for camera in cameras:
                self.mw.videoConfigure.setCamera(camera)
        
        if player := self.mw.pm.getCurrentPlayer():
            if camera := self.mw.cameraPanel.getCamera(player.uri):
                self.mw.videoConfigure.setCamera(camera)

        self.mw.settings.setValue(self.workerKey, worker)
