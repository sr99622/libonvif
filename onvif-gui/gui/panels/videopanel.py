#/********************************************************************
# onvif-gui/gui/panels/modulepanel.py 
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
from gui.components import DirectorySelector

class VideoPanel(QWidget):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw
        self.panel = None
        self.layout = QVBoxLayout(self)
        self.workerKey = "VideoPanel/worker"
        self.engageKey = "VideoPanel/engage"
        self.directoryKey = "VideoPanel/directory"
        self.cmbWorkerConnected = True

        stdLocation = mw.getLocation() + "/modules/video"
        self.dirModules = DirectorySelector(mw, self.directoryKey, "Modules Dir", stdLocation)
        self.dirModules.signals.dirChanged.connect(self.dirModulesChanged)

        self.cmbWorker = QComboBox()
        self.fillModules()
        self.cmbWorker.setCurrentText(mw.settings.value(self.workerKey, "sample.py"))
        self.cmbWorker.currentTextChanged.connect(self.cmbWorkerChanged)
        lblWorkers = QLabel("Python Worker")

        self.chkEngage = QCheckBox("Engage")
        self.chkEngage.setChecked(int(mw.settings.value(self.engageKey, 0)))
        self.chkEngage.stateChanged.connect(self.chkEngageClicked)

        self.lblElapsed = QLabel()

        fixedPanel = QWidget()
        lytFixed = QGridLayout(fixedPanel)
        lytFixed.addWidget(self.dirModules,  0, 0, 1, 2)
        lytFixed.addWidget(lblWorkers,       1, 0, 1, 1)
        lytFixed.addWidget(self.cmbWorker,   1, 1, 1, 1)
        lytFixed.addWidget(self.chkEngage,   2, 0, 1, 1)
        lytFixed.addWidget(self.lblElapsed,  2, 1, 1, 1)
        lytFixed.setColumnStretch(1, 10)
        self.layout.addWidget(fixedPanel)

    def fillModules(self):
        d = self.dirModules.text()
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
        if self.cmbWorkerConnected:
            self.mw.settings.setValue(self.workerKey, worker)
            self.mw.videoFirstPass = True
            self.mw.videoRuntimes.clear()
            self.mw.loadVideoConfigure(worker)
            self.mw.loadVideoWorker(worker)

    def chkEngageClicked(self, state):
        self.mw.settings.setValue(self.engageKey, state)

    def dirModulesChanged(self, path):
        self.cmbWorkerConnected = False
        self.fillModules()
        self.cmbWorkerConnected = True
        self.cmbWorkerChanged(self.cmbWorker.currentText())
