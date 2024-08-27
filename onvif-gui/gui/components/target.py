#********************************************************************
# libonvif/onvif-gui/gui/components/target.py
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

from PyQt6.QtWidgets import QDialog, QGridLayout, QListWidget, QListWidgetItem, \
    QDialogButtonBox, QWidget, QLabel, QPushButton, QMessageBox, QSlider, QCheckBox
from PyQt6.QtCore import Qt, pyqtSignal, QObject
from .warningbar import WarningBar, Indicator
from gui.enums import MediaSource
from loguru import logger

class Target(QListWidgetItem):
    def __init__(self, name, id):
        super().__init__(name)
        self.id = id

class TargetDialog(QDialog):
    def __init__(self, mw):
        super().__init__(mw)
        self.mw = mw
        self.source = MediaSource.CAMERA

        self.targets = {
            0: "person",
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            4: "airplane",
            5: "bus",
            6: "train",
            7: "truck",
            8: "boat",
            14: "bird",
            15: "cat",
            16: "dog",
            17: "horse",
            18: "sheep",
            19: "cow",
            21: "bear"
        }

        self.setModal(True)
        self.setWindowTitle("Add Target")
        self.list = QListWidget()
        for key in self.targets:
            self.list.addItem(Target(self.targets[key], key))

        self.buttonBox = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Close)
        self.buttonBox.button(QDialogButtonBox.StandardButton.Close).setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.buttonBox.rejected.connect(self.reject)

        lytMain = QGridLayout(self)
        lytMain.addWidget(self.list,       0, 0, 1, 2)
        lytMain.addWidget(self.buttonBox,  1, 0, 1, 2)

        self.list.setCurrentRow(0)

    def reject(self):
        self.hide()

class TargetListSignals(QObject):
    delete = pyqtSignal()

class TargetList(QListWidget):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw
        self.signals = TargetListSignals()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Delete:
            self.signals.delete.emit()
        return super().keyPressEvent(event)
    
    def toString(self):
        output = ""
        length = self.count()
        for i in range(length):
            output += str(self.item(i).id)
            if i < length - 1:
                output += ":"
        return output

class TargetSelector(QWidget):
    def __init__(self, mw, module):
        super().__init__()
        self.mw = mw
        self.module = module

        self.lstTargets = TargetList(self.mw)
        self.lstTargets.signals.delete.connect(self.btnDeleteTargetClicked)
        self.lblTargets = QLabel("Targets")
        self.btnAddTarget = QPushButton("+")
        self.btnAddTarget.clicked.connect(self.btnAddTargetClicked)
        self.btnDeleteTarget = QPushButton("-")
        self.btnDeleteTarget.clicked.connect(self.btnDeleteTargetClicked)
        self.dlgTarget = TargetDialog(self.mw)
        self.dlgTarget.list.itemDoubleClicked.connect(self.onAddItemDoubleClicked)
        self.dlgTarget.buttonBox.accepted.connect(self.dlgListAccepted)

        # gui works better if these are on the same panel
        self.barLevel = WarningBar()
        self.indAlarm = Indicator(self.mw)
        self.sldGain = QSlider(Qt.Orientation.Vertical)
        self.sldGain.setMinimum(0)
        self.sldGain.setMaximum(100)
        self.sldGain.setValue(0)
        self.sldGain.valueChanged.connect(self.sldGainValueChanged)
        self.lblGain = QLabel("0")

        pnlTargets = QWidget()
        pnlTargets.setMaximumWidth(200)
        lytTargets = QGridLayout(pnlTargets)

        self.chkShowBoxes = QCheckBox("Show Boxes")
        self.chkShowBoxes.stateChanged.connect(self.chkShowBoxesStateChanged)

        lytTargets.addWidget(self.lblTargets,       0, 0, 1, 1)
        lytTargets.addWidget(self.btnDeleteTarget,  0, 1, 1, 1)
        lytTargets.addWidget(self.btnAddTarget,     0, 2, 1, 1)
        lytTargets.addWidget(self.lstTargets,       1, 0, 2, 3)

        lytMain = QGridLayout(self)
        lytMain.addWidget(pnlTargets,            1, 0, 2, 1)
        lytMain.addWidget(self.lblGain,          1, 1, 1, 1, Qt.AlignmentFlag.AlignHCenter)
        lytMain.addWidget(self.sldGain,          2, 1, 1, 1, Qt.AlignmentFlag.AlignHCenter)
        lytMain.addWidget(QLabel("Limit"),       3, 1, 1, 1, Qt.AlignmentFlag.AlignHCenter)
        lytMain.addWidget(self.indAlarm,         1, 2, 1, 1)
        lytMain.addWidget(self.barLevel,         2, 2, 1, 1)
        lytMain.addWidget(QLabel(),              2, 3, 1, 1)
        lytMain.addWidget(self.chkShowBoxes,     3, 0, 1, 1, Qt.AlignmentFlag.AlignCenter)
        lytMain.setContentsMargins(0, 0, 0, 0)

    def btnAddTargetClicked(self):
        self.dlgTarget.show()

    def btnDeleteTargetClicked(self):
        try:
            if item := self.lstTargets.currentItem():
                ret = QMessageBox.warning(self, "Delete Target: " + item.text(), "You are about to delete target\n"
                                        "Are you sure you want to continue?",
                                        QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
                if ret != QMessageBox.StandardButton.Ok:
                    return
                self.lstTargets.takeItem(self.lstTargets.row(item))

                match self.mw.videoConfigure.source:
                    case MediaSource.CAMERA:
                        if camera := self.mw.cameraPanel.getCurrentCamera():
                            if camera.videoModelSettings:
                                camera.videoModelSettings.setTargets(self.lstTargets.toString())
                    case MediaSource.FILE:
                        if self.mw.filePanel.videoModelSettings:
                            self.mw.filePanel.videoModelSettings.setTargets(self.lstTargets.toString())
        except Exception as ex:
            logger.error(ex)

    def dlgListAccepted(self):
        item = self.dlgTarget.list.item(self.dlgTarget.list.currentRow())
        self.onAddItemDoubleClicked(item)

    def onAddItemDoubleClicked(self, item):
        try:
            target = Target(item.text(), item.id)
            found = False
            for i in range(self.lstTargets.count()):
                if target.text() == self.lstTargets.item(i).text():
                    found = True
                    break
            if not found:
                self.lstTargets.addItem(target)

                match self.mw.videoConfigure.source:
                    case MediaSource.CAMERA:
                        if camera := self.mw.cameraPanel.getCurrentCamera():
                            if camera.videoModelSettings:
                                camera.videoModelSettings.setTargets(self.lstTargets.toString())
                    case MediaSource.FILE:
                        if self.mw.filePanel.videoModelSettings:
                            self.mw.filePanel.videoModelSettings.setTargets(self.lstTargets.toString())
        except Exception as ex:
            logger.error(ex)

    def setTargets(self, targets):
        while self.lstTargets.count() > 0:
            self.lstTargets.takeItem(0)

        for t in targets:
            self.lstTargets.addItem(Target(self.dlgTarget.targets[t], t))

    def getTargets(self):
        output = []
        for i in range(self.lstTargets.count()):
            target = self.lstTargets.item(i)
            output.append(target.id)
        return output

    def sldGainValueChanged(self, value):
        try:
            self.lblGain.setText(f'{value}')
            match self.mw.videoConfigure.source:
                case MediaSource.CAMERA:
                    if camera := self.mw.cameraPanel.getCurrentCamera():
                        if camera.videoModelSettings:
                            camera.videoModelSettings.setModelOutputLimit(value)
                case MediaSource.FILE:
                    if self.mw.filePanel.videoModelSettings:
                        self.mw.filePanel.videoModelSettings.setModelOutputLimit(value)
        except Exception as ex:
            logger.error(ex)

    def chkShowBoxesStateChanged(self, state):
        try:
            match self.mw.videoConfigure.source:
                case MediaSource.CAMERA:
                    if camera := self.mw.cameraPanel.getCurrentCamera():
                        if camera.videoModelSettings:
                            camera.videoModelSettings.setModelShowBoxes(bool(state))
                case MediaSource.FILE:
                    if self.mw.filePanel.videoModelSettings:
                        self.mw.filePanel.videoModelSettings.setModelShowBoxes(bool(state))
        except Exception as ex:
            logger.error(ex)
