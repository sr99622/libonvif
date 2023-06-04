#*******************************************************************************
# onvif-gui/gui/components/labelselector.py
#
# Copyright (c) 2023 Stephen Rhodes 
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
#******************************************************************************/

from PyQt6.QtWidgets import QPushButton, QColorDialog, \
            QGridLayout, QWidget, QCheckBox, QLabel, QComboBox
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from collections import deque

class LabelSelector(QWidget):
    def __init__(self, mw, name, index, labels=None):
        super().__init__()

        self.labels = [ "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", 
                        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
                        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",  "zebra",
                        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
                        "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
                        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
                        "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
                        "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
                        "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
                        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
                        "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" ]
        if labels is not None:
            self.labels = labels
        
        self.mw = mw
        self.index = index
        self.colorKey = "Module/" + name + "/color" + str(index)
        self.enabledKey = "Module/" + name + "/enabled" + str(index)
        self.labelKey = "Module/" + name + "/label" + str(index)
        self.idleColor = QColor("#3B3B3B")
        self.m_color = QColor(self.mw.settings.value(self.colorKey, self.idleColor.name()))
        
        self.cmbLabel = QComboBox()
        self.cmbLabel.addItems(self.labels)
        self.cmbLabel.setCurrentText(self.mw.settings.value(self.labelKey))
        self.cmbLabel.currentTextChanged.connect(self.cmbLabelChanged)

        self.chkBox = QCheckBox()
        self.chkBox.setChecked(int(self.mw.settings.value(self.enabledKey, 0)))
        self.chkBox.stateChanged.connect(self.chkBoxClicked)

        self.btnColor = QPushButton("...")
        self.btnColor.setToolTip("Set Box Color")
        self.btnColor.setToolTipDuration(2000)
        self.btnColor.setMaximumWidth(36)
        self.btnColor.setStyleSheet("QPushButton {background-color: " + self.m_color.name() + "; color: white;}")
        self.btnColor.clicked.connect(self.btnColorClicked)

        self.lblCount = QLabel()
        self.lblCount.setMinimumWidth(60)
        self.lblCount.setAlignment(Qt.AlignmentFlag.AlignRight)

        lytLabel = QGridLayout(self)
        lytLabel.addWidget(self.chkBox,   0, 0, 1, 1)
        lytLabel.addWidget(self.cmbLabel, 0, 1, 1, 1)
        lytLabel.addWidget(self.btnColor, 0, 2, 1, 1)
        lytLabel.addWidget(self.lblCount, 0, 3, 1, 1)
        lytLabel.setColumnStretch(1, 10)
        lytLabel.setContentsMargins(0, 0, 0, 0)

        self.setEnabled(self.chkBox.isChecked())

        self.counts = deque()
        self.tracks = {}
        self.track_count = 0

    def btnColorClicked(self):
        color = QColorDialog.getColor(self.m_color)
        if color.isValid():
            self.m_color = color
            self.btnColor.setStyleSheet("QPushButton {background-color: " + self.m_color.name() + "; color: white;}")
            self.mw.settings.setValue(self.colorKey, self.m_color.name())

    def chkBoxClicked(self, state):
        self.setEnabled(state)
        self.mw.settings.setValue(self.enabledKey, state)
        self.lblCount.setText("")

    def cmbLabelChanged(self, label):
        self.mw.settings.setValue(self.labelKey, label)

    def setEnabled(self, enabled):
        self.chkBox.setChecked(enabled)
        self.cmbLabel.setEnabled(enabled)
        self.btnColor.setEnabled(enabled)
        if enabled:
            self.btnColor.setStyleSheet("QPushButton {background-color: " + self.m_color.name() + "; color: white;}")
        else:
            self.btnColor.setStyleSheet("QPushButton {background-color: " + self.idleColor.name() + "; color: white;}")

    def setCount(self, count):
        self.lblCount.setText(str(count))

    def label(self):
        return self.cmbLabel.currentIndex()
    
    def color(self):
        return [self.m_color.red(), self.m_color.green(), self.m_color.blue()]
    
    def isChecked(self):
        return self.chkBox.isChecked()
    
    def avgCount(self, count, q_size):
        while len(self.counts) > q_size:
            self.counts.popleft()
        self.counts.append(count)

        sum = 0
        for count in self.counts:
            sum += count
        avg_count = sum / len(self.counts)
        self.lblCount.setText("{:.2f}".format(avg_count))

    def trackCount(self, track_id, frame_id):
        if not track_id in self.tracks.keys():
            self.tracks[track_id] = [0, frame_id]
            self.track_count += 1
        self.tracks[track_id][0] += 1
        self.tracks[track_id][1] = frame_id

    def trackPurge(self, frame_id, max_time_lost):
        for track_id in self.tracks.keys():
            if frame_id - self.tracks[track_id][1] > max_time_lost:
                del self.tracks[track_id]
        self.lblCount.setText(str(self.track_count))
