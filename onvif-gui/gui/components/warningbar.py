#********************************************************************
# libonvif/onvif-gui/gui/components/warningbar.py
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

from PyQt6.QtWidgets import QLabel
from PyQt6.QtGui import QPainter, QColorConstants, QLinearGradient, QColor
from PyQt6.QtCore import QRect, QTimer, pyqtSignal, QObject

class WarningBar(QLabel):
    def __init__(self):
        super().__init__()
        self.setMaximumWidth(15)
        self.setStyleSheet("QLabel { border : 1px solid #808D9E; }")
        self.level = 0.0
        self.inverted = False

    def paintEvent(self, event):
        marker = max(int(self.height()*(1-self.level)-2), 0)
        if self.inverted:
            marker = min(int(self.height() * self.level), self.height()-2)
        painter = QPainter(self)
        gradient = QLinearGradient(0,0,0,100)
        gradient.setColorAt(0.0, QColorConstants.Red)
        gradient.setColorAt(0.5, QColorConstants.DarkYellow)
        gradient.setColorAt(1.0, QColorConstants.DarkGreen)
        painter.fillRect(QRect(1, 1, 13, self.height()-2), gradient)
        painter.fillRect(QRect(1, 1, 13, marker), QColor("#3B3B3B"))

    def setLevel(self, level):
        self.level = level
        self.update()

class IndicatorSignals(QObject):
    start = pyqtSignal()

class Indicator(QLabel):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw
        self.setMaximumWidth(15)
        self.setMaximumHeight(10)
        self.setStyleSheet("QLabel { border : 1px solid #808D9E; }")
        self.timer = QTimer()
        self.timer.setInterval(self.mw.settingsPanel.alarm.spnLagTime.value() * 1000)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.timeout)
        self.signals = IndicatorSignals()
        self.signals.start.connect(self.timer.start)
        self.state = 0

    def paintEvent(self, event):
        color = QColor("#3B3B3B")
        if self.state:
            color = QColorConstants.Red
        painter = QPainter(self)
        painter.fillRect(QRect(1, 1, 13, self.height()-2), color)

    def setState(self, state):
        self.state = int(state)
        if state:
            self.signals.start.emit()
        self.update()

    def getState(self):
        return self.state

    def timeout(self):
        self.setState(0)

