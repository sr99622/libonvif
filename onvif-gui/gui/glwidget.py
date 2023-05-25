#********************************************************************
# libavio/samples/pyqt/glwidget.py
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

from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtGui import QPainter, QImage
from PyQt6.QtCore import QRect, QSize
import numpy as np

class GLWidget(QOpenGLWidget):
    def __init__(self):
        super().__init__()
        self.image = QImage()
    
    def sizeHint(self):
        return QSize(640, 480)

    def renderCallback(self, F):
        try :
            ary = np.array(F, copy = True)
            h, w, d = ary.shape
            self.image = QImage(ary.data, w, h, d * w, QImage.Format.Format_RGB888)
            self.update()
        except Exception as ex:
            print (ex)

    def getImageRect(self):
        ratio = min(self.width() / self.image.width(), self.height() / self.image.height())
        w = self.image.width() * ratio
        h = self.image.height() * ratio
        x = (self.width() - w) / 2
        y = (self.height() - h) / 2
        return QRect(int(x), int(y), int(w), int(h))

    def paintGL(self):
        if (not self.image.isNull()):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.drawImage(self.getImageRect(), self.image)

    def clear(self):
        self.image.fill(0)
        self.update()
