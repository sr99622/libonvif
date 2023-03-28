#/********************************************************************
# libonvif/python/modules/retinanet.py 
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

import torchvision
import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from PyQt6.QtWidgets import QPushButton, QColorDialog, \
QGridLayout, QWidget, QCheckBox, QLabel, QComboBox, QSlider
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

transform = transforms.Compose([
    transforms.ToTensor(),
])

class Configure:
    def __init__(self, mw):
        self.mw = mw
        self.panel = QWidget()
        lytMain = QGridLayout(self.panel)

        self.thresholdKey = "Module/retinanet/threshold"

        self.sldThreshold = QSlider(Qt.Orientation.Horizontal)
        self.sldThreshold.setValue(self.mw.settings.value(self.thresholdKey, 35))
        self.sldThreshold.valueChanged.connect(self.sldThresholdChanged)
        lblThreshold = QLabel("Threshold")
        self.lblValue = QLabel(str(self.sldThreshold.value()))
        pnlThreshold = QWidget()
        lytThreshold = QGridLayout(pnlThreshold)
        lytThreshold.addWidget(lblThreshold,          0, 0, 1, 1)
        lytThreshold.addWidget(self.sldThreshold,     0, 1, 1, 1)
        lytThreshold.addWidget(self.lblValue,         0, 2, 1, 1)

        number_of_labels = 5
        self.labels = []
        for i in range(number_of_labels):
            self.labels.append(Label(mw, i+1))

        pnlLabels = QWidget()
        lytLabels = QGridLayout(pnlLabels)
        for i in range(number_of_labels):
            lytLabels.addWidget(self.labels[i], i, 0, 1, 3)
        lytLabels.setContentsMargins(0, 0, 0, 0)
        lblPanel = QLabel("Select classes to be indentified")
        lblPanel.setAlignment(Qt.AlignmentFlag.AlignCenter)

        lytMain.addWidget(pnlThreshold,        0, 0, 1, 4)
        lytMain.addWidget(lblPanel,            1, 0, 1, 4)
        lytMain.addWidget(pnlLabels,           2, 0, 1, 4)
        lytMain.addWidget(QLabel(),            3, 0, 1, 4)
        lytMain.setRowStretch(3, 10)

    def sldThresholdChanged(self, value):
        print(value)
        self.lblValue.setText(str(value))
        self.mw.settings.setValue(self.thresholdKey, value)

class Worker:
    def __init__(self, mw):
        try:
            self.mw = mw
            self.model = torchvision.models.detection.retinanet_resnet50_fpn(weights=torchvision.models.detection.RetinaNet_ResNet50_FPN_Weights.DEFAULT)            
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.eval().to(self.device)
        except Exception as ex:
            print(ex)

    def __call__(self, F):
        try:
            img = np.array(F, copy = False)
            tensor = transform(img).to(self.device)
            tensor = tensor.unsqueeze(0)

            with torch.no_grad():
                outputs = self.model(tensor)

            scores = outputs[0]['scores'].detach().cpu().numpy()
            labels = outputs[0]['labels'].detach().cpu().numpy()
            boxes = outputs[0]['boxes'].detach().cpu().numpy()

            threshold = self.mw.configure.sldThreshold.value() / 100
            labels = labels[np.array(scores) >= threshold]
            boxes = boxes[np.array(scores) >= threshold].astype(np.int32)
            for lbl in self.mw.configure.labels:
                if lbl.chkBox.isChecked():
                    label = lbl.cmbLabel.currentIndex() + 1
                    lbl_boxes = boxes[np.array(labels) == label]
                    r = lbl.color.red()
                    g = lbl.color.green()
                    b = lbl.color.blue()
                    lbl.lblCount.setText(str(lbl_boxes.shape[0]))

                    for box in lbl_boxes:
                        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (r, g, b), 1)

        except Exception as ex:
            print(ex)


class Label(QWidget):
    def __init__(self, mw, index):
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
        
        self.mw = mw
        self.index = index
        self.colorKey = "Module/retinanet/color" + str(index)
        self.enabledKey = "Module/retinanet/enabled" + str(index)
        self.labelKey = "Module/retinanet/label" + str(index)
        self.idleColor = QColor("#3B3B3B")
        self.color = QColor(self.mw.settings.value(self.colorKey, self.idleColor.name()))
        
        self.cmbLabel = QComboBox()
        self.cmbLabel.addItems(self.labels)
        self.cmbLabel.setCurrentText(self.mw.settings.value(self.labelKey))
        self.cmbLabel.currentTextChanged.connect(self.cmbLabelChanged)

        self.chkBox = QCheckBox()
        self.chkBox.setChecked(int(self.mw.settings.value(self.enabledKey, 0)))
        self.chkBox.stateChanged.connect(self.chkBoxClicked)

        self.btnColor = QPushButton("...")
        self.btnColor.setMaximumWidth(36)
        self.btnColor.setStyleSheet("QPushButton {background-color: " + self.color.name() + "; color: white;}")
        self.btnColor.clicked.connect(self.btnColorClicked)

        self.lblCount = QLabel()
        self.lblCount.setMinimumWidth(30)
        self.lblCount.setAlignment(Qt.AlignmentFlag.AlignRight)

        lytLabel = QGridLayout(self)
        lytLabel.addWidget(self.chkBox,   0, 0, 1, 1)
        lytLabel.addWidget(self.cmbLabel, 0, 1, 1, 1)
        lytLabel.addWidget(self.btnColor, 0, 2, 1, 1)
        lytLabel.addWidget(self.lblCount, 0, 3, 1, 1)
        lytLabel.setColumnStretch(1, 10)
        lytLabel.setContentsMargins(0, 0, 0, 0)

        self.setEnabled(self.chkBox.isChecked())

    def btnColorClicked(self):
        color = QColorDialog.getColor(self.color)
        if color.isValid():
            self.color = color
            self.btnColor.setStyleSheet("QPushButton {background-color: " + self.color.name() + "; color: white;}")
            self.mw.settings.setValue(self.colorKey, self.color.name())

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
            self.btnColor.setStyleSheet("QPushButton {background-color: " + self.color.name() + "; color: white;}")
        else:
            self.btnColor.setStyleSheet("QPushButton {background-color: " + self.idleColor.name() + "; color: white;}")
