#*******************************************************************************
# libonvif/onvif-gui/modules/yolox.py
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

import os
import cv2
import torch
from yolox.exp import get_exp
from yolox.utils import postprocess
from torchvision.transforms import functional
import numpy as np
from PyQt6.QtWidgets import QWidget, QLabel, QGridLayout, QComboBox, QCheckBox
from PyQt6.QtCore import Qt
from components.thresholdslider import ThresholdSlider
from components.labelselector import LabelSelector
from components.fileselector import FileSelector

class Configure:
    def __init__(self, mw):
        print("YOLOX Configure.__init__")
        self.mw = mw
        self.panel = QWidget()
        self.resolutionKey = "Module/yolox/resolution"
        self.fp16Key = "Module/yolox/fp16"

        self.txtFilename = FileSelector(mw, "yolox")

        self.cmbResolution = QComboBox()
        self.cmbResolution.addItems(("320", "480", "640", "960", "1280", "1440"))
        self.cmbResolution.setCurrentText(self.mw.settings.value(self.resolutionKey, "640"))
        self.cmbResolution.currentTextChanged.connect(self.cmbResolutionChanged)
        lblRes = QLabel("Model Resolution")
        pnlRes = QWidget()
        lytRes = QGridLayout(pnlRes)
        lytRes.addWidget(lblRes,             0, 0, 1, 1)
        lytRes.addWidget(self.cmbResolution, 0, 1, 1, 1)
        lytRes.setContentsMargins(0, 0, 0, 0)

        self.sldConfidence = ThresholdSlider(mw, "yolox/confidence", "Confidence", 25)
        self.sldNMS = ThresholdSlider(mw, "yolox/nms", "NMS          ", 45)

        self.chkFP16 = QCheckBox("Use Half Precision Math")
        self.chkFP16.setChecked(int(self.mw.settings.value(self.fp16Key, 0)))
        self.chkFP16.stateChanged.connect(self.chkFP16Clicked)

        number_of_labels = 5
        self.labels = []
        for i in range(number_of_labels):
            self.labels.append(LabelSelector(mw, "yolox", i+1))

        pnlLabels = QWidget()
        lytLabels = QGridLayout(pnlLabels)
        lblPanel = QLabel("Select classes to be indentified")
        lblPanel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lytLabels.addWidget(lblPanel,        0, 0, 1, 1)
        for i in range(number_of_labels):
            lytLabels.addWidget(self.labels[i], i+1, 0, 1, 1)
        lytLabels.setContentsMargins(0, 0, 0, 0)

        lytMain = QGridLayout(self.panel)
        lytMain.addWidget(self.txtFilename,         0, 0, 1, 1)
        lytMain.addWidget(pnlRes,                   1, 0, 1, 1)
        lytMain.addWidget(self.sldConfidence,       2, 0, 1, 1)
        lytMain.addWidget(self.sldNMS,              3, 0, 1, 1)
        lytMain.addWidget(self.chkFP16,             4, 0, 1, 1)
        lytMain.addWidget(pnlLabels,                5, 0, 1, 1)
        lytMain.addWidget(QLabel(""),               6, 0, 1, 1)
        lytMain.setRowStretch(6, 10)

    def cmbResolutionChanged(self, text):
        self.mw.settings.setValue(self.resolutionKey, text)

    def chkFP16Clicked(self, state):
        self.mw.settings.setValue(self.fp16Key, state)


class Worker:
    def __init__(self, mw):
        try:
            self.mw = mw
            self.fp16 = False

            ckpt_file = self.mw.configure.txtFilename.text()
            model_name = os.path.splitext(os.path.basename(ckpt_file))[0]

            exp = get_exp(None, model_name)
            self.model = exp.get_model().cuda()
            if self.mw.configure.chkFP16.isChecked():
                self.fp16 = True
                self.model.half()
            self.model.eval()
            ckpt = torch.load(ckpt_file, map_location="cpu")
            self.model.load_state_dict(ckpt["model"])
            self.num_classes = exp.num_classes

        except Exception as ex:
            print("yolox worker init exception:", ex)
            self.mw.modulePanel.chkEngage.setChecked(False)

    def __call__(self, F):
        try :
            img = np.array(F, copy=False)
            res = int(self.mw.configure.cmbResolution.currentText())
            test_size = (res, res)
            ratio = min(test_size[0] / img.shape[0], test_size[1] / img.shape[1])
            inf_shape = (int(img.shape[0] * ratio), int(img.shape[1] * ratio))
            bottom = test_size[0] - inf_shape[0]
            side = test_size[1] - inf_shape[1]
            pad = (0, 0, side, bottom)

            timg = functional.to_tensor(img).cuda()
            timg *= 255
            timg = functional.resize(timg, inf_shape, antialias=True)
            timg = functional.pad(timg, pad, 114)
            timg = timg.unsqueeze(0)

            if self.fp16:
                timg = timg.half()

            with torch.no_grad():
                outputs = self.model(timg)
                confthre = self.mw.configure.sldConfidence.value()
                nmsthre = self.mw.configure.sldNMS.value()
                outputs = postprocess(outputs, self.num_classes, confthre, nmsthre)

            if outputs[0] is not None:
                output = outputs[0].cpu()
                output.cpu()
                boxes = output[:, 0:4] / ratio
                labels = output[:, 6].numpy().astype(int)
                scores = output[:, 4] * output[:, 5]

                for lbl in self.mw.configure.labels:
                    if lbl.chkBox.isChecked():
                        label = lbl.cmbLabel.currentIndex()
                        lbl_boxes = boxes[labels == label]
                        r = lbl.color.red()
                        g = lbl.color.green()
                        b = lbl.color.blue()
                        lbl.lblCount.setText(str(lbl_boxes.shape[0]))
                
                        for box in lbl_boxes:
                            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (r, g, b), 1)

        except Exception as ex:
            print("yolox worker call exception:", ex)        
