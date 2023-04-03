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
import torchvision
import platform
from pathlib import Path
import numpy as np
from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
from torchvision.transforms import functional
import torch.nn as nn
from PyQt6.QtWidgets import QWidget, QLabel, QGridLayout, QCheckBox
from PyQt6.QtCore import Qt
from components import ThresholdSlider, LabelSelector, FileSelector, ComboSelector

class Configure:
    def __init__(self, mw):
        print("YOLOX Configure.__init__")
        self.mw = mw
        self.panel = QWidget()
        self.fp16Key = "Module/yolox/fp16"
        self.autoKey = "Module/yolox/autoDownload"

        self.chkAuto = QCheckBox("Automatically download model")
        self.chkAuto.setChecked(int(self.mw.settings.value(self.autoKey, 1)))
        self.chkAuto.stateChanged.connect(self.chkAutoClicked)

        self.txtFilename = FileSelector(mw, "yolox")
        self.txtFilename.setEnabled(not self.chkAuto.isChecked())

        self.cmbRes = ComboSelector(mw, "Model Size", ("320", "480", "640", "960", "1280", "1440"), "640")
        self.cmbType = ComboSelector(mw, "Model Type", ("yolox_s", "yolox_m", "yolox_l", "yolox_x"), "yolox_s")

        self.sldConfidence = ThresholdSlider(mw, "yolox/confidence", "Confidence", 25)
        self.sldNMS = ThresholdSlider(mw, "yolox/nms", "NMS            ", 45)

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
        lytMain.addWidget(self.chkAuto,             0, 0, 1, 1)
        lytMain.addWidget(self.txtFilename,         1, 0, 1, 1)
        lytMain.addWidget(self.cmbRes,              2, 0, 1, 1)
        lytMain.addWidget(self.cmbType,             3, 0, 1, 1)
        lytMain.addWidget(self.sldConfidence,       4, 0, 1, 1)
        lytMain.addWidget(self.sldNMS,              5, 0, 1, 1)
        lytMain.addWidget(self.chkFP16,             6, 0, 1, 1)
        lytMain.addWidget(pnlLabels,                7, 0, 1, 1)
        lytMain.addWidget(QLabel(""),               8, 0, 1, 1)
        lytMain.setRowStretch(8, 10)

    def chkFP16Clicked(self, state):
        self.mw.settings.setValue(self.fp16Key, state)

    def chkAutoClicked(self, state):
        self.mw.settings.setValue(self.autoKey, state)
        self.txtFilename.setEnabled(not self.chkAuto.isChecked())

    def get_auto_ckpt_filename(self):
        filename = None
        if platform.system() == "win32":
            filename = os.environ['HOMEPATH']
        else:
            filename = os.environ['HOME']

        filename += "/.cache/torch/hub/checkpoints/" + self.cmbType.currentText() + ".pth"
        return filename

class Worker:
    def __init__(self, mw):
        try:
            self.mw = mw
            self.fp16 = False
            self.num_classes = 80
            act = 'silu'

            ckpt_file = self.mw.configure.txtFilename.text()
            model_name = self.mw.configure.cmbType.currentText()

            sizes = {'yolox_s': [0.33, 0.50], 
                     'yolox_m': [0.67, 0.75],
                     'yolox_l': [1.00, 1.00],
                     'yolox_x': [1.33, 1.25]}
            size = sizes[model_name]

            self.model = self.get_model(self.num_classes, size[0], size[1], act).cuda()

            if self.mw.configure.chkFP16.isChecked():
                self.fp16 = True
                self.model.half()

            self.model.eval()

            if self.mw.configure.chkAuto.isChecked():
                ckpt_file = self.mw.configure.get_auto_ckpt_filename()
                cache = Path(ckpt_file)

                if not cache.is_file():
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    link = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth"
                    torch.hub.download_url_to_file(link, ckpt_file)

            ckpt = torch.load(ckpt_file, map_location="cpu")
            self.model.load_state_dict(ckpt["model"])

        except Exception as ex:
            print("yolox worker init exception:", ex)

    def __call__(self, F):
        try :
            img = np.array(F, copy=False)
            res = int(self.mw.configure.cmbRes.currentText())
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
                outputs = self.postprocess(outputs, self.num_classes, confthre, nmsthre)

            output = outputs[0].cpu()
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
                        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (r, g, b), 2)

        except Exception as ex:
            print("yolox worker call exception:", ex)        

    def get_model(self, num_classes, depth, width, act):
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(depth, width, in_channels=in_channels, act=act)
            head = YOLOXHead(num_classes, width, in_channels=in_channels, act=act)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        self.model.train()
        return self.model

    def postprocess(self, prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):

            if not image_pred.size(0):
                continue

            class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

            conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
            detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
            detections = detections[conf_mask]
            if not detections.size(0):
                continue

            if class_agnostic:
                nms_out_index = torchvision.ops.nms(
                    detections[:, :4],
                    detections[:, 4] * detections[:, 5],
                    nms_thre,
                )
            else:
                nms_out_index = torchvision.ops.batched_nms(
                    detections[:, :4],
                    detections[:, 4] * detections[:, 5],
                    detections[:, 6],
                    nms_thre,
                )

            detections = detections[nms_out_index]
            if output[i] is None:
                output[i] = detections
            else:
                output[i] = torch.cat((output[i], detections))

        return output


