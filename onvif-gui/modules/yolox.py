import cv2
import torch
import os
import sys
import numpy as np
from pathlib import Path
from loguru import logger
from torchvision.transforms import functional
import torch.nn as nn

from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
from yolox.utils import postprocess
from components import ComboSelector, FileSelector, LabelSelector, ThresholdSlider
from PyQt6.QtWidgets import QWidget, QGridLayout, QLabel, QCheckBox
from PyQt6.QtCore import Qt

os.environ['KMP_DUPLICATE_LIB_OK']='True'
MODULE_NAME = "yolox"

class Configure(QWidget):
    def __init__(self, mw):
        try:
            super().__init__()
            self.mw = mw
            self.name = MODULE_NAME
            self.autoKey = "Module/" + MODULE_NAME + "/autoDownload"
            self.fp16Key = "Module/" + MODULE_NAME + "/fp16"
            self.trackKey = "Module/" + MODULE_NAME + "/track"

            self.chkAuto = QCheckBox("Automatically download model")
            self.chkAuto.setChecked(int(self.mw.settings.value(self.autoKey, 1)))
            self.chkAuto.stateChanged.connect(self.chkAutoClicked)

            self.txtFilename = FileSelector(mw, MODULE_NAME)
            self.txtFilename.setEnabled(not self.chkAuto.isChecked())

            self.cmbRes = ComboSelector(mw, "Model Size", ("320", "480", "640", "960", "1280", "1440"), "640", MODULE_NAME)
            self.cmbType = ComboSelector(mw, "Model Name", ("yolox_s", "yolox_m", "yolox_l", "yolox_x"), "yolox_s", MODULE_NAME)

            self.chkFP16 = QCheckBox("Use half precision math")
            self.chkFP16.setChecked(int(self.mw.settings.value(self.fp16Key, 1)))
            self.chkFP16.stateChanged.connect(self.chkFP16Clicked)

            self.chkTrack = QCheckBox("Track Objects")
            if sys.platform == "win32":
                self.chkTrack.setChecked(int(self.mw.settings.value(self.trackKey, 0)))
                self.chkTrack.stateChanged.connect(self.chkTrackClicked)
            else:
                self.chkTrack.setChecked(False)
                self.chkTrack.setEnabled(False)

            self.sldConfThre = ThresholdSlider(mw, MODULE_NAME + "/confidence", "Confidence", 25)
            self.sldConfThre.setEnabled(not self.chkTrack.isChecked())

            number_of_labels = 5
            self.labels = []
            for i in range(number_of_labels):
                self.labels.append(LabelSelector(mw, MODULE_NAME, i+1))
            pnlLabels = QWidget()
            lytLabels = QGridLayout(pnlLabels)
            lblPanel = QLabel("Select classes to be identified")
            lblPanel.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lytLabels.addWidget(lblPanel,        0, 0, 1, 1)
            for i in range(number_of_labels):
                lytLabels.addWidget(self.labels[i], i+1, 0, 1, 1)
            lytLabels.setContentsMargins(0, 0, 0, 0)

            lytMain = QGridLayout(self)
            lytMain.addWidget(self.chkAuto,      0, 0, 1, 1)
            lytMain.addWidget(self.txtFilename,  1, 0, 1, 1)
            lytMain.addWidget(self.cmbRes,       2, 0, 1, 1)
            lytMain.addWidget(self.cmbType,      3, 0, 1, 1)
            lytMain.addWidget(self.sldConfThre,  4, 0, 1, 1)
            lytMain.addWidget(self.chkFP16,      5, 0, 1, 1)
            lytMain.addWidget(self.chkTrack,     6, 0, 1, 1)
            lytMain.addWidget(pnlLabels,         7, 0, 1, 1)
            lytMain.addWidget(QLabel(),          8, 0, 1, 1)
            lytMain.setRowStretch(8, 10)

        except:
            logger.exception("yolox configure failed to load")

    def chkAutoClicked(self, state):
        self.mw.settings.setValue(self.autoKey, state)
        self.txtFilename.setEnabled(not self.chkAuto.isChecked())

    def chkFP16Clicked(self, state):
        self.mw.settings.setValue(self.fp16Key, state)

    def chkTrackClicked(self, state):
        self.mw.settings.setValue(self.trackKey, state)
        self.sldConfThre.setEnabled(not self.chkTrack.isChecked())

class Worker:
    def __init__(self, mw):
        try:
            self.mw = mw
            device_name = "cpu"
            if torch.cuda.is_available():
                device_name = "cuda"
            self.device = torch.device(device_name)

            self.num_classes = 80

            self.fp16 = self.mw.configure.chkFP16.isChecked()
            self.track = self.mw.configure.chkTrack.isChecked()

            track_thresh = 0.5
            track_buffer = 30
            match_thresh = 0.8

            size = {'yolox_s': [0.33, 0.50], 
                    'yolox_m': [0.67, 0.75],
                    'yolox_l': [1.00, 1.00],
                    'yolox_x': [1.33, 1.25]}[self.mw.configure.cmbType.currentText()]

            self.model = None
            self.model = self.get_model(self.num_classes, size[0], size[1], None).to(self.device)
            self.model.eval()

            self.ckpt_file = None
            if self.mw.configure.chkAuto.isChecked():
                self.ckpt_file = self.get_auto_ckpt_filename()
                print("cpkt_file:", self.ckpt_file)
                cache = Path(self.ckpt_file)

                if not cache.is_file():
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    model_name = self.mw.configure.cmbType.currentText()
                    link = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/" + model_name + ".pth"
                    torch.hub.download_url_to_file(link, self.ckpt_file)
            else:
                self.ckpt_file = self.mw.configure.txtFilename.text()

            self.model.load_state_dict(torch.load(self.ckpt_file, map_location="cpu")["model"])

            if self.fp16:
                self.model = self.model.half()

            if self.track:
                from yolox.tracker.byte_tracker import BYTETracker
                self.tracker = BYTETracker(track_thresh, track_buffer, match_thresh)

        except:
            logger.exception("yolox initialization failure")

    def __call__(self, F):
        try:
            img = np.array(F, copy=False)
            
            res = int(self.mw.configure.cmbRes.currentText())
            test_size = (res, res)
            ratio = min(test_size[0] / img.shape[0], test_size[1] / img.shape[1])
            inf_shape = (int(img.shape[0] * ratio), int(img.shape[1] * ratio))
            bottom = test_size[0] - inf_shape[0]
            side = test_size[1] - inf_shape[1]
            pad = (0, 0, side, bottom)

            timg = functional.to_tensor(img.copy()).to(self.device)
            timg *= 255
            timg = functional.resize(timg, inf_shape)
            timg = functional.pad(timg, pad, 114)
            timg = timg.unsqueeze(0)

            if self.fp16:
                timg = timg.half()  # to FP16

            tmp = None
            if self.mw.configure.chkAuto.isChecked():
                tmp = self.get_auto_ckpt_filename()
            else:
                tmp = self.mw.configure.txtFilename.text()
            if self.ckpt_file != tmp:
                self.__init__(self.mw)


            if self.mw.configure.chkTrack.isChecked():
                confthre = 0.001
            else:
                confthre = self.mw.configure.sldConfThre.value()
            
            nmsthre = 0.65

            label_filter = []
            for lbl in self.mw.configure.labels:
                if lbl.chkBox.isChecked():
                    label_filter.append(lbl.label())

            with torch.no_grad():
                outputs = self.model(timg)
                outputs = postprocess(outputs, self.num_classes, confthre, nmsthre)

            if self.mw.configure.name != MODULE_NAME:
                return
            
            if outputs[0] is not None:
                output = outputs[0].cpu()
                if self.mw.configure.chkTrack.isChecked():
                    labels = output[:, 6].numpy().astype(int)
                    mask = np.in1d(labels, label_filter)
                    output = output[mask]
                    online_targets = self.tracker.update(output, [img.shape[0], img.shape[1]], test_size)
                    self.draw_track_boxes(img, online_targets)
                else:
                    self.draw_plain_boxes(img, output, ratio)

        except:
            logger.exception("yolox runtime error")

    def draw_plain_boxes(self, img, output, ratio):
        boxes = output[:, 0:4] / ratio
        labels = output[:, 6].numpy().astype(int)
        for lbl in self.mw.configure.labels:
            if lbl.chkBox.isChecked():
                lbl_boxes = boxes[labels == lbl.label()].numpy().astype(int)
                lbl.setCount(lbl_boxes.shape[0])

                for box in lbl_boxes:
                    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), lbl.color(), 2)

    def draw_track_boxes(self, img, online_targets):
        label_colors = {}
        count = {}

        for lbl in self.mw.configure.labels:
            if lbl.chkBox.isChecked():
                label_colors[lbl.label()] = lbl.color()
                count[lbl.label()] = 0

        for t in online_targets:
            count[t.label] += 1

            track_id = int(t.track_id)
            id_text = '{}'.format(int(track_id)).zfill(5)
            color = ((37 * track_id) % 255, (17 * track_id) % 255, (29 * track_id) % 255)

            tlwh = t.tlwh
            x, y, w, h = tlwh.astype(int)
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, id_text, (x, y), cv2.FONT_HERSHEY_PLAIN, 2, label_colors[t.label], 2)

        for lbl in self.mw.configure.labels:
            if lbl.chkBox.isChecked():
                lbl.setCount(count[lbl.label()])


    def get_auto_ckpt_filename(self):
        filename = None
        if sys.platform == "win32":
            filename = os.environ['HOMEPATH']
        else:
            filename = os.environ['HOME']

        filename += "/.cache/torch/hub/checkpoints/" + self.mw.configure.cmbType.currentText() + ".pth"
        return filename

    def get_model(self, num_classes, depth, width, act):
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        in_channels = [256, 512, 1024]
        backbone = YOLOPAFPN(depth, width, in_channels=in_channels)
        head = YOLOXHead(num_classes, width, in_channels=in_channels)
        model = YOLOX(backbone, head)

        model.apply(init_yolo)
        model.head.initialize_biases(1e-2)
        return model