#/********************************************************************
# onvif-gui/modules/video/yolov7.py 
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

IMPORT_ERROR = ""
try:
    import cv2
    import os
    import sys
    import gc
    import numpy as np
    from pathlib import Path
    from loguru import logger

    from gui.components import ComboSelector, FileSelector, LabelSelector, ThresholdSlider
    from PyQt6.QtWidgets import QWidget, QGridLayout, QLabel, QCheckBox, QMessageBox
    from PyQt6.QtCore import Qt

    import cv2
    import torch
    from tracker.byte_tracker import BYTETracker

    sys.path.append("yolov7")
    for path in sys.path:
        tmp = os.path.join(path, "yolov7")
        if os.path.isdir(tmp):
            sys.path.append(tmp)

    from models.experimental import attempt_load
    from utils.datasets import letterbox
    from utils.general import non_max_suppression, scale_coords
    from utils.torch_utils import select_device

except ModuleNotFoundError as ex:
    IMPORT_ERROR = str(ex)
    print("Import Error has occurred, missing modules need to be installed, please consult documentation: ", ex)

os.environ['KMP_DUPLICATE_LIB_OK']='True'
MODULE_NAME = "yolov7"

class VideoConfigure(QWidget):
    def __init__(self, mw):
        try:
            super().__init__()
            self.mw = mw
            self.name = MODULE_NAME
            self.autoKey = "Module/" + MODULE_NAME + "/autoDownload"
            self.trackKey = "Module/" + MODULE_NAME + "/track"

            self.chkAuto = QCheckBox("Automatically download model")
            self.chkAuto.setChecked(int(self.mw.settings.value(self.autoKey, 1)))
            self.chkAuto.stateChanged.connect(self.chkAutoClicked)

            self.txtFilename = FileSelector(mw, MODULE_NAME)
            self.txtFilename.setEnabled(not self.chkAuto.isChecked())

            self.cmbRes = ComboSelector(mw, "Model Size", ("320", "480", "640", "960", "1280"), "640", MODULE_NAME)
            self.cmbType = ComboSelector(mw, "Model Name", ("yolov7", "yolov7x"), "yolov7", MODULE_NAME)

            self.chkTrack = QCheckBox("Track Objects")
            self.chkTrack.setChecked(int(self.mw.settings.value(self.trackKey, 0)))
            self.chkTrack.stateChanged.connect(self.chkTrackClicked)

            self.sldConfThre = ThresholdSlider(mw, MODULE_NAME + "/confidence", "Confidence", 25)

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
            lytMain.addWidget(self.cmbType,      1, 0, 1, 1)
            lytMain.addWidget(self.txtFilename,  2, 0, 1, 1)
            lytMain.addWidget(self.cmbRes,       3, 0, 1, 1)
            lytMain.addWidget(self.sldConfThre,  4, 0, 1, 1)
            lytMain.addWidget(self.chkTrack,     6, 0, 1, 1)
            lytMain.addWidget(pnlLabels,         7, 0, 1, 1)
            lytMain.addWidget(QLabel(),          8, 0, 1, 1)
            lytMain.setRowStretch(8, 10)

            if len(IMPORT_ERROR) > 0:
                QMessageBox.critical(None, MODULE_NAME + " Import Error", "Modules required for running this function are missing: " + IMPORT_ERROR)

        except:
            logger.exception(MODULE_NAME + " configure failed to load")

    def chkAutoClicked(self, state):
        self.mw.settings.setValue(self.autoKey, state)
        self.txtFilename.setEnabled(not self.chkAuto.isChecked())

    def chkTrackClicked(self, state):
        self.mw.settings.setValue(self.trackKey, state)

    def getLabel(self, cls):
        for lbl in self.labels:
            if lbl.label() == cls:
                return lbl

class VideoWorker:
    def __init__(self, mw):
        try:
            self.mw = mw
            self.last_ex = ""

            self.ckpt_file = None
            if self.mw.configure.chkAuto.isChecked():
                self.ckpt_file = self.get_auto_ckpt_filename()
                print("cpkt_file:", self.ckpt_file)
                cache = Path(self.ckpt_file)

                if not cache.is_file():
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    model_name = self.mw.configure.cmbType.currentText()
                    link = "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/" + model_name + ".pt"
                    torch.hub.download_url_to_file(link, self.ckpt_file)
            else:
                self.ckpt_file = self.mw.configure.txtFilename.text()


            weights = self.ckpt_file
            res = int(self.mw.configure.cmbRes.currentText())
            self.iou_thres = 0.45

            self.device = select_device('')
            self.half = self.device.type != 'cpu'

            self.model = None
            gc.collect()
            with torch.no_grad():
                torch.cuda.empty_cache()

            self.model = attempt_load(weights, map_location=self.device)
            self.stride = int(self.model.stride.max())
            if self.half:
                self.model.half()
            self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
            with torch.no_grad():
                self.model(torch.zeros(1, 3, res, res).to(self.device).type_as(next(self.model.parameters())))

            self.track_thresh = self.mw.configure.sldConfThre.value()
            self.track_buffer = 30
            self.match_thresh = 0.8

            self.tracker = BYTETracker(self.track_thresh, self.track_buffer, self.match_thresh)

        except:
            logger.exception(MODULE_NAME + " initialization failure")

    def __call__(self, F):
        try:
            original_img = np.array(F, copy=False)

            res = int(self.mw.configure.cmbRes.currentText())
            img = letterbox(original_img, res, stride=self.stride)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()
            img /= 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            with torch.no_grad():
                pred = self.model(img, augment=False)[0]

            conf_thres = self.mw.configure.sldConfThre.value()
            if self.mw.configure.chkTrack.isChecked():
                conf_thres = 0.001

            label_counts = {}
            label_filter = []
            for lbl in self.mw.configure.labels:
                if lbl.chkBox.isChecked():
                    label_filter.append(lbl.label())
                    label_counts[lbl.label()] = 0

            pred = non_max_suppression(pred, conf_thres, self.iou_thres, classes=label_filter, agnostic=False)

            boxes = pred[0]
            if len(boxes):
                boxes[:, :4] = scale_coords(img.shape[2:], boxes[:, :4], original_img.shape).round()
                boxes = boxes.cpu().numpy()

                if self.mw.configure.chkTrack.isChecked():
                    w = original_img.shape[0]
                    h = original_img.shape[1]
                    if self.track_thresh != self.mw.configure.sldConfThre.value():
                        self.track_thresh = self.mw.configure.sldConfThre.value()
                        self.tracker = BYTETracker(self.track_thresh, self.track_buffer, self.match_thresh)

                    online_targets = self.tracker.update(boxes, [w * res / h, res], (res, res))
                    for t in online_targets:
                        label_counts[t.label] += 1
                        track_id = int(t.track_id)
                        id_text = '{}'.format(int(track_id)).zfill(5)
                        box_color = ((37 * track_id) % 255, (17 * track_id) % 255, (29 * track_id) % 255)

                        x, y, w, h = t.tlwh.astype(int)
                        cv2.rectangle(original_img, (x, y), (x+w, y+h), box_color, 2)
                        label_color = self.mw.configure.getLabel(t.label).color()
                        cv2.putText(original_img, id_text, (x, y), cv2.FONT_HERSHEY_PLAIN, 2, label_color, 2)

                else:
                    for box in boxes:
                        x1, y1, x2, y2 = box[:4].astype(int)
                        cls = box[5].astype(int)
                        label_counts[cls] += 1
                        color = self.mw.configure.getLabel(cls).color()
                        cv2.rectangle(original_img, (x1, y1), (x2, y2), color, 2)

            for lbl in label_filter:
                self.mw.configure.getLabel(lbl).setCount(label_counts[lbl])

            tmp = None
            if self.mw.configure.chkAuto.isChecked():
                tmp = self.get_auto_ckpt_filename()
            else:
                tmp = self.mw.configure.txtFilename.text()
            if self.ckpt_file != tmp:
                self.__init__(self.mw)

        except Exception as ex:
            if self.last_ex != str(ex) and self.mw.configure.name == MODULE_NAME:
                logger.exception(MODULE_NAME + " runtime error")
            self.last_ex = str(ex)

    def get_auto_ckpt_filename(self):
        filename = None
        if sys.platform == "win32":
            filename = os.environ['HOMEPATH']
        else:
            filename = os.environ['HOME']

        filename += "/.cache/torch/hub/checkpoints/" + self.mw.configure.cmbType.currentText() + ".pt"
        return filename

        