#/********************************************************************
# onvif-gui/modules/video/yolox.py 
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
    import os
    import sys
    from loguru import logger
    if sys.platform == "win32":
        filename = os.environ['HOMEPATH'] + "/.cache/onvif-gui/errors.txt"
    else:
        filename = os.environ['HOME'] + "/.cache/onvif-gui/errors.txt"
    logger.add(filename, retention="10 days")

    import cv2
    import numpy as np
    from pathlib import Path
    from datetime import datetime
    from gui.components import ComboSelector, FileSelector, LabelSelector, ThresholdSlider
    from PyQt6.QtWidgets import QWidget, QGridLayout, QLabel, QCheckBox, QMessageBox, QLineEdit
    from PyQt6.QtCore import Qt, QRegularExpression
    from PyQt6.QtGui import QRegularExpressionValidator

    import torch
    from torchvision.transforms import functional
    import torch.nn as nn
    
    from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
    from yolox.utils import postprocess
    from tracker.byte_tracker import BYTETracker

except ModuleNotFoundError as ex:
    IMPORT_ERROR = str(ex)
    logger.debug("Import Error has occurred, missing modules need to be installed, please consult documentation: ", ex)

os.environ['KMP_DUPLICATE_LIB_OK']='True'
MODULE_NAME = "yolox"

class VideoConfigure(QWidget):
    def __init__(self, mw):
        try:
            super().__init__()
            self.mw = mw
            self.name = MODULE_NAME
            self.autoKey = "Module/" + MODULE_NAME + "/autoDownload"
            self.trackKey = "Module/" + MODULE_NAME + "/track"
            self.showIDKey = "Module/" + MODULE_NAME + "/showID"
            self.logCountKey = "Module/" + MODULE_NAME + "/logCount"
            self.countIntervalKey = "Module/" + MODULE_NAME + "/countInterval"

            self.mw.signals.started.connect(self.onMediaStarted)
            
            self.chkAuto = QCheckBox("Automatically download model")
            self.chkAuto.setChecked(int(self.mw.settings.value(self.autoKey, 1)))
            self.chkAuto.stateChanged.connect(self.chkAutoClicked)

            self.txtFilename = FileSelector(mw, MODULE_NAME)
            self.txtFilename.setEnabled(not self.chkAuto.isChecked())

            self.cmbRes = ComboSelector(mw, "Model Size", ("320", "480", "640", "960", "1280", "1440"), "640", MODULE_NAME)
            self.cmbType = ComboSelector(mw, "Model Name", ("yolox_s", "yolox_m", "yolox_l", "yolox_x"), "yolox_s", MODULE_NAME)

            self.chkTrack = QCheckBox("Track Objects")
            self.chkTrack.setChecked(int(self.mw.settings.value(self.trackKey, 0)))
            self.chkTrack.stateChanged.connect(self.chkTrackClicked)

            self.chkShowID = QCheckBox("Show Object ID")
            self.chkShowID.setChecked(int(self.mw.settings.value(self.showIDKey, 1)))
            self.chkShowID.stateChanged.connect(self.chkShowIDClicked)

            self.sldConfThre = ThresholdSlider(mw, MODULE_NAME + "/confidence", "Confidence", 25)

            self.chkShowID.setVisible(self.chkTrack.isChecked())

            pnlCount = QWidget()
            lblCount = QLabel("Count Interval (seconds)")
            self.txtCountInterval = QLineEdit()
            self.txtCountInterval.setText(self.mw.settings.value(self.countIntervalKey, ""))
            self.txtCountInterval.textChanged.connect(self.countIntervalChanged)
            numRegex = QRegularExpression("[0-9]*")
            numValidator = QRegularExpressionValidator(numRegex, self)
            self.txtCountInterval.setValidator(numValidator)        
            self.chkLogCount = QCheckBox("Log Counts")
            self.chkLogCount.setChecked(int(self.mw.settings.value(self.logCountKey, 0)))
            self.chkLogCount.stateChanged.connect(self.chkLogCountClicked)
            lytCount = QGridLayout(pnlCount)
            lytCount.addWidget(lblCount,              0, 0, 1, 1)
            lytCount.addWidget(self.txtCountInterval, 0, 1, 1, 1)
            lytCount.addWidget(self.chkLogCount,      0, 2, 1, 1)
            lytCount.setContentsMargins(0, 0, 0, 0)

            number_of_labels = 5
            self.labels = []
            for i in range(number_of_labels):
                self.labels.append(LabelSelector(mw, MODULE_NAME, i+1))
            pnlLabels = QWidget()
            lytLabels = QGridLayout(pnlLabels)
            lblPanel = QLabel("Select classes to be identified and counted")
            lblPanel.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lytLabels.addWidget(lblPanel,        0, 0, 1, 1)
            for i in range(number_of_labels):
                lytLabels.addWidget(self.labels[i], i+1, 0, 1, 1)
            lytLabels.setContentsMargins(0, 0, 0, 0)

            lytMain = QGridLayout(self)
            lytMain.addWidget(self.chkAuto,      0, 0, 1, 2)
            lytMain.addWidget(self.cmbType,      1, 0, 1, 2)
            lytMain.addWidget(self.txtFilename,  2, 0, 1, 2)
            lytMain.addWidget(self.cmbRes,       3, 0, 1, 2)
            lytMain.addWidget(self.sldConfThre,  4, 0, 1, 2)
            lytMain.addWidget(self.chkTrack,     6, 0, 1, 1)
            lytMain.addWidget(self.chkShowID,    6, 1, 1, 1)
            lytMain.addWidget(pnlCount,          7, 0, 1, 2)
            lytMain.addWidget(pnlLabels,         8, 0, 1, 2)
            lytMain.addWidget(QLabel(),          9, 0, 1, 2)
            lytMain.setRowStretch(9, 10)

            self.first_pass = True

            if len(IMPORT_ERROR) > 0:
                QMessageBox.critical(None, MODULE_NAME + " Import Error", "Modules required for running this function are missing: " + IMPORT_ERROR)

        except:
            logger.exception(MODULE_NAME + " configure failed to load")

    def chkAutoClicked(self, state):
        self.mw.settings.setValue(self.autoKey, state)
        self.txtFilename.setEnabled(not self.chkAuto.isChecked())

    def chkTrackClicked(self, state):
        self.mw.settings.setValue(self.trackKey, state)
        self.chkShowID.setVisible(state)

    def chkShowIDClicked(self, state):
        self.mw.settings.setValue(self.showIDKey, state)

    def chkLogCountClicked(self, state):
        self.mw.settings.setValue(self.logCountKey, state)

    def countIntervalChanged(self, txt):
        self.mw.settings.setValue(self.countIntervalKey, txt)

    def getCountInterval(self):
        result = 0
        if len(self.txtCountInterval.text()) > 0:
            result = int(self.txtCountInterval.text())
        return result

    def onMediaStarted(self, n):
        self.first_pass = True

class VideoWorker:
    def __init__(self, mw):
        try:
            self.mw = mw
            self.last_ex = ""

            if self.mw.configure.name != MODULE_NAME or len(IMPORT_ERROR) > 0:
                return
            
            self.mw.signals.showWait.emit()
            device_name = "cpu"
            if torch.cuda.is_available():
                device_name = "cuda"
            self.device = torch.device(device_name)

            self.num_classes = 80

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
                cache = Path(self.ckpt_file)

                if not cache.is_file():
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    model_name = self.mw.configure.cmbType.currentText()
                    link = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/" + model_name + ".pth"
                    if os.path.split(sys.executable)[1] == "pythonw.exe":
                        torch.hub.download_url_to_file(link, self.ckpt_file, progress=False)
                    else:
                        torch.hub.download_url_to_file(link, self.ckpt_file)
            else:
                self.ckpt_file = self.mw.configure.txtFilename.text()

            self.model.load_state_dict(torch.load(self.ckpt_file, map_location="cpu")["model"])

            res = int(self.mw.configure.cmbRes.currentText())
            self.model(torch.zeros(1, 3, res, res).to(self.device))

            self.track_thresh = self.mw.configure.sldConfThre.value()
            self.track_buffer = 30
            self.match_thresh = 0.8
            framerate = self.mw.getVideoFrameRate()
            if framerate == 0: framerate = 30
            self.tracker = BYTETracker(self.track_thresh, self.track_buffer, self.match_thresh, frame_rate=framerate)

            self.count_interval_start = 0
            self.rts = 0
            self.log_filename = ""

            self.mw.signals.hideWait.emit()

        except:
            logger.exception(MODULE_NAME + " initialization failure")
            self.mw.signals.hideWait.emit()
            self.mw.signals.error.emit(MODULE_NAME + " initialization failure, please check logs for details")

    def __call__(self, F):
        try:
            img = np.array(F, copy=False)
            self.rts = F.m_rts

            if self.mw.configure.name != MODULE_NAME:
                return
            
            if self.mw.configure.first_pass:
                self.count_interval_start = self.rts
                self.mw.configure.first_pass = False
                self.log_filename = ""

            res = int(self.mw.configure.cmbRes.currentText())
            test_size = (res, res)
            ratio = min(test_size[0] / img.shape[0], test_size[1] / img.shape[1])
            inf_shape = (int(img.shape[0] * ratio), int(img.shape[1] * ratio))
            bottom = test_size[0] - inf_shape[0]
            side = test_size[1] - inf_shape[1]
            pad = (0, 0, side, bottom)

            #timg = functional.to_tensor(img.copy()).to(self.device)
            timg = functional.to_tensor(img).to(self.device)
            timg *= 255
            timg = functional.resize(timg, inf_shape)
            timg = functional.pad(timg, pad, 114)
            timg = timg.unsqueeze(0)

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

            if outputs[0] is not None:
                output = outputs[0].cpu()
                if self.mw.configure.chkTrack.isChecked():
                    labels = output[:, 6].numpy().astype(int)
                    mask = np.in1d(labels, label_filter)
                    output = output[mask]
                    output = output.cpu().numpy()
                    
                    if self.track_thresh != self.mw.configure.sldConfThre.value():
                        self.track_thresh = self.mw.configure.sldConfThre.value()
                        framerate = self.mw.getVideoFrameRate()
                        if framerate == 0: framerate = 30
                        self.tracker = BYTETracker(self.track_thresh, self.track_buffer, self.match_thresh, frame_rate=framerate)

                    online_targets = self.tracker.update(output, [img.shape[0], img.shape[1]], test_size)
                    self.draw_track_boxes(img, online_targets)
                else:
                    self.draw_plain_boxes(img, output, ratio)

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

    def draw_plain_boxes(self, img, output, ratio):
        interval = self.mw.configure.getCountInterval()
        q_size = self.mw.getVideoFrameRate() * interval

        boxes = output[:, 0:4] / ratio
        labels = output[:, 6].numpy().astype(int)

        for lbl in self.mw.configure.labels:
            if lbl.chkBox.isChecked():
                lbl_boxes = boxes[labels == lbl.label()].numpy().astype(int)
                for box in lbl_boxes:
                    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), lbl.color(), 2)
                if self.mw.configure.getCountInterval() > 0:
                    lbl.avgCount(lbl_boxes.shape[0], q_size)
                else:
                    lbl.setCount(lbl_boxes.shape[0])

        if  self.rts - self.count_interval_start >= interval * 1000:
            self.count_interval_start = self.rts
            if self.mw.configure.chkLogCount.isChecked():
                self.writeLog()
            else:
                self.log_filename = ""

    def draw_track_boxes(self, img, online_targets):
        interval = self.mw.configure.getCountInterval()
        q_size = self.mw.getVideoFrameRate() * interval
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
            if self.mw.configure.chkShowID.isChecked():
                cv2.putText(img, id_text, (x, y), cv2.FONT_HERSHEY_PLAIN, 2, label_colors[t.label], 2)

        for lbl in self.mw.configure.labels:
            if lbl.chkBox.isChecked():
                if self.mw.configure.getCountInterval() > 0:
                    lbl.avgCount(count[lbl.label()], q_size)
                else:
                    lbl.setCount(count[lbl.label()])

        if  self.rts - self.count_interval_start >= interval * 1000:
            self.count_interval_start = self.rts
            if self.mw.configure.chkLogCount.isChecked():
                self.writeLog()
            else:
                self.log_filename = ""

    def writeLog(self):
        if len(self.log_filename) == 0:
            self.log_filename = self.mw.get_log_filename()
            dir = os.path.dirname(self.log_filename)
            if not os.path.exists(dir):
                os.makedirs(dir)
            if not os.path.exists(self.log_filename):
                with open(self.log_filename, "a") as f:
                    f.write("milliseconds, timestamp, class, count\n")
        for lbl in self.mw.configure.labels:
            if lbl.chkBox.isChecked():
                if self.mw.configure.chkLogCount.isChecked():
                    msg = str(self.rts) + " , "
                    msg += datetime.now().strftime("%m/%d/%Y %H:%M:%S") + " , "
                    msg += lbl.cmbLabel.currentText() + " , "
                    msg += lbl.lblCount.text() + "\n"
                    with open(self.log_filename, "a") as f: 
                        f.write(msg)

    def get_auto_ckpt_filename(self):
        return torch.hub.get_dir() + "/checkpoints/" + self.mw.configure.cmbType.currentText() + ".pth"

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
