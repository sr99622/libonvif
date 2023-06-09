#/********************************************************************
# onvif-gui/modules/video/yolov8.py 
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
    from datetime import datetime
    from pathlib import Path
    from gui.components import ComboSelector, FileSelector, LabelSelector, ThresholdSlider
    from PyQt6.QtWidgets import QWidget, QGridLayout, QLabel, QCheckBox, QMessageBox, QLineEdit
    from PyQt6.QtCore import Qt, QRegularExpression
    from PyQt6.QtGui import QRegularExpressionValidator

    import torch
    from ultralytics import YOLO
    from tracker.byte_tracker import BYTETracker

except ModuleNotFoundError as ex:
    IMPORT_ERROR = str(ex)
    logger.debug("Import Error has occurred, missing modules need to be installed, please consult documentation: " + IMPORT_ERROR)

os.environ['KMP_DUPLICATE_LIB_OK']='True'
MODULE_NAME = "yolov8"

class VideoConfigure(QWidget):
    def __init__(self, mw):
        try:
            super().__init__()
            self.mw = mw
            self.name = MODULE_NAME
            self.last_ex = ""
            self.autoKey = "Module/" + MODULE_NAME + "/autoDownload"
            self.trackKey = "Module/" + MODULE_NAME + "/track"
            self.showIDKey = "Module/" + MODULE_NAME + "/showID"
            self.logCountKey = "Module/" + MODULE_NAME + "/logCount"
            self.countIntervalKey = "Module/" + MODULE_NAME + "/countInterval"

            self.mw.signals.started.connect(self.onMediaStarted)
            
            self.model_names = {"nano" : "yolov8n.pt", "small" : "yolov8s.pt", "medium" : "yolov8m.pt", "large" : "yolov8l.pt", "XL" : "yolov8x.pt"}

            self.chkAuto = QCheckBox("Automatically download model")
            self.chkAuto.setChecked(int(self.mw.settings.value(self.autoKey, 1)))
            self.chkAuto.stateChanged.connect(self.chkAutoClicked)

            self.cmbModel = ComboSelector(mw, "Model Name", self.model_names.keys(), "nano", MODULE_NAME)
            self.cmbModel.setEnabled(self.chkAuto.isChecked())

            self.cmbRes = ComboSelector(mw, "Model Size", ("320", "480", "640", "960", "1280", "1440"), "320", MODULE_NAME)

            self.txtModelFile = FileSelector(mw, MODULE_NAME)
            self.txtModelFile.setEnabled(not self.chkAuto.isChecked())

            self.sldConfThre = ThresholdSlider(mw, MODULE_NAME + "/confidence", "Confidence", 25)

            self.chkTrack = QCheckBox("Track Objects")
            self.chkTrack.setChecked(int(self.mw.settings.value(self.trackKey, 0)))
            self.chkTrack.stateChanged.connect(self.chkTrackClicked)

            self.chkShowID = QCheckBox("Show Object ID")
            self.chkShowID.setChecked(int(self.mw.settings.value(self.showIDKey, 1)))
            self.chkShowID.stateChanged.connect(self.chkShowIDClicked)

            if not self.chkTrack.isChecked():
                self.chkShowID.setVisible(False)

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
            lytMain.addWidget(self.cmbModel,     1, 0, 1, 2)
            lytMain.addWidget(self.txtModelFile, 2, 0, 1, 2)
            lytMain.addWidget(self.cmbRes,       3, 0, 1, 2)
            lytMain.addWidget(self.sldConfThre,  4, 0, 1, 2)
            lytMain.addWidget(self.chkTrack,     5, 0, 1, 1)
            lytMain.addWidget(self.chkShowID,    5, 1, 1, 1)
            lytMain.addWidget(pnlCount,          6, 0, 1, 2)
            lytMain.addWidget(pnlLabels,         7, 0, 1, 2)
            lytMain.addWidget(QLabel(),          8, 0, 1, 2)
            lytMain.setRowStretch(8, 10)

            self.first_pass = True

            if len(IMPORT_ERROR) > 0:
                QMessageBox.critical(None, MODULE_NAME + " Import Error", "Modules required for running this function are missing: " + IMPORT_ERROR)

        except:
            logger.exception(MODULE_NAME + " configure failed to load")

    def chkAutoClicked(self, state):
        self.mw.settings.setValue(self.autoKey, state)
        self.cmbModel.setEnabled(self.chkAuto.isChecked())
        self.txtModelFile.setEnabled(not self.chkAuto.isChecked())

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

    def getModelName(self):
        if self.chkAuto.isChecked():
            return self.model_names[self.cmbModel.currentText()]
        else:
            return self.txtModelFile.text()
        
    def getLabel(self, class_id):
        for lbl in self.labels:
            if lbl.label() == class_id:
                return lbl
            
class VideoWorker:
    def __init__(self, mw):
        try:
            self.mw = mw
            self.last_ex = ""

            if self.mw.configure.name != MODULE_NAME or len(IMPORT_ERROR) > 0:
                return
            
            self.mw.signals.showWait.emit()

            self.ckpt_file = None
            if self.mw.configure.chkAuto.isChecked():
                self.ckpt_file = self.get_auto_ckpt_filename()
                cache = Path(self.ckpt_file)

                if not cache.is_file():
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    model_name = self.mw.configure.getModelName()
                    link = "https://github.com/ultralytics/assets/releases/download/v0.0.0/" + model_name
                    if os.path.split(sys.executable)[1] == "pythonw.exe":
                        torch.hub.download_url_to_file(link, self.ckpt_file, progress=False)
                    else:
                        torch.hub.download_url_to_file(link, self.ckpt_file)
            else:
                self.ckpt_file = self.configure.txtFilename.text()

            self.model_name = self.mw.configure.getModelName()
            self.model = YOLO(Path(self.ckpt_file))
            self.model.predict(np.zeros([1280, 720, 3], dtype=np.uint8), stream=True, verbose=False)

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

            label_counts = {}
            label_filter = []
            for lbl in self.mw.configure.labels:
                if lbl.chkBox.isChecked():
                    label_filter.append(lbl.label())
                    label_counts[lbl.label()] = 0

            confthre = self.mw.configure.sldConfThre.value()
            if self.mw.configure.chkTrack.isChecked():
                confthre = 0.001

            if self.model_name != self.mw.configure.getModelName():
                self.model_name = self.mw.configure.getModelName()
                with torch.no_grad():
                    torch.cuda.empty_cache()
                self.__init__(self.mw)

            res = int(self.mw.configure.cmbRes.currentText())
                
            results = self.model.predict(img, stream=True, verbose=False, 
                                         classes=label_filter,
                                         conf=confthre, 
                                         imgsz=res)

            interval = self.mw.configure.getCountInterval()
            q_size = self.mw.getVideoFrameRate() * interval

            for result in results:
                if self.mw.configure.chkTrack.isChecked():

                    output = result.boxes.xyxy
                    scores = result.boxes.conf.reshape(-1, 1)
                    labels = result.boxes.cls.reshape(-1, 1)
                    output = torch.hstack((output, scores))
                    output = torch.hstack((output, labels))
                    output = output.cpu().numpy()

                    if self.track_thresh != self.mw.configure.sldConfThre.value():
                        self.track_thresh = self.mw.configure.sldConfThre.value()
                        framerate = self.mw.getVideoFrameRate()
                        if framerate == 0: framerate = 30
                        self.tracker = BYTETracker(self.track_thresh, self.track_buffer, self.match_thresh, frame_rate=framerate)

                    online_targets = self.tracker.update(output, [img.shape[0] * res / img.shape[1], res], (res, res))

                    for t in online_targets:
                        label_counts[t.label] += 1

                        track_id = int(t.track_id)
                        id_text = '{}'.format(int(track_id)).zfill(5)
                        box_color = ((37 * track_id) % 255, (17 * track_id) % 255, (29 * track_id) % 255)

                        x, y, w, h = t.tlwh.astype(int)
                        cv2.rectangle(img, (x, y), (x+w, y+h), box_color, 2)
                        if self.mw.configure.chkShowID.isChecked():
                            label_color = self.mw.configure.getLabel(t.label).color()
                            cv2.putText(img, id_text, (x, y), cv2.FONT_HERSHEY_PLAIN, 2, label_color, 2)
                else:
                    boxes = result.boxes.cpu().numpy()
                    for box in boxes:
                        r = box.xyxy[0].astype(int)
                        class_id = int(box.cls[0])
                        color = self.mw.configure.getLabel(class_id).color()
                        label_counts[class_id] += 1
                        cv2.rectangle(img, r[:2], r[2:], color, 2)

            for lbl in self.mw.configure.labels:
                if lbl.isChecked():
                    if self.mw.configure.getCountInterval() > 0:
                        lbl.avgCount(label_counts[lbl.label()], q_size)
                    else:
                        lbl.setCount(label_counts[lbl.label()])

            if  self.rts - self.count_interval_start >= interval * 1000:
                self.count_interval_start = self.rts
                if self.mw.configure.chkLogCount.isChecked():
                    self.writeLog()
                else:
                    self.log_filename = ""

        except Exception as ex:
            if self.last_ex != str(ex) and self.mw.configure.name == MODULE_NAME:
                logger.exception(MODULE_NAME + " runtime error")
            self.last_ex = str(ex)

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
        return torch.hub.get_dir() +  "/checkpoints/" + self.mw.configure.getModelName()

