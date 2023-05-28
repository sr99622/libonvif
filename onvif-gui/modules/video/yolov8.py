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
    import cv2
    import sys
    import os
    import numpy as np
    from loguru import logger
    from pathlib import Path

    from gui.components import ComboSelector, FileSelector, LabelSelector, ThresholdSlider
    from PyQt6.QtWidgets import QWidget, QGridLayout, QLabel, QCheckBox, QMessageBox
    from PyQt6.QtCore import Qt, QObject, pyqtSignal

    import torch
    from ultralytics import YOLO
    from tracker.byte_tracker import BYTETracker

except ModuleNotFoundError as ex:
    IMPORT_ERROR = str(ex)
    print("Import Error has occurred, missing modules need to be installed, please consult documentation: ", ex)

os.environ['KMP_DUPLICATE_LIB_OK']='True'
MODULE_NAME = "yolov8"

class VideoConfigure(QWidget):
    def __init__(self, mw):
        try:
            super().__init__()
            self.mw = mw
            logger.add("errors.txt", retention="1 days")
            self.name = MODULE_NAME
            self.last_ex = ""
            self.autoKey = "Module/" + MODULE_NAME + "/autoDownload"
            self.trackKey = "Module/" + MODULE_NAME + "/track"
            self.showIDKey = "Module/" + MODULE_NAME + "/showID"

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
            lytMain.addWidget(self.chkAuto,      0, 0, 1, 2)
            lytMain.addWidget(self.cmbModel,     1, 0, 1, 2)
            lytMain.addWidget(self.txtModelFile, 2, 0, 1, 2)
            lytMain.addWidget(self.cmbRes,       3, 0, 1, 2)
            lytMain.addWidget(self.sldConfThre,  4, 0, 1, 2)
            lytMain.addWidget(self.chkTrack,     5, 0, 1, 1)
            lytMain.addWidget(self.chkShowID,    5, 1, 1, 1)
            lytMain.addWidget(pnlLabels,         6, 0, 1, 2)
            lytMain.addWidget(QLabel(),          7, 0, 1, 2)
            lytMain.setRowStretch(7, 10)

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

            self.ckpt_file = None
            if self.mw.configure.chkAuto.isChecked():
                self.ckpt_file = self.get_auto_ckpt_filename()
                cache = Path(self.ckpt_file)

                if not cache.is_file():
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    model_name = self.mw.configure.getModelName()
                    link = "https://github.com/ultralytics/assets/releases/download/v0.0.0/" + model_name
                    if os.path.split(sys.executable)[1] == "pythonw.exe":
                        self.mw.signals.showWait.emit()
                        torch.hub.download_url_to_file(link, self.ckpt_file, progress=False)
                        self.mw.signals.hideWait.emit()
                    else:
                        torch.hub.download_url_to_file(link, self.ckpt_file)
            else:
                self.ckpt_file = self.configure.txtFilename.text()

            self.model_name = self.mw.configure.getModelName()
            self.model = YOLO(Path(self.ckpt_file))

            self.track_thresh = self.mw.configure.sldConfThre.value()
            self.track_buffer = 30
            self.match_thresh = 0.8

            self.tracker = BYTETracker(self.track_thresh, self.track_buffer, self.match_thresh)

        except:
            logger.exception(MODULE_NAME + " initialization failure")

    def __call__(self, F):
        try:
            img = np.array(F, copy=False)

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
                        self.tracker = BYTETracker(self.track_thresh, self.track_buffer, self.match_thresh)

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

                for lbl in label_filter:
                    self.mw.configure.getLabel(lbl).setCount(label_counts[lbl])
            
        except Exception as ex:
            if self.last_ex != str(ex) and self.mw.configure.name == MODULE_NAME:
                logger.exception(MODULE_NAME + " runtime error")
            self.last_ex = str(ex)

    def get_auto_ckpt_filename(self):
        return torch.hub.get_dir() +  "/checkpoints/" + self.mw.configure.getModelName()

