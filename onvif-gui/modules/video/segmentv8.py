#/********************************************************************
# onvif-gui/modules/video/segmentv8.py 
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
    import cv2
    from loguru import logger
    if sys.platform == "win32":
        filename = os.environ['HOMEPATH'] + "/.cache/onvif-gui/errors.txt"
    else:
        filename = os.environ['HOME'] + "/.cache/onvif-gui/errors.txt"
    logger.add(filename, retention="10 days")

    import cv2
    import numpy as np
    from pathlib import Path
    from gui.components import ComboSelector, FileSelector, LabelSelector, ThresholdSlider
    from PyQt6.QtWidgets import QWidget, QGridLayout, QLabel, QCheckBox, QMessageBox, QLineEdit
    from PyQt6.QtCore import Qt

    import torch
    from ultralytics import YOLO

except ModuleNotFoundError as ex:
    IMPORT_ERROR = str(ex)
    logger.debug("Import Error has occurred, missing modules need to be installed, please consult documentation: " + IMPORT_ERROR)

os.environ['KMP_DUPLICATE_LIB_OK']='True'
MODULE_NAME = "segmentv8"

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
            self.showBoxKey = "Module/" + MODULE_NAME + "/showBox"

            self.model_names = {"nano" : "yolov8n-seg.pt", "small" : "yolov8s-seg.pt", "medium" : "yolov8m-seg.pt", "large" : "yolov8l-seg.pt", "XL" : "yolov8x-seg.pt"}

            self.chkAuto = QCheckBox("Automatically download model")
            self.chkAuto.setChecked(int(self.mw.settings.value(self.autoKey, 1)))
            self.chkAuto.stateChanged.connect(self.chkAutoClicked)

            self.cmbModel = ComboSelector(mw, "Model Name", self.model_names.keys(), "small", MODULE_NAME)
            self.cmbModel.setEnabled(self.chkAuto.isChecked())

            self.cmbRes = ComboSelector(mw, "Model Size", ("320", "480", "640", "960", "1280", "1440", "1920"), "640", MODULE_NAME)

            self.txtModelFile = FileSelector(mw, MODULE_NAME)
            self.txtModelFile.setEnabled(not self.chkAuto.isChecked())

            self.sldConfThre = ThresholdSlider(mw, MODULE_NAME + "/confidence", "Confidence", 25)

            self.chkShowBox = QCheckBox("Show Detection Box")
            self.chkShowBox.setChecked(int(self.mw.settings.value(self.showBoxKey, 0)))
            self.chkShowBox.stateChanged.connect(self.chkShowBoxClicked)

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
            lblPanel = QLabel("Select classes to be segmented")
            lblPanel.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lytLabels.addWidget(lblPanel,        0, 0, 1, 1)
            for i in range(number_of_labels):
                self.labels[i].btnColor.setVisible(self.chkShowBox.isChecked())
                lytLabels.addWidget(self.labels[i], i+1, 0, 1, 1)
            lytLabels.setContentsMargins(0, 0, 0, 0)

            lytMain = QGridLayout(self)
            lytMain.addWidget(self.chkAuto,      0, 0, 1, 2)
            lytMain.addWidget(self.cmbModel,     1, 0, 1, 2)
            lytMain.addWidget(self.txtModelFile, 2, 0, 1, 2)
            lytMain.addWidget(self.cmbRes,       3, 0, 1, 2)
            lytMain.addWidget(self.sldConfThre,  4, 0, 1, 2)
            lytMain.addWidget(self.chkShowBox,   5, 0, 1, 2)
            lytMain.addWidget(self.chkTrack,     6, 0, 1, 1)
            lytMain.addWidget(self.chkShowID,    6, 1, 1, 1)
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

    def chkShowBoxClicked(self, state):
        self.mw.settings.setValue(self.showBoxKey, state)
        for label in self.labels:
            label.btnColor.setVisible(state)

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
            self.model(np.zeros([1280, 720, 3]), imgsz=1920, verbose=False)
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
            
            label_counts = {}
            label_filter = []
            for lbl in self.mw.configure.labels:
                if lbl.chkBox.isChecked():
                    label_filter.append(lbl.label())
                    label_counts[lbl.label()] = 0

            if self.model_name != self.mw.configure.getModelName():
                self.model_name = self.mw.configure.getModelName()
                with torch.no_grad():
                    torch.cuda.empty_cache()
                self.__init__(self.mw)

            confthre = self.mw.configure.sldConfThre.value()
            res = int(self.mw.configure.cmbRes.currentText())

            composite = None
            results = None

            if self.mw.configure.chkTrack.isChecked():
                results = self.model.track(img, stream=True, verbose=False,
                                classes=label_filter, persist=True,
                                conf=confthre, imgsz=res)
            else:
                results = self.model.predict(img, stream=True, verbose=False,
                                classes=label_filter, conf=confthre, imgsz=res)
            
            boxes = None
            for result in results:
                boxes = result.boxes.cpu().numpy()
                if result.masks is not None:
                    for mask in result.masks:
                        m = torch.squeeze(mask.data)
                        if composite is None:
                            composite = torch.zeros_like(m).cuda()
                        composite += m

                    composite = torch.gt(composite, 0)
                    composite = torch.stack((composite, composite, composite), 2).cpu().numpy().astype(np.uint8)
                    composite = cv2.resize(composite, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

            if composite is not None:
                img *= composite

            if self.mw.configure.chkShowBox.isChecked():
                if boxes is not None:
                    for box in boxes:
                        r = box.xyxy[0].astype(int)
                        class_id = int(box.cls[0])
                        class_color = self.mw.configure.getLabel(class_id).color()
                        if self.mw.configure.chkTrack.isChecked():
                            id_text = '{}'.format(int(box.id)).zfill(4)
                            box_color = (int((37 * box.id) % 255), int((17 * box.id) % 255), int((29 * box.id) % 255))
                            cv2.rectangle(img, r[:2], r[2:], box_color, 2)
                            if self.mw.configure.chkShowID.isChecked():
                                cv2.putText(img, id_text, r[:2], cv2.FONT_HERSHEY_PLAIN, 2, class_color, 2)
                        else:
                            cv2.rectangle(img, r[:2], r[2:], class_color, 2)

        except Exception as ex:
            if self.last_ex != str(ex) and self.mw.configure.name == MODULE_NAME:
                logger.exception(MODULE_NAME + " runtime error")
            self.last_ex = str(ex)

    def get_auto_ckpt_filename(self):
        return torch.hub.get_dir() +  "/checkpoints/" + self.mw.configure.getModelName()

