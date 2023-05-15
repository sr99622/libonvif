IMPORT_ERROR = ""
try:
    import cv2
    import os
    import sys
    import numpy as np
    from loguru import logger

    from gui.components import ComboSelector, FileSelector, LabelSelector, ThresholdSlider
    from PyQt6.QtWidgets import QWidget, QGridLayout, QLabel, QCheckBox, QMessageBox
    from PyQt6.QtCore import Qt

    import cv2
    import torch
    from numpy import random

    sys.path.append("yolov7")
    from models.experimental import attempt_load
    from utils.datasets import letterbox
    from utils.general import check_img_size, non_max_suppression, scale_coords
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
            self.chkTrack.setChecked(int(self.mw.settings.value(self.trackKey, 0)))
            self.chkTrack.stateChanged.connect(self.chkTrackClicked)

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

            if len(IMPORT_ERROR) > 0:
                QMessageBox.critical(None, MODULE_NAME + " Import Error", "Modules required for running this function are missing: " + IMPORT_ERROR)

        except:
            logger.exception(MODULE_NAME + " configure failed to load")

    def chkAutoClicked(self, state):
        self.mw.settings.setValue(self.autoKey, state)
        self.txtFilename.setEnabled(not self.chkAuto.isChecked())

    def chkFP16Clicked(self, state):
        self.mw.settings.setValue(self.fp16Key, state)

    def chkTrackClicked(self, state):
        self.mw.settings.setValue(self.trackKey, state)
        self.sldConfThre.setEnabled(not self.chkTrack.isChecked())

class VideoWorker:
    def __init__(self, mw):
        try:
            self.mw = mw
            self.last_ex = ""

            weights=['yolov7.pt']
            self.imgsz=640
            self.conf_thres=0.25
            self.iou_thres=0.45
            self.classes = [0, 2]

            self.device = select_device('')
            self.half = self.device.type != 'cpu'
            self.model = attempt_load(weights, map_location=self.device)
            self.stride = int(self.model.stride.max())
            self.imgsz = check_img_size(self.imgsz, s=self.stride)
            if self.half:
                self.model.half()
            self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
            self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
            with torch.no_grad():
                self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))

        except:
            logger.exception(MODULE_NAME + " initialization failure")

    def __call__(self, F):
        try:
            original_img = np.array(F, copy=False)
            img = letterbox(original_img, self.imgsz, stride=self.stride)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()
            img /= 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            with torch.no_grad():
                pred = self.model(img, augment=False)[0]

            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=False)

            boxes = pred[0]
            if len(boxes):
                boxes[:, :4] = scale_coords(img.shape[2:], boxes[:, :4], original_img.shape).round()
                boxes = boxes.cpu().numpy()

                for box in boxes:
                    x1, y1, x2, y2 = box[:4].astype(int)
                    cls = box[5].astype(int)
                    cv2.rectangle(original_img, (x1, y1), (x2, y2), self.colors[cls], 2)
            else:
                print("len(boxes) == 0")

        except Exception as ex:
            if self.last_ex != str(ex) and self.mw.configure.name == MODULE_NAME:
                logger.exception(MODULE_NAME + " runtime error")
            self.last_ex = str(ex)

        