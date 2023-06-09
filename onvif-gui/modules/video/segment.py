#/********************************************************************
#libonvif/gui/modules/video/segment.py 
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
    from PyQt6.QtWidgets import QWidget, QGridLayout, QLabel, QMessageBox
    from PyQt6.QtCore import Qt
    from gui.components import ThresholdSlider, LabelSelector

    import torch
    from detectron2.config import get_cfg
    from detectron2.predictor import Predictor
    from detectron2.tracker import DetectedInstance, SimpleTracker

    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

except ModuleNotFoundError as ex:
    IMPORT_ERROR = str(ex)
    logger.debug("Import Error has occurred, missing modules need to be installed, please consult documentation: " + IMPORT_ERROR)

MODULE_NAME = "detectron2/segment"

class VideoConfigure(QWidget):
    def __init__(self, mw):
        try:
            super().__init__()
            self.mw = mw
            self.name = MODULE_NAME
            self.sldThreshold = ThresholdSlider(mw, MODULE_NAME, "Confidence", 50)

            self.number_of_labels = 5
            self.labels = []
            for i in range(self.number_of_labels):
                lbl = LabelSelector(mw, self.name, i)
                lbl.btnColor.setVisible(False)
                self.labels.append(lbl)

            pnlLabel = QWidget()
            lytLabels = QGridLayout(pnlLabel)
            lblPanel = QLabel("Select classes to be identified")
            lblPanel.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lytLabels.addWidget(lblPanel,    0, 0, 1, 1)
            for i in range(self.number_of_labels):
                lytLabels.addWidget(self.labels[i], i+1, 0, 1, 1)
            lytLabels.setContentsMargins(0, 0, 0, 0)

            lytMain = QGridLayout(self)
            lytMain.addWidget(self.sldThreshold,  0, 0, 1, 1)
            lytMain.addWidget(pnlLabel,           1, 0, 1, 1)
            lytMain.addWidget(QLabel(""),         2, 0, 1, 1)
            lytMain.setRowStretch(2, 10)

            self.mw.signals.started.connect(self.disableThresholdSlider)
            self.mw.signals.stopped.connect(self.enableThresholdSlider)
            if self.mw.playing:
                self.sldThreshold.setEnabled(False)

            if len(IMPORT_ERROR) > 0:
                QMessageBox.critical(None, "Detectron2 Import Error", "Modules required for running this function are missing: " + IMPORT_ERROR)

        except:
            logger.exception("instance segmentation configuration load error")

    def disableThresholdSlider(self):
        self.sldThreshold.setEnabled(False)

    def enableThresholdSlider(self):
        self.sldThreshold.setEnabled(True)

class VideoWorker:
    def __init__(self, mw):
        try:
            self.mw = mw
            self.last_ex = ""

            if self.mw.configure.name != MODULE_NAME or len(IMPORT_ERROR) > 0:
                return
            
            self.mw.signals.showWait.emit()

            self.CONFIDENCE_THRESHOLD = self.mw.configure.sldThreshold.value()
            ckpt_file = 'auto'
            fp16 = True
            self.simple = True
            self.draw_overlay = False
            self.first_pass = True

            if ckpt_file is not None:
                if ckpt_file.lower() == "auto":
                    ckpt_file = self.get_auto_ckpt_filename()
                    print("ckpt_file:", ckpt_file)
                    cache = Path(ckpt_file)

                    if not cache.is_file():
                        cache.parent.mkdir(parents=True, exist_ok=True)
                        link = "https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
                        if sys.platform == "win32":
                            torch.hub.download_url_to_file(link, ckpt_file, progress=False)
                        else:
                            torch.hub.download_url_to_file(link, ckpt_file)

            cfg = get_cfg()
            config_file = ""
            yaml_file = 'detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
            if not os.path.isfile(yaml_file):
                for path in sys.path:
                    config_file = os.path.join(path, yaml_file)
                    if os.path.isfile(config_file):
                        break
            else:
                config_file = yaml_file

            cfg.merge_from_file(config_file)
            cfg.MODEL.WEIGHTS = ckpt_file

            cfg.MODEL.RETINANET.SCORE_THRESH_TEST = self.CONFIDENCE_THRESHOLD
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.CONFIDENCE_THRESHOLD
            cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = self.CONFIDENCE_THRESHOLD
            cfg.freeze()

            self.tracker = SimpleTracker()
            self.predictor = Predictor(cfg, fp16)

            self.predictor(np.zeros([1280, 720, 3]))

            self.first_pass = False
            self.mw.signals.stopped.connect(self.stopped)

            self.mw.signals.hideWait.emit()

        except:
            logger.exception("Instance Segmentation initialization error")
            self.mw.signals.hideWait.emit()
            self.mw.signals.error.emit("Instance Segmentation initialization error, please check logs for details")

    def __call__(self, F):
        try:

            if self.first_pass:
                if self.mw.configure.sldThreshold.value() != self.CONFIDENCE_THRESHOLD:
                    self.predictor = None
                    self.__init__(self.mw)

            img = np.array(F, copy=False)

            if self.mw.configure.name != MODULE_NAME:
                return
            
            predictions = self.predictor(img)["instances"]

            test_classes = predictions.pred_classes.cpu().numpy()
            test_boxes = predictions.pred_boxes.tensor.cpu().numpy().astype(int)

            masks = []
            classes = []
            boxes = []
            filter_classes = []
            counts = {}
            for lbl in self.mw.configure.labels:
                if lbl.isChecked():
                    filter_classes.append(lbl.label())
                    counts[lbl.label()] = 0

            for i in range(len(test_classes)):
                if test_classes[i] in filter_classes:
                    classes.append(test_classes[i])
                    masks.append(predictions.pred_masks[i])
                    boxes.append(test_boxes[i])
                    counts[test_classes[i]] += 1

            for lbl in self.mw.configure.labels:
                if lbl.isChecked():
                    lbl.setCount(counts[lbl.label()])

            detected = []
            for i in range(len(classes)):
                detected.append(DetectedInstance(classes[i], boxes[i], mask_rle=None, color=None, ttl=8))
            if len(detected):
                colors = self.tracker.assign_colors(detected)
            
            composite = torch.zeros((img.shape[0], img.shape[1])).cuda()
            for mask in masks:
                composite += mask
            composite = torch.gt(composite, 0)
            composite = torch.stack((composite, composite, composite), 2).cpu().numpy().astype(np.uint8)
            img *= composite

            for index, box in enumerate(boxes):
                color = (colors[index] * 255).astype(int).tolist()
                #cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.line(img, (box[0], box[3]), (box[2], box[3]), color, 2)

        except Exception as ex:
            if self.last_ex != str(ex) and self.mw.configure.name == MODULE_NAME:
                logger.exception("Instance Segmentation runtime error")
            self.last_ex = str(ex)

    def get_auto_ckpt_filename(self):
        filename = None
        if sys.platform == "win32":
            filename = os.environ['HOMEPATH'] + "/.cache/torch/hub/checkpoints/model_final_f10217.pkl"
        else:
            filename = os.environ['HOME'] + "/.cache/torch/hub/checkpoints/model_final_f10217.pkl"
        return filename


    def stopped(self):
        print("segment stopped")
        self.first_pass = True