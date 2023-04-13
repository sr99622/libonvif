# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch
import os
import cv2
from loguru import logger
from sys import platform
from pathlib import Path
from detectron2.config import get_cfg
from detectron2.predictor import Predictor
from detectron2.tracker import DetectedInstance, SimpleTracker
from PyQt6.QtWidgets import QWidget, QGridLayout, QLabel
from PyQt6.QtCore import Qt
from components import ThresholdSlider, LabelSelector

# constants
MODULE_NAME = "detectron2/segment"

class Configure(QWidget):
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

        except:
            logger.exception("instance segmentation configuration load error")

    def disableThresholdSlider(self):
        self.sldThreshold.setEnabled(False)

    def enableThresholdSlider(self):
        self.sldThreshold.setEnabled(True)

class Worker:
    def __init__(self, mw):
        try:
            self.mw = mw
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
                        torch.hub.download_url_to_file("https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl", ckpt_file)

            cfg = get_cfg()
            cfg.merge_from_file('./detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
            cfg.MODEL.WEIGHTS = ckpt_file

            self.CONFIDENCE_THRESHOLD = self.mw.configure.sldThreshold.value()
            cfg.MODEL.RETINANET.SCORE_THRESH_TEST = self.CONFIDENCE_THRESHOLD
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.CONFIDENCE_THRESHOLD
            cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = self.CONFIDENCE_THRESHOLD
            cfg.freeze()

            self.tracker = SimpleTracker()
            self.predictor = Predictor(cfg, fp16)

            self.first_pass = False
            self.mw.signals.stopped.connect(self.stopped)

        except:
            logger.exception("Instance Segmentation initialization error")

    def __call__(self, F):
        try:

            if self.first_pass:
                if self.mw.configure.sldThreshold.value() != self.CONFIDENCE_THRESHOLD:
                    self.predictor = None
                    self.__init__(self.mw)

            img = np.array(F, copy=False)

            predictions = self.predictor(img)["instances"]

            test_classes = predictions.pred_classes.cpu().numpy()
            test_boxes = predictions.pred_boxes.tensor.cpu().numpy().astype(int)

            if self.mw.configure.name != MODULE_NAME:
                return

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

        except:
            logger.exception("Instance Segmentation runtime error")

    def get_auto_ckpt_filename(self):
        filename = None
        if platform == "win32":
            filename = os.environ['HOMEPATH'] + "/.cache/torch/hub/checkpoints/model_final_f10217.pkl"
        else:
            filename = os.environ['HOME'] + "/.cache/torch/hub/checkpoints/model_final_f10217.pkl"
        return filename


    def stopped(self):
        print("segment stopped")
        self.first_pass = True