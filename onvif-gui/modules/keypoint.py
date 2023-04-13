# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch
import cv2
import math
import os
from loguru import logger
from sys import platform
from pathlib import Path
from detectron2.config import get_cfg
from detectron2.predictor import Predictor
from detectron2.tracker import DetectedInstance, SimpleTracker
from PyQt6.QtWidgets import QWidget, QGridLayout, QLabel
from components import ThresholdSlider

# constants
CONFIDENCE_THRESHOLD = 0.50
MODULE_NAME = "detectron2/keypoint"

class Configure(QWidget):
    def __init__(self, mw):
        try:
            super().__init__()
            self.mw = mw
            self.name = MODULE_NAME
            self.sldThreshold = ThresholdSlider(mw, MODULE_NAME, "Confidence", 50)
            lytMain = QGridLayout(self)
            lytMain.addWidget(self.sldThreshold,   0, 0, 1, 1)
            lytMain.addWidget(QLabel(""),          1, 0, 1, 1)
            lytMain.setRowStretch(1, 10)

            self.mw.signals.started.connect(self.disableThresholdSlider)
            self.mw.signals.stopped.connect(self.enableThresholdSlider)
            if self.mw.playing:
                self.sldThreshold.setEnabled(False)

        except:
            logger.exception("keypoints configuration load error")

    def disableThresholdSlider(self):
        self.sldThreshold.setEnabled(False)

    def enableThresholdSlider(self):
        self.sldThreshold.setEnabled(True)

class Worker:
    def __init__(self, mw):
        try:
            self.mw = mw
            ckpt_file = "auto"
            fp16 = True
            self.no_back = True
            self.simple = True

            if ckpt_file is not None:
                if ckpt_file.lower() == "auto":
                    ckpt_file = self.get_auto_ckpt_filename()
                    print("ckpt_file:", ckpt_file)
                    cache = Path(ckpt_file)

                    if not cache.is_file():
                        cache.parent.mkdir(parents=True, exist_ok=True)
                        torch.hub.download_url_to_file("https://dl.fbaipublicfiles.com/detectron2/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x/137849621/model_final_a6e10b.pkl", ckpt_file)

            cfg = get_cfg()
            cfg.merge_from_file('./detectron2/configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml')
            cfg.MODEL.RETINANET.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
            #cfg.MODEL.WEIGHTS = './detectron2/models/COCO-Keypoints/model_final_a6e10b.pkl'
            cfg.MODEL.WEIGHTS = ckpt_file
            cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = CONFIDENCE_THRESHOLD
            cfg.freeze()

            self.tracker = SimpleTracker()
            self.predictor = Predictor(cfg, fp16)
        except:
            logger.exception("Keypoints initialization error")


    def __call__(self, F):
        try:
            img = np.array(F, copy=False)

            predictions = self.predictor(img)["instances"].to(torch.device('cpu'))
            keypoints = predictions.pred_keypoints.numpy().astype(int)

            if self.no_back:
                img *= 0

            boxes = predictions.pred_boxes.tensor.numpy()
            classes = predictions.pred_classes.numpy()

            detected = []
            for i in range(len(predictions)):
                detected.append(DetectedInstance(classes[i], boxes[i], mask_rle=None, color=None, ttl=8))

            if self.mw.configure.name != MODULE_NAME:
                return

            if len(detected):
                colors = np.asarray(self.tracker.assign_colors(detected)) * 255
                colors = colors.astype(np.int32)
                for idx, keypoint in enumerate(keypoints):
                    self.draw_keypoint(img, keypoint, colors[idx])

        except:
            logger.exception("Keypoints runtime error")

    def get_auto_ckpt_filename(self):
        filename = None
        if platform == "win32":
            filename = os.environ['HOMEPATH'] + "/.cache/torch/hub/checkpoints/model_final_a6e10b.pkl"
        else:
            filename = os.environ['HOME'] + "/.cache/torch/hub/checkpoints/model_final_a6e10b.pkl"
        return filename

    def draw_keypoint(self, img, keypoint, color):
        color = (int(color[0]), int(color[1]), int(color[2]))
        kp = keypoint[:, :2]
        nose = kp[0]
        left_eye = kp[1]
        right_eye = kp[2]
        left_ear = kp[3]
        right_ear = kp[4]
        left_shoulder = kp[5]
        right_shoulder = kp[6]
        left_elbow = kp[7]
        right_elbow = kp[8]
        left_wrist = kp[9]
        right_wrist = kp[10]
        left_hip = kp[11]
        right_hip = kp[12]
        left_knee = kp[13]
        right_knee = kp[14]
        left_ankle = kp[15]
        right_ankle = kp[16]

        mid_hip = (left_hip[0] - int((left_hip[0] - right_hip[0]) / 2),
            left_hip[1] - int((left_hip[1] - right_hip[1]) / 2))

        mid_shoulder = (left_shoulder[0] - int((left_shoulder[0] - right_shoulder[0]) / 2),
            left_shoulder[1] - int((left_shoulder[1] - right_shoulder[1]) / 2))

        a = left_ear[0] - right_ear[0]
        b = left_ear[1] - right_ear[1]
        c = math.sqrt(a*a + b*b)
        head_radius = int(c/2)

        cv2.line(img, left_ankle, left_knee, color, 3)
        cv2.line(img, right_ankle, right_knee, color, 3)
        cv2.line(img, left_knee, left_hip, color, 3)
        cv2.line(img, right_knee, right_hip, color, 3)
        cv2.line(img, left_hip, right_hip, color, 3)
        cv2.line(img, mid_hip, mid_shoulder, color, 3)
        cv2.line(img, left_shoulder, right_shoulder, color, 3)
        cv2.line(img, left_shoulder, left_elbow, color, 3)
        cv2.line(img, right_shoulder, right_elbow, color, 3)
        cv2.line(img, left_elbow, left_wrist, color, 3)
        cv2.line(img, right_elbow, right_wrist, color, 3)
        cv2.circle(img, nose, head_radius, color, 3)
