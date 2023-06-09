#/********************************************************************
# onvif-gui/modules/video/keypoint.py 
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

    import numpy as np
    import cv2
    import math
    from pathlib import Path
    from PyQt6.QtWidgets import QWidget, QGridLayout, QLabel, QMessageBox
    from gui.components import ThresholdSlider

    import torch
    from detectron2.config import get_cfg
    from detectron2.predictor import Predictor
    from detectron2.tracker import DetectedInstance, SimpleTracker

    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

except ModuleNotFoundError as ex:
    IMPORT_ERROR = str(ex)
    logger.debug("Import Error has occurred, missing modules need to be installed, please consult documentation: " + IMPORT_ERROR)

MODULE_NAME = "detectron2/keypoint"

class VideoConfigure(QWidget):
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

            if len(IMPORT_ERROR) > 0:
                QMessageBox.critical(None, "Detectron2 Import Error", "Modules required for running this function are missing: " + IMPORT_ERROR)

        except:
            logger.exception("keypoints configuration load error")

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
                        link = "https://dl.fbaipublicfiles.com/detectron2/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x/137849621/model_final_a6e10b.pkl"
                        if sys.platform == "win32":
                            torch.hub.download_url_to_file(link, ckpt_file, progress=False)
                        else:
                            torch.hub.download_url_to_file(link, ckpt_file)

            cfg = get_cfg()
            config_file = ""
            yaml_file = 'detectron2/configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml'
            if not os.path.isfile(yaml_file):
                for path in sys.path:
                    config_file = os.path.join(path, yaml_file)
                    if os.path.isfile(config_file):
                        break
            else:
                config_file = yaml_file

            cfg.merge_from_file(config_file)
            cfg.MODEL.RETINANET.SCORE_THRESH_TEST = self.CONFIDENCE_THRESHOLD
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.CONFIDENCE_THRESHOLD
            cfg.MODEL.WEIGHTS = ckpt_file
            cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = self.CONFIDENCE_THRESHOLD
            cfg.freeze()

            self.tracker = SimpleTracker()
            self.predictor = Predictor(cfg, fp16)
            self.predictor(np.zeros([1280, 720, 3]))

            self.mw.signals.hideWait.emit()

        except:
            logger.exception("Keypoints initialization error")
            self.mw.signals.hideWait.emit()
            self.mw.signals.error.emit("Keypoints initialization error, please check logs for details")


    def __call__(self, F):
        try:
            img = np.array(F, copy=False)

            if self.mw.configure.name != MODULE_NAME:
                return
            
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


        except Exception as ex:
            if self.last_ex != str(ex) and self.mw.configure.name == MODULE_NAME:
                logger.exception("Keypoints runtime error")
            self.last_ex = str(ex)

    def get_auto_ckpt_filename(self):
        filename = None
        if sys.platform == "win32":
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
