#/********************************************************************
# onvif-gui/modules/video/RyzenAI.py 
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
MODULE_NAME = "RyzenAI"

try:
    import os
    import cv2
    from loguru import logger
    import numpy as np
    from gui.components import ComboSelector, FileSelector, ThresholdSlider, TargetSelector
    from gui.enums import MediaSource
    from PyQt6.QtWidgets import QWidget, QGridLayout, QLabel, QCheckBox, QMessageBox, \
        QGroupBox, QDialog, QSpinBox
    from PyQt6.QtCore import Qt, QSize, QObject, pyqtSignal
    from PyQt6.QtGui import QMovie
    import time
    import torch
    import sys
    from pathlib import Path
    import onnxruntime as ort

except ModuleNotFoundError as ex:
    IMPORT_ERROR = str(ex)
    logger.debug("Import Error has occurred, missing modules need to be installed, please consult documentation: ", ex)
    QMessageBox.critical(None, MODULE_NAME + " Import Error", "Modules required for running this function are missing: " + IMPORT_ERROR)

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class RyzenAIWaitDialog(QDialog):
    def __init__(self, p):
        super().__init__(p)
        self.lblMessage = QLabel("Please wait for model to download")
        self.lblProgress = QLabel()
        self.movie = QMovie("image:spinner.gif")
        self.movie.setScaledSize(QSize(50, 50))
        self.lblProgress.setMovie(self.movie)
        self.setWindowTitle("yolox")

        lytMain = QGridLayout(self)
        lytMain.addWidget(self.lblMessage,  0, 1, 1, 1, Qt.AlignmentFlag.AlignCenter)
        lytMain.addWidget(self.lblProgress, 1, 1, 1, 1, Qt.AlignmentFlag.AlignCenter)

        self.movie.start()
        self.setModal(True)

    def sizeHint(self):
        return QSize(300, 100)
    
class RyzenAISettings():
    def __init__(self, mw, camera=None):
        self.camera = camera
        self.mw = mw
        self.id = "File"
        if camera:
            self.id = camera.serial_number()

        self.targets = self.getTargetsForPlayer()
        self.limit = self.getModelOutputLimit()
        self.confidence = self.getModelConfidence()
        self.show = self.getModelShowBoxes()
        self.skipFrames = self.getSkipFrames()
        self.skipCounter = 0
        self.sampleSize = self.getSampleSize()
        self.orig_img = None

    def getTargets(self):
        key = f'{self.id}/{MODULE_NAME}/Targets'
        return str(self.mw.settings.value(key, "")).strip()
    
    def getTargetsForPlayer(self):
        var = self.getTargets()
        ary = []
        if len(var):
            tmp = var.split(":")
            for t in tmp:
                ary.append(int(t))
        return ary    

    def setTargets(self, targets):
        key = f'{self.id}/{MODULE_NAME}/Targets'
        self.targets.clear()
        if len(targets):
            tmp = targets.split(":")
            for t in tmp:
                self.targets.append(int(t))
        self.mw.settings.setValue(key, targets)

    def getModelConfidence(self):
        key = f'{self.id}/{MODULE_NAME}/ConfidenceThreshold'
        return int(self.mw.settings.value(key, 50))
    
    def setModelConfidence(self, value):
        key = f'{self.id}/{MODULE_NAME}/ConfidenceThreshold'
        self.confidence = value
        self.mw.settings.setValue(key, value)

    def getModelOutputLimit(self):
        key = f'{self.id}/{MODULE_NAME}/ModelOutputLimit'
        return int(self.mw.settings.value(key, 0))
    
    def setModelOutputLimit(self, value):
        key = f'{self.id}/{MODULE_NAME}/ModelOutputLimit'
        self.limit = value
        self.mw.settings.setValue(key, value)

    def getModelShowBoxes(self):
        key = f'{self.id}/{MODULE_NAME}/ModelShowBoxes'
        return bool(int(self.mw.settings.value(key, 1)))
    
    def setModelShowBoxes(self, value):
        key = f'{self.id}/{MODULE_NAME}/ModelShowBoxes'
        self.show = value
        self.mw.settings.setValue(key, int(value))

    def getSkipFrames(self):
        key = f'{self.id}/{MODULE_NAME}/SkipFrames'
        return int(self.mw.settings.value(key, 0))

    def setSkipFrames(self, value):
        key = f'{self.id}/{MODULE_NAME}/SkipFrames'
        self.skipFrames = int(value)
        self.mw.settings.setValue(key, int(value))

    def getSampleSize(self):
        key = f'{self.id}/{MODULE_NAME}/SampleSize'
        return int(self.mw.settings.value(key, 1))

    def setSampleSize(self, value):
        key = f'{self.id}/{MODULE_NAME}/SampleSize'
        self.sampleSize = int(value)
        self.mw.settings.setValue(key, int(value))        

class RyzenAISignals(QObject):
    showWaitDialog = pyqtSignal()
    hideWaitDialog = pyqtSignal()


class VideoConfigure(QWidget):
    def __init__(self, mw):
        try:
            print("video configure init")
            super().__init__()
            self.mw = mw
            self.name = MODULE_NAME
            self.source = None
            self.media = None
            self.initialized = False
            self.autoKey = "Module/" + MODULE_NAME + "/autoDownload"

            if len(IMPORT_ERROR):
                self.mw.videoPanel.lblCamera.setText(f'Configuration error - {IMPORT_ERROR}')
                return
            
            self.dlgWait = RyzenAIWaitDialog(self.mw)
            self.signals = RyzenAISignals()
            self.signals.showWaitDialog.connect(self.showWaitDialog)
            self.signals.hideWaitDialog.connect(self.hideWaitDialog)
            
            self.chkAuto = QCheckBox("Automatically download model")
            self.chkAuto.setChecked(int(self.mw.settings.value(self.autoKey, 1)))
            self.chkAuto.stateChanged.connect(self.chkAutoClicked)
            self.txtFilename = FileSelector(mw, MODULE_NAME)
            self.txtFilename.setEnabled(not self.chkAuto.isChecked())
            self.cmbRes = ComboSelector(mw, "Size", ("640",), "640", MODULE_NAME)
            self.cmbModelName = ComboSelector(mw, "Name", ("yolox-s-int8",), "yolox-s-int8", MODULE_NAME)
            self.txtConfigFile = FileSelector(mw, "VAIP_CONFIG")
            self.txtConfigFile.lblSelect.setText("VAIP")
            
            apis = ["RyzenAI"]
            
            self.cmbAPI = ComboSelector(mw, "API", apis, "PyTorch", MODULE_NAME)
            self.cmbAPI.cmbBox.currentTextChanged.connect(self.cmbAPIChanged)

            self.cmbDevice = ComboSelector(mw, "Device", self.getDevices(self.cmbAPI.currentText()), "AUTO", MODULE_NAME)

            self.sldConfThre = ThresholdSlider(mw, "Confidence", MODULE_NAME)
            self.selTargets = TargetSelector(self.mw, MODULE_NAME)

            self.spnSkipFrames = QSpinBox()
            self.lblSkipFrames = QLabel("Skip Frames")
            self.spnSkipFrames.setValue(0)
            self.spnSkipFrames.valueChanged.connect(self.spnSkipFramesChanged)

            self.spnSampleSize = QSpinBox()
            self.lblSampleSize = QLabel("Sample Size")
            self.spnSampleSize.setMinimum(1)
            self.spnSampleSize.setValue(1)
            self.spnSampleSize.valueChanged.connect(self.spnSampleSizeChanged)

            grpSystem = QGroupBox("System wide model parameters")
            lytSystem = QGridLayout(grpSystem)
            lytSystem.addWidget(self.chkAuto,       0, 0, 1, 4)
            lytSystem.addWidget(self.txtFilename,   1, 0, 1, 4)
            lytSystem.addWidget(self.cmbModelName,  2, 0, 1, 2)
            lytSystem.addWidget(self.cmbRes,        2, 2, 1, 2)
            lytSystem.addWidget(self.cmbAPI,        3, 0, 1, 2)
            lytSystem.addWidget(self.cmbDevice,     3, 2, 1, 2)
            lytSystem.addWidget(self.txtConfigFile, 4, 0, 1, 4)

            self.grpCamera = QGroupBox("Check camera video alarm to enable")
            lytCamera = QGridLayout(self.grpCamera)
            lytCamera.addWidget(self.sldConfThre,    0, 0, 1, 4)
            lytCamera.addWidget(self.lblSkipFrames,  1, 0, 1, 1)
            lytCamera.addWidget(self.spnSkipFrames,  1, 1, 1, 1)
            lytCamera.addWidget(self.lblSampleSize,  1, 2, 1, 1)
            lytCamera.addWidget(self.spnSampleSize,  1, 3, 1, 1)
            lytCamera.addWidget(QLabel(),            2, 0, 1, 4)
            lytCamera.addWidget(self.selTargets,     3, 0, 1, 4)

            lytMain = QGridLayout(self)
            lytMain.addWidget(grpSystem,         0, 0, 1, 1)
            lytMain.addWidget(QLabel(),          1, 0, 1, 1)
            lytMain.addWidget(self.grpCamera,    2, 0, 1, 1)
            lytMain.addWidget(QLabel(),          3, 0, 1, 1)
            lytMain.setRowStretch(3, 10)

            self.enableControls(False)
            if camera := self.mw.cameraPanel.getCurrentCamera():
                self.setCamera(camera)
            else:
                if file := self.mw.filePanel.getCurrentFileURI():
                    self.setFile(file)

            self.initialized = True


        except Exception as ex:
            logger.exception(MODULE_NAME + " configure failed to load")
            QMessageBox.critical(None, f'{MODULE_NAME} Error', f'{MODULE_NAME} configure failed to initialize: {ex}')

    def chkAutoClicked(self, state):
        self.mw.settings.setValue(self.autoKey, state)
        self.txtFilename.setEnabled(not self.chkAuto.isChecked())

    def spnSkipFramesChanged(self, value):
        if self.source == MediaSource.CAMERA:
            if self.media:
                if self.media.videoModelSettings:
                    self.media.videoModelSettings.setSkipFrames(value)
        if self.source == MediaSource.FILE:
            if self.mw.filePanel.videoModelSettings:
                self.mw.filePanel.videoModelSettings.setSkipFrames(value)

    def spnSampleSizeChanged(self, value):
        if self.source == MediaSource.CAMERA:
            if self.media:
                if self.media.videoModelSettings:
                    self.media.videoModelSettings.setSampleSize(value)
        if self.source == MediaSource.FILE:
            if self.mw.filePanel.videoModelSettings:
                self.mw.filePanel.videoModelSettings.setSampleSize(value)
        self.selTargets.sldGain.setMaximum(value)

    def getDevices(self, api):
        devices = ["NPU"]
        return devices

    def cmbAPIChanged(self, text):
        self.cmbDevice.clear()
        self.cmbDevice.addItems(self.getDevices(text))

    def setCamera(self, camera):
        self.source = MediaSource.CAMERA
        self.media = camera

        if camera and not len(IMPORT_ERROR):
            if not self.isModelSettings(camera.videoModelSettings):
                camera.videoModelSettings = RyzenAISettings(self.mw, camera)
            self.mw.videoPanel.lblCamera.setText(f'Camera - {camera.name()}')
            self.sldConfThre.setValue(camera.videoModelSettings.confidence)
            self.spnSkipFrames.setValue(camera.videoModelSettings.skipFrames)
            self.spnSampleSize.setValue(camera.videoModelSettings.sampleSize)
            self.selTargets.setModelParameters(camera.videoModelSettings)
            if profile := self.mw.cameraPanel.getProfile(camera.uri()):
                self.enableControls(profile.getAnalyzeVideo())

    def setFile(self, file):
        self.source = MediaSource.FILE
        self.media = file

        if file and not len(IMPORT_ERROR):
            if not self.isModelSettings(self.mw.filePanel.videoModelSettings):
                self.mw.filePanel.videoModelSettings = RyzenAISettings(self.mw)
            file_dir = "File"
            if os.path.isdir(file):
                file_dir = "Directory"
            self.mw.videoPanel.lblCamera.setText(f'{file_dir} - {os.path.split(file)[1]}')
            self.sldConfThre.setValue(self.mw.filePanel.videoModelSettings.confidence)
            self.spnSkipFrames.setValue(self.mw.filePanel.videoModelSettings.skipFrames)
            self.spnSampleSize.setValue(self.mw.filePanel.videoModelSettings.sampleSize)
            self.selTargets.setModelParameters(self.mw.filePanel.videoModelSettings)
            self.enableControls(self.mw.videoPanel.chkEnableFile.isChecked())

    def isModelSettings(self, arg):
        return type(arg) == RyzenAISettings
    
    def enableControls(self, state):
        self.grpCamera.setEnabled(bool(state))
        if self.source == MediaSource.CAMERA:
            if state:
                self.grpCamera.setTitle("Camera Parameters")
            else:
                self.grpCamera.setTitle("Check camera video alarm to enable")

    def showWaitDialog(self):
        self.dlgWait.exec()

    def hideWaitDialog(self):
        self.dlgWait.hide()

class VideoWorker:
    def __init__(self, mw):
        try:
            self.mw = mw
            self.last_ex = ""
            self.api = self.mw.videoConfigure.cmbAPI.currentText()

            if self.mw.videoConfigure.name != MODULE_NAME or len(IMPORT_ERROR) > 0 or self.mw.glWidget.model_loading:
                return
            
            if self.mw.videoConfigure.name != MODULE_NAME or len(IMPORT_ERROR) > 0 or self.mw.glWidget.model_loading:
                return
            
            self.mw.glWidget.model_loading = True
            time.sleep(1)

            for player in self.mw.pm.players:
                player.boxes = []

            self.mw.videoConfigure.selTargets.indAlarm.setState(0)
            self.mw.videoConfigure.selTargets.barLevel.setLevel(0)

            self.ckpt_file = None
            if self.mw.videoConfigure.chkAuto.isChecked():
                self.ckpt_file = self.get_auto_ckpt_filename()
                cache = Path(self.ckpt_file)

                if not cache.is_file():
                    self.mw.videoConfigure.signals.showWaitDialog.emit()
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    link = "https://huggingface.co/amd/yolox-s/resolve/main/yolox-s-int8.onnx"
                    if os.path.split(sys.executable)[1] == "pythonw.exe":
                        torch.hub.download_url_to_file(link, self.ckpt_file, progress=False)
                    else:
                        torch.hub.download_url_to_file(link, self.ckpt_file)
                    self.mw.videoConfigure.signals.hideWaitDialog.emit()
            else:
                self.ckpt_file = self.mw.videoConfigure.txtFilename.text()

            vaip_config = self.mw.videoConfigure.txtConfigFile.text()
            if not len(vaip_config):
                raise(Exception("VAIP setting is required"))

            providers = ["VitisAIExecutionProvider"]
            provider_options = [{"config_file": vaip_config}]
            self.session = ort.InferenceSession(self.ckpt_file, providers=providers, provider_options=provider_options)

        except Exception as ex:
            logger.exception(MODULE_NAME + " initialization failure")
            self.mw.signals.error.emit(f'{MODULE_NAME} initialization failure - {ex}')

        self.mw.glWidget.model_loading = False
     
    def preprocess(self, img, input_shape, swap=(2, 0, 1)):
        if len(img.shape) == 3:
            padded_img = np.ones((input_shape[0], input_shape[1], 3), dtype=np.uint8) * 114
        else:
            padded_img = np.ones(input_shape, dtype=np.uint8) * 114
        ratio = min(input_shape[0] / img.shape[0], input_shape[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * ratio), int(img.shape[0] * ratio)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[: int(img.shape[0] * ratio), : int(img.shape[1] * ratio)] = resized_img
        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, ratio
    
    def postprocess(self, outputs, input_shape, ratio, player):
        outputs = [out.reshape(*out.shape[:2], -1).transpose(0,2,1) for out in outputs]
        outputs = np.concatenate(outputs, axis=1)
        outputs[..., 4:] = self.sigmoid(outputs[..., 4:])
        predictions = self.demo_postprocess(outputs, input_shape, p6=False)[0]
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
        boxes_xyxy /= ratio
        confthre = player.videoModelSettings.confidence / 100
        dets = self.multiclass_nms_class_agnostic(player, boxes_xyxy, scores, nms_thr=0.45, score_thr=confthre)
        return dets

    def multiclass_nms_class_agnostic(self, player, boxes, scores, nms_thr, score_thr):
        """Multiclass NMS implemented in Numpy. Class-agnostic version."""
        cls_inds = scores.argmax(1)
        cls_scores = scores[np.arange(len(cls_inds)), cls_inds]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            return None
        valid_scores = cls_scores[valid_score_mask]
        valid_boxes = boxes[valid_score_mask]
        valid_cls_inds = cls_inds[valid_score_mask]
        keep = self.nms(valid_boxes, valid_scores, nms_thr)
        if keep:
            dets = np.concatenate(
                [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
            )
        return dets

    def nms(self, boxes, scores, nms_thr):
        """Single class NMS implemented in Numpy."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= nms_thr)[0]
            order = order[inds + 1]
        return keep

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def demo_postprocess(self, outputs, img_size, p6=False):
        grids = []
        expanded_strides = []
        if not p6:
            strides = [8, 16, 32]
        else:
            strides = [8, 16, 32, 64]
        hsizes = [img_size[0] // stride for stride in strides]
        wsizes = [img_size[1] // stride for stride in strides]
        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))
        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides
        return outputs
    
    def get_auto_ckpt_filename(self):
        return Path(f'{torch.hub.get_dir()}/checkpoints/yolox-s-int8.onnx').absolute()

    def __call__(self, F, player):
        try:
            if len(IMPORT_ERROR) or self.mw.glWidget.model_loading or not self.mw.videoConfigure:
                return
            
            if not F or not player or self.mw.videoConfigure.name != MODULE_NAME:
                self.mw.videoConfigure.selTargets.barLevel.setLevel(0)
                self.mw.videoConfigure.selTargets.indAlarm.setState(0)
                return
            
            camera = self.mw.cameraPanel.getCamera(player.uri)
            if not self.mw.videoConfigure.isModelSettings(player.videoModelSettings):
                if player.isCameraStream():
                    if camera:
                        if not self.mw.videoConfigure.isModelSettings(camera.videoModelSettings):
                            self.mw.cameraPanel.setCurrentCamera(camera)
                        player.videoModelSettings = camera.videoModelSettings
                else:
                    if not self.mw.videoConfigure.isModelSettings(self.mw.filePanel.videoModelSettings):
                        self.mw.filePanel.videoModelSettings = RyzenAISettings(self.mw)
                    player.videoModelSettings = self.mw.filePanel.videoModelSettings

            if not player.videoModelSettings:
                raise Exception("Unable to set video model parameters for player")

            if player.videoModelSettings.skipCounter < player.videoModelSettings.skipFrames:
                player.videoModelSettings.skipCounter += 1
                return
            player.videoModelSettings.skipCounter = 0

            player.lock()
            player.videoModelSettings.orig_img = np.array(F, copy=False)

            res = int(self.mw.videoConfigure.cmbRes.currentText())
            input_shape = [res, res]
            input_img = np.array(F, copy=False)
            img, ratio = self.preprocess(input_img, input_shape)
            ort_inputs = {self.session.get_inputs()[0].name: np.transpose(img[None, :, :, :], (0, 2 ,3, 1))}

            outputs = self.session.run(None, ort_inputs)

            outputs = [np.transpose(out, (0, 3, 1, 2)) for out in outputs]
            dets = self.postprocess(outputs, input_shape, ratio, player)
            boxes = []
            if dets is not None:
                final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
                for i in range(len(final_cls_inds)):
                    if int(final_cls_inds[i]) in player.videoModelSettings.targets:
                        boxes.append(final_boxes[i])
            player.boxes = boxes

            result = player.processModelOutput()
            alarmState = result >= player.videoModelSettings.limit if result else False
            player.handleAlarm(alarmState)

            show_alarm = False
            if camera := self.mw.cameraPanel.getCamera(player.uri):
                if camera.isFocus():
                    show_alarm = True
            if not player.isCameraStream():
                    show_alarm = True

            if show_alarm:
                level = 0
                if player.videoModelSettings.limit:
                    level = result / player.videoModelSettings.limit
                else:
                    if result:
                        level = 1.0

                self.mw.videoConfigure.selTargets.barLevel.setLevel(level)

                if alarmState:
                    self.mw.videoConfigure.selTargets.indAlarm.setState(1)

        except Exception as ex:
            if self.last_ex != str(ex) and self.mw.videoConfigure.name == MODULE_NAME:
                logger.exception(f'{MODULE_NAME} runtime error - {ex}')
            self.last_ex = str(ex)

        player.unlock()
