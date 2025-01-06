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

#/*********************************************************************
#
# Portions of this code include work developed by ultralytics and is 
# licensed under the The GNU Affero General Public License Version 3
# 
#    https://www.gnu.org/licenses/licenses.html#AGPL
#
#**********************************************************************/

IMPORT_ERROR = ""
MODULE_NAME = "yolov8"

try:
    import os
    import time
    import sys
    from loguru import logger
    from typing import Tuple
    import numpy as np
    from pathlib import Path
    from gui.components import ComboSelector, FileSelector, ThresholdSlider, TargetSelector
    from gui.onvif.datastructures import MediaSource
    from PyQt6.QtWidgets import QWidget, QGridLayout, QLabel, QCheckBox, QMessageBox, \
        QGroupBox, QSpinBox, QDialog
    from PyQt6.QtCore import Qt, QSize, QObject, pyqtSignal
    from PyQt6.QtGui import QMovie
    import cv2
    import torch
    from torchvision.transforms import functional
    from ultralytics import YOLO
    from ultralytics.utils import ops
    if sys.platform != "darwin":
        import openvino as ov

except ModuleNotFoundError as ex:
    IMPORT_ERROR = str(ex)
    logger.debug("Import Error has occurred, missing modules need to be installed, please consult documentation: ", ex)
    QMessageBox.critical(None, MODULE_NAME + " Import Error", "Modules required for running this function are missing: " + IMPORT_ERROR)

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class YoloV8WaitDialog(QDialog):
    def __init__(self, p):
        super().__init__(p)
        self.lblMessage = QLabel("Please wait for model to download")
        self.lblProgress = QLabel()
        self.movie = QMovie("image:spinner.gif")
        self.movie.setScaledSize(QSize(50, 50))
        self.lblProgress.setMovie(self.movie)
        self.setWindowTitle("yolov8")

        lytMain = QGridLayout(self)
        lytMain.addWidget(self.lblMessage,  0, 1, 1, 1, Qt.AlignmentFlag.AlignCenter)
        lytMain.addWidget(self.lblProgress, 1, 1, 1, 1, Qt.AlignmentFlag.AlignCenter)

        self.movie.start()
        self.setModal(True)

    def sizeHint(self):
        return QSize(300, 100)
    
class YoloV8Settings():
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
        self.input_hw = None

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

class YoloV8Signals(QObject):
    showWaitDialog = pyqtSignal()
    hideWaitDialog = pyqtSignal()

class VideoConfigure(QWidget):
    def __init__(self, mw):
        try:
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

            self.dlgWait = YoloV8WaitDialog(self.mw)
            self.signals = YoloV8Signals()
            self.signals.showWaitDialog.connect(self.showWaitDialog)
            self.signals.hideWaitDialog.connect(self.hideWaitDialog)
            
            self.chkAuto = QCheckBox("Automatically download model")
            self.chkAuto.setChecked(int(self.mw.settings.value(self.autoKey, 1)))
            self.chkAuto.stateChanged.connect(self.chkAutoClicked)
            self.txtFilename = FileSelector(mw, MODULE_NAME)
            self.txtFilename.setEnabled(not self.chkAuto.isChecked())
            self.cmbRes = ComboSelector(mw, "Size", ("320", "480", "640", "960", "1280", "1440"), "640", MODULE_NAME)

            self.model_names = {"nano" : "yolov8n.pt", "small" : "yolov8s.pt", "medium" : "yolov8m.pt", "large" : "yolov8l.pt", "XL" : "yolov8x.pt"}
            self.cmbModelName = ComboSelector(mw, "Name", self.model_names.keys(), "small", MODULE_NAME)

            apis = ["PyTorch"]
            if sys.platform != "darwin":
                apis.append("OpenVINO")
            
            self.cmbAPI = ComboSelector(mw, "API", apis, "PyTorch", MODULE_NAME)
            self.cmbAPI.cmbBox.currentTextChanged.connect(self.cmbAPIChanged)
            self.fixRes()

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
            lytSystem.addWidget(self.chkAuto,      0, 0, 1, 4)
            lytSystem.addWidget(self.txtFilename,  1, 0, 1, 4)
            lytSystem.addWidget(self.cmbModelName, 2, 0, 1, 2)
            lytSystem.addWidget(self.cmbRes,       2, 2, 1, 2)
            lytSystem.addWidget(self.cmbAPI,       3, 0, 1, 2)
            lytSystem.addWidget(self.cmbDevice,    3, 2, 1, 2)

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
        devices = []
        if api == "OpenVINO" and sys.platform != "darwin":
            devices = ["AUTO"] + ov.Core().available_devices
        if api == "PyTorch":
            devices = ["auto", "cpu"]
            if torch.cuda.is_available():
                devices.append("cuda")
            if torch.backends.mps.is_available():
                devices.append("mps")
        return devices

    def cmbAPIChanged(self, text):
        self.cmbDevice.clear()
        self.cmbDevice.addItems(self.getDevices(text))
        self.fixRes()

    def fixRes(self):
        # yolov8 on OpenVINO seems to only support 640 resolution
        api = self.cmbAPI.currentText()
        if api == "OpenVINO":
            self.cmbRes.cmbBox.setCurrentText("640")
            self.cmbRes.cmbBox.setEnabled(False)
        else:
            self.cmbRes.cmbBox.setEnabled(True)

    def setCamera(self, camera):
        self.source = MediaSource.CAMERA
        self.media = camera

        if camera and not len(IMPORT_ERROR):
            if not self.isModelSettings(camera.videoModelSettings):
                camera.videoModelSettings = YoloV8Settings(self.mw, camera)
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
                self.mw.filePanel.videoModelSettings = YoloV8Settings(self.mw)
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
        return type(arg) == YoloV8Settings
    
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

    def getModelName(self):
        key = self.mw.videoConfigure.cmbModelName.currentText()
        return self.mw.videoConfigure.model_names[key]

class VideoWorker:
    def __init__(self, mw):
        try:
            self.mw = mw
            self.last_ex = ""

            if self.mw.videoConfigure.name != MODULE_NAME or len(IMPORT_ERROR) > 0 or self.mw.glWidget.model_loading:
                return
            
            self.mw.glWidget.model_loading = True
            time.sleep(1)

            for player in self.mw.pm.players:
                player.boxes = []

            self.mw.videoConfigure.selTargets.indAlarm.setState(0)
            self.mw.videoConfigure.selTargets.barLevel.setLevel(0)

            self.model = None
            self.torch_device_name = None
            self.ov_device = None
            self.compiled_model = None
            ov_model = None

            self.api = self.mw.videoConfigure.cmbAPI.currentText()
            self.res = int(self.mw.videoConfigure.cmbRes.currentText())
            initializer_data = torch.rand(1, 3, self.res, self.res)
            self.model_name = self.mw.videoConfigure.getModelName()

            self.api = self.mw.videoConfigure.cmbAPI.currentText()
            if self.api == "PyTorch":
                self.torch_device_name = self.mw.videoConfigure.cmbDevice.currentText()
            if self.api == "OpenVINO":
                self.ov_device = self.mw.videoConfigure.cmbDevice.currentText()

            if self.api == "OpenVINO" and Path(self.get_ov_model_filename()).is_file() and sys.platform != "darwin":
                ov_model = ov.Core().read_model(self.get_ov_model_filename())
            
            if (self.api == "OpenVINO" and not ov_model) or self.api == "PyTorch":
                self.torch_device_name = "cpu"
                if self.api == "PyTorch":
                    if torch.cuda.is_available():
                        self.torch_device_name = "cuda"
                    if torch.backends.mps.is_available():
                        self.torch_device_name = "mps"
                    if self.mw.videoConfigure.cmbDevice.currentText() == "cpu":
                        self.torch_device_name = "cpu"

                self.ckpt_file = None
                if self.mw.videoConfigure.chkAuto.isChecked():
                    self.ckpt_file = self.get_auto_ckpt_filename()
                    cache = Path(self.ckpt_file)
                    if not cache.is_file():
                        self.mw.videoConfigure.signals.showWaitDialog.emit()
                        cache.parent.mkdir(parents=True, exist_ok=True)
                        model_name = self.mw.videoConfigure.getModelName()
                        link = "https://github.com/ultralytics/assets/releases/download/v0.0.0/" + model_name
                        if os.path.split(sys.executable)[1] == "pythonw.exe":
                            torch.hub.download_url_to_file(link, self.ckpt_file, progress=False)
                        else:
                            torch.hub.download_url_to_file(link, self.ckpt_file)
                        self.mw.videoConfigure.signals.hideWaitDialog.emit()
                else:
                    self.ckpt_file = self.configure.txtFilename.text()

                self.model = YOLO(Path(self.ckpt_file))
                self.model.predict(initializer_data, stream=True, verbose=False, device=self.torch_device_name)

            if self.api == "OpenVINO" and sys.platform != "darwin":
                if not ov_model:
                    self.model.export(format='openvino')
                    ov_path = Path(self.get_ov_model_filename()).absolute()
                    if ov_path.exists():
                        core = ov.Core()
                        ov_model = core.read_model(ov_path)
                        if self.ov_device != "CPU":
                            ov_model.reshape({0: [1, 3, self.res, self.res]})

                ov_config = {}
                if "GPU" in self.ov_device or ("AUTO" in self.ov_device and "GPU" in ov.Core().available_devices):
                    ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}

                self.compiled_model = ov.compile_model(ov_model, self.ov_device, ov_config)
                self.compiled_model(initializer_data)
                self.infer_queue = ov.AsyncInferQueue(self.compiled_model)
                self.infer_queue.set_callback(self.callback)

        except Exception as ex:
            logger.exception(MODULE_NAME + " initialization failure")
            self.mw.signals.error.emit(f'{MODULE_NAME} initialization failure - {ex}')

        self.mw.glWidget.model_loading = False

    def callback(self, infer_request, player):
        try:
            if self.mw.glWidget.model_loading:
                return
            
            player.lock()
            results = infer_request.get_output_tensor(0).data

            boxes = []
            detections = self.postprocess(pred_boxes=results, input_hw=[self.res, self.res], orig_img=player.videoModelSettings.orig_img)
            detections = detections[0]['det']
            if not isinstance(detections, list):
                detections = detections.cpu().numpy()
                for detection in detections:
                    if detection[5] in player.videoModelSettings.targets:
                        boxes.append(detection)

            player.boxes = boxes
            result = player.processModelOutput()
            alarmState = result >= player.videoModelSettings.limit if result else False
            player.handleAlarm(alarmState)

            if camera := self.mw.cameraPanel.getCamera(player.uri):
                if camera.isFocus():
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
            logger.exception(f'{MODULE_NAME} callback error: {ex}')

        player.unlock()

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
                        self.mw.filePanel.videoModelSettings = YoloV8Settings(self.mw)
                    player.videoModelSettings = self.mw.filePanel.videoModelSettings

            if not player.videoModelSettings:
                raise Exception("Unable to set video model parameters for player")

            if player.videoModelSettings.skipCounter < player.videoModelSettings.skipFrames:
                player.videoModelSettings.skipCounter += 1
                return
            player.videoModelSettings.skipCounter = 0

            conf_thre = player.videoModelSettings.confidence / 100
            targets = player.videoModelSettings.targets
            res = int(self.mw.videoConfigure.cmbRes.currentText())

            player.lock()
            player.videoModelSettings.orig_img = np.array(F, copy=False)

            if self.api == "PyTorch" and len(targets):
                results = self.model.predict(player.videoModelSettings.orig_img, stream=True, verbose=False,
                                         classes=targets, conf=conf_thre,
                                         imgsz=res, device=self.torch_device_name)
                for result in results:
                    player.boxes = result.boxes.xyxy.cpu().numpy()
            
            if self.api == "OpenVINO":
                preprocessed_image = self.preprocess_image(player.videoModelSettings.orig_img)
                input_tensor = self.image_to_tensor(preprocessed_image)
                player.videoModelSettings.input_hw = input_tensor.shape[2:]
                self.infer_queue.wait_all()
                self.infer_queue.start_async({0: input_tensor}, player, False)

            if self.api == "PyTorch":
                result = player.processModelOutput()
                alarmState = result >= player.videoModelSettings.limit if result else False
                player.handleAlarm(alarmState)

                if camera:
                    if camera.isFocus():
                        level = 0
                        if player.videoModelSettings.limit:
                            level = result / player.videoModelSettings.limit
                        else:
                            if result:
                                level = 1.0
                        self.mw.videoConfigure.selTargets.barLevel.setLevel(level)
                        if alarmState:
                            self.mw.videoConfigure.selTargets.indAlarm.setState(1)

            if self.parameters_changed():
                self.__init__(self.mw)
            
        except Exception as ex:
            if self.last_ex != str(ex) and self.mw.videoConfigure.name == MODULE_NAME:
                logger.exception(f'{MODULE_NAME} runtime error - {ex}')
            self.last_ex = str(ex)

        player.unlock()

    def parameters_changed(self):
        result = False

        api = self.api == self.mw.videoConfigure.cmbAPI.currentText()
        name = self.model_name == self.mw.videoConfigure.getModelName()
        res = str(self.res) == self.mw.videoConfigure.cmbRes.currentText()

        dev = False
        if self.api == "PyTorch":
            dev = True
            if self.mw.videoConfigure.cmbDevice.currentText() != "auto":
                dev = self.torch_device_name == self.mw.videoConfigure.cmbDevice.currentText()
        if self.api == "OpenVINO":
            dev = self.ov_device == self.mw.videoConfigure.cmbDevice.currentText()

        if not api or not name or not res or not dev:
            result = True

        return result

    def postprocess(self,
        pred_boxes:np.ndarray, 
        input_hw:Tuple[int, int], 
        orig_img:np.ndarray, 
        min_conf_threshold:float = 0.75, 
        nms_iou_threshold:float = 0.7, 
        agnosting_nms:bool = False, 
        max_detections:int = 300,
    ):
        nms_kwargs = {"agnostic": agnosting_nms, "max_det":max_detections}
        preds = ops.non_max_suppression(
            torch.from_numpy(pred_boxes),
            min_conf_threshold,
            nms_iou_threshold,
            nc=80,
            **nms_kwargs
        )

        results = []
        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if isinstance(orig_img, list) else orig_img.shape
            if not len(pred):
                results.append({"det": [], "segment": []})
                continue
            pred[:, :4] = ops.scale_boxes(input_hw, pred[:, :4], shape).round()
            results.append({"det": pred})

        return results

    def image_to_tensor(self, image:np.ndarray):
        input_tensor = image.astype(np.float32)
        input_tensor /= 255.0
        
        if input_tensor.ndim == 3:
            input_tensor = np.expand_dims(input_tensor, 0)
        return input_tensor

    def preprocess_image(self, img0: np.ndarray):
        img = self.letterbox(img0)[0]
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        return img

    def letterbox(self, img: np.ndarray, new_shape:Tuple[int, int] = (640, 640), color:Tuple[int, int, int] = (114, 114, 114), auto:bool = False, scale_fill:bool = False, scaleup:bool = False, stride:int = 32):
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scale_fill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)

    def get_ov_model_filename(self):
        model_name = Path(self.mw.videoConfigure.getModelName()).stem
        return Path(f'{torch.hub.get_dir()}/checkpoints/{model_name}_openvino_model/{model_name}.xml').absolute() 

    def get_auto_ckpt_filename(self):
        return Path(f'{torch.hub.get_dir()}/checkpoints/{self.mw.videoConfigure.getModelName()}').absolute()

