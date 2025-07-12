#/********************************************************************
# libonvif/onvif-gui/panels/video/modules/yolox.py 
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
MODULE_NAME = "yolox"

try:
    import os
    import sys
    from loguru import logger
    import numpy as np
    from pathlib import Path
    from onvif_gui.components import ComboSelector, FileSelector, ThresholdSlider, TargetSelector
    from onvif_gui.enums import MediaSource
    from PyQt6.QtWidgets import QWidget, QGridLayout, QLabel, QCheckBox, QMessageBox, \
        QGroupBox, QSpinBox
    from onvif_gui.panels.video.modules.common.yolosettings import YoloSettings
    import time
    import torch
    import requests
    from torchvision.transforms import functional
    import torch.nn as nn
    from onvif_gui.panels.video.modules.yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
    import onvif_gui.panels.video.modules.yolox.utils
    if sys.platform != "darwin":
        import openvino as ov

except ModuleNotFoundError as ex:
    IMPORT_ERROR = str(ex)
    logger.debug("Import Error has occurred, missing modules need to be installed, please consult documentation: ", ex)
    #QMessageBox.critical(None, MODULE_NAME + " Import Error", "Modules required for running this function are missing: " + IMPORT_ERROR)

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class YoloxSettings(YoloSettings):
    def __init__(self, mw, camera=None):
        super().__init__(mw, camera, MODULE_NAME)

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

            self.chkAuto = QCheckBox("Automatically download model")
            self.chkAuto.setChecked(int(self.mw.settings.value(self.autoKey, 1)))
            self.chkAuto.stateChanged.connect(self.chkAutoClicked)
            self.txtFilename = FileSelector(mw, MODULE_NAME)
            self.txtFilename.setEnabled(not self.chkAuto.isChecked())
            self.cmbRes = ComboSelector(mw, "Size", ("160", "320", "480", "640", "960", "1280"), "640", MODULE_NAME)
            self.cmbModelName = ComboSelector(mw, "Name", ("yolox_tiny", "yolox_s", "yolox_m", "yolox_l", "yolox_x"), "yolox_s", MODULE_NAME)
            
            apis = None
            if sys.platform == "darwin":
                apis = ["PyTorch"]
                default_api = "PyTorch"
            elif sys.platform == "win32":
                apis = ["OpenVINO", "PyTorch"]
                default_api = "OpenVINO"
            else:
                apis = ["PyTorch", "OpenVINO"]            
                if self.mw.haveNvidia():
                    default_api = "PyTorch"
                else:
                    default_api = "OpenVINO"
            
            self.cmbAPI = ComboSelector(mw, "API", apis, default_api, MODULE_NAME)
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
            lytSystem.addWidget(self.chkAuto,      0, 0, 1, 4)
            lytSystem.addWidget(self.txtFilename,  1, 0, 1, 4)
            lytSystem.addWidget(self.cmbModelName, 2, 0, 1, 2)
            lytSystem.addWidget(self.cmbRes,       2, 2, 1, 2)
            lytSystem.addWidget(self.cmbAPI,       3, 0, 1, 2)
            lytSystem.addWidget(self.cmbDevice,    3, 2, 1, 2)

            self.grpCamera = QGroupBox("Check to enable")
            self.grpCamera.setCheckable(True)
            self.grpCamera.clicked.connect(self.mw.cameraPanel.tabVideo.chkAnalyzeVideoChecked)
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

    def setCamera(self, camera):
        self.source = MediaSource.CAMERA
        self.media = camera

        if camera and not len(IMPORT_ERROR):
            if not self.isModelSettings(camera.videoModelSettings):
                camera.videoModelSettings = YoloxSettings(self.mw, camera)
            self.mw.videoPanel.lblCamera.setText(f'Camera - {camera.name()}')
            self.sldConfThre.setValue(camera.videoModelSettings.confidence)
            self.spnSkipFrames.setValue(camera.videoModelSettings.skipFrames)
            self.spnSampleSize.setValue(camera.videoModelSettings.sampleSize)
            self.selTargets.setModelParameters(camera.videoModelSettings)
            if profile := self.mw.cameraPanel.getProfile(camera.uri()):
                self.enableControls(profile.getAnalyzeVideo())

    def isModelSettings(self, arg):
        return type(arg) == YoloxSettings
    
    def enableControls(self, state):
        self.grpCamera.setChecked(bool(state))
        self.grpCamera.setEnabled(False)
        if self.source == MediaSource.CAMERA:
            if camera := self.mw.cameraPanel.getCurrentCamera():
                self.grpCamera.setEnabled(True)
                if state:
                    self.grpCamera.setTitle("Video analysis enabled")
                else:
                    self.grpCamera.setTitle("Check to enable")

class VideoWorker:
    def __init__(self, mw):
        try:
            self.mw = mw
            self.last_ex = ""
            self.api = self.mw.videoConfigure.cmbAPI.currentText()

            if self.mw.videoConfigure.name != MODULE_NAME or len(IMPORT_ERROR) > 0 or self.mw.glWidget.model_loading:
                return
            
            self.mw.glWidget.model_loading = True
            time.sleep(1)

            for player in self.mw.pm.players:
                player.boxes = []

            self.mw.videoConfigure.selTargets.indAlarm.setState(0)
            self.mw.videoConfigure.selTargets.barLevel.setLevel(0)

            self.torch_device = None
            self.torch_device_name = None
            self.ov_device = None
            self.compiled_model = None
            ov_model = None

            self.num_classes = 80
            self.res = int(self.mw.videoConfigure.cmbRes.currentText())
            initializer_data = torch.rand(1, 3, self.res, self.res)
            self.model_name = self.mw.videoConfigure.cmbModelName.currentText()

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
                self.torch_device = torch.device(self.torch_device_name)

                size = {'yolox_tiny': [0.33, 0.375],
                        'yolox_s': [0.33, 0.50], 
                        'yolox_m': [0.67, 0.75],
                        'yolox_l': [1.00, 1.00],
                        'yolox_x': [1.33, 1.25]}[self.model_name]

                self.model = None
                self.model = self.get_model(self.num_classes, size[0], size[1], None).to(self.torch_device)
                self.model.eval()

                self.ckpt_file = None
                if self.mw.videoConfigure.chkAuto.isChecked():
                    self.ckpt_file = self.get_auto_ckpt_filename()
                    cache = Path(self.ckpt_file)
                    if not cache.is_file():
                        self.mw.signals.showWaitDialog.emit("Please wait for yolox model to download")
                        try:
                            cache.parent.mkdir(parents=True, exist_ok=True)
                            id = f'{self.model_name}.pth'
                            link = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/" + id
                            response = requests.get(link, allow_redirects=True, timeout=(10, 120))
                            if not response:
                                raise RuntimeError(f'Error downloading {link}: {response.status_code}')
                            with open(self.ckpt_file, 'wb') as content:
                                content.write(response.content)
                            if os.path.isfile(self.ckpt_file):

                                hashes = {
                                    "yolox_tiny.pth": "9de513de589ac98bb92d3bca53b5af7b9acfa9b0bacb831f7999d0f7afaee8f0",
                                    "yolox_s.pth": "f55ded7181e1b0c13285c56e7790b8f0e8f8db590fe4edb37f0b7f345c913a30",
                                    "yolox_m.pth": "60076992b32da82951c90cfa7bd6ab70eba9eda243e08b940a396f60ac2d19b6",
                                    "yolox_l.pth": "1e6b7fa6240375370b2a8a8eab9066b3cdd43fd1d0bfa8d2027fd3a51def2917",
                                    "yolox_x.pth": "5652330b6ae860043f091b8f550a60c10e1129f416edfdb65c259be6caf355cf"
                                }

                                verified = False
                                if hash := self.mw.calculate_sha256(self.ckpt_file):
                                    if hash == hashes.get(id, None):
                                        verified = True
                                if not verified:
                                    os.remove(self.ckpt_file)
                                    raise RuntimeError(f'Error verifying {self.ckpt_file}')

                                logger.debug(f'model {self.ckpt_file} was downloaded succesfully')
                        except Exception as ex:
                            logger.error(f'YOLO model {link} download failure: {ex}')
                        self.mw.signals.hideWaitDialog.emit()
                else:
                    self.ckpt_file = self.mw.videoConfigure.txtFilename.text()

                self.model.load_state_dict(torch.load(self.ckpt_file, map_location="cpu")["model"])
                self.model(initializer_data.to(self.torch_device))

            if self.api == "OpenVINO" and sys.platform != "darwin":
                if not ov_model:
                    ov_model = ov.convert_model(self.model, example_input=initializer_data)
                    ov.save_model(ov_model, self.get_ov_model_filename())

                self.ov_device = self.mw.videoConfigure.cmbDevice.currentText()
                if self.ov_device != "CPU":
                    ov_model.reshape({0: [1, 3, self.res, self.res]})
                ov_config = {}
                if "GPU" in self.ov_device or ("AUTO" in self.ov_device and "GPU" in ov.Core().available_devices):
                    ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}

                self.compiled_model = ov.compile_model(ov_model, self.ov_device, ov_config)
                self.compiled_model(initializer_data)
                self.infer_queue = ov.AsyncInferQueue(self.compiled_model)
                self.infer_queue.set_callback(self.callback)

            if not self.torch_device:
                self.torch_device = torch.device("cpu")

        except Exception as ex:
            logger.exception(MODULE_NAME + " initialization failure")
            self.mw.signals.error.emit(f'{MODULE_NAME} initialization failure - {ex}')

        self.mw.glWidget.model_loading = False

    def preprocess(self, player):
        h = player.videoModelSettings.orig_img.shape[0]
        w = player.videoModelSettings.orig_img.shape[1]

        test_size = (self.res, self.res)
        ratio = min(test_size[0] / h, test_size[1] / w)
        inf_shape = (int(h * ratio), int(w * ratio))
        bottom = test_size[0] - inf_shape[0]
        side = test_size[1] - inf_shape[1]
        pad = (0, 0, side, bottom)

        timg = functional.to_tensor(player.videoModelSettings.orig_img).to(self.torch_device)
        timg *= 255
        timg = functional.resize(timg, inf_shape)
        timg = functional.pad(timg, pad, 114)
        timg = timg.unsqueeze(0)
        return timg
    
    def postprocess(self, outputs, player):
        confthre = player.videoModelSettings.confidence / 100
        nmsthre = 0.65
        if isinstance(outputs, np.ndarray):
            outputs = torch.from_numpy(outputs)
        outputs = onvif_gui.panels.video.modules.yolox.utils.postprocess(outputs, self.num_classes, confthre, nmsthre)
        output = None
        boxes = []
        test_size = (self.res, self.res)
        h = player.videoModelSettings.orig_img.shape[0]
        w = player.videoModelSettings.orig_img.shape[1]
        ratio = min(test_size[0] / h, test_size[1] / w)

        if outputs[0] is not None:
            output = outputs[0].cpu().numpy().astype(float)
            output[:, 0:4] /= ratio
            output[:, 4] *= output[:, 5]
            output = np.delete(output, 5, 1)

            labels = output[:, 5].astype(int)
            for i in range(len(labels)):
                if labels[i] in player.videoModelSettings.targets:
                    boxes.append(output[i, 0:4])

        player.boxes = boxes
        result = player.processModelOutput()
        alarmState = result >= player.videoModelSettings.limit if result else False
        player.handleAlarm(alarmState)

        show_alarm = False
        if camera := self.mw.cameraPanel.getCamera(player.uri):
            if camera.isCurrent():
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

    def callback(self, infer_request, player):
        try:
            if not self.mw.glWidget.model_loading:
                player.lock()
                outputs = infer_request.get_output_tensor(0).data
                self.postprocess(outputs, player)

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
            
            if not self.mw.videoConfigure.isModelSettings(player.videoModelSettings):
                if player.isCameraStream():
                    if camera := self.mw.cameraPanel.getCamera(player.uri):
                        if not self.mw.videoConfigure.isModelSettings(camera.videoModelSettings):
                            self.mw.videoConfigure.setCamera(camera)
                        player.videoModelSettings = camera.videoModelSettings

            if not player.videoModelSettings:
                raise Exception("Unable to set video model parameters for player")

            if player.videoModelSettings.skipCounter < player.videoModelSettings.skipFrames:
                player.videoModelSettings.skipCounter += 1
                return
            player.videoModelSettings.skipCounter = 0

            player.lock()
            player.videoModelSettings.orig_img = np.array(F, copy=False)

            if self.api == "OpenVINO" and self.compiled_model:
                timg = self.preprocess(player)
                
                self.infer_queue.wait_all()
                self.infer_queue.start_async({0: timg}, player, False)
            
            if self.api == "PyTorch" and self.model:
                timg = self.preprocess(player)
                with torch.no_grad():
                    outputs = self.model(timg)

                self.postprocess(outputs, player)

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
        name = self.model_name == self.mw.videoConfigure.cmbModelName.currentText()
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

    def get_ov_model_filename(self):
        model_name = self.mw.videoConfigure.cmbModelName.currentText()
        openvino_device = self.mw.videoConfigure.cmbDevice.currentText()
        path = os.path.join(self.mw.getCacheLocation(), "checkpoints", model_name, openvino_device, "model.xml")
        return path

    def get_auto_ckpt_filename(self):
        model_name = self.mw.videoConfigure.cmbModelName.currentText()
        path = os.path.join(self.mw.getCacheLocation(), "checkpoints", f'{model_name}.pth')
        return path

    def get_model(self, num_classes, depth, width, act):
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        in_channels = [256, 512, 1024]
        backbone = YOLOPAFPN(depth, width, in_channels=in_channels)
        head = YOLOXHead(num_classes, width, in_channels=in_channels)
        model = YOLOX(backbone, head)

        model.apply(init_yolo)
        model.head.initialize_biases(1e-2)
        return model
