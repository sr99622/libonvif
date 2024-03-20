#/********************************************************************
# libonvif/onvif-gui/modules/video/yolox.py 
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
    import numpy as np
    from pathlib import Path
    from gui.components import ComboSelector, FileSelector, ThresholdSlider, TargetSelector
    from gui.onvif.datastructures import MediaSource
    from PyQt6.QtWidgets import QWidget, QGridLayout, QLabel, QCheckBox, QMessageBox, \
        QGroupBox, QDialog
    from PyQt6.QtCore import Qt, QSize, QObject, pyqtSignal
    from PyQt6.QtGui import QMovie
    from time import sleep
    import torch
    from torchvision.transforms import functional
    import torch.nn as nn
    from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
    from yolox.utils import postprocess

except ModuleNotFoundError as ex:
    IMPORT_ERROR = str(ex)
    logger.debug("Import Error has occurred, missing modules need to be installed, please consult documentation: ", ex)

os.environ['KMP_DUPLICATE_LIB_OK']='True'
MODULE_NAME = "yolox"

class YoloxWaitDialog(QDialog):
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
    
class YoloxSettings():
    def __init__(self, mw, camera=None):
        self.camera = camera
        self.mw = mw
        self.id = "File"
        if camera:
            self.id = camera.serial_number()

        self.targets = self.getTargetsForPlayer()
        self.gain = self.getModelOutputGain()
        self.confidence = self.getModelConfidence()
        self.show = self.getModelShowBoxes()

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

    def getModelOutputGain(self):
        key = f'{self.id}/{MODULE_NAME}/ModelOutputGain'
        return int(self.mw.settings.value(key, 50))
    
    def setModelOutputGain(self, value):
        key = f'{self.id}/{MODULE_NAME}/ModelOutputGain'
        self.gain = value
        self.mw.settings.setValue(key, value)

    def getModelShowBoxes(self):
        key = f'{self.id}/{MODULE_NAME}/ModelShowBoxes'
        return bool(int(self.mw.settings.value(key, 1)))
    
    def setModelShowBoxes(self, value):
        key = f'{self.id}/{MODULE_NAME}/ModelShowBoxes'
        self.show = value
        self.mw.settings.setValue(key, int(value))

class YoloxSignals(QObject):
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
            self.autoKey = "Module/" + MODULE_NAME + "/autoDownload"

            self.dlgWait = YoloxWaitDialog(self.mw)
            self.signals = YoloxSignals()
            self.signals.showWaitDialog.connect(self.showWaitDialog)
            self.signals.hideWaitDialog.connect(self.hideWaitDialog)
            
            self.chkAuto = QCheckBox("Automatically download model")
            self.chkAuto.setChecked(int(self.mw.settings.value(self.autoKey, 1)))
            self.chkAuto.stateChanged.connect(self.chkAutoClicked)

            self.txtFilename = FileSelector(mw, MODULE_NAME)
            self.txtFilename.setEnabled(not self.chkAuto.isChecked())
            self.cmbRes = ComboSelector(mw, "Model Size", ("320", "480", "640", "960", "1280", "1440"), "640", MODULE_NAME)
            self.cmbType = ComboSelector(mw, "Model Name", ("yolox_s", "yolox_m", "yolox_l", "yolox_x"), "yolox_s", MODULE_NAME)

            self.txtFilename.setEnabled(not self.chkAuto.isChecked())

            self.sldConfThre = ThresholdSlider(mw, "Confidence", MODULE_NAME)
            self.selTargets = TargetSelector(self.mw, MODULE_NAME)

            grpSystem = QGroupBox("System wide model parameters")
            lytSystem = QGridLayout(grpSystem)
            lytSystem.addWidget(self.chkAuto,      0, 0, 1, 1)
            lytSystem.addWidget(self.cmbType,      1, 0, 1, 1)
            lytSystem.addWidget(self.txtFilename,  2, 0, 1, 1)
            lytSystem.addWidget(self.cmbRes,       3, 0, 1, 1)

            self.grpCamera = QGroupBox("Check camera video alarm to enable")
            lytCamera = QGridLayout(self.grpCamera)
            lytCamera.addWidget(self.sldConfThre,  0, 0, 1, 1)
            lytCamera.addWidget(QLabel(),          1, 0, 1, 1)
            lytCamera.addWidget(self.selTargets,   2, 0, 1, 1)

            lytMain = QGridLayout(self)
            lytMain.addWidget(grpSystem,         0, 0, 1, 1)
            lytMain.addWidget(QLabel(),          1, 0, 1, 1)
            lytMain.addWidget(self.grpCamera,    2, 0, 1, 1)
            lytMain.addWidget(QLabel(),          3, 0, 1, 1)
            lytMain.setRowStretch(3, 10)

            self.enableControls(False)

            if len(IMPORT_ERROR) > 0:
                QMessageBox.critical(None, MODULE_NAME + " Import Error", "Modules required for running this function are missing: " + IMPORT_ERROR)

        except:
            logger.exception(MODULE_NAME + " configure failed to load")

    def chkAutoClicked(self, state):
        self.mw.settings.setValue(self.autoKey, state)
        self.txtFilename.setEnabled(not self.chkAuto.isChecked())

    def setCamera(self, camera):
        self.source = MediaSource.CAMERA
        self.media = camera

        if camera:
            if not self.isModelSettings(camera.videoModelSettings):
                camera.videoModelSettings = YoloxSettings(self.mw, camera)
            self.mw.videoPanel.lblCamera.setText(f'Camera - {camera.name()}')
            self.selTargets.setTargets(camera.videoModelSettings.targets)
            self.sldConfThre.setValue(camera.videoModelSettings.confidence)
            self.selTargets.sldGain.setValue(camera.videoModelSettings.gain)
            self.selTargets.chkShowBoxes.setChecked(camera.videoModelSettings.show)
            self.selTargets.barLevel.setLevel(0)
            self.selTargets.indAlarm.setState(0)
            profile = self.mw.cameraPanel.getProfile(camera.uri())
            if profile:
                self.enableControls(profile.getAnalyzeVideo())

    def setFile(self, file):
        self.source = MediaSource.FILE
        self.media = file

        if file:
            if not self.isModelSettings(self.mw.filePanel.videoModelSettings):
                self.mw.filePanel.videoModelSettings = YoloxSettings(self.mw)
            self.mw.videoPanel.lblCamera.setText(f'File - {os.path.split(file)[1]}')
            self.selTargets.setTargets(self.mw.filePanel.videoModelSettings.targets)
            self.sldConfThre.setValue(self.mw.filePanel.videoModelSettings.confidence)
            self.selTargets.sldGain.setValue(self.mw.filePanel.videoModelSettings.gain)
            self.selTargets.chkShowBoxes.setChecked(self.mw.filePanel.videoModelSettings.show)
            self.selTargets.barLevel.setLevel(0)
            self.selTargets.indAlarm.setState(0)
            self.enableControls(self.mw.videoPanel.chkEnableFile.isChecked())

    def isModelSettings(self, arg):
        return type(arg) == YoloxSettings
    
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
            self.lock = False

            if self.mw.videoConfigure.name != MODULE_NAME or len(IMPORT_ERROR) > 0:
                return
            
            device_name = "cpu"
            if torch.cuda.is_available():
                device_name = "cuda"
            self.device = torch.device(device_name)

            self.mw.glWidget.model_loading = True

            self.num_classes = 80

            size = {'yolox_s': [0.33, 0.50], 
                    'yolox_m': [0.67, 0.75],
                    'yolox_l': [1.00, 1.00],
                    'yolox_x': [1.33, 1.25]}[self.mw.videoConfigure.cmbType.currentText()]

            self.model = None
            self.model = self.get_model(self.num_classes, size[0], size[1], None).to(self.device)
            self.model.eval()

            self.ckpt_file = None
            if self.mw.videoConfigure.chkAuto.isChecked():
                self.ckpt_file = self.get_auto_ckpt_filename()
                cache = Path(self.ckpt_file)

                if not cache.is_file():
                    self.mw.videoConfigure.signals.showWaitDialog.emit()
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    model_name = self.mw.videoConfigure.cmbType.currentText()
                    link = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/" + model_name + ".pth"
                    if os.path.split(sys.executable)[1] == "pythonw.exe":
                        torch.hub.download_url_to_file(link, self.ckpt_file, progress=False)
                    else:
                        torch.hub.download_url_to_file(link, self.ckpt_file)
                    self.mw.videoConfigure.signals.hideWaitDialog.emit()
            else:
                self.ckpt_file = self.mw.videoConfigure.txtFilename.text()

            self.model.load_state_dict(torch.load(self.ckpt_file, map_location="cpu")["model"])

            res = int(self.mw.videoConfigure.cmbRes.currentText())
            self.model(torch.zeros(1, 3, res, res).to(self.device))
            self.mw.glWidget.model_loading = False

        except:
            logger.exception(MODULE_NAME + " initialization failure")
            self.mw.signals.error.emit(MODULE_NAME + " initialization failure, please check logs for details")
            for player in self.mw.pm.players:
                player.request_reconnect = False
                player.running = False
            self.mw.glWidget.model_loading = False
                
    def __call__(self, F, player):
        try:

            if not F or not player or self.mw.videoConfigure.name != MODULE_NAME:
                self.mw.videoConfigure.selTargets.barLevel.setLevel(0)
                self.mw.videoConfigure.selTargets.indAlarm.setState(0)
                return

            img = np.array(F, copy=False)

            res = int(self.mw.videoConfigure.cmbRes.currentText())
            test_size = (res, res)
            ratio = min(test_size[0] / img.shape[0], test_size[1] / img.shape[1])
            inf_shape = (int(img.shape[0] * ratio), int(img.shape[1] * ratio))
            bottom = test_size[0] - inf_shape[0]
            side = test_size[1] - inf_shape[1]
            pad = (0, 0, side, bottom)

            timg = functional.to_tensor(img).to(self.device)
            timg *= 255
            timg = functional.resize(timg, inf_shape)
            timg = functional.pad(timg, pad, 114)
            timg = timg.unsqueeze(0)

            camera = self.mw.cameraPanel.getCamera(player.uri)
            if not self.mw.videoConfigure.isModelSettings(player.videoModelSettings):
                if player.isCameraStream():
                    if camera:
                        if not self.mw.videoConfigure.isModelSettings(camera.videoModelSettings):
                            camera.videoModelSettings = YoloxSettings(self.mw, camera)
                        player.videoModelSettings = camera.videoModelSettings
                else:
                    if not self.mw.videoConfigure.isModelSettings(self.mw.filePanel.videoModelSettings):
                        self.mw.filePanel.videoModelSettings = YoloxSettings(self.mw)
                    player.videoModelSettings = self.mw.filePanel.videoModelSettings

            if not player.videoModelSettings:
                raise Exception("Unable to set video model parameters for player")

            confthre = player.videoModelSettings.confidence / 100
            nmsthre = 0.65

            while self.lock:
                sleep(0.001)
            
            self.lock = True
            with torch.no_grad():
                outputs = self.model(timg)
                outputs = postprocess(outputs, self.num_classes, confthre, nmsthre)
            self.lock = False

            output = None
            if outputs[0] is not None:
                output = outputs[0].cpu().numpy().astype(float)
                output[:, 0:4] /= ratio
                output[:, 4] *= output[:, 5]
                output = np.delete(output, 5, 1)

            result = player.processModelOutput(output)
            frame_rate = player.getVideoFrameRate()
            if frame_rate <= 0:
                profile = self.mw.cameraPanel.getProfile(player.uri)
                if profile:
                    frame_rate = profile.frame_rate()

            gain = 1
            if frame_rate:
                gain = player.videoModelSettings.gain / frame_rate

            alarmState = result * gain >= 1.0

            if self.mw.glWidget.focused_uri == player.uri:
                self.mw.videoConfigure.selTargets.barLevel.setLevel(result * gain)
                if alarmState:
                    self.mw.videoConfigure.selTargets.indAlarm.setState(1)
                    
            player.handleAlarm(alarmState)

            # restart the model if changed during run time
            tmp = None
            if self.mw.videoConfigure.chkAuto.isChecked():
                tmp = self.get_auto_ckpt_filename()
            else:
                tmp = self.mw.videoConfigure.txtFilename.text()
            if self.ckpt_file != tmp:
                self.__init__(self.mw)

        except Exception as ex:
            if self.last_ex != str(ex) and self.mw.videoConfigure.name == MODULE_NAME:
                logger.exception(MODULE_NAME + " runtime error")
            self.last_ex = str(ex)

    def get_auto_ckpt_filename(self):
        return torch.hub.get_dir() + "/checkpoints/" + self.mw.videoConfigure.cmbType.currentText() + ".pth"

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
