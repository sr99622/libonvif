from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtCore import pyqtSignal, QObject
import libonvif as onvif
from loguru import logger

class ClientProtocolSignals(QObject):
    error = pyqtSignal(str)

class ClientProtocols():
    def __init__(self, mw):
        self.mw = mw
        self.signals = ClientProtocolSignals()
        self.signals.error.connect(self.showMsgBox)

    def callback(self, msg):
        configs = msg.split("\n\n")
        cmd = configs.pop(0)

        if cmd == "GET CAMERAS":
            for config in configs:
                if len(config):
                    profiles = config.split("\n")
                    onvif_data = None
                    for idx, profile in enumerate(profiles):
                        if idx == 0:
                            onvif_data = onvif.Data(profile)
                        data = onvif.Data(profile)
                        onvif_data.addProfile(data)
                    if onvif_data:
                        self.mw.cameraPanel.getProxyData(onvif_data)
            self.mw.viewer_cameras_filled = True

        if cmd == "UPDATE":
            data = onvif.Data(configs[0])
            if camera := self.mw.cameraPanel.getCameraBySerialNumber(data.serial_number()):
                camera.syncData(data)

    def error(self, msg):
        logger.error(f'Client protocol error: {msg}')
        self.signals.error.emit(msg)

    def showMsgBox(self, msg):
        QMessageBox.critical(self.mw, "Unable to complete request", msg)
