#********************************************************************
# onvif-gui/onvif_gui/protocols/client.py
#
# Copyright (c) 2025  Stephen Rhodes
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

from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtCore import pyqtSignal, QObject
import libonvif as onvif
from loguru import logger
from pathlib import Path

class ClientProtocolSignals(QObject):
    error = pyqtSignal(str)

class ClientProtocols():
    def __init__(self, mw):
        self.mw = mw
        self.signals = ClientProtocolSignals()
        self.signals.error.connect(self.showMsgBox)
        self.msg_box = QMessageBox(mw)
        self.msg_box.setIcon(QMessageBox.Icon.Critical)  # Sets the critical icon
        self.msg_box.setWindowTitle("client error")  # Sets the window title
        self.msg_box.setText("Unable to complete request")  # Sets the main text
        #self.msg_box.setInformativeText(msg)  # Sets the informative text (if needed)
        self.msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)  # Adds an OK button
        self.msg_box.setMinimumWidth(650)
    def callback(self, arg):
        try:
            #print("LEN ARG", len(arg))
            index = bytearray(arg).find(b'\r\n')
            #print(index)
            msg = bytearray(arg[:index]).decode('utf-8')
            payload_size = (len(arg) - index - 2)
            #print("payload size", payload_size)
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

            if cmd == "SNAPSHOT":
                #print("SNAPSHOT SERVER RETURN")
                filename = configs[0]
                #print("filename", filename)
                if payload_size:
                    index += 2
                    with open(filename, 'wb') as file:
                        file.write(bytearray(arg[index:]))
                else:
                    path = Path(filename)
                    camera_name = path.parent.name
                    #print("CAMERA NAME", camera_name)
                    if camera := self.mw.cameraPanel.getCameraByName(camera_name):
                        if profile := camera.getDisplayProfile():
                            if player := self.mw.pm.getPlayer(profile.uri()):
                                #print("FOUND PLAYER")
                                if player.image:
                                    player.image.save(filename)

        except Exception as ex:
            print("EXCEPTION ", ex)
            return

    def error(self, msg):
        logger.error(f'Client protocol error: {msg}')
        #self.signals.error.emit(msg)

    def showMsgBox(self, msg):
        #QMessageBox.critical(self.mw, "Unable to complete request", msg)
        self.msg_box.setText(msg) 
        self.msg_box.exec()
