#********************************************************************
# onvif-gui/onvif_gui/protocols/server.py
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

import libonvif as onvif
from loguru import logger
import numpy as np
import io
from onvif_gui.panels.camera import Snapshot

class ServerProtocols():
    def __init__(self, mw):
        self.mw = mw
        self.snapshot = Snapshot(mw)

    def callback(self, msg):
        buffer = io.BytesIO()
        args = msg.split("\n\n")

        match args[0]:
            case "GET CAMERAS":
                lstCamera = self.mw.cameraPanel.lstCamera
                cameras = [lstCamera.item(x) for x in range(lstCamera.count())]
                buffer.write(bytearray("GET CAMERAS\n\n", 'utf-8'))
                for c_idx, camera in enumerate(cameras):
                    for p_idx, profile in enumerate(camera.profiles):
                        buffer.write(bytearray(profile.toJSON(), 'utf-8'))
                        if c_idx < len(cameras) - 1 :
                            buffer.write(bytearray("\n", 'utf-8'))
                        else:
                            if p_idx < len(camera.profiles) - 1:
                                buffer.write(bytearray("\n", 'utf-8'))
                    if c_idx < len(cameras) - 1:
                        buffer.write(bytearray("\n", 'utf-8'))
                buffer.write(bytearray("\r\n", 'utf-8'))

            case "UPDATE VIDEO":
                data = onvif.Data(args[1])
                data.updateVideo()
                buffer.write(self.resolve(data))

            case "UPDATE AUDIO":
                data = onvif.Data(args[1])
                data.updateAudio()
                buffer.write(self.resolve(data))

            case "UPDATE IMAGE":
                data = onvif.Data(args[1])
                data.updateImage()
                buffer.write(self.resolve(data))

            case "MOVE":
                data = onvif.Data(args[1])
                data.move()
                buffer.write(bytearray("PTZ\r\n", 'utf-8'))

            case "STOP":
                data = onvif.Data(args[1])
                data.stop()
                buffer.write(bytearray("PTZ\r\n", 'utf-8'))

            case "GOTO PRESET":
                data = onvif.Data(args[1])
                data.set()
                buffer.write(bytearray("GOTO PRESET\r\n", 'utf-8'))

            case "SET PRESET":
                data = onvif.Data(args[1])
                data.setGotoPreset()
                buffer.write(bytearray("SET PRESET\r\n", 'utf-8'))

            case "REBOOT":
                data = onvif.Data(args[1])
                data.reboot()
                buffer.write(bytearray("REBOOT\r\n", 'utf-8'))

            case "SYNC TIME":
                data = onvif.Data(args[1])
                if camera := self.mw.cameraPanel.getCameraBySerialNumber(data.serial_number()):
                    camera.onvif_data.updateTime()
                buffer.write(bytearray("SYNC TIME\r\n", 'utf-8'))

            case "SNAPSHOT":
                profile = onvif.Data(args[1])
                remote = self.mw.settingsPanel.proxy.lblServer.text().split()[0].strip()
                key = f'{remote}{profile.serial_number()}/{profile.profile()}'
                if camera := self.mw.cameraPanel.getCamera(key):
                    buffer.write(bytearray("SNAPSHOT\n\n", 'utf-8'))
                    buffer.write(bytearray(profile.user_data() + "\r\n", 'utf-8'))
                    self.snapshot.getBufferedSnapshot(profile, buffer, camera)

        return np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    
    def resolve(self, data):
        if camera := self.mw.cameraPanel.getCameraBySerialNumber(data.serial_number()):
            camera.syncData(data)
        return bytearray("UPDATE\n\n" + data.toJSON() + "\r\n", 'utf-8')

    def error(self, msg):
        logger.error(f"server protocol error: {msg}")