import libonvif as onvif

class ServerProtocols():
    def __init__(self, mw):
        self.mw = mw

    def callback(self, msg):
        #print("server protocol callback", msg)

        args = msg.split("\n\n")

        if args[0] == "GET CAMERAS":
            lstCamera = self.mw.cameraPanel.lstCamera
            cameras = [lstCamera.item(x) for x in range(lstCamera.count())]
            result = "GET CAMERAS\n\n"
            for c_idx, camera in enumerate(cameras):
                for p_idx, profile in enumerate(camera.profiles):
                    result += profile.toJSON()
                    if c_idx < len(cameras) - 1 :
                        result += "\n"
                    else:
                        if p_idx < len(camera.profiles) - 1:
                            result += "\n"
                if c_idx < len(cameras) - 1:
                    result += "\n"

        if args[0] == "UPDATE VIDEO":
            data = onvif.Data(args[1])
            data.updateVideo()
            result = self.resolve(data)

        if args[0] == "UPDATE AUDIO":
            data = onvif.Data(args[1])
            data.updateAudio()
            result = self.resolve(data)

        if args[0] == "UPDATE IMAGE":
            data = onvif.Data(args[1])
            data.updateImage()
            result = self.resolve(data)

        if args[0] == "MOVE":
            data = onvif.Data(args[1])
            data.move()
            result = "PTZ"

        if args[0] == "STOP":
            data = onvif.Data(args[1])
            data.stop()
            result = "PTZ"

        if args[0] == "GOTO PRESET":
            data = onvif.Data(args[1])
            data.set()
            result = "GOTO PRESET"

        if args[0] == "SET PRESET":
            data = onvif.Data(args[1])
            data.setGotoPreset()
            result = "SET PRESET"

        if args[0] == "REBOOT":
            data = onvif.Data(args[1])
            data.reboot()
            result = "REBOOT"

        if args[0] == "SYNC TIME":
            data = onvif.Data(args[1])
            if camera := self.mw.cameraPanel.getCameraBySerialNumber(data.serial_number()):
                camera.onvif_data.updateTime()
            result = "SYNC TIME"

        #print("length", len(result))
        return result
    
    def resolve(self, data):
        if camera := self.mw.cameraPanel.getCameraBySerialNumber(data.serial_number()):
            camera.syncData(data)
        return "UPDATE\n\n" + data.toJSON()


    def error(self, msg):
        print("server protocol error", msg)