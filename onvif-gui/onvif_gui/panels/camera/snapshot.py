from time import sleep
from loguru import logger
import libonvif as onvif
from pathlib import Path
import os
import subprocess
from datetime import datetime
from onvif_gui.enums import ProxyType, SnapshotAuth
import platform
import webbrowser
import requests
from requests.auth import HTTPDigestAuth
from urllib.parse import urlparse, parse_qs
import threading


class Snapshot():
    def __init__(self, mw):
        self.mw = mw

    def getSnapshot(self, profile, filename, auth_type):
        parsed = urlparse(profile.snapshot_uri())

        if auth_type == SnapshotAuth.NOT_WORKING:
            return False

        if auth_type == SnapshotAuth.UNKNOWN:
            if self.getSnapshot(profile, filename, SnapshotAuth.BASIC):
                camera = self.mw.cameraPanel.getCamera(profile.uri())
                camera.setSnapshotAuth(SnapshotAuth.BASIC)
                logger.debug(f"camera {camera.name()} snapshot auth type set to BASIC")
                return True

            else:
                if self.getSnapshot(profile, filename, SnapshotAuth.DIGEST):
                    camera = self.mw.cameraPanel.getCamera(profile.uri())
                    camera.setSnapshotAuth(SnapshotAuth.DIGEST)
                    logger.debug(f"camera {camera.name()} snapshot auth type set to DIGEST")
                    return True

        if auth_type == SnapshotAuth.BASIC:
            try:
                simple_url = f"{parsed.scheme}://{profile.username()}:{profile.password()}@{parsed.netloc}{parsed.path}?{parsed.query}"
                response = requests.get(simple_url, timeout=5)
                if response.status_code == 200:
                    with open(filename, 'wb') as file:
                        file.write(response.content)
                    #logger.debug(f"Image downloaded successfully as {filename}")
                    return True
                else:
                    logger.debug(f"get snapshot BASIC auth status code: {response}")
                    return False
            except Exception as ex:
                logger.debug(f'set snapshot BASIC auth exception {ex}')
            return False

        if auth_type == SnapshotAuth.DIGEST:
            try:
                base_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                params = {k: v[0] for k, v in parse_qs(parsed.query).items()}
                response = requests.get(base_url, params=params, auth=HTTPDigestAuth(profile.username(), profile.password()), timeout=5)
                if response.status_code == 200:
                    with open(filename, 'wb') as file:
                        file.write(response.content)
                    #logger.debug(f"Image downloaded successfully as {filename}")
                    return True
                else:
                    logger.debug(f"get snapshot DIGEST auth status code: {response}")
                    return False
            except Exception as ex:
                logger.debug(f'set snapshot DIGEST auth exception {ex}')
            return False

        return False

    def getSnapshotBuffered(self, profile, buffer, auth_type):
        print("Here we are at getSnapshotBuffered", auth_type)
        parsed = urlparse(profile.snapshot_uri())

        if auth_type == SnapshotAuth.NOT_WORKING:
            return False

        if auth_type == SnapshotAuth.UNKNOWN:
            if self.getSnapshotBuffered(profile, buffer, SnapshotAuth.BASIC):
                #camera = self.mw.cameraPanel.getCamera(profile.uri())
                #camera.setSnapshotAuth(SnapshotAuth.BASIC)
                #logger.debug(f"camera {camera.name()} snapshot auth type set to BASIC")
                return True

            else:
                if self.getSnapshotBuffered(profile, buffer, SnapshotAuth.DIGEST):
                    #camera = self.mw.cameraPanel.getCamera(profile.uri())
                    #camera.setSnapshotAuth(SnapshotAuth.DIGEST)
                    #logger.debug(f"camera {camera.name()} snapshot auth type set to DIGEST")
                    return True

        if auth_type == SnapshotAuth.BASIC:
            try:
                simple_url = f"{parsed.scheme}://{profile.username()}:{profile.password()}@{parsed.netloc}{parsed.path}?{parsed.query}"
                response = requests.get(simple_url, timeout=5)
                if response.status_code == 200:
                    
                    print("Writing buffer from BASIC AUTH")
                    buffer.write(response.content)
                    
                    #logger.debug(f"Image downloaded successfully as {filename}")
                    return True
                else:
                    logger.debug(f"get snapshot BASIC auth status code: {response}")
                    return False
            except Exception as ex:
                logger.debug(f'set snapshot BASIC auth exception {ex}')
            return False

        if auth_type == SnapshotAuth.DIGEST:
            try:
                base_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                print("BASE URL:", base_url)
                params = {k: v[0] for k, v in parse_qs(parsed.query).items()}
                response = requests.get(base_url, params=params, auth=HTTPDigestAuth(profile.username(), profile.password()), timeout=5)
                if response.status_code == 200:
                    
                    print("Writing buffer from DIGEST AUTH")
                    buffer.write(response.content)
                    #with open("/home/stephen/Pictures/sample.jpg", "wb") as file:
                        #file.write(response.content)
                        #file.write(buffer.getvalue())
                    #print("where is the file")

                    #logger.debug(f"Image downloaded successfully as {filename}")
                    return True
                else:
                    logger.debug(f"get snapshot DIGEST auth status code: {response}")
                    return False
            except Exception as ex:
                logger.debug(f'set snapshot DIGEST auth exception {ex}')
            return False

        return False

    def getBufferedSnapshot(self, profile, buffer, camera):
        camera.setSnapshotAuth(SnapshotAuth.UNKNOWN)
        print("call came into getBufferedSnapshot", camera.name(), camera.snapshotAuth)
        if not self.getSnapshotBuffered(profile, buffer, camera.snapshotAuth):
            if camera.snapshotAuth is not SnapshotAuth.NOT_WORKING:
                logger.debug(f'camera {camera.name()} snapshot auth type set to NOT_WORKING')
                camera.setSnapshotAuth(SnapshotAuth.NOT_WORKING)
            #player.image.save(filename)
            #return buffer


    def __call__(self, profile, filename, camera, player):
        if not camera.systemTabSettings.remote_snapshot:
            player.image.save(filename)
            return
        if self.mw.settingsPanel.proxy.proxyType == ProxyType.CLIENT:
            print("CLIENT PROXY", camera.name())
            profile.setUserData(filename)
            self.mw.client.transmit(bytearray(f"SNAPSHOT\n\n{profile.toJSON()}\r\n", 'utf-8'))
            return
        if not self.getSnapshot(profile, filename, camera.snapshotAuth):
            if camera.snapshotAuth is not SnapshotAuth.NOT_WORKING:
                logger.debug(f'camera {camera.name()} snapshot auth type set to NOT_WORKING')
                camera.setSnapshotAuth(SnapshotAuth.NOT_WORKING)
            player.image.save(filename)
            #logger.debug(f'Raw stream image saved as {filename}')
