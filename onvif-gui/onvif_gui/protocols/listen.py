#********************************************************************
# libonvif/onvif-gui/onvif_gui/protocols/listen.py
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

from loguru import logger
from time import sleep
from datetime import datetime

class Detection():
    def __init__(self, boxes, alarm, width, height, timestamp):
        self.boxes = boxes
        self.alarm = alarm
        self.width = width
        self.height = height
        self.timestamp = timestamp

class ListenProtocols():
    def __init__(self, mw):
        self.mw = mw
        self.cameras = {}
        self.thread_lock = False
        self.detections = {}
        self.last_timestamp = ""

    def error(self, msg):
        if msg.find("WSACancelBlockingCall") < 0:
            logger.error(msg)

    def lock(self):
        while self.thread_lock:
            sleep(0.001)
        self.thread_lock = True

    def unlock(self):
        self.thread_lock = False

    def getDetection(self, uri):
        result = None
        self.lock()
        result = self.detections.get(uri)
        self.unlock()
        return result
    
    def setDetection(self, uri, detection):
        self.lock()
        self.detections[uri] = detection
        self.unlock()

    def callback(self, msg):
        arguments = msg.split("\n\n")
        timestamp = arguments.pop(0)
        if timestamp == self.last_timestamp:
            #print("DUPLICATE PACKET FOUND")
            return
        
        self.last_timestamp = timestamp
        cmd = arguments.pop(0)

        if cmd == "ALARMS":
            #print("ALARMS", arguments)
            self.mw.alarm_states = arguments
            self.mw.last_alarm = datetime.now()
