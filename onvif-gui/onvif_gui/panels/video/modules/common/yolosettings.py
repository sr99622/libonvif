#/********************************************************************
# libonvif/onvif-gui/panels/video/modules/common/yolosettings.py 
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

class YoloSettings():
    def __init__(self, mw, camera=None, module_name=None):
        self.camera = camera
        self.mw = mw
        self.module_name = module_name
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
        key = f'{self.id}/{self.module_name}/Targets'
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
        key = f'{self.id}/{self.module_name}/Targets'
        self.targets.clear()
        if len(targets):
            tmp = targets.split(":")
            for t in tmp:
                self.targets.append(int(t))
        self.mw.settings.setValue(key, targets)

    def getModelConfidence(self):
        key = f'{self.id}/{self.module_name}/ConfidenceThreshold'
        return int(self.mw.settings.value(key, 50))
    
    def setModelConfidence(self, value):
        key = f'{self.id}/{self.module_name}/ConfidenceThreshold'
        self.confidence = value
        self.mw.settings.setValue(key, value)

    def getModelOutputLimit(self):
        key = f'{self.id}/{self.module_name}/ModelOutputLimit'
        return int(self.mw.settings.value(key, 0))
    
    def setModelOutputLimit(self, value):
        key = f'{self.id}/{self.module_name}/ModelOutputLimit'
        self.limit = value
        self.mw.settings.setValue(key, value)

    def getModelShowBoxes(self):
        key = f'{self.id}/{self.module_name}/ModelShowBoxes'
        return bool(int(self.mw.settings.value(key, 1)))
    
    def setModelShowBoxes(self, value):
        key = f'{self.id}/{self.module_name}/ModelShowBoxes'
        self.show = value
        self.mw.settings.setValue(key, int(value))

    def getSkipFrames(self):
        key = f'{self.id}/{self.module_name}/SkipFrames'
        return int(self.mw.settings.value(key, 0))

    def setSkipFrames(self, value):
        key = f'{self.id}/{self.module_name}/SkipFrames'
        self.skipFrames = int(value)
        self.mw.settings.setValue(key, int(value))

    def getSampleSize(self):
        key = f'{self.id}/{self.module_name}/SampleSize'
        return int(self.mw.settings.value(key, 1))

    def setSampleSize(self, value):
        key = f'{self.id}/{self.module_name}/SampleSize'
        self.sampleSize = int(value)
        self.mw.settings.setValue(key, int(value))        

