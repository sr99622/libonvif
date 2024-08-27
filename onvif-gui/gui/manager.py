#/********************************************************************
# libonvif/onvif-gui/gui/manager.py 
#
# Copyright (c) 2024  Stephen Rhodes
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

from time import sleep
from loguru import logger
from PyQt6.QtCore import QRectF, QSize, Qt, QSizeF, QPointF

class Manager():
    def __init__(self, mw):
        self.players = []
        self.ordinals = {}
        self.sizes = {}
        self.start_lock = False
        self.remove_lock = False
        self.mw = mw
        self.auto_start_mode = False

    def startPlayer(self, player):
        while self.start_lock:
            sleep(0.001)
        self.start_lock = True

        if not player.disable_video:
            if not player.uri in self.ordinals.keys():
                ordinal = self.getOrdinal()
                if player.isCameraStream():
                    camera = self.mw.cameraPanel.getCamera(player.uri)
                    if camera:
                        if self.auto_start_mode and camera.ordinal > -1:
                            duplicate = False
                            keys = self.ordinals.keys()
                            for key in keys:
                                c = self.mw.cameraPanel.getCamera(key)
                                if camera.ordinal == self.ordinals[key] and camera.serial_number() != c.serial_number():
                                    duplicate = True
                            if duplicate:
                                camera.ordinal = ordinal
                            else:
                                ordinal = camera.ordinal
                        else:
                            comp_uri = camera.companionURI(player.uri)
                            if comp_uri:
                                if comp_uri in self.ordinals.keys():
                                    ordinal = self.ordinals[comp_uri]
                            camera.setOrdinal(ordinal)

                self.ordinals[player.uri] = ordinal

        self.players.append(player)
        player.start()
        self.start_lock = False

    def getUniqueOrdinals(self):
        result = []
        values = self.ordinals.values()
        for value in values:
            if value not in result:
                result.append(value)
        return result

    def getOrdinal(self):
        ordinal = -1

        values = self.getUniqueOrdinals()
        for i in range(len(values)):
            if not i in values:
                ordinal = i
                break

        if ordinal == -1:
            ordinal = len(values)

        return ordinal
        
    def getPlayer(self, uri):
        result = None
        if uri:
            for player in self.players:
                if player.uri == uri:
                    result = player
                    break
        return result
    
    def getPlayerByOrdinal(self, ordinal):
        result = None
        for key, value in self.ordinals.items():
            if value == ordinal:
                result = self.getPlayer(key)
                break
        return result

    def getCurrentPlayer(self):
        return self.getPlayer(self.mw.glWidget.focused_uri)
    
    def getStreamPairProfiles(self, uri):
        result = []
        if uri:
            camera = self.mw.cameraPanel.getCamera(uri)
            if camera:
                displayProfile = camera.getDisplayProfile()
                if displayProfile:
                    result.append(displayProfile)
                if camera.displayProfileIndex() != camera.recordProfileIndex():
                    recordProfile = camera.getRecordProfile()
                    if recordProfile:
                        result.append(recordProfile)

        return result
    
    def getStreamPairURIs(self, uri):
        result = []
        profiles = self.getStreamPairProfiles(uri)
        for profile in profiles:
            result.append(profile.uri())
        return result
    
    def getStreamPairPlayers(self, uri):
        result = []
        profiles = self.getStreamPairProfiles(uri)
        for profile in profiles:
            player = self.getPlayer(profile.uri())
            if player:
                result.append(player)
        return result
    
    def getStreamPairTimers(self, uri):
        result = []
        profiles = self.getStreamPairProfiles(uri)
        for profile in profiles:
            timer = self.mw.timers.get(profile.uri(), None)
            if timer:
                result.append(timer)
        return result
    
    def removeKeys(self, uri):
        if uri in self.ordinals.keys():
            del self.ordinals[uri]
        if uri in self.sizes.keys():
            del self.sizes[uri]
    
    def removePlayer(self, uri):
        for player in self.players:
            if player.uri == uri:
                while player.rendering:
                    sleep(0.001)

                if not player.request_reconnect:
                    self.removeKeys(uri)

                self.players.remove(player)

    def playerShutdownWait(self, uri):
        player = self.getPlayer(uri)
        if player:
            player.requestShutdown()
            count = 0
            while self.getPlayer(uri):
                sleep(0.01)
                count += 1
                if count > 200:
                    logger.error(f'Player did not complete shut down during allocated time interval: {uri}')
                    break

    def getMostCommonAspectRatio(self):
        ratio_counter = {}
        for size in self.sizes.values():
            ratio = round(1000 * size.width() / size.height())
            if not ratio in ratio_counter.keys():
                ratio_counter[ratio] = 1
            else:
                ratio_counter[ratio] += 1

        highest_count_key = -1
        if ratio_counter:
            keys = list(ratio_counter.keys())
            if len(keys):
                highest_count_key = keys[0]
                highest_count = ratio_counter[highest_count_key]
            for key in keys:
                if ratio_counter[key] > highest_count:
                    highest_count = ratio_counter[key]
                    highest_count_key = key

        return highest_count_key
    
    def computeRowsCols(self, size_canvas, aspect_ratio):

        num_cells = len(self.getUniqueOrdinals())

        if self.auto_start_mode:
            num_cells = len(self.mw.cameraPanel.cached_serial_numbers)

        valid_layouts = []
        for i in range(1, num_cells+1):
            for j in range(num_cells, 0, -1):
                if ((i * j) >= num_cells):
                    if (((i-1)*j) < num_cells) and ((i*(j-1)) < num_cells):
                        valid_layouts.append(QSize(i, j))

        index = -1
        min_ratio = 0
        first_pass = True
        for i, layout in enumerate(valid_layouts):
            composite = (aspect_ratio * layout.height()) / layout.width()
            ratio = (size_canvas.width() / size_canvas.height()) / composite
            optimize = abs(1 - ratio)
            if first_pass:
                first_pass = False
                min_ratio = optimize
                index = i
            else:
                if optimize < min_ratio:
                    min_ratio = optimize
                    index = i

        if index == -1:
            return 0, 0
        
        return valid_layouts[index].width(), valid_layouts[index].height()

    def displayRect(self, uri, canvas_size):
        ar = self.getMostCommonAspectRatio()
        num_rows, num_cols = self.computeRowsCols(canvas_size, ar / 1000)
        if num_cols == 0:
            return QRectF(QPointF(0, 0), QSizeF(canvas_size))

        ordinal = -1
        if uri in self.ordinals.keys():
            ordinal = self.ordinals[uri]
        else:
            return QRectF(QPointF(0, 0), QSizeF(canvas_size))

        if ordinal > num_rows * num_cols - 1:
            ordinal = self.getOrdinal()
            uris = self.getStreamPairURIs(uri)
            for u in uris:
                if u in self.ordinals.keys():
                    self.ordinals[u] = ordinal
        
        col = ordinal % num_cols
        row = int(ordinal / num_cols)

        composite_size = QSizeF()
        if num_rows:
            composite_size = QSizeF(num_cols * ar / 1000, num_rows)
            composite_size.scale(QSizeF(canvas_size), Qt.AspectRatioMode.KeepAspectRatio)

        cell_width = composite_size.width() / num_cols
        cell_height = composite_size.height() / num_rows

        image_size = QSizeF(ar, 1000)
        if uri in self.sizes.keys():
            image_size = QSizeF(self.sizes[uri])

        image_size.scale(cell_width, cell_height, Qt.AspectRatioMode.KeepAspectRatio)
        w = image_size.width()
        h = image_size.height()

        x_offset = (canvas_size.width() - composite_size.width() + (cell_width - w)) / 2
        y_offset = (canvas_size.height() - composite_size.height() + (cell_height - h)) / 2
        
        x = (col * cell_width) + x_offset
        y = (row * cell_height) + y_offset

        return QRectF(x, y, w, h)
