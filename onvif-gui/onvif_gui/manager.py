#/********************************************************************
# onvif-gui/onvif_gui/manager.py 
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
        self.players = {}
        self.ordinals = {}
        self.sizes = {}
        self.thread_lock = False
        self.mw = mw

    def lock(self):
        # the lock protects the player list
        while self.thread_lock:
            sleep(0.001)
        self.thread_lock = True

    def unlock(self):
        self.thread_lock = False

    def startPlayer(self, player):
        self.lock()
        if not player.uri in self.ordinals.keys():
            ordinal = self.nextOrdinal()
    
            if player.isCameraStream():
                if camera := self.mw.cameraPanel.getCamera(player.uri):
                    if camera.companionURI(player.uri) in self.ordinals.keys():
                        ordinal = self.ordinals[camera.companionURI(player.uri)]
                    camera.setOrdinal(ordinal)
            
            self.ordinals[player.uri] = ordinal

        self.players[player.uri] = player
        player.start()
        self.unlock()

    def nextOrdinal(self):
        values = set(self.ordinals.values())
        size = len(values)
        for i in range(size):
            if not i in values:
                return i
        return size
        
    def getPlayer(self, uri):
        if not uri: return None
        return self.players.get(uri)
    
    def getCurrentPlayer(self):
        return self.getPlayer(self.mw.glWidget.focused_uri)
    
    def countPlayers(self):
        return len(self.players)
    
    def waitForAllPlayersClosed(self):
        count = 0
        while self.countPlayers():
            sleep(0.1)
            count += 1
            if count > 50:
                raise RuntimeError("Not all players closed in the alloted time")
    
    def hasFilePlayers(self):
        result = False
        self.lock()
        for player in self.players.values():
            if not player.isCameraStream():
                result = True
                break
        self.unlock()
        return result
    
    def getStreamPairProfiles(self, uri):
        result = []
        if uri:
            if camera := self.mw.cameraPanel.getCamera(uri):
                if displayProfile := camera.getDisplayProfile():
                    result.append(displayProfile)
                if camera.displayProfileIndex() != camera.recordProfileIndex():
                    if recordProfile := camera.getRecordProfile():
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
            if player := self.getPlayer(profile.uri()):
                result.append(player)
        return result
    
    def getStreamPairTimers(self, uri):
        result = []
        profiles = self.getStreamPairProfiles(uri)
        for profile in profiles:
            if timer := self.mw.timers.get(profile.uri(), None):
                result.append(timer)
        return result
    
    def removeKeys(self, uri):
        if not uri: return
        if uri in self.ordinals:
            del self.ordinals[uri]
        if uri in self.sizes:
            del self.sizes[uri]
    
    def recording(self):
        self.lock()
        result = False
        for player in self.players.values():
            if player.isRecording():
                result = True
                break
        self.unlock()
        return result

    def removePlayer(self, uri):
        if not uri: return
        self.lock()
        if uri in self.players:
            if not self.players[uri].request_reconnect:
                self.removeKeys(uri)
            self.players[uri].lock()
            del self.players[uri]

        recording = False
        for player in self.players.values():
            if player.isRecording():
                recording = True
                break
        if not recording:
            self.mw.enable_disk_mgmt = False

        if not len(self.players):
            self.mw.glWidget.force_clear = True

        self.unlock()

    def playerShutdownWait(self, uri):
        if not uri: return
        if player := self.getPlayer(uri):
            player.requestShutdown()
            count = 0
            while uri in self.players:
                sleep(0.01)
                count += 1
                if count > 100:
                    logger.error(f'Player did not complete shut down during allocated time interval: {uri}')
                    break

    def getMostCommonAspectRatio(self):
        ratio_counter = {}
        for size in self.sizes.values():
            ratio = int(100 * size.width() / size.height())
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
    
    def getBlankSpace(self, canvas_size: QSize) -> list[QRectF]:
        # after determining the size of the composite image which is the aggregate containing all streams, 
        # compute the rectangles required to fill out the blank space not occupied by any current stream

        ar = self.getMostCommonAspectRatio()
        if ar == -1:
            ar = 177
            
        aspect_ratio = ar / 100
        blanks = []
        num_rows, num_cols = self.computeRowsCols(canvas_size, aspect_ratio)
        if not num_rows:
            blanks.append(QRectF(QPointF(0, 0), QSizeF(canvas_size)))
            return blanks
        
        composite_size = QSizeF(num_cols * aspect_ratio, num_rows)
        composite_size.scale(QSizeF(canvas_size), Qt.AspectRatioMode.KeepAspectRatio)
        im_w = composite_size.toSize().width()
        im_h = composite_size.toSize().height()
        cv_w = canvas_size.width()
        cv_h = canvas_size.height()
        if im_h == cv_h:
            blank_w = (cv_w - im_w)/2
            blanks.append(QRectF(0, 0, blank_w, im_h))
            blanks.append(QRectF(im_w + blank_w, 0, blank_w, im_h))
        if im_w == cv_w:
            blank_h = (cv_h - im_h)/2
            blanks.append(QRectF(0, 0, im_w, blank_h))
            blanks.append(QRectF(0, im_h + blank_h, im_w, blank_h))
        for i in range(num_rows * num_cols):
            if i not in self.ordinals.values():
                blanks.append(self.rectForOrdinal(i, canvas_size, aspect_ratio, num_rows, num_cols))
        return blanks

    def rectForOrdinal(self, ordinal: int, canvas_size: QSize, aspect_ratio: float, num_rows: int, num_cols: int) -> QRectF:
        if not num_rows or ordinal < 0:
            return QRectF(QPointF(0, 0, QSizeF(canvas_size)))

        col = ordinal % num_cols
        row = int(ordinal / num_cols)

        composite_size = QSizeF(num_cols * aspect_ratio, num_rows)
        composite_size.scale(QSizeF(canvas_size), Qt.AspectRatioMode.KeepAspectRatio)

        cell_width = composite_size.width() / num_cols
        cell_height = composite_size.height() / num_rows

        image_size = QSizeF(aspect_ratio, 1)
        image_size.scale(cell_width, cell_height, Qt.AspectRatioMode.KeepAspectRatio)
        w = image_size.width()
        h = image_size.height()

        x_offset = (canvas_size.width() - composite_size.width() + (cell_width - w)) / 2
        y_offset = (canvas_size.height() - composite_size.height() + (cell_height - h)) / 2
        
        x = (col * cell_width) + x_offset
        y = (row * cell_height) + y_offset

        return QRectF(x, y, w, h)


    def computeRowsCols(self, size_canvas, aspect_ratio):
        num_cells = len(set(self.ordinals.values()))

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
            if not layout.width():
                continue
            composite = (aspect_ratio * layout.height()) / layout.width()
            if not composite or not size_canvas.height():
                continue
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
        if ar == -1:
            if player := self.getPlayer(uri):
                if player.desired_aspect:
                    ar = player.desired_aspect
                else:
                    ar = player.width() / player.height()
        
        num_rows, num_cols = self.computeRowsCols(canvas_size, ar / 100)
        if num_cols == 0:
            return QRectF(QPointF(0, 0), QSizeF(canvas_size))

        ordinal = -1
        if uri in self.ordinals.keys():
            ordinal = self.ordinals[uri]
        else:
            return QRectF(QPointF(0, 0), QSizeF(canvas_size))

        if ordinal > num_rows * num_cols - 1:
            ordinal = self.nextOrdinal()
            uris = self.getStreamPairURIs(uri)
            for u in uris:
                if u in self.ordinals.keys():
                    self.ordinals[u] = ordinal
        
        col = ordinal % num_cols
        row = int(ordinal / num_cols)

        composite_size = QSizeF()
        if num_rows:
            composite_size = QSizeF(num_cols * ar / 100, num_rows)
            composite_size.scale(QSizeF(canvas_size), Qt.AspectRatioMode.KeepAspectRatio)

        cell_width = composite_size.width() / num_cols
        cell_height = composite_size.height() / num_rows

        image_size = QSizeF(ar, 100)
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
