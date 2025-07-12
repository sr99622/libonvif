#/********************************************************************
# libonvif/onvif-gui/onvif_gui/panels/file/treeview.py 
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

import os
from PyQt6.QtWidgets import QTreeView
from PyQt6.QtCore import Qt

class TreeView(QTreeView):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw

    def keyPressEvent(self, event):

        pass_through = True

        match event.key():
            case Qt.Key.Key_Return:
                index = self.currentIndex()
                if index.isValid():
                    fileInfo = self.model().fileInfo(index)
                    if fileInfo.isFile():
                        if self.model().isReadOnly():
                            for player in self.mw.pm.players:
                                if not player.isCameraStream():
                                    self.mw.pm.playerShutdownWait(player.uri)
                            self.mw.filePanel.control.btnPlayClicked()
                    else:
                        if self.isExpanded(index):
                            self.collapse(index)
                        else:
                            self.expand(index)

            case Qt.Key.Key_Space:
                index = self.currentIndex()
                if index.isValid():
                    fileInfo = self.model().fileInfo(index)
                    if fileInfo.isFile():
                        if self.model().isReadOnly():
                            if player := self.mw.filePanel.getCurrentlyPlayingFile():
                                player.togglePaused()

            case Qt.Key.Key_Escape:
                if self.model().isReadOnly():
                    self.mw.filePanel.control.btnStopClicked()
                else:
                    self.model().setReadOnly(True)
        
            case Qt.Key.Key_F1:
                self.mw.filePanel.onMenuInfo()

            case Qt.Key.Key_F2:
                self.mw.filePanel.onMenuRename()

            case Qt.Key.Key_Delete:
                self.mw.filePanel.onMenuRemove()

            case Qt.Key.Key_Left:
                self.mw.filePanel.rewind()
                pass_through = False

            case Qt.Key.Key_Right:
                self.mw.filePanel.fastForward()
                pass_through = False
        
        if pass_through:
            return super().keyPressEvent(event)

    def currentChanged(self, newItem, oldItem):
        if newItem.data():
            fullPath = os.path.join(self.model().rootPath(), newItem.data())
            if os.path.isfile(fullPath):
                if player := self.mw.pm.getPlayer(str(fullPath)):
                    self.mw.glWidget.focused_uri = player.uri
            self.scrollTo(self.currentIndex())
