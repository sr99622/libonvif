#/********************************************************************
# onvif-gui/onvif_gui/panels/camera/camerapanel.py 
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

from time import sleep
from PyQt6.QtWidgets import QPushButton, QGridLayout, QWidget, QSlider, \
    QListWidget, QTabWidget, QMessageBox, QMenu, QFileDialog
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QTimer, QSettings
from . import NetworkTab, ImageTab, VideoTab, PTZTab, SystemTab, LoginDialog, \
    Session, Camera
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

class CameraList(QListWidget):
    def __init__(self, mw):
        super().__init__()
        #self.signals = CameraPanelSignals()
        self.setSortingEnabled(True)
        self.mw = mw

    def focusInEvent(self, event):
        if self.currentRow() == -1:
            self.setCurrentRow(0)
        super().focusInEvent(event)

    def keyPressEvent(self, event):
        match event.key():
            case Qt.Key.Key_Return:
                if camera := self.currentItem():
                    if not camera.editing():
                        self.itemDoubleClicked.emit(camera)
                        self.setFocus()

            case Qt.Key.Key_Escape:
                if self.mw.isFullScreen():
                    self.mw.showNormal()
                elif self.mw.focus_window and self.mw.focus_window.isVisible():
                    self.mw.focus_window.close()
                elif camera := self.currentItem():
                    players = self.mw.pm.getStreamPairPlayers(camera.uri())
                    if len(players):
                        self.mw.cameraPanel.btnStopClicked()

            case Qt.Key.Key_Delete:
                self.remove()

            case Qt.Key.Key_F2:
                self.rename()

            case Qt.Key.Key_F1:
                self.info()

            case Qt.Key.Key_A:
                if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                    self.mw.cameraPanel.btnStopAllClicked()
            case Qt.Key.Key_F:
                if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                    self.mw.cameraPanel.btnHistoryClicked()
            case Qt.Key.Key_S:
                if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                    self.mw.cameraPanel.btnSnapshotClicked()
            case Qt.Key.Key_R:
                if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                    self.mw.cameraPanel.btnRecordClicked()

        return super().keyPressEvent(event)
    
    def remove(self):
        if camera := self.currentItem():
            if self.mw.pm.getPlayer(camera.uri()):
                ret = QMessageBox.warning(self, camera.name(),
                                            "Camera is currently playing. Please stop before deleting.",
                                            QMessageBox.StandardButton.Ok)

                return
            else:
                ret = QMessageBox.warning(self, camera.name(),
                                            "Removing this camera from the list.\n"
                                            "Are you sure you want to continue?",
                                            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)

                if ret != QMessageBox.StandardButton.Ok:
                    return

            row = self.currentRow()
            if row > -1:
                if camera.filled:
                    camera = self.takeItem(row)
                    for profile in camera.profiles:
                        self.mw.proxies.pop(profile.stream_uri(), None)

                    if self.mw.settingsPanel.proxy.proxyType == ProxyType.SERVER:
                        self.mw.settingsPanel.proxy.setMediaMTXProxies()

                else:
                    ret = QMessageBox.warning(self, camera.name(),
                                                "The program is currently communicating with the camera. Please wait before deleting.",
                                                QMessageBox.StandardButton.Ok)

        if not self.count():
            data = onvif.Data()
            self.mw.cameraPanel.signals.fill.emit(data)

        self.mw.cameraPanel.saveCameraList()

    def info(self):
        msg = ""
        if camera := self.currentItem():
            players = self.mw.pm.getStreamPairPlayers(camera.uri())
            if not len(players):
                msg = "Start camera to get stream info"
            for i, player in enumerate(players):
                if i == 0:
                    msg += "<h2>Display Stream</h2>"
                    msg += player.getStreamInfo()
                    msg += "\n"
                if i == 1:
                    msg += "<h2>Record Stream</h2>"
                    msg += player.getStreamInfo()
                    msg += "\n"
        msgBox = QMessageBox(self)
        msgBox.setWindowTitle("Stream Info")
        msgBox.setText(msg)
        msgBox.setTextFormat(Qt.TextFormat.RichText)
        msgBox.exec()
    
    def rename(self):
        if camera := self.currentItem():
            camera.setFlags(camera.flags() | Qt.ItemFlag.ItemIsEditable)
            index = self.currentIndex()
            if index.isValid():
                self.edit(index)

    def password(self):
        if camera := self.currentItem():
            self.mw.settings.setValue(f'{camera.xaddrs()}/alternateUsername', camera.onvif_data.username())
            self.mw.settings.setValue(f'{camera.xaddrs()}/alternatePassword', camera.onvif_data.password())
            logger.debug(f'Alternate password set for camera {camera.name()}')

    def closeEditor(self, editor, hint):
        if camera := self.currentItem():
            camera.onvif_data.alias = editor.text()
            self.mw.settings.setValue(f'{camera.serial_number()}/Alias', editor.text())
            camera.setFlags(camera.flags() & ~Qt.ItemFlag.ItemIsEditable)
        return super().closeEditor(editor, hint)
    
    def startCamera(self):
        if camera := self.currentItem():
            self.mw.cameraPanel.onItemDoubleClicked(camera)

    def stopCamera(self):
        if camera := self.currentItem():
            self.mw.cameraPanel.onItemDoubleClicked(camera)

