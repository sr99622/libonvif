#/********************************************************************
# onvif-gui/onvif_gui/panels/settings/settingspanel.py 
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

from PyQt6.QtWidgets import QGridLayout, QWidget, QTabWidget

from . import DiscoverOptions, GeneralOptions, StorageOptions, \
    AlarmOptions, ProxyOptions

class SettingsPanel(QWidget):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw

        self.tab = QTabWidget()

        self.general = GeneralOptions(mw)
        self.discover = DiscoverOptions(mw)
        self.storage = StorageOptions(mw)
        self.proxy = ProxyOptions(mw)
        self.alarm = AlarmOptions(mw)

        self.tab.addTab(self.general, "General")
        self.tab.addTab(self.discover, "Discover")
        self.tab.addTab(self.storage, "Storage")
        self.tab.addTab(self.proxy, "Proxy")
        self.tab.addTab(self.alarm, "Alarm")

        lytMain = QGridLayout(self)
        lytMain.addWidget(self.tab,   0, 0, 1, 1)

    def onMediaStarted(self, uri):
        if self.mw.pm.countPlayers():
            self.general.btnCloseAll.setText("Close All")

    def onMediaStopped(self, uri):
        if not self.mw.pm.countPlayers():
            self.general.btnCloseAll.setText("Start All")
