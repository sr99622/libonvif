#********************************************************************
# libonvif/onvif-gui/onvif_gui/enums.py
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

from enum import Enum

class ProxyType(Enum):
    STAND_ALONE = 0
    SERVER = 1
    CLIENT = 2

class StreamState(Enum):
    IDLE = 0
    CONNECTING = 1
    CONNECTED = 2
    INVALID = 3

class MediaSource(Enum):
    CAMERA = 0
    FILE = 1
    
class Style(Enum):
    DARK = 0
    LIGHT = 1

class Occurence(Enum):
    BEFORE = 0
    DURING = 1
    AFTER = 2

class PkgType(Enum):
    NATIVE = 0
    FLATPAK = 1
    SNAP = 2
