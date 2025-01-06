#*******************************************************************************
# libonvif/onvif-gui/run.py
#
# Copyright (c) 2023 Stephen Rhodes 
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
#******************************************************************************/

import sys
from PyQt6.QtWidgets import QApplication
from gui import MainWindow

clear_settings = False
if len(sys.argv) > 1:
    if str(sys.argv[1]) == "--clear":
        clear_settings = True

app = QApplication(sys.argv)

app.setStyle('Fusion')
window = MainWindow(clear_settings)

window.show()
app.exec()
