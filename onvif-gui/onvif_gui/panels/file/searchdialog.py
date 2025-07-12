#/********************************************************************
# libonvif/onvif-gui/onvif_gui/panels/file/filesearchdialog.py 
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
from PyQt6.QtWidgets import QGridLayout, QWidget, \
    QLabel, QMessageBox, QAbstractItemView, \
    QDialog, QCalendarWidget, QDialogButtonBox, QComboBox, \
    QAbstractItemView
from PyQt6.QtGui import QBrush
from PyQt6.QtCore import Qt
from loguru import logger
from datetime import datetime
from onvif_gui.enums import Occurence

FORMAT = "%Y%m%d%H%M%S"

class FileSearchDialog(QDialog):
    def __init__(self, mw):
        super().__init__(mw)
        self.mw = mw
        self.geometryKey = "FileSearchDialog/geometry"
        self.timeFormat = ""
        self.positionInitialized = False
        self.setWindowTitle("File Search")

        self.matching_file = None
        self.closest_before = None
        self.closest_after = None

        if rect := self.mw.settings.value(self.geometryKey):
            if rect.width() and rect.height():
                self.setGeometry(rect)
                self.positionInitialized = True

        self.cameras = QComboBox()
        self.pnlCameras = QWidget()
        lytCamera = QGridLayout(self.pnlCameras)
        lytCamera.addWidget(QLabel("Camera"),  0, 0, 1, 1)
        lytCamera.addWidget(self.cameras,      0, 1, 1, 1)
        lytCamera.setColumnStretch(1, 10)

        self.calendar = QCalendarWidget()
        self.calendar.setVerticalHeaderFormat(QCalendarWidget.VerticalHeaderFormat.NoVerticalHeader)
        self.calendar.setStyleSheet("QTableView{selection-background-color: darkGreen; selection-color: lightGray}")
        format = self.calendar.weekdayTextFormat(Qt.DayOfWeek.Saturday)
        format.setForeground(QBrush(Qt.GlobalColor.white, Qt.BrushStyle.SolidPattern))
        self.calendar.setWeekdayTextFormat(Qt.DayOfWeek.Saturday, format)
        self.calendar.setWeekdayTextFormat(Qt.DayOfWeek.Sunday, format)
        
        self.hour = QComboBox()
        self.hour.setMinimumWidth(50)
        hours = []
        for i in range(1, 13):
            hours.append(str(i))
        self.hour.addItems(hours)
        self.hour.setCurrentText(datetime.now().strftime("%I").lstrip('0'))

        self.minute = QComboBox()
        self.minute.setMinimumWidth(50)
        minutes = []
        for i in range(0, 60):
            if i < 10:
                minutes.append(f'0{i}')
            else:
                minutes.append(str(i))
        self.minute.addItems(minutes)
        # no good solutions for size problem found
        #self.minute.setStyleSheet(" QComboBox { combobox-popup: 0 } ")
        #self.minute.setStyleSheet(" QComboBox QListView { max-height: 100px;}")
        self.minute.setCurrentText(datetime.now().strftime("%M"))

        self.AM_PM = QComboBox()
        self.AM_PM.setMinimumWidth(60)
        self.AM_PM.addItems(["AM", "PM"])
        self.AM_PM.setCurrentText(datetime.now().strftime("%p"))

        self.pnlTime = QWidget()
        lytTime = QGridLayout(self.pnlTime)
        lytTime.addWidget(QLabel(),        0, 0, 1, 1)
        lytTime.addWidget(self.hour,       0, 1, 1, 1)
        lytTime.addWidget(QLabel(" : "),   0, 2, 1, 1)
        lytTime.addWidget(self.minute,     0, 3, 1, 1)
        lytTime.addWidget(self.AM_PM,      0, 4, 1, 1)
        lytTime.addWidget(QLabel(),        0, 5, 1, 1)
        lytTime.setColumnStretch(0, 2)
        lytTime.setColumnStretch(0, 5)
        lytTime.setColumnStretch(0, 0)
        lytTime.setColumnStretch(0, 5)
        lytTime.setColumnStretch(0, 2)

        self.buttonBox = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttonBox.button(QDialogButtonBox.StandardButton.Cancel).setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.buttonBox.button(QDialogButtonBox.StandardButton.Ok).setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.buttonBox.rejected.connect(self.reject)
        self.buttonBox.accepted.connect(self.accept)

        instruct = QLabel("Select a Camera, Date and Time")

        lytMain = QGridLayout(self)
        lytMain.addWidget(instruct,           0, 0, 1, 1, Qt.AlignmentFlag.AlignCenter)
        lytMain.addWidget(self.pnlCameras,    1, 0, 1, 1)
        lytMain.addWidget(self.calendar,      2, 0, 1, 1)
        lytMain.addWidget(self.pnlTime,       3, 0, 1, 1)
        lytMain.addWidget(self.buttonBox,     4, 0, 1, 1)

    def moveEvent(self, event):
        # for adding persistence in the future maybe
        return super().moveEvent(event)

    def resizeEvent(self, event):
        # for adding persistence in the future maybe
        return super().resizeEvent(event)

    def reject(self):
        self.matching_file = None
        self.closest_before = None
        self.closest_after = None
        self.hide()

    def accept(self):
        self.matching_file = None
        self.closest_before = None
        self.closest_after = None
        try:
            selected = self.getSelectedDate()
            main_directory = self.mw.filePanel.dirArchive.txtDirectory.text()
            sub_directory = self.cameras.currentText()
            self.findFileForEventTime(selected, main_directory,  sub_directory)

            if self.matching_file:
                self.selectFileInTree(os.path.join(main_directory, sub_directory), self.matching_file)
            self.hide()

            if self.matching_file:
                answer = QMessageBox.question(self.mw, "Found Event Time", "The program found a match, would you like to start the playback?")
                if answer == QMessageBox.StandardButton.Yes:
                    main_directory = self.mw.filePanel.dirArchive.txtDirectory.text()
                    sub_directory = self.cameras.currentText()
                    tree = self.mw.filePanel.tree
                    model = tree.model()
                    path = os.path.join(main_directory, sub_directory, self.matching_file)
                    if file_idx := model.index(path):
                        if file_idx.isValid():
                            # repeat select file for first pass bug
                            self.selectFileInTree(os.path.join(main_directory, sub_directory), self.matching_file)
                            
                            file_start_time = self.fileAsDate(self.matching_file).timestamp()
                            file_end_time = self.endTimestamp(os.path.join(main_directory, sub_directory), self.matching_file)
                            file_seek_time = selected.timestamp()
                            pct = 0
                            num = file_seek_time - file_start_time
                            den = file_end_time - file_start_time
                            if den:
                                pct = num/den
                            if pct > .99:
                                pct = 0
                            self.mw.filePanel.control.startPlayer(file_start_from_seek=pct)
            else:
                file_to_index = None
                dist_to_before = None
                dist_to_after = None
                if self.closest_before:
                    dist_to_before = selected.timestamp() - self.endTimestamp(os.path.join(main_directory, sub_directory), self.closest_before)

                if self.closest_after:
                    dist_to_after = self.startTimestamp(self.closest_after) - selected.timestamp()

                found = False
                if dist_to_before and not dist_to_after:
                    found = True
                    file_to_index = self.closest_before
                if not dist_to_before and dist_to_after:
                    found = True
                    file_to_index = self.closest_after
                if dist_to_before and dist_to_after:
                    found = True
                    if dist_to_before < dist_to_after:
                        file_to_index = self.closest_before
                    else:
                        file_to_index = self.closest_after

                if not found:
                    QMessageBox.warning(self.mw, "Algorithm Error", "The program was not able to resolve the file")
                else:
                    self.selectFileInTree(os.path.join(main_directory, sub_directory), file_to_index)
                    answer = QMessageBox.question(self.mw, "Closest Result", "The program could not find an exact match, the closest result is highlighted, would you like to open it?")
                    self.selectFileInTree(os.path.join(main_directory, sub_directory), file_to_index)
                    if answer == QMessageBox.StandardButton.Yes:
                        self.mw.filePanel.control.startPlayer()

        except Exception as ex:
            logger.error(f'File search error: {ex}')

        self.hide()

    def selectFileInTree(self, path, filename):
        tree = self.mw.filePanel.tree
        model = tree.model()
        if camera_idx := model.index(path):
            if camera_idx.isValid():
                if not tree.isExpanded(camera_idx):
                    tree.setExpanded(camera_idx, True)
                if file_idx := model.index(os.path.join(path, filename)):
                    if file_idx.isValid():
                        tree.setCurrentIndex(file_idx)
                        tree.scrollTo(file_idx, QAbstractItemView.ScrollHint.PositionAtCenter)

    def qualifiedFileName(self, path, name):
        if not os.path.isfile(os.path.join(path, name)):
            return False
        components = os.path.splitext(name)
        if len(components) != 2:
            return False
        if components[1] != ".mov" and components[1] != ".mp4":
            return False
        if not components[0].isdigit():
            return False
        if len(components[0]) != 14:
            return False
        return True

    def isBefore(self, target, filename):
        result = False
        reference = datetime.strptime(os.path.splitext(filename)[0], FORMAT)
        reference = reference.replace(second=0)
        if target <= reference:
            result = True
        return result
    
    def isAfter(self, target, path, filename):
        result = False
        reference = datetime.fromtimestamp(os.path.getmtime(os.path.join(path, filename)))
        reference = reference.replace(second=59)
        if target >= reference:
            result = True
        return result
    
    def startTimestamp(self, filename):
        # file start time is deduced from file name
        return datetime.strptime(os.path.splitext(filename)[0], FORMAT).timestamp()
    
    def endTimestamp(self, path, filename):
        # file end time is the os file creation time
        return datetime.fromtimestamp(os.path.getmtime(os.path.join(path, filename))).timestamp()
    
    def fileAsDate(self, file):
        return datetime.strptime(os.path.splitext(file)[0], FORMAT)

    def getOccurence(self, target, path, filename):
        result = Occurence.DURING
        if self.isBefore(target, filename):
            result = Occurence.BEFORE
        if self.isAfter(target, path, filename):
            result = Occurence.AFTER
        return result
    
    def getSelectedDate(self):
        result = None
        date = self.calendar.selectedDate()
        h = int(self.hour.currentText())
        if self.AM_PM.currentText() == "PM" and h < 12:
            h += 12
        if self.AM_PM.currentText() == "AM" and h == 12:
            h = 0
        tmp = f'{date.year()}{date.month():02}{date.day():02}{h:02}{self.minute.currentText()}30'
        result = datetime.strptime(tmp, FORMAT)
        return result

    def guessFileIndex(self, selected, path, files, max_idx, min_idx, last_idx):
        idx = int(min_idx + (max_idx - min_idx) / 2)
  
        if idx == last_idx:
            # algorithm converged without finding match
            if self.getOccurence(selected, path, files[idx]) == Occurence.BEFORE:
                if idx > 0:
                    self.closest_before = files[idx-1]
                self.closest_after = files[idx]

            if self.getOccurence(selected, path, files[idx]) == Occurence.AFTER:
                self.closest_before = files[idx]
                if idx < len(files) - 1:
                    self.closest_after = files[idx+1]
            return

        last_idx = idx
        
        match self.getOccurence(selected, path, files[idx]):
            case Occurence.BEFORE:
                self.guessFileIndex(selected, path, files, idx, min_idx, last_idx)
            case Occurence.AFTER:
                self.guessFileIndex(selected, path, files, max_idx, idx, last_idx)
            case Occurence.DURING:
                self.matching_file = files[idx]
        return
    
    def findFileForEventTime(self, target_time, main_directory, sub_directory):
        path = os.path.join(main_directory, sub_directory)
        files = os.listdir(path)
        files = [f for f in files if self.qualifiedFileName(path, f)]
        files.sort()
    
        self.matching_file = None
        self.closest_before = None
        self.closest_after = None

        inside_range = True
        if self.isBefore(target_time, files[0]):
            inside_range = False
            self.closest_after = files[0]
        if self.isAfter(target_time, path, files[-1]):
            inside_range = False
            self.closest_before = files[-1]
        if inside_range:
            self.guessFileIndex(target_time, path, files, len(files)-1, 0, -1)

