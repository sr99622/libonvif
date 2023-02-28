import os
import sys
import numpy as np
import cv2
from time import sleep
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, \
QGridLayout, QWidget, QSlider, QLabel, QMessageBox
from PyQt6.QtCore import Qt, pyqtSignal, QObject


sys.path.append("../build/libonvif")
import onvif


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.boss = onvif.Manager()

        self.btnPlay = QPushButton("play")
        self.btnPlay.clicked.connect(self.btnPlayClicked)
        pnlMain = QWidget()
        lytMain = QGridLayout(pnlMain)
        lytMain.addWidget(self.btnPlay, 0, 0, 1, 1, Qt.AlignmentFlag.AlignCenter)
        self.setCentralWidget(pnlMain)

    def btnPlayClicked(self):
        print("btnPplayClicked")
        self.boss.finished = lambda : self.finished()
        self.boss.startTest()

    def finished(self):
        print("python discover finished")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()