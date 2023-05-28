from PyQt6.QtWidgets import QDialog, QLabel, QGridLayout
from PyQt6.QtCore import Qt, QSize

class WaitDialog(QDialog):
    def __init__(self, p):
        super().__init__(p)
        self.setMinimumWidth(400)
        self.lblMessage = QLabel("Please wait while model is being downloaded")
        self.setWindowTitle("onvif-gui")

        lytMain = QGridLayout(self)
        lytMain.addWidget(self.lblMessage, 0, 0, 1, 1, Qt.AlignmentFlag.AlignCenter)

    def exec(self):
        self.cancelled = not super().exec()

    def sizeHint(self):
        return QSize(300, 100)
