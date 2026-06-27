from PyQt6.QtWidgets import QGridLayout, QLabel, \
    QDialog, QPushButton
from PyQt6.QtCore import Qt

class InfoDialog(QDialog):
    def __init__(self, p):
        super().__init__(p)
        self.setWindowTitle("info")
        self.lblMessage = QLabel()
        self.btnOK = QPushButton("OK")
        self.btnOK.clicked.connect(self.hide)
        lytMain = QGridLayout(self)
        lytMain.addWidget(self.lblMessage, 0, 0, 1, 1)
        lytMain.addWidget(QLabel(),        1, 0, 1, 1)
        lytMain.addWidget(self.btnOK,      2, 0, 1, 1, Qt.AlignmentFlag.AlignCenter)
