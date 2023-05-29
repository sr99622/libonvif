from PyQt6.QtWidgets import QDialog, QLabel, QGridLayout
from PyQt6.QtGui import QMovie
from PyQt6.QtCore import Qt, QSize

class WaitDialog(QDialog):
    def __init__(self, p):
        super().__init__(p)
        self.lblMessage = QLabel("Please wait for operations to be completed")
        self.lblProgress = QLabel()
        self.movie = QMovie("image:spinner.gif")
        self.movie.setScaledSize(QSize(50, 50))
        self.lblProgress.setMovie(self.movie)
        self.setWindowTitle("onvif-gui")

        lytMain = QGridLayout(self)
        lytMain.addWidget(self.lblMessage,  0, 1, 1, 1, Qt.AlignmentFlag.AlignCenter)
        lytMain.addWidget(self.lblProgress, 1, 1, 1, 1, Qt.AlignmentFlag.AlignCenter)

        self.movie.start()
        self.setModal(True)

    def sizeHint(self):
        return QSize(300, 100)
    
