from PyQt6.QtWidgets import QDialogButtonBox, QLineEdit, QGridLayout, QDialog, QLabel
from PyQt6.QtCore import Qt

class LoginDialog(QDialog):
    def __init__(self, p):
        super().__init__(p)
        self.active = False
        self.lblCameraIP = QLabel()
        self.lblCameraName = QLabel()
        buttonBox = QDialogButtonBox( \
            QDialogButtonBox.StandardButton.Ok | \
            QDialogButtonBox.StandardButton.Cancel)
        self.txtUsername = QLineEdit()
        lblUsername = QLabel("Username")
        self.txtPassword = QLineEdit()
        lblPassword = QLabel("Password")

        lytMain = QGridLayout(self)
        lytMain.addWidget(self.lblCameraName,  0, 0, 1, 2, Qt.AlignmentFlag.AlignCenter)
        lytMain.addWidget(self.lblCameraIP,    1, 0, 1, 2, Qt.AlignmentFlag.AlignCenter)
        lytMain.addWidget(lblUsername,         2, 0, 1, 1)
        lytMain.addWidget(self.txtUsername,    2, 1, 1, 1)
        lytMain.addWidget(lblPassword,         3, 0, 1, 1)
        lytMain.addWidget(self.txtPassword,    3, 1, 1, 1)
        lytMain.addWidget(buttonBox,           4, 0, 1, 2)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def exec(self, onvif_data):
        self.lblCameraName.setText(onvif_data.camera_name())
        self.lblCameraIP.setText(onvif_data.host())
        self.txtUsername.setText("")
        self.txtPassword.setText("")
        self.txtUsername.setFocus()
        onvif_data.cancelled = not super().exec()
        onvif_data.setUsername(self.txtUsername.text())
        onvif_data.setPassword(self.txtPassword.text())
        self.active = False

