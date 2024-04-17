import os
from PyQt6.QtWidgets import QGridLayout, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout
from PyQt6.QtCore import Qt

class AlertPanel(QWidget):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw
        self.layout = QVBoxLayout(self)
        
        self.lblBotId = QLabel("BOT ID")
        self.txtBotId = QLineEdit()
        self.txtBotId.setEchoMode(QLineEdit.EchoMode.Password)
        
        self.lblChatId = QLabel("Your_CHAT_ID")
        self.txtChatId = QLineEdit()
        self.txtChatId.setEchoMode(QLineEdit.EchoMode.Password)

        self.btnTest = QPushButton("Test")
        self.btnTest.clicked.connect(self.testConnection)

        fixedPanel = QWidget()
        lytFixed = QGridLayout(fixedPanel)
        lytFixed.addWidget(self.lblBotId,    0, 0, 1, 1)
        lytFixed.addWidget(self.txtBotId,    0, 1, 1, 1)
        lytFixed.addWidget(self.lblChatId,   1, 0, 1, 1)
        lytFixed.addWidget(self.txtChatId,   1, 1, 1, 1)
        lytFixed.addWidget(self.btnTest,     2, 0, 1, 2, Qt.AlignmentFlag.AlignCenter)
        lytFixed.setColumnStretch(1, 10)
        self.layout.addWidget(fixedPanel)

    def testConnection(self):
        # Implement connection testing logic
        bot_id = self.txtBotId.text()
        chat_id = self.txtChatId.text()
        print(f"Testing connection with BOT ID: {bot_id} and CHAT ID: {chat_id}")
        # Additional logic to actually test the connection can be implemented here
