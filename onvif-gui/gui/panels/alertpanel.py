import os
import asyncio
from telegram import Bot
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QGridLayout, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QCheckBox, QMessageBox, QApplication
class AlertPanel(QWidget):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw
        self.layout = QVBoxLayout(self)

        
        self.chkEnableAlert = QCheckBox("If you enable this, you will receive Alert's on Telegram along with the image of detected object")
        self.chkEnableAlert.stateChanged.connect(self.enableAlertChanged)
        self.layout.addWidget(self.chkEnableAlert) 
        
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

        
    def enableAlertChanged(self, checked):
        # Enable or disable text inputs and button based on the checkbox state
        self.txtBotId.setEnabled(checked)
        self.txtChatId.setEnabled(checked)
        self.btnTest.setEnabled(checked)

    # def testConnection(self):
    #     if self.txtBotId.isEnabled():
    #         # Implement connection testing logic
    #         bot_id = self.txtBotId.text()
    #         chat_id = self.txtChatId.text()
    #         bot = Bot(token=bot_id)

    #         try:
                
    #             bot.send_message(chat_id=chat_id, text="Successfully enabled Telegram Alerts")
    #             # You can update the UI or notify the user of success here
    #         except Exception as e:
    #             # Handle exceptions, e.g., invalid token or chat ID
    #             print(f"Failed to send message: {str(e)}")

    def testConnection(self):
        if self.txtBotId.isEnabled():
            asyncio.run(self.async_testConnection())

    async def async_testConnection(self):
        bot_id = self.txtBotId.text()
        chat_id = self.txtChatId.text()
        bot = Bot(token=bot_id)

        try:
            await bot.send_message(chat_id=chat_id, text="Successfully enabled Telegram Alerts")
            QMessageBox.information(self, "Success", "Message sent successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to send message: {str(e)}")
        finally:
            QApplication.processEvents()