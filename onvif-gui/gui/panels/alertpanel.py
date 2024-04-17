import os
import asyncio
from telegram import Bot
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QGridLayout, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QCheckBox, QMessageBox, QApplication, QFileDialog
class AlertPanel(QWidget):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw
        self.layout = QVBoxLayout(self)

        self.chkEnableAlert = QCheckBox("Enable alerts on Telegram with the image of detected object")
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

                # Checkbox and button for saving detected images
        self.chkSaveImages = QCheckBox("Save detected images locally")
        self.chkSaveImages.stateChanged.connect(self.saveImagesChanged)
        self.layout.addWidget(self.chkSaveImages)
        
        self.btnSavePath = QPushButton("Select save directory")
        self.btnSavePath.clicked.connect(self.selectSaveDirectory)
        self.btnSavePath.setEnabled(False)  # Disabled by default
        self.layout.addWidget(self.btnSavePath)

        self.selectedSavePath = ""  # To store the selected directorys

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
        self.txtBotId.setEnabled(checked)
        self.txtChatId.setEnabled(checked)
        self.btnTest.setEnabled(checked)

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

    def saveImagesChanged(self, checked):
        # Enable or disable the save path button
        self.btnSavePath.setEnabled(checked)

    def selectSaveDirectory(self):
        # Open a dialog to select a directory
        path = QFileDialog.getExistingDirectory(self, "Select Save Directory")
        if path:
            self.selectedSavePath = path
            QMessageBox.information(self, "Directory Selected", f"Files will be saved to: {path}")