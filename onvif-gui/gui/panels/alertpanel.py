import os
import asyncio
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QGridLayout, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout,
    QCheckBox, QMessageBox, QApplication, QFileDialog
)
from telegram import Bot
class AlertPanel(QWidget):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw
        self.layout = QVBoxLayout(self)

        # Keys for settings
        self.enableAlertKey = "AlertPanel/enableAlert"
        self.saveImagesKey = "AlertPanel/saveImages"
        self.savePathKey = "AlertPanel/savePath"
        self.botIdKey = "AlertPanel/botId"
        self.chatIdKey = "AlertPanel/chatId"

        # Telegram alert setup
        self.chkEnableAlert = QCheckBox("Enable alerts on Telegram with the image of detected object")
        self.chkEnableAlert.stateChanged.connect(self.enableAlertChanged)
        self.layout.addWidget(self.chkEnableAlert)

        # UI for Telegram bot credentials
        self.setupTelegramUI()

        # Checkbox and button for saving detected images
        self.chkSaveImages = QCheckBox("Save detected images locally")
        self.chkSaveImages.stateChanged.connect(self.saveImagesChanged)
        self.layout.addWidget(self.chkSaveImages)
        
        self.btnSavePath = QPushButton("Select save directory")
        self.btnSavePath.clicked.connect(self.selectSaveDirectory)
        self.btnSavePath.setEnabled(False)  # Disabled by default
        self.layout.addWidget(self.btnSavePath)

        # Button to clear the selected directory
        self.btnClearSavePath = QPushButton("Clear Selected Directory")
        self.btnClearSavePath.clicked.connect(self.clearSaveDirectory)
        self.btnClearSavePath.setEnabled(False)  # Disabled by default
        self.layout.addWidget(self.btnClearSavePath)

        self.selectedSavePath = ""  # To store the selected directory

    def setupTelegramUI(self):
        fixedPanel = QWidget()
        lytFixed = QGridLayout(fixedPanel)
        
        self.lblBotId = QLabel("BOT ID")
        self.txtBotId = QLineEdit()
        self.txtBotId.setEchoMode(QLineEdit.EchoMode.Password)
        self.txtBotId.setEnabled(False)
        self.txtBotId.textChanged.connect(self.checkCredentials)
        self.txtBotId.textChanged.connect(self.updateButtonsState)

        self.lblChatId = QLabel("Your_CHAT_ID")
        self.txtChatId = QLineEdit()
        self.txtChatId.setEchoMode(QLineEdit.EchoMode.Password)
        self.txtChatId.setEnabled(False)
        self.txtChatId.textChanged.connect(self.checkCredentials)
        self.txtChatId.textChanged.connect(self.updateButtonsState)
        
        self.btnTest = QPushButton("Test Configuration")
        self.btnTest.clicked.connect(self.testConnection)
        self.btnTest.setEnabled(False)

        self.btnClearConfig = QPushButton("Clear Configuration")
        self.btnClearConfig.clicked.connect(self.clearConfiguration)
        self.btnClearConfig.setEnabled(False)
        
        self.btnToggleVisibility = QPushButton("Toggle Text Visibility")
        self.btnToggleVisibility.clicked.connect(self.toggleVisibility)
        self.btnToggleVisibility.setEnabled(False)
        
        lytFixed.addWidget(self.lblBotId,    0, 0, 1, 1)
        lytFixed.addWidget(self.txtBotId,    0, 1, 1, 1)
        lytFixed.addWidget(self.lblChatId,   1, 0, 1, 1)
        lytFixed.addWidget(self.txtChatId,   1, 1, 1, 1)
        lytFixed.addWidget(self.btnTest,     2, 0, 1, 2, Qt.AlignmentFlag.AlignCenter)
        lytFixed.addWidget(self.btnClearConfig, 3, 0, 1, 2, Qt.AlignmentFlag.AlignCenter)
        lytFixed.addWidget(self.btnToggleVisibility, 4, 0, 1, 2, Qt.AlignmentFlag.AlignCenter)
        lytFixed.setColumnStretch(1, 10)
        self.layout.addWidget(fixedPanel)

    def updateButtonsState(self):
        # Enable or disable the buttons based on text fields content
        has_text = bool(self.txtBotId.text() or self.txtChatId.text())
        self.btnClearConfig.setEnabled(has_text)
        self.btnToggleVisibility.setEnabled(has_text)

    def clearConfiguration(self):
        # Clear the configuration in both text fields
        self.txtBotId.clear()
        self.txtChatId.clear()

    def toggleVisibility(self):
        # Toggle the EchoMode between Password and Normal to show/hide text
        if self.txtBotId.echoMode() == QLineEdit.EchoMode.Password:
            self.txtBotId.setEchoMode(QLineEdit.EchoMode.Normal)
            self.txtChatId.setEchoMode(QLineEdit.EchoMode.Normal)
        else:
            self.txtBotId.setEchoMode(QLineEdit.EchoMode.Password)
            self.txtChatId.setEchoMode(QLineEdit.EchoMode.Password)

    def enableAlertChanged(self, checked):
        # Enable or disable Telegram settings based on checkbox
        self.txtBotId.setEnabled(checked)
        self.txtChatId.setEnabled(checked)
        self.btnTest.setEnabled(checked)
        self.btnTest.setEnabled(checked and bool(self.txtBotId.text() and self.txtChatId.text()))
    
    def checkCredentials(self):
        # Check both text fields to determine if the test button should be enabled
        is_filled = bool(self.txtBotId.text() and self.txtChatId.text())
        self.btnTest.setEnabled(is_filled)

    def testConnection(self):
        # Simple Telegram message test
        asyncio.run(self.async_testConnection())

    async def async_testConnection(self):
        bot = Bot(token=self.txtBotId.text())
        try:
            await bot.send_message(chat_id=self.txtChatId.text(), text="Successfully enabled Telegram Alerts")
            QMessageBox.information(self, "Success", "Message sent successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to send message: {str(e)}")

    def saveImagesChanged(self, checked):
        # Enable or disable the save path button
        self.btnSavePath.setEnabled(checked)

    def selectSaveDirectory(self):
        # Open a dialog to select a directory
        path = QFileDialog.getExistingDirectory(self, "Select Save Directory")
        if path:
            self.selectedSavePath = path
            QMessageBox.information(self, "Directory Selected", f"Files will be saved to: {path}")
            self.btnSavePath.setText("Directory Selected")
            self.btnClearSavePath.setEnabled(True)  # Enable the clear button when a directory is selected

    def clearSaveDirectory(self):
        # Clear the currently selected directory
        self.selectedSavePath = ""
        self.btnSavePath.setText("Select save directory")
        self.btnClearSavePath.setEnabled(False)  # Disable the clear button once the selection is cleared
        QMessageBox.information(self, "Clear Directory", "Selected save directory has been cleared.")
