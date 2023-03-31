from PyQt6.QtWidgets import QWidget, QLineEdit, QPushButton, \
    QLabel, QGridLayout, QFileDialog

class FileSelector(QWidget):
    def __init__(self, mw, name):
        super().__init__()
        self.mw = mw
        self.filenameKey = "Module/" + name + "/filename"

        self.txtFilename = QLineEdit()
        self.txtFilename.setText(self.mw.settings.value(self.filenameKey))
        self.txtFilename.textEdited.connect(self.txtFilenameChanged)
        self.btnSelect = QPushButton("...")
        self.btnSelect.clicked.connect(self.btnSelectClicked)
        self.btnSelect.setMaximumWidth(36)
        lblSelect = QLabel("Model")

        lytMain = QGridLayout(self)
        lytMain.addWidget(lblSelect,          0, 0, 1, 1)
        lytMain.addWidget(self.txtFilename,   0, 1, 1, 1)
        lytMain.addWidget(self.btnSelect,     0, 2, 1, 1)
        lytMain.setContentsMargins(0, 0, 0, 0)

    def btnSelectClicked(self):
        print("btnSelctClicked")
        filename = QFileDialog.getOpenFileName(self, "Select Model File", 
                                               self.txtFilename.text())[0]
        print(filename)
        if len(filename) > 0:
            self.txtFilename.setText(filename)
            self.mw.settings.setValue(self.filenameKey, filename)

    def txtFilenameChanged(self, text):
        self.mw.settings.setValue(self.filenameKey, text)

    def text(self):
        return self.txtFilename.text()
