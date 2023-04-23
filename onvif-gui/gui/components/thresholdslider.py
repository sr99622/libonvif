from PyQt6.QtWidgets import QWidget, QSlider, QLabel, QGridLayout
from PyQt6.QtCore import Qt

class ThresholdSlider(QWidget):
    def __init__(self, mw, name, title, initValue):
        super().__init__()
        self.mw = mw
        self.thresholdKey = "Module/" + name + "/threshold"
        self.sldThreshold = QSlider(Qt.Orientation.Horizontal)
        value = int(self.mw.settings.value(self.thresholdKey, initValue))
        self.sldThreshold.setValue(value)
        self.sldThreshold.valueChanged.connect(self.sldThresholdChanged)
        lblThreshold = QLabel(title)
        self.lblValue = QLabel(str(self.sldThreshold.value()))
        lytThreshold = QGridLayout(self)
        lytThreshold.addWidget(lblThreshold,          0, 0, 1, 1)
        lytThreshold.addWidget(self.sldThreshold,     0, 1, 1, 1)
        lytThreshold.addWidget(self.lblValue,         0, 2, 1, 1)
        lytThreshold.setContentsMargins(0, 0, 0, 0)

    def sldThresholdChanged(self, value):
        self.lblValue.setText(str(value))
        self.mw.settings.setValue(self.thresholdKey, value)

    def value(self):
        return self.sldThreshold.value() / 100