import sys
from PyQt6.QtWidgets import QApplication
from gui import MainWindow

clear_settings = False
if len(sys.argv) > 1:
    if str(sys.argv[1]) == "--clear":
        clear_settings = True

app = QApplication(sys.argv)
app.setStyle('Fusion')
window = MainWindow(clear_settings)
window.style()
window.show()
app.exec()
