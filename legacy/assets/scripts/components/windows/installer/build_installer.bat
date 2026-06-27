rem this file needs to be run in admin mode
mkdir %HOMEPATH%\installer
cd %HOMEPATH%\installer
if not exist "%ProgramFiles(x86)%\NSIS\" (
    curl -OL https://prdownloads.sourceforge.net/nsis/nsis-3.11-setup.exe?download
    nsis-3.11-setup.exe
    echo "Wait for the NSIS installer to finish, then type the enter key"
    pause
)
if not exist cpython-3.13.3+20250517-x86_64-pc-windows-msvc-install_only.tar.gz (
    curl -OL https://github.com/astral-sh/python-build-standalone/releases/download/20250517/cpython-3.13.3+20250517-x86_64-pc-windows-msvc-install_only.tar.gz
)
mkdir "%ProgramFiles(x86)%\Onvif GUI"
tar -xvzf cpython-3.13.3+20250517-x86_64-pc-windows-msvc-install_only.tar.gz -C "%ProgramFiles(x86)%\Onvif GUI"
cd "%ProgramFiles(x86)%\Onvif GUI"
python\python -m venv onvif-gui-env
onvif-gui-env\Scripts\pip install %HOMEPATH%\libonvif\onvif-gui torch torchvision openvino
cd %HOMEPATH%\installer
copy %HOMEPATH%\libonvif\onvif-gui\onvif_gui\resources\onvif-gui.ico .
copy %HOMEPATH%\libonvif\assets\scripts\components\windows\installer\onvif-gui.nsi .
copy %HOMEPATH%\libonvif\assets\scripts\components\windows\installer\license.txt .
"%ProgramFiles(x86)%\NSIS\makensis" onvif-gui.nsi
rmdir /q /s "%ProgramFiles(x86)%\Onvif GUI"
cd %HOMEPATH%