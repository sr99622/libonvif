@echo off

if not exist "%ALLUSERSPROFILE%\chocolatey\bin\" (
    @"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -InputFormat None -ExecutionPolicy Bypass -Command "[System.Net.ServicePointManager]::SecurityProtocol = 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))" && SET "PATH=%PATH%;%ALLUSERSPROFILE%\chocolatey\bin"
)
if not exist "%ProgramFiles%\Git\" (
    choco install -y git
)
if not exist "%ProgramFiles%\CMake\" (
    choco install -y cmake --installargs 'ADD_CMAKE_TO_PATH=System' --apply-install-arguments-to-dependencies
)
if not exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\" (
    winget install Microsoft.VisualStudio.2022.BuildTools --silent --override "--wait --quiet --add ProductLang En-us --add Microsoft.VisualStudio.Workload.VCTools;includeRecommended"
)

cd %HOMEPATH%
if not exist onvif-gui-win-libs\ (
    git clone https://github.com/sr99622/onvif-gui-win-libs
)
cd %HOMEPATH%\onvif-gui-win-libs
git pull
cd %HOMEPATH%
if not exist libonvif\ (
    git clone --recursive http://10.1.1.14:3000/stephen/libonvif
)
cd %HOMEPATH%\libonvif
git pull --recurse-submodules
if exist dist\ (
    del /q dist\*
)

call assets\scripts\components\windows\python\install
call assets\scripts\components\windows\env_variables
call assets\scripts\components\windows\copy_libs

set list=(310 311 312 313)
for %%v in %list% do (
    cd %HOMEPATH%
    %LOCALAPPDATA%\Programs\Python\Python%%v\python -m venv py%%v
    call py%%v\Scripts\activate
    python.exe -m pip install --upgrade pip
    pip uninstall -y libonvif
    pip uninstall -y avio
    pip uninstall -y kankakee
    pip uninstall -y onvif-gui
    cd libonvif
    call assets\scripts\build_pkgs
    call deactivate
)
