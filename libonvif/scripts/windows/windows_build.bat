@echo off
echo Delete system python from host machine before compiling, otherwise linking will not work
echo Build and test this module from a directory other than the project directory
echo During first run, use Administrator privilege to install tools, standard prompt ok after that

if not exist "%PROGRAMDATA%\chocolatey\bin\" (
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

setlocal

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
for %%i in ("%SCRIPT_DIR%\..\..") do set "PROJECT_DIR=%%~fi"
if not exist "%PROJECT_DIR%" (
    echo Project directory %PROJECT_DIR% does not exist 1>&2
    exit /b 1
)
if not exist "%PROJECT_DIR%\pyproject.toml" if not exist "%PROJECT_DIR%\setup.py" (
    echo Invalid Python project directory, did not find either pyproject.toml or setup.py in %PROJECT_DIR% 1>&2
    exit /b 1
)

echo Project Directory: %PROJECT_DIR% 1>&2

set BASE=%CD%
echo Base Directory: %BASE% 1>&2

if not exist %BASE%\onvif-gui-win-libs\ (
    git clone https://github.com/sr99622/onvif-gui-win-libs
)

cd %PROJECT_DIR%

if exist %BASE%\dist\ (
    del /q %BASE%\dist\*
)
if exist wheelhouse\ (
    del /q %BASE%\wheelhouse\*
)

call %PROJECT_DIR%\scripts\windows\python\install

set LIBXML2_INCLUDE_DIRS=%BASE%\onvif-gui-win-libs\libxml2\include\libxml2
set LIBXML2_LIBRARIES=%BASE%\onvif-gui-win-libs\libxml2\lib\libxml2.lib

rem set list=(310 311 312 313 314)
set list=(313)
rem list is set to the single parameter when developing, full list is for build

for %%v in %list% do (
    cd %BASE%
    %LOCALAPPDATA%\Programs\Python\Python%%v\python -m venv py%%v
    call py%%v\Scripts\activate
    python.exe -m pip install --upgrade pip
    pip uninstall -y libonvif
    cd %PROJECT_DIR%
    pip install build delvewheel
    python -m build --outdir %BASE%\dist
    delvewheel repair %BASE%\dist\*cp%%v-cp%%v-*.whl --add-path %BASE%\onvif-gui-win-libs\libxml2\bin --wheel-dir %BASE%\wheelhouse
    for %%F in (%BASE%\wheelhouse\*cp%%v-cp%%v-*.whl) do (
        pip install --force-reinstall "%%F"
    )
    call deactivate
)

cd %BASE%