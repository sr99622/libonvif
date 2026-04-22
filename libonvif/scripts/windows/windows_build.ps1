$ErrorActionPreference = "Stop"

trap {
    Write-Error "Build failed in $($MyInvocation.ScriptName) at line $($_.InvocationInfo.ScriptLineNumber)"
    exit 1
}

Write-Host "Delete system python from host machine before compiling, otherwise linking will not work"
Write-Host "Build and test this module from a directory other than the project directory"
Write-Host "During first run, use Administrator privilege to install tools, standard prompt ok after that"

$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_DIR = (Resolve-Path (Join-Path $SCRIPT_DIR "..\..")).Path

if (-not (Test-Path $PROJECT_DIR -PathType Container)) {
    throw "Project directory $PROJECT_DIR does not exist"
}

$pyproject = Join-Path $PROJECT_DIR "pyproject.toml"
$setupPy   = Join-Path $PROJECT_DIR "setup.py"

if (-not (Test-Path $pyproject) -and -not (Test-Path $setupPy)) {
    throw "Invalid Python project directory, did not find either pyproject.toml or setup.py in $PROJECT_DIR"
}

Write-Host "Project Directory: $PROJECT_DIR"

$BASE = (Get-Location).Path
Write-Host "Base Directory: $BASE"

$chocoBin = Join-Path $env:ProgramData "chocolatey\bin"
if (-not (Test-Path $chocoBin)) {
    & "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" `
        -NoProfile `
        -InputFormat None `
        -ExecutionPolicy Bypass `
        -Command "[System.Net.ServicePointManager]::SecurityProtocol = 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))"

    $env:Path += ";$chocoBin"
}

if (-not (Test-Path "$env:ProgramFiles\Git")) {
    choco install -y git
}

if (-not (Test-Path "$env:ProgramFiles\CMake")) {
    choco install -y cmake --installargs 'ADD_CMAKE_TO_PATH=System' --apply-install-arguments-to-dependencies
}

if (-not (Test-Path "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools")) {
    winget install Microsoft.VisualStudio.2022.BuildTools --silent --override "--wait --quiet --add ProductLang En-us --add Microsoft.VisualStudio.Workload.VCTools;includeRecommended"
}

$libsDir = Join-Path $BASE "onvif-gui-win-libs"
if (-not (Test-Path $libsDir)) {
    git clone https://github.com/sr99622/onvif-gui-win-libs $libsDir
}

Set-Location $PROJECT_DIR

$distDir = Join-Path $BASE "dist"
$wheelhouseDir = Join-Path $BASE "wheelhouse"

if (Test-Path $distDir) {
    Remove-Item -Recurse -Force $distDir
}
if (Test-Path $wheelhouseDir) {
    Remove-Item -Recurse -Force $wheelhouseDir
}

New-Item -ItemType Directory -Force -Path $distDir | Out-Null
New-Item -ItemType Directory -Force -Path $wheelhouseDir | Out-Null

& (Join-Path $PROJECT_DIR "scripts\windows\python\install.ps1")

$env:LIBXML2_INCLUDE_DIRS = Join-Path $BASE "onvif-gui-win-libs\libxml2\include\libxml2"
$env:LIBXML2_LIBRARIES    = Join-Path $BASE "onvif-gui-win-libs\libxml2\lib\libxml2.lib"

# Full build list:
# $pyList = @(310, 311, 312, 313, 314)

# Development list:
$pyList = @(313)

foreach ($v in $pyList) {
    Set-Location $BASE

    $pythonExe  = Join-Path $env:LOCALAPPDATA "Programs\Python\Python$v\python.exe"
    $venvDir    = Join-Path $BASE "py$v"
    $venvPython = Join-Path $venvDir "Scripts\python.exe"

    & $pythonExe -m venv $venvDir
    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip uninstall -y libonvif

    Set-Location $PROJECT_DIR

    & $venvPython -m pip install build delvewheel
    & $venvPython -m build --outdir $distDir

    $wheel = Get-ChildItem (Join-Path $distDir "libonvif*cp$v-cp$v-*.whl") | Select-Object -Last 1
    $libxml2Bin = Join-Path $BASE "onvif-gui-win-libs\libxml2\bin"

    & $venvPython -m delvewheel repair $wheel.FullName `
        --add-path $libxml2Bin `
        --wheel-dir $wheelhouseDir

    $fixedWheels = Get-ChildItem (Join-Path $wheelhouseDir "libonvif*cp$v-cp$v-*.whl") -ErrorAction SilentlyContinue
    foreach ($fixedWheel in $fixedWheels) {
        & $venvPython -m pip install --force-reinstall $fixedWheel.FullName
    }
}

Set-Location $BASE