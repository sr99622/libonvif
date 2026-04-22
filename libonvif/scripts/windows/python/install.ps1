$ErrorActionPreference = "Stop"

trap {
    Write-Error "Install failed in $($MyInvocation.ScriptName) at line $($_.InvocationInfo.ScriptLineNumber)"
    exit 1
}

function Install-PythonIfMissing {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Version,
        [Parameter(Mandatory = $true)]
        [string]$InstallerName
    )

    $shortVersion = ($Version -split '\.')[0..1] -join ''
    $installDir = Join-Path $env:LOCALAPPDATA "Programs\Python\Python$shortVersion"

    if (-not (Test-Path $installDir)) {
        $url = "https://www.python.org/ftp/python/$Version/$InstallerName"
        Write-Host "Downloading $InstallerName"
        Invoke-WebRequest -Uri $url -OutFile $InstallerName
        Start-Process -FilePath ".\$InstallerName" -ArgumentList "/passive", "/quiet" -Wait
    }
}

Install-PythonIfMissing -Version "3.10.11" -InstallerName "python-3.10.11-amd64.exe"
Install-PythonIfMissing -Version "3.11.9"  -InstallerName "python-3.11.9-amd64.exe"
Install-PythonIfMissing -Version "3.12.9"  -InstallerName "python-3.12.9-amd64.exe"
Install-PythonIfMissing -Version "3.13.11" -InstallerName "python-3.13.11-amd64.exe"
Install-PythonIfMissing -Version "3.14.2"  -InstallerName "python-3.14.2-amd64.exe"