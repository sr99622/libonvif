
if not exist "%LOCALAPPDATA%\Programs\Python\Python310\" (
    curl --output python-3.10.11-amd64.exe https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe
    python-3.10.11-amd64.exe /passive /quiet
)
if not exist "%LOCALAPPDATA%\Programs\Python\Python311\" (
    curl --output python-3.11.9-amd64.exe https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe
    python-3.11.9-amd64.exe /passive /quiet
)
if not exist "%LOCALAPPDATA%\Programs\Python\Python312\" (
    curl --output python-3.12.9-amd64.exe https://www.python.org/ftp/python/3.12.9/python-3.12.9-amd64.exe
    python-3.12.9-amd64.exe /passive /quiet
)
if not exist "%LOCALAPPDATA%\Programs\Python\Python313\" (
    curl --output python-3.13.2-amd64.exe https://www.python.org/ftp/python/3.13.2/python-3.13.2-amd64.exe
    python-3.13.2-amd64.exe /passive /quiet
)
