pip install build
set LIBXML2_INCLUDE_DIRS=%HOMEPATH%\onvif-gui-win-libs\libxml2\include\libxml2
set LIBXML2_LIBRARIES=%HOMEPATH%\onvif-gui-win-libs\libxml2\lib\libxml2.lib
if not exist dist/ (
    mkdir dist
)
cd libonvif
rmdir /q /s build
python -m build
for /f %%F in ('dir /b /a-d dist\*whl') do (
    pip install dist\%%F
)
move dist\* ..\dist
cd ..
