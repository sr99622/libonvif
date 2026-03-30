set LIBXML2_INCLUDE_DIRS=%HOMEPATH%\onvif-gui-win-libs\libxml2\include\libxml2
set LIBXML2_LIBRARIES=%HOMEPATH%\onvif-gui-win-libs\libxml2\lib\libxml2.lib
cd libonvif
if exist build\ (
    rmdir /s /q build
)
if exist libonvif.egg-info\ (
    rmdir /s /q libonvif.egg-info
)
pip install -v .
cd ..
