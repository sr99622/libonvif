set FFMPEG_INSTALL_DIR=%HOMEPATH%\onvif-gui-win-libs\ffmpeg
set SDL2_INSTALL_DIR=%HOMEPATH%\onvif-gui-win-libs\sdl
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
cd ../libavio
if exist build\ (
    rmdir /s /q build
)
if exist avio.egg-info\ (
    rmdir /s /q avio.egg-info
)
pip install -v .
cd ../kankakee
if exist build\ (
    rmdir /s /q build
)
if exist kankakee.egg-info\ (
    rmdir /s /q kankakee.egg-info
)
pip install -v .
cd ../onvif-gui
if exist build\ (
    rmdir /s /q build
)
if exist onvif_gui.egg-info\ (
    rmdir /s /q onvif_gui.egg-info
)
pip install .
cd ..
