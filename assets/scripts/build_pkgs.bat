pip install build
cd libonvif
set CMAKE_CURRENT_SOURCE_DIR=%CD%
python -m build
cd ..
cd libavio
set CMAKE_CURRENT_SOURCE_DIR=%CD%
set FFMPEG_INSTALL_DIR=%CD%/ffmpeg
set SDL2_INSTALL_DIR=%CD%/sdl
python -m build
cd ..
cd kankakee
set CMAKE_CURRENT_SOURCE_DIR=%CD%
python -m build
cd ..
cd onvif-gui
python -m build
cd ..
