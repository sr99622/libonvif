#*******************************************************************************
# onvif-gui/CMakeLists.txt
#
# Copyright (c) 2022 Stephen Rhodes 
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
#******************************************************************************/

cmake_minimum_required(VERSION 3.17)
project(avio)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-D__STDC_CONSTANT_MACROS")
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(Qt5 COMPONENTS Widgets REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
find_package(FFmpeg REQUIRED)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

add_executable(onvif-gui
    src/admintab.cpp
    src/camera.cpp
    src/cameradialogtab.cpp
    src/cameralistmodel.cpp
    src/cameralistview.cpp
    src/camerapanel.cpp
    src/discovery.cpp
    src/filepanel.cpp
    src/imagetab.cpp
    src/logindialog.cpp
    src/main.cpp
    src/mainwindow.cpp
    src/messagepanel.cpp
    src/networktab.cpp
    src/onvifmanager.cpp
    src/ptztab.cpp
    src/settingspanel.cpp
    src/stylepanel.cpp
    src/videotab.cpp
    include/admintab.h
    include/camera.h
    include/cameradialogtab.h
    include/cameralistmodel.h
    include/cameralistview.h
    include/camerapanel.h
    include/discovery.h
    include/filepanel.h
    include/imagetab.h
    include/logindialog.h
    include/mainwindow.h
    include/messagepanel.h
    include/networktab.h
    include/onvifmanager.h
    include/ptztab.h
    include/settingspanel.h
    include/stylepanel.h
    include/videotab.h

    resources/resources.qrc
)

target_link_libraries(onvif-gui PRIVATE
    Qt5::Widgets
    FFmpeg::FFmpeg
    Python3::Python
    onvif
    avio
)

target_include_directories(onvif-gui PUBLIC
    libonvif/include
    include
)

