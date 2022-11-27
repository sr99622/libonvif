#*******************************************************************************
# libonvif/onvif-camadmin/CMakeLists.txt
#
# Copyright (c) 2020 Stephen Rhodes 
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

project(onvif-camadmin)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_AUTOMOC ON)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)

find_package(Qt5 COMPONENTS Widgets REQUIRED)
find_package(LibXml2 REQUIRED)
find_package(LibOnvif REQUIRED)
find_package(Avio REQUIRED)
find_package(SDL2 REQUIRED)
find_package(FFmpeg REQUIRED)

set(MOC_INCLUDES
    include/admintab.h
    include/camera.h
    include/cameradialogtab.h
    include/cameralistmodel.h
    include/cameralistview.h
    include/camerapanel.h
    include/configtab.h
    include/discovery.h
    include/imagetab.h
    include/logindialog.h
    include/mainwindow.h
    include/networktab.h
    include/onvifmanager.h
    include/ptztab.h
    include/videotab.h
)

add_executable(onvif-camadmin
    src/admintab.cpp
    src/camera.cpp
    src/cameradialogtab.cpp
    src/cameralistmodel.cpp
    src/cameralistview.cpp
    src/camerapanel.cpp
    src/configtab.cpp
    src/discovery.cpp
    src/imagetab.cpp
    src/logindialog.cpp
    src/main.cpp
    src/mainwindow.cpp
    src/networktab.cpp
    src/onvifmanager.cpp
    src/ptztab.cpp
    src/videotab.cpp
    ${MOC_INCLUDES}
)

set_target_properties(onvif-camadmin PROPERTIES
    OUTPUT_NAME onvif-camadmin
    ARCHIVE_OUTPUT_DIRECTORY onvif-camadmin
)

target_link_libraries(onvif-camadmin PRIVATE
    Qt5::Widgets
    SDL2::SDL2
    ${FFMPEG_LIBRARIES}
    ${AVIO_LIBRARY}
    ${LIBONVIF_LIBRARIES}
    ${LIBXML2_LIBRARIES}
)

target_include_directories(onvif-camadmin PUBLIC
    include
    ${LIBONVIF_INCLUDE_DIRS}
    ${LIBXML2_INCLUDE_DIRS}
    ${AVIO_INCLUDE_DIR}
)

install(TARGETS onvif-camadmin DESTINATION bin)
