#*******************************************************************************
# libonvif/sample/CMakeLists.txt
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

project(discover)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)

find_package(Qt5 COMPONENTS Widgets REQUIRED)
if (Qt5_FOUND)
    message("-- Found Qt")
endif()

find_package(LibXml2)
if (NOT LibXml2_FOUND)
    message("-- Looking for installation of libxml2")
    find_path(LIBXML2_INCLUDE_DIRS libxml/parser.h
        HINTS
            $ENV{CONDA_PREFIX}/Library/include/libxml2
    )
    find_library(LIBXML2_LIBRARIES NAMES xml2
        HINTS
            $ENV{CONDA_PREFIX}/lib
    )
    message("-- Found LIBXML2_INCLUDE_DIRS ${LIBXML2_INCLUDE_DIRS}")
    message("-- Found LIBXML2_LIBRARIES ${LIBXML2_LIBRARIES}")
endif()

find_package(LibOnvif)
if (LibOnvif_FOUND)
    message("-- Found LIBONVIF_INCLUDE_DIRS ${LIBONVIF_INCLUDE_DIRS}")
    message("-- Found LIBONVIF_LIBRARIES ${LIBONVIF_LIBRARIES}")
endif()

find_package(Iconv)
if (NOT Iconv_FOUND)
    message("--Could not find Iconv. If you are using libxml2 compiled without iconv, this message may be ignored")
endif()

if (FFMPEG_INCLUDE_DIRS AND FFMPEG_LIBRARIES)
    message("-- FFmpeg is using include dirs ${FFMPEG_INCLUDE_DIRS} and libs ${FFMPEG_LIBRARIES}")
else()
    find_package(FFmpeg)
    if (FFMPEG_INCLUDE_DIRS)
        message("-- Found FFmpeg Include Directories ${FFMPEG_INCLUDE_DIRS}")
    else()    
        message("-- **WARNING** Unable to find FFmpeg Include Directories automatically.  Use -DFFMPEG_INCLUDE_DIRS location manually")
    endif()
    if (FFMPEG_LIBRARIES)
        message("-- Found FFmpeg Libraries ${FFMPEG_LIBRARIES}")
    else()    
        message("-- **WARNING** Unable to find FFmpeg Libraries automatically.  Use -DFFMPEG_LIBRARIES to set location manually")
    endif()
endif()

if (UNIX)
    message("finding alsa")
    set (ALSA_INCLUDE_DIR_CANDIDATE $ENV{CONDA_PREFIX}/include/alsa)
    if (EXISTS "${ALSA_INCLUDE_DIR_CANDIDATE}")
        set (ALSA_INCLUDE_DIR ${ALSA_INCLUDE_DIR_CANDIDATE})
        message("-- Found ALSA include directory ${ALSA_INCLUDE_DIR}")
    endif()
    set (ALSA_LIBRARIES_CANDIDATE $ENV{CONDA_PREFIX}/lib/libasound.so)
    if (EXISTS "${ALSA_LIBRARIES_CANDIDATE}")
        set (ALSA_LIBRARIES ${ALSA_LIBRARIES_CANDIDATE})
        message("-- Found ALSA libraries ${ALSA_LIBRARIES}")
    endif()
endif()

if (SDL2_LIBRARIES AND SDL2_INCLUDE_DIRS)
    message("-- SDL2 is using include dirs ${SDL2_INCLUDE_DIRS} and lib ${SDL2_LIBRARIES}")
else()
    find_package(SDL2)
    if (SDL2_FOUND)
        set(SDL2_LIBRARIES SDL2::SDL2)
        message("-- Found SDL2: ${SDL2_INCLUDE_DIRS}")
    else()
        message("-- **WARNING** Unable to find SDL2 automatically.  Use -DSDL2_INCLUDE_DIRS and -DSDL2_LIBRARIES to set locations manually")
    endif()
endif()

find_package(Avio)
if (Avio_FOUND)
    message("-- Found AVIO_INCLUDE_DIRS ${AVIO_INCLUDE_DIRS}")
    message("-- Found AVIO_LIBRARIES ${AVIO_LIBRARIES}")
endif()


add_executable(onvif
    admintab.cpp
    camera.cpp
    cameradialogtab.cpp
    cameralistmodel.cpp
    cameralistview.cpp
    camerapanel.cpp
    configtab.cpp
    discovery.cpp
    imagetab.cpp
    logindialog.cpp
    main.cpp
    mainwindow.cpp
    networktab.cpp
    onvifmanager.cpp
    ptztab.cpp
    videotab.cpp
)

target_link_libraries(onvif PRIVATE
    Qt5::Widgets
    ${LIBONVIF_LIBRARIES}
    ${AVIO_LIBRARIES}
    ${LIBXML2_LIBRARIES}
    ${Iconv_LIBRARY}
    ${FFMPEG_LIBRARIES}
    ${SDL2_LIBRARIES}
)

target_include_directories(onvif PUBLIC
    include
    ${FFMPEG_INCLUDE_DIRS}
    ${SDL2_INCLUDE_DIRS}
    ${LIBONVIF_INCLUDE_DIRS}
    ${AVIO_INCLUDE_DIRS}
    ${LIBXML2_INCLUDE_DIRS}
    ${Iconv_INCLUDE_DIR}
)

