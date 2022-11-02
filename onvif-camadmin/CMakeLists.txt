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

project(onvif-camadmin)

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

find_package(QtAV REQUIRED)

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

add_executable(onvif-camadmin
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

set_target_properties(onvif-camadmin PROPERTIES
    OUTPUT_NAME onvif-camadmin
    ARCHIVE_OUTPUT_DIRECTORY onvif-camadmin
)

target_link_libraries(onvif-camadmin PRIVATE
    Qt5::Widgets
    ${QTAVWIDGETS_LIBRARIES}
    ${LIBONVIF_LIBRARIES}
    ${LIBXML2_LIBRARIES}
    ${Iconv_LIBRARY}
)

target_include_directories(onvif-camadmin PUBLIC
    ${QTAVWIDGETS_INCLUDE_DIRS}
    ${LIBONVIF_INCLUDE_DIRS}
    ${LIBXML2_INCLUDE_DIRS}
    ${Iconv_INCLUDE_DIR}
)

install(TARGETS onvif-camadmin DESTINATION bin)
