#*******************************************************************************
# libonvif/example/CmakeLists.txt
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
set(win_path "C:/Program Files (x86)")

if (WIN32)
    set(xml_lib_name "libxml2")
endif()

if (UNIX)
    set(xml_lib_name "xml2")
endif()

project(onvif-discover)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)

find_package(LibXml2)
if (NOT LibXml2_FOUND)
    message("-- Looking for installation of libxml2")
    find_path(LIBXML2_INCLUDE_DIRS libxml/parser.h
        HINTS
            $ENV{CONDA_PREFIX}/include/libxml2
            ${win_path}/libxml2/include/libxml2
    )
    find_library(LIBXML2_LIBRARIES NAMES ${xml_lib_name}
        HINTS
            $ENV{CONDA_PREFIX}/lib
            ${win_path}/libxml2/lib
    )
    message("-- Found LIBXML2_INCLUDE_DIRS ${LIBXML2_INCLUDE_DIRS}")
    message("-- Found LIBXML2_LIBRARIES ${LIBXML2_LIBRARIES}")
endif()

find_package(LibOnvif)
if (LibOnvif_FOUND)
    message("-- Found LIBONVIF_INCLUDE_DIRS ${LIBONVIF_INCLUDE_DIRS}")
    message("-- Found LIBONVIF_LIBRARIES ${LIBONVIF_LIBRARIES}")
endif()

add_executable(onvif-util
    onvif-util.cpp
)

target_link_libraries(onvif-util PRIVATE
    ${LIBONVIF_LIBRARIES}
    ${LIBXML2_LIBRARIES}
)

target_include_directories(onvif-util PUBLIC
    ${LIBONVIF_INCLUDE_DIRS}
    ${LIBXML2_INCLUDE_DIRS}
)

install(TARGETS onvif-util DESTINATION bin)