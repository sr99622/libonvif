#*******************************************************************************
# libonvif/libonvif/CMakeLists.txt
#
# Copyright (c) 2023 Stephen Rhodes 
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

project(libonvif VERSION 2.0.7)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(WIN32)
    find_package(Iconv REQUIRED)
endif()

add_definitions(-w)

find_package(LibXml2 REQUIRED)

if (NOT WITHOUT_LIBS)

    add_library(libonvif SHARED
        src/onvif.c
        src/cencode.c
        src/sha1.c
        src/onvifboss.cpp
    )

    target_link_libraries(libonvif PRIVATE
        ${LIBXML2_LIBRARIES}
    )

    if (UNIX)
        set_target_properties(libonvif PROPERTIES
            OUTPUT_NAME onvif
            SOVERSION 1
        )
    endif()

    target_include_directories(libonvif PUBLIC
        include
        ${Iconv_INCLUDE_DIRS}
        ${LIBXML2_INCLUDE_DIRS}
    )

endif()

IF (NOT WITHOUT_PYTHON)

    add_subdirectory(pybind11)

    pybind11_add_module(pyonvif
        src/onvif.cpp
        src/onvif.c
        src/cencode.c
        src/sha1.c
        src/onvifboss.cpp
    )

    set_target_properties(pyonvif PROPERTIES
        OUTPUT_NAME libonvif
    )

    target_link_libraries(pyonvif PRIVATE
        ${LIBXML2_LIBRARIES}
    )

    target_include_directories(pyonvif PUBLIC
        include
        ${Iconv_INCLUDE_DIRS}
        ${LIBXML2_INCLUDE_DIRS}
    )

endif()
