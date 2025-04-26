#*******************************************************************************
# libonvif/libonvif/CMakeLists.txt
#
# Copyright (c) 2023, 2024 Stephen Rhodes 
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

project(libonvif VERSION 3.2.6)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if (DEFINED ENV{LIBXML2_INCLUDE_DIRS})
    set(LIBXML2_INCLUDE_DIRS $ENV{LIBXML2_INCLUDE_DIRS})
    set(LIBXML2_LIBRARIES $ENV{LIBXML2_LIBRARIES})
else()
    find_package(LibXml2 REQUIRED)
endif()

add_subdirectory(pybind11)

pybind11_add_module(pyonvif
    src/onvif.cpp
    src/onvif.c
    src/cencode.c
    src/sha1.c
)

set_target_properties(pyonvif PROPERTIES
    OUTPUT_NAME libonvif
)

target_link_libraries(pyonvif PRIVATE
    ${LIBXML2_LIBRARIES}
)

message("-- LIBXML2_INCLUDE_DIRS: " ${LIBXML2_INCLUDE_DIRS})
message("-- LIBXML2_LIBRARIES: " ${LIBXML2_LIBRARIES})

target_include_directories(pyonvif PUBLIC
    include
    ${LIBXML2_INCLUDE_DIRS}
)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set_target_properties(pyonvif PROPERTIES
        BUILD_RPATH "$ORIGIN"
        BUILD_RPATH_USE_ORIGIN TRUE
        INSTALL_RPATH "$ORIGIN"
        INSTALL_RPATH_USE_ORIGIN TRUE
    )
endif()
