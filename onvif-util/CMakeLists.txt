#*******************************************************************************
# libonvif/onvif-util/CMakeLists.txt
#
# Copyright (c) 2022, 2025 Stephen Rhodes 
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

project(onvif-util VERSION 1.4.9)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if (DEFINED ENV{LIBXML2_INCLUDE_DIRS})
    set(LIBXML2_INCLUDE_DIRS $ENV{LIBXML2_INCLUDE_DIRS})
    set(LIBXML2_LIBRARIES $ENV{LIBXML2_LIBRARIES})
else()
    find_package(LibXml2 REQUIRED)
endif()

add_library(libonvif STATIC
    ../libonvif/src/onvif.c
    ../libonvif/src/cencode.c
    ../libonvif/src/sha1.c
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
    ../libonvif/include
    ${LIBXML2_INCLUDE_DIRS}
)

add_executable(onvif-util
    src/onvif-util.cpp
)

target_link_libraries(onvif-util PRIVATE
    libonvif
)
