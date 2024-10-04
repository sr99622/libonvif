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

project(libonvif VERSION 3.2.2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_definitions(-w)

# pass current dir in from host when building in virtual env
set (MY_CURRENT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
if (DEFINED ENV{CMAKE_CURRENT_SOURCE_DIR})
    set(MY_CURRENT_DIR $ENV{CMAKE_CURRENT_SOURCE_DIR})
endif()
file(TO_CMAKE_PATH "${MY_CURRENT_DIR}" MY_DIR_VAR)

if (WIN32)
    set(LIBXML2_INCLUDE_DIRS ${MY_DIR_VAR}/libxml2/include/libxml2)
    set(LIBXML2_LIBRARIES ${MY_DIR_VAR}/libxml2/lib/libxml2.lib)
else()
    find_package(LibXml2 REQUIRED)
endif()

if (NOT WITHOUT_LIBS)

    add_library(libonvif SHARED
        src/onvif.c
        src/cencode.c
        src/sha1.c
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

endif()

IF (BUILD_TEST)
    add_executable(onvif-test
        test/onvif-test.cpp
    )

    target_link_libraries(onvif-test PRIVATE
        libonvif
    )

    target_include_directories(onvif-test PUBLIC
        include
    )

endif()