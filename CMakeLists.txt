#*******************************************************************************
# libonvif/CMakeLists.txt
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

project(libonvif)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

if(MSVC)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
    set(BUILD_SHARED_LIBS TRUE)
endif()

include(GNUInstallDirs)
# set arch-specific include dir
set(CMAKE_INSTALL_INCLUDEDIR "include/${CMAKE_LIBRARY_ARCHITECTURE}")

find_package(LibXml2)
if (NOT LibXml2_FOUND)
    message("-- Looking for conda installation of libxml2")
    find_path(LIBXML2_INCLUDE_DIRS libxml/parser.h
        HINTS
            $ENV{CONDA_PREFIX}/include/libxml2
    )
    find_library(LIBXML2_LIBRARIES NAMES xml2
        HINTS
            $ENV{CONDA_PREFIX}/lib
    )
    message("-- Found LIBXML2_INCLUDE_DIRS ${LIBXML2_INCLUDE_DIRS}")
    message("-- Found LIBXML2_LIBRARIES ${LIBXML2_LIBRARIES}")
endif()

add_library(onvif_static STATIC 
    src/onvif.c
    src/cencode.c
    src/sha1.c
)

set_target_properties(onvif_static PROPERTIES
    OUTPUT_NAME onvif_static
    ARCHIVE_OUTPUT_DIRECTORY onvif_static
)

target_link_libraries(onvif_static PRIVATE
    ${LIBXML2_LIBRARIES}
)

target_include_directories(onvif_static PUBLIC
    include
    ${LIBXML2_INCLUDE_DIRS}
    $ENV{CONDA_PREFIX}/Library/include
)

add_library(onvif_shared SHARED
    src/onvif.c
    src/cencode.c
    src/sha1.c
)

set_target_properties(onvif_shared PROPERTIES
    OUTPUT_NAME onvif
    ARCHIVE_OUTPUT_DIRECTORY onvif_shared
)

target_link_libraries(onvif_shared PRIVATE
    ${LIBXML2_LIBRARIES}
)

set_target_properties(onvif_shared PROPERTIES
    SOVERSION 0
)

target_include_directories(onvif_shared PUBLIC
    include
    ${LIBXML2_INCLUDE_DIRS}
    $ENV{CONDA_PREFIX}/Library/include
)

add_subdirectory(onvif-util)

if (BUILD_GUI)
    add_subdirectory(onvif-camadmin)
endif()

if (MSVC)
    install(TARGETS onvif_shared 
                    LIBRARY DESTINATION lib
                    ARCHIVE DESTINATION lib
                    RUNTIME DESTINATION bin
                    COMPONENT library)
else()
    install(TARGETS onvif_shared DESTINATION "${CMAKE_INSTALL_LIBDIR}/")
endif()
install(TARGETS onvif_static DESTINATION "${CMAKE_INSTALL_LIBDIR}/")
install(FILES include/onvif.h DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
INSTALL(FILES onvif-util/onvif-util.1 DESTINATION ${CMAKE_INSTALL_PREFIX}/man/man1)
INSTALL(FILES onvif-camadmin/onvif-camadmin.1 DESTINATION ${CMAKE_INSTALL_PREFIX}/man/man1)
