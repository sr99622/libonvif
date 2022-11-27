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

project(libonvif VERSION 1.4.0)

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

find_package(LibXml2 REQUIRED)

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
    SOVERSION 1
)

target_include_directories(onvif_shared PUBLIC
    include
    ${LIBXML2_INCLUDE_DIRS}
)

add_executable(onvif-util
    onvif-util/src/onvif-util.cpp
)

target_link_libraries(onvif-util PRIVATE
    onvif_shared
    ${LIBXML2_LIBRARIES}
)

target_include_directories(onvif-util PUBLIC
    include
    ${LIBXML2_INCLUDE_DIRS}
)

if (BUILD_GUI)

    set(CMAKE_AUTOMOC ON)
    find_package(Qt5 COMPONENTS Widgets REQUIRED)
    find_package(Avio REQUIRED)
    find_package(SDL2 REQUIRED)
    find_package(FFmpeg REQUIRED)

    set(MOC_INCLUDES
        onvif-gui/include/admintab.h
        onvif-gui/include/camera.h
        onvif-gui/include/cameradialogtab.h
        onvif-gui/include/cameralistmodel.h
        onvif-gui/include/cameralistview.h
        onvif-gui/include/camerapanel.h
        onvif-gui/include/configtab.h
        onvif-gui/include/discovery.h
        onvif-gui/include/imagetab.h
        onvif-gui/include/logindialog.h
        onvif-gui/include/mainwindow.h
        onvif-gui/include/networktab.h
        onvif-gui/include/onvifmanager.h
        onvif-gui/include/ptztab.h
        onvif-gui/include/videotab.h
    )

    add_executable(onvif-gui
        onvif-gui/src/admintab.cpp
        onvif-gui/src/camera.cpp
        onvif-gui/src/cameradialogtab.cpp
        onvif-gui/src/cameralistmodel.cpp
        onvif-gui/src/cameralistview.cpp
        onvif-gui/src/camerapanel.cpp
        onvif-gui/src/configtab.cpp
        onvif-gui/src/discovery.cpp
        onvif-gui/src/imagetab.cpp
        onvif-gui/src/logindialog.cpp
        onvif-gui/src/main.cpp
        onvif-gui/src/mainwindow.cpp
        onvif-gui/src/networktab.cpp
        onvif-gui/src/onvifmanager.cpp
        onvif-gui/src/ptztab.cpp
        onvif-gui/src/videotab.cpp
        ${MOC_INCLUDES}
    )

    set_target_properties(onvif-gui PROPERTIES
        OUTPUT_NAME onvif-gui
        ARCHIVE_OUTPUT_DIRECTORY onvif-gui
    )

    target_link_libraries(onvif-gui PRIVATE
        Qt5::Widgets
        onvif_static
        ${LIBXML2_LIBRARIES}
        ${FFMPEG_LIBRARIES}
        ${AVIO_LIBRARY}
        SDL2::SDL2
    )

    target_include_directories(onvif-gui PUBLIC
        include
        onvif-gui/include
        ${LIBXML2_INCLUDE_DIRS}
        ${AVIO_INCLUDE_DIR}
    )

    install(TARGETS onvif-gui DESTINATION bin)

endif()

install(TARGETS onvif-util DESTINATION bin)
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
INSTALL(FILES onvif-util/docs/onvif-util.1 DESTINATION ${CMAKE_INSTALL_PREFIX}/share/man/man1)
INSTALL(FILES onvif-gui/docs/onvif-gui.1 DESTINATION ${CMAKE_INSTALL_PREFIX}/share/man/man1)
