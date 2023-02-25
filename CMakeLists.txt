#*******************************************************************************
# libonvif/CMakeLists.txt
#
# Copyright (c) 2022 Stephen Rhodes 
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

project(libonvif-pkg VERSION 1.4.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(WIN32)
    add_compile_options("/EHsc")
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
    set(BUILD_SHARED_LIBS TRUE)
else()
    set(CMAKE_CXX_FLAGS "-D_FORTIFY_SOURCE=2 -fstack-protector-strong -Wformat -Werror=format-security")
endif()

include(GNUInstallDirs)
set(CMAKE_INSTALL_INCLUDEDIR "include/${CMAKE_LIBRARY_ARCHITECTURE}")

add_subdirectory(libonvif)
add_subdirectory(libonvifboss)
add_subdirectory(onvif-util)

if (BUILD_GUI)

    add_subdirectory(libavio)
    add_subdirectory(onvif-gui)

    install(TARGETS onvif-gui DESTINATION bin)
    if (WIN32)
        install(TARGETS libavio 
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib
            RUNTIME DESTINATION bin
            COMPONENT library
        )
    else()
        install(TARGETS libavio DESTINATION "${CMAKE_INSTALL_LIBDIR}/")
    endif()

endif()

install(TARGETS onvif-util DESTINATION bin)

if (WIN32)
    install(TARGETS onvif 
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        COMPONENT library
    )
    install(TARGETS onvifboss
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        COMPONENT library
    )
else()
    install(TARGETS onvif DESTINATION "${CMAKE_INSTALL_LIBDIR}/")
    install(TARGETS libonvifboss DESTINATION "${CMAKE_INSTALL_LIBDIR}/")
endif()

install(FILES libonvif/include/onvif.h DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(FILES onvif-util/docs/onvif-util.1 DESTINATION "${CMAKE_INSTALL_PREFIX}/share/man/man1")
install(FILES onvif-gui/docs/onvif-gui.1 DESTINATION "${CMAKE_INSTALL_PREFIX}/share/man/man1")
