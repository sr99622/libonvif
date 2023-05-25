#*******************************************************************************
# libonvif/CMakeLists.txt
#
# Copyright (c) 2022 Stephen Rhodes 
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
#
#******************************************************************************/

cmake_minimum_required(VERSION 3.17)

project(libonvif-pkg VERSION 1.4.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

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
add_subdirectory(onvif-util)
IF (NOT WITHOUT_PYTHON)
    add_subdirectory(libavio)
endif()

if (WIN32)
    install(TARGETS libonvif 
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        COMPONENT library
    )
    IF (NOT WITHOUT_PYTHON)
        install(TARGETS libavio 
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib
            RUNTIME DESTINATION bin
            COMPONENT library
        )
    endif()
else()
    install(TARGETS libonvif DESTINATION "${CMAKE_INSTALL_LIBDIR}/")
    IF (NOT WITHOUT_PYTHON)
        install(TARGETS libavio DESTINATION "${CMAKE_INSTALL_LIBDIR}/")
    endif()
endif()


install(TARGETS onvif-util DESTINATION bin)
install(FILES libonvif/include/onvif.h libonvif/include/onvifboss.h DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(FILES onvif-util/docs/onvif-util.1 DESTINATION "${CMAKE_INSTALL_PREFIX}/share/man/man1")
