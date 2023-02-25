#*******************************************************************************
# libonvif/libonvif/CMakeLists.txt
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

project(libonvifmanager VERSION 1.4.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(WIN32)
    add_compile_options("/EHsc")
#    find_package(Iconv REQUIRED)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
    set(BUILD_SHARED_LIBS TRUE)
endif()

add_library(onvifboss SHARED
    src/onvifboss.cpp
)

target_link_libraries(onvifboss PRIVATE
    onvif
)

set_target_properties(onvifboss PROPERTIES
    SOVERSION 1
)

target_include_directories(onvifboss PUBLIC
    include
)
