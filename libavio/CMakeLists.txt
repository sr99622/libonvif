#*******************************************************************************
# libavio/CMakeLists.txt
#
# Copyright (c) 2022 Stephen Rhodes 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#******************************************************************************/

cmake_minimum_required(VERSION 3.17)

project(libavio VERSION 1.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__STDC_CONSTANT_MACROS")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

if(WIN32)
    add_compile_options("/EHsc")
#    find_package(Iconv REQUIRED)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
    set(BUILD_SHARED_LIBS TRUE)
endif()

find_package(SDL2 REQUIRED)
find_package(FFmpeg REQUIRED)
find_package(Qt5 COMPONENTS Widgets REQUIRED)

set(CMAKE_AUTOMOC ON)

add_library(avio SHARED
    src/Reader.cpp
    src/Exception.cpp
    src/Frame.cpp
    src/Decoder.cpp
    src/Filter.cpp
    src/Clock.cpp
    src/Display.cpp
    src/Encoder.cpp
    src/Writer.cpp
    src/Pipe.cpp
    src/GLWidget.cpp
    include/GLWidget.h
)

target_link_libraries(avio PRIVATE
    Qt5::Widgets
    FFmpeg::FFmpeg
    SDL2::SDL2
    pthread
)

set_target_properties(avio PROPERTIES
    SOVERSION 1
)

target_include_directories(avio PUBLIC
    include
)