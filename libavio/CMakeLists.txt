#********************************************************************
# libavio/CMakeLists.txt
#
# Copyright (c) 2022  Stephen Rhodes
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
#********************************************************************/

cmake_minimum_required(VERSION 3.17)

project(avio)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-D__STDC_CONSTANT_MACROS")
if (WIN32)
    add_compile_options("/EHsc")
endif()
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

if(MSVC)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
    set(BUILD_SHARED_LIBS TRUE)
endif()

include(GNUInstallDirs)
# set arch-specific include dir
set(CMAKE_INSTALL_INCLUDEDIR "include/${CMAKE_LIBRARY_ARCHITECTURE}")

if (UNIX)
    message("finding alsa")
    set (ALSA_INCLUDE_DIR_CANDIDATE $ENV{CONDA_PREFIX}/include/alsa)
    if (EXISTS "${ALSA_INCLUDE_DIR_CANDIDATE}")
        set (ALSA_INCLUDE_DIR ${ALSA_INCLUDE_DIR_CANDIDATE})
        message("-- Found ALSA include directory ${ALSA_INCLUDE_DIR}")
    endif()
    set (ALSA_LIBRARIES_CANDIDATE $ENV{CONDA_PREFIX}/lib/libasound.so)
    if (EXISTS "${ALSA_LIBRARIES_CANDIDATE}")
        set (ALSA_LIBRARIES ${ALSA_LIBRARIES_CANDIDATE})
        message("-- Found ALSA libraries ${ALSA_LIBRARIES}")
    endif()
endif()

find_package(SDL2)
find_package(FFmpeg)
find_package(Qt5 COMPONENTS Widgets REQUIRED)

set(CMAKE_AUTOMOC ON)
set(MOC_INCLUDES
    include/GLWidget.h
)

add_library(avio SHARED 
                        src/Reader.cpp
                        src/Exception.cpp
                        src/Event.cpp
                        src/Frame.cpp
                        src/Decoder.cpp
                        src/Filter.cpp
                        src/Clock.cpp
                        src/Display.cpp
                        src/Encoder.cpp
                        src/Writer.cpp
                        src/Pipe.cpp
                        src/GLWidget.cpp
                        ${MOC_INCLUDES}
                    )
                
target_link_libraries(avio PRIVATE 
                SDL2::SDL2
                Qt5::Widgets
                ${FFMPEG_LIBRARIES}
             )

target_include_directories(avio PUBLIC
                    include
                    ${FFMPEG_INCLUDE_DIRS}
                  )

if (MSVC)
    install(TARGETS avio
                    LIBRARY DESTINATION lib
                    ARCHIVE DESTINATION lib
                    RUNTIME DESTINATION bin
                    COMPONENT library)
else()
    install(TARGETS avio DESTINATION "${CMAKE_INSTALL_LIBDIR}/")
endif()
install(FILES 
            include/avio.h 
            include/Clock.h
            include/Decoder.h
            include/Display.h
            include/Encoder.h
            include/Event.h
            include/Exception.h
            include/Filter.h
            include/Frame.h
            include/Pipe.h
            include/Queue.h
            include/Reader.h
            include/Writer.h
            include/GLWidget.h
            
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
