#*******************************************************************************
# FindFFmpeg.cmake
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

macro(find_component_library var library_name)
    find_library(${var} NAME ${library_name}
        HINTS
            $ENV{CONDA_PREFIX}
    )
endmacro()

macro(find_component_include_dir var header_name)
    find_path(${var} ${header_name}
        HINTS
            $ENV{CONDA_PREFIX}
    )
endmacro()

find_package(PkgConfig QUIET)
if (PKG_CONFIG_FOUND)
    pkg_check_modules(FFMPEG 
        libavcodec 
        libavfilter 
        libavformat 
        libavutil 
        libswscale 
        libswresample
    )
else()
    find_component_library(LIBAVCODEC_LIBRARY avcodec)
    find_component_include_dir(LIBAVCODEC_INCLUDE_DIR libavcodec/avcodec.h)

    find_component_library(LIBAVDEVICE_LIBRARY avdevice)
    find_component_include_dir(LIBAVDEVICE_INCLUDE_DIR libavdevice/avdevice.h)
    
    find_component_library(LIBAVFILTER_LIBRARY avfilter)
    find_component_include_dir(LIBAVFILTER_INCLUDE_DIR libavfilter/avfilter.h)
    
    find_component_library(LIBAVFORMAT_LIBRARY avformat)
    find_component_include_dir(LIBAVFORMAT_INCLUDE_DIR libavformat/avformat.h)
    
    find_component_library(LIBAVUTIL_LIBRARY avutil)
    find_component_include_dir(LIBAVUTIL_INCLUDE_DIR libavutil/avutil.h)
    
    find_component_library(LIBSWRESAMPLE_LIBRARY swresample)
    find_component_include_dir(LIBSWRESAMPLE_INCLUDE_DIR libswresample/swresample.h)
    
    find_component_library(LIBSWSCALE_LIBRARY swscale)
    find_component_include_dir(LIBSWSCALE_INCLUDE_DIR libswscale/swscale.h)

    set(FFMPEG_LINK_LIBRARIES
        ${LIBAVCODEC_LIBRARY}
        ${LIBAVDEVICE_LIBRARY}
        ${LIBAVFILTER_LIBRARY}
        ${LIBAVFORMAT_LIBRARY}
        ${LIBAVUTIL_LIBRARY}
        ${LIBSWRESAMPLE_LIBRARY}
        ${LIBSWSCALE_LIBRARY}
    )

    set(FFMPEG_INCLUDE_DIRS
        ${LIBAVCODEC_INCLUDE_DIR}
        ${LIBAVDEVICE_INCLUDE_DIR}
        ${LIBAVFILTER_INCLUDE_DIR}
        ${LIBAVFORMAT_INCLUDE_DIR}
        ${LIBVUTIL_INCLUDE_DIRS}
        ${LIBSWRESAMPLE_INCLUDE_DIR}
        ${LIBSWSCALE_INCLUDE_DIR}
    )

endif()

include(FindPackageHandleStandardArgs)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(FFmpeg
    REQUIRED_VARS FFMPEG_INCLUDE_DIRS FFMPEG_LINK_LIBRARIES
    VERSION_VAR FFMPEG_VERSION_STRING
)

if (FFMPEG_FOUND)
    add_library(FFmpeg::FFmpeg INTERFACE IMPORTED)
    set_target_properties(FFmpeg::FFmpeg PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${FFMPEG_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES "${FFMPEG_LINK_LIBRARIES}")
endif()
