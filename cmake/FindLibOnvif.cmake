#*******************************************************************************
# FindLibOnvif.cmake
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

set(win_path "C:/Program Files (x86)")

find_path(LIBONVIF_INCLUDE_DIR onvif.h
    HINTS
        /usr/local/include
        /usr/include
        $ENV{CONDA_PREFIX}/include
        $ENV{CONDA_PREFIX}/Library/include
        ${win_path}/libonvif/include
)

find_library(LIBONVIF_LIBRARY NAMES onvif
    HINTS
        /usr/local/lib
        /usr/lib/x86_64-linux-gnu/lib
        $ENV{CONDA_PREFIX}/lib
        $ENV{CONDA_PREFIX}/Library/lib
        ${win_path}/libonvif/lib
)

set(LIBONVIF_INCLUDE_DIRS ${LIBONVIF_INCLUDE_DIR})
set(LIBONVIF_LIBRARIES ${LIBONVIF_LIBRARY})
set(LIBONVIF_VERSION_STRING 1.4.1)

include(FindPackageHandleStandardArgs)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(LibOnvif
    REQUIRED_VARS LIBONVIF_INCLUDE_DIRS LIBONVIF_LIBRARIES
    VERSION_VAR LIBONVIF_VERSION_STRING
)

mark_as_advanced(LIBONVIF_INCLUDE_DIR LIBONVIF_LIBRARY)
