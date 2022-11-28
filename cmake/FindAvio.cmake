#********************************************************************
# FindAvio.cmake
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

set(win_path "C:/Program Files (x86)")

find_path(AVIO_INCLUDE_DIR avio.h
    HINTS
        /usr/local/include
        /usr/include
        $ENV{CONDA_PREFIX}/include
        $ENV{CONDA_PREFIX}/Library/include
        ${win_path}/avio/include
)

find_library(AVIO_LIBRARY NAMES avio
    HINTS
        /usr/local/lib
        /usr/lib/x86_64-linux-gnu/lib
        $ENV{CONDA_PREFIX}/lib
        $ENV{CONDA_PREFIX}/Library/lib
        ${win_path}/avio/lib
)

set(AVIO_INCLUDE_DIRS ${AVIO_INCLUDE_DIR})
set(AVIO_LIBRARIES ${AVIO_LIBRARY})
set(AVIO_VERSION_STRING 1.2.0)

include(FindPackageHandleStandardArgs)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(Avio
    REQUIRED_VARS AVIO_INCLUDE_DIRS AVIO_LIBRARIES
    VERSION_VAR AVIO_VERSION_STRING
)

mark_as_advanced(AVIO_INCLUDE_DIR AVIO_LIBRARY)
