#/********************************************************************
# libavio/cmake/FindSDL2.cmake
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
#*********************************************************************/

macro(find_component_library var library_name)
    find_library(${var} NAME ${library_name}
        HINTS
            $ENV{LIBXML2_INSTALL_DIR}/lib
            $ENV{CONDA_PREFIX}
    )
endmacro()

macro(find_component_include_dir var header_name)
    find_path(${var} ${header_name}
        HINTS
            $ENV{LIBXML2_INSTALL_DIR}/include/libxml2
            $ENV{CONDA_PREFIX}
    )
endmacro()

find_package(PkgConfig QUIET)
if (PKG_CONFIG_FOUND)
        pkg_check_modules(LIBXML2 
        LIBXML2
    )
else()
    find_component_library(LIBXML2_LIBRARY LIBXML2)
    find_component_include_dir(LIBXML2_INCLUDE_DIR libxml/xpath.h)

    set(LIBXML2_LINK_LIBRARIES
        ${LIBXML2_LIBRARY}
    )

    set(LIBXML2_INCLUDE_DIRS
        ${LIBXML2_INCLUDE_DIR}/include/libxml2
    )

endif()

include(FindPackageHandleStandardArgs)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(LIBXML2
    REQUIRED_VARS LIBXML2_INCLUDE_DIRS LIBXML2_LINK_LIBRARIES
    VERSION_VAR LIBXML2_VERSION_STRING
)

if (LIBXML2_FOUND)
    add_library(LibXml2::LibXml2 INTERFACE IMPORTED)
    set_target_properties(LibXml2::LibXml2 PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${LIBXML2_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES "${LIBXML2_LINK_LIBRARIES}")
endif()
