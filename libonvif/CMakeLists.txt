cmake_minimum_required(VERSION 3.17)

project(libonvif VERSION 3.2.9 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Preserve the old setup.py Windows behavior so FindLibXml2 can succeed.
if(WIN32)
    set(_LOCAL_LIBXML2_DIR "${CMAKE_SOURCE_DIR}/libxml2")
    if(EXISTS "${_LOCAL_LIBXML2_DIR}")
        set(ENV{PATH} "${_LOCAL_LIBXML2_DIR};$ENV{PATH}")
    endif()
endif()

if(DEFINED ENV{LIBXML2_INCLUDE_DIRS})
    set(LIBXML2_INCLUDE_DIRS $ENV{LIBXML2_INCLUDE_DIRS})
    set(LIBXML2_LIBRARIES $ENV{LIBXML2_LIBRARIES})
else()
    find_package(LibXml2 REQUIRED)
endif()

set(PYBIND11_FINDPYTHON ON)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(pyonvif
    src/onvif.cpp
    src/onvif.c
    src/cencode.c
    src/sha1.c
)

set_target_properties(pyonvif PROPERTIES
    OUTPUT_NAME libonvif
)

target_link_libraries(pyonvif PRIVATE
    ${LIBXML2_LIBRARIES}
)

target_include_directories(pyonvif PRIVATE
    include
    ${LIBXML2_INCLUDE_DIRS}
)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set_target_properties(pyonvif PROPERTIES
        BUILD_RPATH "$ORIGIN"
        BUILD_RPATH_USE_ORIGIN TRUE
        INSTALL_RPATH "$ORIGIN"
        INSTALL_RPATH_USE_ORIGIN TRUE
    )
endif()

install(TARGETS pyonvif
    LIBRARY DESTINATION libonvif
    RUNTIME DESTINATION libonvif
    ARCHIVE DESTINATION libonvif
)