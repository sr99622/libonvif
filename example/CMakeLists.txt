cmake_minimum_required(VERSION 3.17)

project(test_onvif)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)

find_package(LibXml2)
if (NOT LibXml2_FOUND)
    message("-- Looking for conda installation of libxml2")
    find_path(LIBXML2_INCLUDE_DIRS libxml/parser.h
        HINTS
            $ENV{CONDA_PREFIX}/include/libxml2
    )
    find_library(LIBXML2_LIBRARIES NAMES xml2
        HINTS
            $ENV{CONDA_PREFIX}/lib
    )
    message("-- Found LIBXML2_INCLUDE_DIRS ${LIBXML2_INCLUDE_DIRS}")
    message("-- Found LIBXML2_LIBRARIES ${LIBXML2_LIBRARIES}")
endif()

find_package(LibOnvif)
if (LibOnvif_FOUND)
    message("-- Found LIBONVIF_INCLUDE_DIRS ${LIBONVIF_INCLUDE_DIRS}")
    message("-- Found LIBONVIF_LIBRARIES ${LIBONVIF_LIBRARIES}")
endif()

add_executable(test_onvif
                test_onvif.c
            )

target_link_libraries(test_onvif PRIVATE
                        ${LIBONVIF_LIBRARIES}
                        ${LIBXML2_LIBRARIES}
                    )

target_include_directories(test_onvif PUBLIC
                            ${LIBONVIF_INCLUDE_DIRS}
                            ${LIBXML2_INCLUDE_DIRS}
                        )
