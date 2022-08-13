cmake_minimum_required(VERSION 3.17)

project(libonvif)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

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

add_library(onvif SHARED
                src/onvif.c
                src/cencode.c
                src/sha1.c
            )

target_link_libraries(onvif PRIVATE
                        ${LIBXML2_LIBRARIES}
                    )

target_include_directories(onvif PUBLIC
                            include
                            ${LIBXML2_INCLUDE_DIRS}
                        )

install(TARGETS onvif DESTINATION lib)
install(FILES include/onvif.h DESTINATION include)