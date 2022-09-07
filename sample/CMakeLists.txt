cmake_minimum_required(VERSION 3.17)
set(win_path "C:/Program Files (x86)")

if (WIN32)
    set(xml_lib_name "libxml2_a")
endif()

if (UNIX)
    set(xml_lib_name "xml2")
endif()

project(discover)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)

find_package(Qt5 COMPONENTS Widgets REQUIRED)
if (Qt5_FOUND)
    message("-- Found Qt")
endif()

#find_package(LibXml2)
if (NOT LibXml2_FOUND)
    message("-- Looking for installation of libxml2")
    find_path(LIBXML2_INCLUDE_DIRS libxml/parser.h
        HINTS
            $ENV{CONDA_PREFIX}/Library/include/libxml2
            ${win_path}/libxml2/include/libxml2
    )
    find_library(LIBXML2_LIBRARIES NAMES ${xml_lib_name}
        HINTS
            $ENV{CONDA_PREFIX}/lib
            ${win_path}/libxml2/lib
    )
    message("-- Found LIBXML2_INCLUDE_DIRS ${LIBXML2_INCLUDE_DIRS}")
    message("-- Found LIBXML2_LIBRARIES ${LIBXML2_LIBRARIES}")
endif()

find_package(LibOnvif)
if (LibOnvif_FOUND)
    message("-- Found LIBONVIF_INCLUDE_DIRS ${LIBONVIF_INCLUDE_DIRS}")
    message("-- Found LIBONVIF_LIBRARIES ${LIBONVIF_LIBRARIES}")
endif()

find_package(Iconv)
if (NOT Iconv_FOUND)
    message("--Could not find Iconv. If you are using libxml2 compiled without iconv, this message may be ignored")
endif()

add_executable(onvif
    admintab.cpp
    camera.cpp
    cameradialogtab.cpp
    cameralistmodel.cpp
    cameralistview.cpp
    camerapanel.cpp
    configtab.cpp
    discovery.cpp
    imagetab.cpp
    logindialog.cpp
    main.cpp
    mainwindow.cpp
    networktab.cpp
    onvifmanager.cpp
    ptztab.cpp
    videotab.cpp
)

target_link_libraries(onvif PRIVATE
    Qt5::Widgets
    ${LIBONVIF_LIBRARIES}
    ${LIBXML2_LIBRARIES}
    ${Iconv_LIBRARY}
)

target_include_directories(onvif PUBLIC
    ${LIBONVIF_INCLUDE_DIRS}
    ${LIBXML2_INCLUDE_DIRS}
    ${Iconv_INCLUDE_DIR}
)

