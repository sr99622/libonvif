cmake_minimum_required(VERSION 3.17)

project(libonvif)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(GNUInstallDirs)
# set arch-specific include dir
set(CMAKE_INSTALL_INCLUDEDIR "include/${CMAKE_LIBRARY_ARCHITECTURE}")

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

add_library(onvif_static STATIC 
    src/onvif.c
    src/cencode.c
    src/sha1.c
)

set_target_properties(onvif_static PROPERTIES
    OUTPUT_NAME onvif
    ARCHIVE_OUTPUT_DIRECTORY onvif_static
)

target_link_libraries(onvif_static PRIVATE
    ${LIBXML2_LIBRARIES}
)

target_include_directories(onvif_static PUBLIC
    include
    ${LIBXML2_INCLUDE_DIRS}
    $ENV{CONDA_PREFIX}/Library/include
)

add_library(onvif SHARED
    src/onvif.c
    src/cencode.c
    src/sha1.c
)

target_link_libraries(onvif PRIVATE
    ${LIBXML2_LIBRARIES}
)

set_target_properties(onvif PROPERTIES
    SOVERSION 0
)

target_include_directories(onvif PUBLIC
    include
    ${LIBXML2_INCLUDE_DIRS}
    $ENV{CONDA_PREFIX}/Library/include
)

add_executable(onvif-discover
    example/onvif-discover.c
)

target_link_libraries(onvif-discover PRIVATE
    onvif_static
    ${LIBXML2_LIBRARIES}
)

target_include_directories(onvif-discover PUBLIC
    include
    ${LIBXML2_INCLUDE_DIRS}
)

add_executable(onvif-getall
    example/onvif-getall.c
)

target_link_libraries(onvif-getall PRIVATE
    onvif_static
    ${LIBXML2_LIBRARIES}
)

target_include_directories(onvif-getall PUBLIC
    include
    ${LIBXML2_INCLUDE_DIRS}
)

if (BUILD_GUI)
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_AUTORCC ON)
    set(CMAKE_AUTOUIC ON)
    find_package(Qt5 COMPONENTS Widgets REQUIRED)

    add_executable(onvif_exe
        sample/admintab.cpp
        sample/camera.cpp
        sample/cameradialogtab.cpp
        sample/cameralistmodel.cpp
        sample/cameralistview.cpp
        sample/camerapanel.cpp
        sample/configtab.cpp
        sample/discovery.cpp
        sample/imagetab.cpp
        sample/logindialog.cpp
        sample/main.cpp
        sample/mainwindow.cpp
        sample/networktab.cpp
        sample/onvifmanager.cpp
        sample/ptztab.cpp
        sample/videotab.cpp
    )

    set_target_properties(onvif_exe PROPERTIES
        OUTPUT_NAME onvif
        ARCHIVE_OUTPUT_DIRECTORY onvif_exe
    )

    target_link_libraries(onvif_exe PRIVATE
        Qt5::Widgets
        onvif_static
        ${LIBXML2_LIBRARIES}
    )

    target_include_directories(onvif_exe PUBLIC
        include
        ${LIBXML2_INCLUDE_DIRS}
    )

    install(TARGETS onvif_exe DESTINATION bin)
endif()

install(TARGETS onvif-discover DESTINATION bin)
install(TARGETS onvif-getall DESTINATION bin)
install(TARGETS onvif DESTINATION "${CMAKE_INSTALL_LIBDIR}/")
install(TARGETS onvif_static DESTINATION "${CMAKE_INSTALL_LIBDIR}/")
install(FILES include/onvif.h DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
