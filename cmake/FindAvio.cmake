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
