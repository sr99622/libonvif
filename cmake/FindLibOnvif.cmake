find_path(LIBONVIF_INCLUDE_DIR onvif.h
    HINTS
        /usr/local/include
        /usr/include
        $ENV{CONDA_PREFIX}/include
)

find_library(LIBONVIF_LIBRARY NAMES onvif
    HINTS
        /usr/local/lib
        /usr/lib/x86_64-linux-gnu/lib
        $ENV{CONDA_PREFIX}/lib
)

set(LIBONVIF_INCLUDE_DIRS ${LIBONVIF_INCLUDE_DIR})
set(LIBONVIF_LIBRARIES ${LIBONVIF_LIBRARY})
set(LIBONVIF_VERSION_STRING 1.2.0)

include(FindPackageHandleStandardArgs)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(LibOnvif
                                    REQUIRED_VARS LIBONVIF_INCLUDE_DIRS LIBONVIF_LIBRARIES
                                    VERSION_VAR LIBONVIF_VERSION_STRING)

mark_as_advanced(LIBONVIF_INCLUDE_DIR
                 LIBONVIF_LIBRARY)
