#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])

AC_INIT([libonvif], [1.1.0], [BUG-REPORT-ADDRESS])
AC_CONFIG_SRCDIR([source/onvif.c])
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_MACRO_DIRS([m4])
AM_INIT_AUTOMAKE([-Wall foreign])
AM_PROG_AR
LT_INIT([win32-dll])

echo host is $host
have_w32_system=no

case "${host}" in
  *-*-mingw32*)
    have_w32_system=yes

    ;;
esac
AM_CONDITIONAL(MINGW32, test "$have_w32_system" = yes)

# Checks for programs.
AC_PROG_CC
AC_PROG_LIBTOOL
PKG_PROG_PKG_CONFIG

# Checks for libraries.
PKG_CHECK_MODULES([LIBXML2], [libxml-2.0 >= 2.5])

# Checks for header files.
AC_CHECK_HEADERS([arpa/inet.h fcntl.h float.h inttypes.h limits.h locale.h memory.h netdb.h netinet/in.h stdint.h stdlib.h string.h strings.h sys/ioctl.h sys/socket.h sys/time.h sys/timeb.h unistd.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_CHECK_HEADER_STDBOOL
AC_TYPE_SIZE_T

# Checks for library functions.
#AC_FUNC_MALLOC
AC_FUNC_MKTIME
#AC_FUNC_REALLOC
AC_FUNC_STRERROR_R
AC_FUNC_STRTOD
AC_CHECK_FUNCS([ftime gethostbyname gettimeofday localtime_r memset select socket strchr strdup strerror strstr strtod])

AC_CONFIG_FILES([Makefile build/Makefile])
AC_CONFIG_FILES([build/libonvif.pc])

AC_OUTPUT
