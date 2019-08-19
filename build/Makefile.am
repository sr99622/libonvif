# Makefile for libonvif

AM_CFLAGS = -I$(INCLUDE) -I$(INCLUDE)/libxml2 -I../source
ACLOCAL_AMFLAGS = -I m4

pkginclude_HEADERS = ../source/onvif.h ../source/sha1.h ../source/cencode.h
INCLUDE = @includedir@
LIB = @libdir@

lib_LTLIBRARIES = libonvif.la

if MINGW32
libonvif_la_LIBADD = -lws2_32 -liphlpapi -lxml2
else
libonvif_la_LIBADD = -lxml2
endif

libonvif_la_LDFLAGS = -no-undefined
libonvif_la_SOURCES = ../source/onvif.c ../source/sha1.c ../source/cencode.c

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = libonvif.pc
