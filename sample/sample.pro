#*******************************************************************************
# sample.pro
#
# Copyright (c) 2020 Stephen Rhodes 
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
#******************************************************************************/

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

SOURCES += \
    admintab.cpp \
    camera.cpp \
    cameradialogtab.cpp \
    cameralistmodel.cpp \
    cameralistview.cpp \
    camerapanel.cpp \
    configtab.cpp \
    discovery.cpp \
    imagetab.cpp \
    logindialog.cpp \
    main.cpp \
    mainwindow.cpp \
    networktab.cpp \
    onvifmanager.cpp \
    ptztab.cpp \
    videotab.cpp

HEADERS += \
    admintab.h \
    camera.h \
    cameradialogtab.h \
    cameralistmodel.h \
    cameralistview.h \
    camerapanel.h \
    configtab.h \
    discovery.h \
    imagetab.h \
    logindialog.h \
    mainwindow.h \
    networktab.h \
    onvifmanager.h \
    ptztab.h \
    videotab.h

unix: LIBS += -L/usr/local/lib/ -lonvif -lxml2

win32: LIBS += -L'C:/Program Files (x86)/libonvif/lib/' -lonvif \
               -L'C:/Program Files (x86)/libxml2/lib/' -llibxml2

unix: INCLUDEPATH += /usr/local/include \
                     /usr/include/libxml2 \

win32: INCLUDEPATH += 'C:/Program Files (x86)/libonvif/include' \
                      'C:/Program Files (x86)/libxml2/include/libxml2'

DISTFILES += \
    CMakeLists.txt \
    sample.pro.user
