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
