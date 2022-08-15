QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

SOURCES += \
    main.cpp \
    mainwindow.cpp

HEADERS += \
    mainwindow.h

unix: LIBS += -L/usr/local/lib/ -lonvif -lxml2

win32: LIBS += -L'C:/Program Files (x86)/libonvif/lib/' -lonvif \
               -L'C:/Program Files (x86)/libxml2/lib/' -llibxml2

unix: INCLUDEPATH += /usr/local/include \
                          /usr/include/libxml2

win32: INCLUDEPATH += 'C:/Program Files (x86)/libonvif/include' \
                      'C:/Program Files (x86)/libxml2/include/libxml2'
