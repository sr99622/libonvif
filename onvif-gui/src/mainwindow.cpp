/*******************************************************************************
* mainwindow.cpp
*
* Copyright (c) 2020 Stephen Rhodes
*
* This program is free software; you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation; either version 2 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along
* with this program; if not, write to the Free Software Foundation, Inc.,
* 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*
*******************************************************************************/

#include <iostream>
#include "mainwindow.h"
#include <QGridLayout>
#include <QApplication>
#include <QScreen>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    setWindowTitle("onvif-gui version 1.4.1");
    settings = new QSettings("libonvif", "onvif");

    glWidget = new avio::GLWidget();

    cameraPanel = new CameraPanel(this);
    setMinimumWidth(840);

    QWidget* layoutPanel = new QWidget();
    QGridLayout* layout = new QGridLayout();

    layout->addWidget(glWidget,        0, 0, 2, 1);
    layout->addWidget(cameraPanel,     0, 1, 1, 1);
    layout->setColumnStretch(0, 10);
    layoutPanel->setLayout(layout);
    setCentralWidget(layoutPanel);

    QList<QScreen*> screens = QGuiApplication::screens();
    QSize screenSize = screens[0]->size();
    int x = (screenSize.width() - width()) / 2;
    int y = (screenSize.height() - height()) / 2;
    move(x, y);
    std::cout << "w: " << screenSize.width() << " h: " << screenSize.height() << std::endl;

    QRect savedGeometry = settings->value("geometry").toRect();
    if (savedGeometry.isValid()) {
        setGeometry(savedGeometry);
    }
    else {
        QList<QScreen*> screens = QGuiApplication::screens();
        QSize screenSize = screens[0]->size();
        int x = (screenSize.width() - width()) / 2;
        int y = (screenSize.height() - height()) / 2;
        move(x, y);
        std::cout << "w: " << screenSize.width() << " h: " << screenSize.height() << std::endl;
    }
}

MainWindow::~MainWindow()
{
}

void MainWindow::closeEvent(QCloseEvent* event)
{
    Q_UNUSED(event);
    settings->setValue("geometry", geometry());
}

void MainWindow::msg(QString str)
{
    std::cout << (const char*)str.toLatin1() << std::endl;
}
