/*******************************************************************************
* main.cpp
*
* Copyright (c) 2022 Stephen Rhodes
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

#include "PyRunner.h"

#include <functional>
#include <iostream>

#include "mainwindow.h"
#include <QApplication>


Q_DECLARE_METATYPE(std::string);

int main(int argc, char *argv[])
{
    /*
    std::string python_dir = "C:/Users/sr996/source/libonvif/onvif-gui/python";
    std::string python_file = "echo";
    std::string python_class = "Echo";
    std::string python_args = "key1=value1";
    avio::PyRunner runner;
    import_array();
    runner.initialize(python_dir, python_file, python_class, python_args);
    */

    qRegisterMetaType<std::string>();

    QApplication a(argc, argv);
    MainWindow w;

    //using namespace std::placeholders;
    //w.glWidget->initPy = std::bind(&avio::PyRunner::initialize, &runner, _1, _2, _3, _4);
    //w.glWidget->runPy = std::bind(&avio::PyRunner::run, &runner, _1);
    
	w.show();
    return a.exec();
}
