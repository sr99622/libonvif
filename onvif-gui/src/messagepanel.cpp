/*******************************************************************************
* messagepanel.cpp
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

#include "messagepanel.h"
#include "mainwindow.h"
#include <QGridLayout>
#include <QGuiApplication>

MessagePanel::MessagePanel(QMainWindow *parent)
{
    mainWindow = parent;
    cp = QGuiApplication::clipboard();
    msg = new QTextEdit();
    btnCopy = new QPushButton("Copy");
    connect(btnCopy, SIGNAL(clicked()), this, SLOT(onBtnCopyClicked()));
    QGridLayout *layout = new QGridLayout(this);
    layout->addWidget(msg,     0, 0, 1, 1);
    layout->addWidget(btnCopy, 1, 0, 1, 1, Qt::AlignRight);
}

void MessagePanel::onBtnCopyClicked()
{
    //std::cout << "test" << std::endl;
    cp->setText(msg->toPlainText());
}
