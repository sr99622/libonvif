/*******************************************************************************
* logindialog.cpp
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
#include "logindialog.h"
#include "camerapanel.h"

#include <QGridLayout>
#include <QLabel>

LoginDialog::LoginDialog(QWidget *parent)
{
    cameraPanel = parent;
    setWindowTitle("Login");
    cameraIP = new QLabel();
    cameraName = new QLabel("Camera Name:");
    buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    username = new QLineEdit;
    password = new QLineEdit;
    QGridLayout *layout = new QGridLayout;
    layout->addWidget(cameraIP,               0, 0, 1, 2);
    layout->addWidget(cameraName,             1, 0, 1, 2);
    layout->addWidget(new QLabel("Username"), 2, 0, 1, 1);
    layout->addWidget(username,               2, 1, 1, 1);
    layout->addWidget(new QLabel("Password"), 3, 0, 1, 1);
    layout->addWidget(password,               3, 1, 1, 1);
    layout->addWidget(buttonBox,              4, 0, 1, 2);
    setLayout(layout);

    connect(buttonBox, SIGNAL(accepted()), this, SLOT(accept()));
    connect(buttonBox, SIGNAL(rejected()), this, SLOT(reject()));
}

int LoginDialog::exec()
{
    QRect rect = CP->geometry();

    setMinimumWidth(300);

    QPoint global = CP->mapToGlobal(QPoint(rect.x(), rect.y()));

    int x = global.x() + rect.width() / 2 - 150;
    int y = global.y() + rect.height() / 2 - 150;

    move(x, y);

    username->setText("");
    password->setText("");
    username->setFocus();
    return QDialog::exec();
}

