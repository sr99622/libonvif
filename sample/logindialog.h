/*******************************************************************************
* logindialog.h
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

#ifndef LOGINDIALOG_H
#define LOGINDIALOG_H

#include <QDialog>
#include <QLineEdit>
#include <QPushButton>
#include <QDialogButtonBox>
#include <QMainWindow>
#include <QLabel>
#include <QCheckBox>

class Credential
{
public:
    char camera_name[1024];
    char username[128];
    char password[128];
    bool accept_requested;
};

class LoginDialog : public QDialog
{
    Q_OBJECT

public:
    LoginDialog(QWidget *parent);
    int exec() override;

    QWidget *cameraPanel;
    QLabel *cameraName;
    QDialogButtonBox *buttonBox;
    QLineEdit *username;
    QLineEdit *password;
    Credential credential;

};

#endif // LOGINDIALOG_H
