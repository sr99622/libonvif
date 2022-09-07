/*******************************************************************************
* discovery.h
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

#ifndef DISCOVERY_H
#define DISCOVERY_H

#include "onvif.h"
#include "logindialog.h"
#include <QObject>
#include <QMutex>
#include <QThread>
#include <QHash>
#include <QMainWindow>
#include <QWaitCondition>

class Discovery : public QObject
{
    Q_OBJECT

public:
    Discovery(QWidget *parent);
    ~Discovery();

    void start();
    void stop();
    void resume();
    void discover();
    bool isRunning();
    bool alreadyLoggedIn(OnvifData *onvif_data);
    void addCamera(OnvifData *onvif_data);

    QWidget *cameraPanel;
    char *username;
    char *password;
    QHash<QString, QString> cameraAlias;
    LoginDialog *loginDialog;
    Credential credential;

private:
    bool running;
    QThread *thread;
    QMutex mutex;
    QWaitCondition waitCondition;

signals:
    void starting();
    void stopping();
    void found(OnvifData *onvif_data);
    void login(Credential*);
    void msg(QString);

public slots:
    void run();

};

#endif // DISCOVERY_H
