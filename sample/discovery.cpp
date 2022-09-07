/*******************************************************************************
* discovery.cpp
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

#include "discovery.h"
#include "camerapanel.h"

Discovery::Discovery(QWidget *parent)
{
    cameraPanel = parent;

    thread = new QThread;
    moveToThread(thread);

    connect(this, SIGNAL(starting()), thread, SLOT(start()));
    connect(this, SIGNAL(stopping()), thread, SLOT(quit()));
    connect(thread, SIGNAL(started()), this, SLOT(run()));

    connect(this, SIGNAL(msg(QString)), CP->mainWindow, SLOT(msg(QString)));

    running = false;

    loginDialog = new LoginDialog(cameraPanel);
    connect(this, SIGNAL(login(Credential*)), cameraPanel, SLOT(showLoginDialog(Credential*)));
    connect(this, SIGNAL(found(OnvifData*)), cameraPanel, SLOT(receiveOnvifData(OnvifData*)));
}

Discovery::~Discovery()
{
    if (running)
        stop();
    thread->wait();
}

void Discovery::start()
{
    mutex.lock();
    running = true;
    mutex.unlock();
    emit starting();
}

void Discovery::stop()
{
    mutex.lock();
    running = false;
    mutex.unlock();
    emit stopping();
}

void Discovery::resume()
{
    waitCondition.wakeAll();
}

bool Discovery::isRunning()
{
    return running;
}

void Discovery::run()
{
    discover();
}

void Discovery::discover()
{
    int nb_loops = 1;
    if (CP->configTab->multiBroadcast->isChecked())
        nb_loops = CP->configTab->broadcastRepeat->value();

    for (int k=0; k<nb_loops; k++) {
        OnvifSession *onvif_session = (OnvifSession*)malloc(sizeof(OnvifSession));
        ConfigTab *configTab = CP->configTab;

        QString str = "Discovery started\n";
        onvif_session->discovery_msg_id = k;

        initializeSession(onvif_session);
        int number_of_cameras = broadcast(onvif_session);
        str.append(QString("libonvif found %1 cameras\n").arg(QString::number(number_of_cameras)));
        emit msg(str);

        for (int i=0; i<number_of_cameras; i++) {
            if (running) {
                OnvifData *onvif_data = (OnvifData*)malloc(sizeof(OnvifData));
                memset(onvif_data, 0, sizeof(OnvifData));
                prepareOnvifData(i, onvif_session, onvif_data);
                emit msg(QString("Connecting to camera %1 at %2").arg(onvif_data->camera_name, onvif_data->xaddrs));
                QString username = configTab->commonUsername->text();
                QString password = configTab->commonPassword->text();
                strncpy(onvif_data->username, username.toLatin1(), username.length());
                strncpy(onvif_data->password, password.toLatin1(), password.length());

                bool loggedIn = alreadyLoggedIn(onvif_data);
                if (loggedIn) {
                    emit msg(QString("Duplicate discovery packet for camera %1\n").arg(onvif_data->camera_name));
                }

                while (!loggedIn) {
                    if (fillRTSP(onvif_data) == 0) {
                        loggedIn = true;
                        addCamera(onvif_data);
                    }
                    else {
                        QString error_msg = onvif_data->last_error;
                        if (error_msg.contains("ter:NotAuthorized") || error_msg.contains("Unauthorized")) {
                            memset(&credential, 0, sizeof(credential));
                            strncpy(credential.camera_name, onvif_data->camera_name, strlen(onvif_data->camera_name));
                            emit login(&credential);

                            emit msg("starting login");
                            mutex.lock();
                            waitCondition.wait(&mutex);
                            mutex.unlock();

                            if (credential.accept_requested) {
                                strncpy(onvif_data->username, credential.username, strlen(credential.username));
                                strncpy(onvif_data->password, credential.password, strlen(credential.password));

                                if (fillRTSP(onvif_data) == 0) {
                                    loggedIn = true;
                                    addCamera(onvif_data);
                                }
                                else {
                                    emit msg(QString("Login failure for camera %1\n").arg(onvif_data->camera_name));
                                }
                            }
                            else {
                                emit msg(QString("Login cancelled for camera %1\n").arg(onvif_data->camera_name));
                                break;
                            }
                        }
                        else {
                            emit msg(QString("ONVIF error %1\n").arg(onvif_data->last_error));
                            break;
                        }
                    }
                }
            }
            else {
                break;
            }
        }

        closeSession(onvif_session);
        free(onvif_session);
        thread->msleep(200);
    }
    stop();
}

void Discovery::addCamera(OnvifData *onvif_data)
{
    getProfile(onvif_data);
    getDeviceInformation(onvif_data);

    QString str;
    str.append(QString("%1\n").arg(onvif_data->stream_uri));
    str.append(QString("serial number: %1\nmfgr name: %2\n").arg(onvif_data->serial_number, onvif_data->camera_name));

    QString key = onvif_data->serial_number;
    QString alias = cameraAlias.value(key);
    if (alias.length() > 0)
        strncpy(onvif_data->camera_name, alias.toLatin1().data(), alias.length());

    emit found(onvif_data);

    str.append(QString("display name: %1\n").arg(onvif_data->camera_name));
    emit msg(str);
}

bool Discovery::alreadyLoggedIn(OnvifData *onvif_data)
{
    bool result = false;

    QVector<Camera *> cameras = CP->cameraList->cameraListModel->cameras;
    for (int i = 0; i < cameras.size(); i++) {
        Camera *camera = cameras[i];
        QString currentXaddrs = onvif_data->xaddrs;
        QString cameraXaddrs = camera->onvif_data->xaddrs;
        if (currentXaddrs == cameraXaddrs)
            result = true;
    }

    return result;
}
