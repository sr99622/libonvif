/*******************************************************************************
* camerapanel.h
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

#ifndef CAMERAPANEL_H
#define CAMERAPANEL_H

#include "videotab.h"
#include "imagetab.h"
#include "networktab.h"
#include "ptztab.h"
#include "admintab.h"
#include "onvifmanager.h"
#include "camera.h"
#include "cameralistview.h"
#include "discovery.h"
#include "logindialog.h"

#include <QObject>
#include <QDialog>
#include <QTabWidget>
#include <QPushButton>
#include <QMainWindow>
#include <QSettings>

#define CP dynamic_cast<CameraPanel*>(cameraPanel)

class CameraPanel : public QWidget
{
    Q_OBJECT

public:
    CameraPanel(QMainWindow *parent);
    ~CameraPanel();
    void refreshList();
    void saveUsername();
    void savePassword();
    void saveAutoDiscovery();
    void saveMultiBroadcast();
    void saveNetIntf(const QString& name);
    void autoLoadClicked(bool checked);
    void autoCameraChanged(int index);
    void saveBroadcastRepeat(int value);

    Camera *camera;
    QTabWidget *tabWidget;
    QSlider* volumeSlider;
    QPushButton *applyButton;
    QPushButton *discoverButton;
    QPushButton *recordButton;
    VideoTab *videoTab;
    ImageTab *imageTab;
    NetworkTab *networkTab;
    PTZTab *ptzTab;
    AdminTab *adminTab;
    QMainWindow *mainWindow;
    Filler *filler;
    CameraListView *cameraList;
    Discovery *discovery;
    LoginDialog *loginDialog = nullptr;
    QSettings *cameraNames;
    OnvifSession *onvif_session;
    QPushButton *btnMute;

    const QString volumeKey     = "Application/volume";
    const QString muteKey       = "Application/mute";

    //QString MW->currentStreamingMediaName;
    bool connecting = false;
    std::string uri;
    char buf[256];

    QProcess process;

signals:
    void msg(QString str);

public slots:
    void fillData();
    void showData();
    void receiveOnvifData(OnvifData*);
    void showLoginDialog(Credential*);
    void applyButtonClicked();
    void discoverButtonClicked();
    void viewButtonClicked();
    void recordButtonClicked();
    void discoveryFinished();
    void adjustVolume(int);
    void streamStarting();
    void onBtnMuteClicked();
    void cameraTimeout();
    void connectFailed(const QString&);

};

#endif // CAMERAPANEL_H
