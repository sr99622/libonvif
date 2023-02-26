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
#include "logindialog.h"

#include <QObject>
#include <QDialog>
#include <QTabWidget>
#include <QPushButton>
#include <QMainWindow>
#include <QSettings>
#include <QListWidget>
#include <QListWidgetItem>

#include "avio.h"
#include "onvifboss.h"


#define CP dynamic_cast<CameraPanel*>(cameraPanel)

class CameraPanel : public QWidget
{
    Q_OBJECT

public:
    CameraPanel(QMainWindow *parent);
    ~CameraPanel();
    void refreshList();
    void setMuteButton(bool);
    void setPlayButton();
    void setRecordButton();

    QTabWidget *tabWidget;
    QSlider* sldVolume;
    QPushButton *btnApply;
    QPushButton *btnDiscover;
    QPushButton *btnRecord;
    QPushButton *btnPlay;
    QPushButton *btnMute;
    VideoTab *videoTab;
    ImageTab *imageTab;
    NetworkTab *networkTab;
    PTZTab *ptzTab;
    AdminTab *adminTab;
    QMainWindow *mainWindow;
    QListWidget *cameraList;
    QSettings *cameraNames;

    bool connecting = false;
    bool recording = false;
    std::string uri;
    char buf[256];

    std::vector<onvif::Data> devices;
    LoginDialog* loginDlg = nullptr;
    onvif::Data last_data;

    void discoverFinished();
    bool getCredential(onvif::Data&);
    void getData(onvif::Data&);
    void fillData(onvif::Data&, int);
    int currentStreamingRow = -1;
    int currentDataRow = -1;

signals:
    void msg(QString str);
    void showError(const QString&);
    void showLogin();
    void initTabs();

public slots:
    void fillData();
    void showData();
    void btnApplyClicked();
    void btnDiscoverClicked();
    void btnPlayClicked();
    void btnRecordClicked();
    void cameraListDoubleClicked(QListWidgetItem*);
    void cameraListClicked(QListWidgetItem*);
    void disableToolTips(bool);
    void onUpdateUI();
    void onShowLogin();
    void onInitTabs();

};

#endif // CAMERAPANEL_H
