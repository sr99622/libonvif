/*******************************************************************************
* camerapanel.cpp
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

#include <sstream>
#include "camerapanel.h"
#include "mainwindow.h"
#include <QGridLayout>
#include <QThreadPool>
#include <QMessageBox>
#include <QDateTime>

CameraPanel::CameraPanel(QMainWindow *parent)
{
    mainWindow = parent;

    videoTab = new VideoTab(this);
    imageTab = new ImageTab(this);
    networkTab = new NetworkTab(this);
    ptzTab = new PTZTab(this);
    adminTab = new AdminTab(this);

    tabWidget = new QTabWidget();
    tabWidget->addTab(videoTab, "Video");
    tabWidget->addTab(imageTab, "Image");
    tabWidget->addTab(networkTab, "Network");
    tabWidget->addTab(ptzTab, "PTZ");
    tabWidget->addTab(adminTab, "Admin");

    btnApply = new QPushButton(this);
    btnApply->setStyleSheet(MW->getButtonStyle("apply"));
    connect(btnApply, SIGNAL(clicked()), this, SLOT(btnApplyClicked()));

    btnDiscover = new QPushButton(this);
    btnDiscover->setStyleSheet(MW->getButtonStyle("discover"));
    connect(btnDiscover, SIGNAL(clicked()), this, SLOT(btnDiscoverClicked()));

    btnPlay = new QPushButton(this);
    btnPlay->setStyleSheet(MW->getButtonStyle("play"));
    connect(btnPlay, SIGNAL(clicked()), this, SLOT(btnPlayClicked()));

    btnRecord = new QPushButton(this);
    btnRecord->setStyleSheet(MW->getButtonStyle("record"));
    connect(btnRecord, SIGNAL(clicked()), this, SLOT(btnRecordClicked()));

    sldVolume = new QSlider(Qt::Horizontal, this);
    connect(sldVolume, SIGNAL(valueChanged(int)), MW, SLOT(setPlayerVolume(int)));

    btnMute = new QPushButton();
    connect(btnMute, SIGNAL(clicked()), MW, SLOT(togglePlayerMute()));

    QWidget *controlPanel = new QWidget(this);
    QGridLayout* controlLayout = new QGridLayout(controlPanel);
    controlLayout->addWidget(btnPlay,      0, 0, 1, 1);
    controlLayout->addWidget(btnRecord,    0, 1, 1, 1);
    controlLayout->addWidget(btnMute,         0, 2, 1, 1);
    controlLayout->addWidget(sldVolume,    0, 3, 1, 1);
    controlLayout->addWidget(btnDiscover,  0, 4, 1, 1);
    controlLayout->addWidget(btnApply,     0, 5, 1 ,1);
    controlPanel->setMaximumHeight(60);

    cameraList = new CameraListView(mainWindow);

    QGridLayout *layout = new QGridLayout();

    layout->addWidget(cameraList,     0, 0, 1, 1);
    layout->addWidget(tabWidget,      1, 0, 1, 1);
    layout->addWidget(controlPanel,   2, 0, 1, 1);
    layout->setRowStretch(0, 10);

    setLayout(layout);

    filler = new Filler(this);
    connect(filler, SIGNAL(done()), this, SLOT(showData()));

    videoTab->setActive(false);
    imageTab->setActive(false);
    networkTab->setActive(false);
    ptzTab->setActive(false);
    adminTab->setActive(false);
    btnApply->setEnabled(false);

    connect(MW, SIGNAL(updateUI()), this, SLOT(onUpdateUI()));
    connect(this, SIGNAL(msg(const QString&)), mainWindow, SLOT(msg(const QString&)));

    CameraListModel *cameraListModel = cameraList->cameraListModel;
    connect(cameraListModel, SIGNAL(showCameraData()), this, SLOT(showData()));
    connect(cameraListModel, SIGNAL(getCameraData()), this, SLOT(fillData()));

    onvif_session = (OnvifSession*)calloc(sizeof(OnvifSession), 1);
    initializeSession(onvif_session);
    discovery = new Discovery(this, MW->settingsPanel);
    connect(discovery, SIGNAL(stopping()), this, SLOT(discoveryFinished()));
    cameraNames = new QSettings("Onvif", "Camera Names");
    foreach(QString key, cameraNames->allKeys()) {
        discovery->cameraAlias.insert(key, cameraNames->value(key).toString());
    }

    disableToolTips(MW->settingsPanel->hideToolTips->isChecked());

    if (MW->settingsPanel->autoDiscovery->isChecked()) {
        discovery->start();
    }
}

CameraPanel::~CameraPanel()
{
    closeSession(onvif_session);
    free(onvif_session);
}

void CameraPanel::receiveOnvifData(OnvifData *onvif_data)
{
    cameraList->cameraListModel->pushCamera(onvif_data);
}

void CameraPanel::btnDiscoverClicked()
{
    discovery->start();
}

void CameraPanel::setPlayButton()
{
    if (MW->player) 
        btnPlay->setStyleSheet(MW->getButtonStyle("stop"));
    else
        btnPlay->setStyleSheet(MW->getButtonStyle("play"));
}

void CameraPanel::setMuteButton(bool mute)
{
    if (mute)
        btnMute->setStyleSheet(MW->getButtonStyle("mute"));
    else
        btnMute->setStyleSheet(MW->getButtonStyle("audio"));
}

void CameraPanel::setRecordButton()
{
    if (MW->player) {
        if (MW->player->isPiping()) {
            btnRecord->setStyleSheet(MW->getButtonStyle("recording"));
        }
        else {
            btnRecord->setStyleSheet(MW->getButtonStyle("record"));
        }
    }
    else {
        btnRecord->setStyleSheet(MW->getButtonStyle("record"));
    }
}

void CameraPanel::cameraListDoubleClicked()
{
    if (connecting) {
        std::cout << "currently attempting to connect to " << MW->currentMedia.toLatin1().data() << " please wait" << std::endl;
    }
    else {
        MW->currentMedia = cameraList->getCurrentCamera()->getCameraName();
        std::cout << "attempting to connect to " << MW->currentMedia.toLatin1().data() << std::endl;
        std::stringstream ss_uri;
        OnvifData* onvif_data = cameraList->getCurrentCamera()->onvif_data;
        std::string uri(onvif_data->stream_uri);
        ss_uri << uri.substr(0, 7) << onvif_data->username << ":" << onvif_data->password << "@" << uri.substr(7);
        uri = ss_uri.str();
        connecting = true;
        MW->playerStart(uri.c_str());
    }
}

void CameraPanel::btnPlayClicked()
{
    std::cout << "btnPlayClicked" << std::endl;
    if (MW->player) {
        MW->playerStop();
    }
    else {
        if (cameraList->getCurrentCamera()) {
            QString name = cameraList->getCurrentCamera()->getCameraName();
            cameraListDoubleClicked();
        }
    }
}

void CameraPanel::showLoginDialog(Credential *credential)
{
    if (loginDialog == nullptr)
        loginDialog = new LoginDialog(this);

    loginDialog->setStyleSheet(MW->style);

    QString host = QString(credential->host_name);
    int start = host.indexOf("//") + 2;
    int stop = host.indexOf("/", start);
    int len = stop - start;
    QString ip = host.mid(start, len);

    loginDialog->cameraIP->setText(QString("Camera IP: ").append(ip));
    loginDialog->cameraName->setText(QString("Camera Name: ").append(credential->camera_name));
    
    if (loginDialog->exec()) {
        QString username = loginDialog->username->text();
        strncpy(credential->username, username.toLatin1(), username.length());
        QString password = loginDialog->password->text();
        strncpy(credential->password, password.toLatin1(), password.length());
        credential->accept_requested = true;
    }
    else {
        emit msg("login cancelled");
        memset(credential->username, 0, 128);
        memset(credential->password, 0, 128);
        credential->accept_requested = false;
    }
    discovery->resume();
}

void CameraPanel::btnApplyClicked()
{
    CameraDialogTab *tab = (CameraDialogTab *)tabWidget->currentWidget();
    tab->update();
}

void CameraPanel::btnRecordClicked()
{
    std::cout << "record button clicked - use environment variable QT_FILESYSTEMMODEL_WATCH_FILES for file size updates" << std::endl;
    recording = !recording;
    QString filename = MW->filePanel->directorySetter->directory;
    if (MW->settingsPanel->generateFilename->isChecked()) 
        filename.append("/").append(QDateTime::currentDateTime().toString("yyyyMMddhhmmss")).append(".mp4");
    else 
        filename.append("/").append("out.mp4");

    if (MW->player) {
        MW->player->togglePiping(filename.toLatin1().data());
    }
    setRecordButton();
}

void CameraPanel::disableToolTips(bool arg)
{
    if (!arg)
    {
        btnApply->setToolTip("Apply");
        btnDiscover->setToolTip("Discover");
        btnPlay->setToolTip("Play");
        btnRecord->setToolTip("Record");
        btnMute->setToolTip("Mute");
    }
    else {
        btnApply->setToolTip("");
        btnDiscover->setToolTip("");
        btnPlay->setToolTip("");
        btnRecord->setToolTip("");
        btnMute->setToolTip("");
    }
}

void CameraPanel::fillData()
{
    videoTab->clear();
    imageTab->clear();
    networkTab->clear();
    adminTab->clear();
    videoTab->setActive(false);
    imageTab->setActive(false);
    networkTab->setActive(false);
    adminTab->setActive(false);
    btnApply->setEnabled(false);
    QThreadPool::globalInstance()->tryStart(filler);
}

void CameraPanel::showData()
{
    videoTab->initialize();
    imageTab->initialize();
    networkTab->initialize();
    adminTab->initialize();

    videoTab->setActive(true);
    imageTab->setActive(true);
    networkTab->setActive(true);
    adminTab->setActive(true);
    ptzTab->setActive(camera->hasPTZ());
    camera->onvif_data_read = true;
    btnApply->setEnabled(false);   
}

void CameraPanel::discoveryFinished()
{
    emit msg("discovery is completed");
}

void CameraPanel::refreshList()
{
    cameraList->refresh();
}

void CameraPanel::onUpdateUI()
{
    setPlayButton();
}
