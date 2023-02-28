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
#include <algorithm>
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
    controlLayout->addWidget(btnMute,      0, 2, 1, 1);
    controlLayout->addWidget(sldVolume,    0, 3, 1, 1);
    controlLayout->addWidget(btnDiscover,  0, 4, 1, 1);
    controlLayout->addWidget(btnApply,     0, 5, 1 ,1);
    controlPanel->setMaximumHeight(60);

    cameraList = new QListWidget(this);
    connect(cameraList, SIGNAL(itemDoubleClicked(QListWidgetItem*)), this, SLOT(cameraListDoubleClicked(QListWidgetItem*)));
    connect(cameraList, SIGNAL(itemClicked(QListWidgetItem*)), this, SLOT(cameraListClicked(QListWidgetItem*)));

    QGridLayout *layout = new QGridLayout(this);
    layout->addWidget(cameraList,     0, 0, 1, 1);
    layout->addWidget(tabWidget,      1, 0, 1, 1);
    layout->addWidget(controlPanel,   2, 0, 1, 1);
    layout->setRowStretch(0, 10);

    videoTab->setActive(false);
    imageTab->setActive(false);
    networkTab->setActive(false);
    ptzTab->setActive(false);
    adminTab->setActive(false);
    btnApply->setEnabled(false);

    connect(MW, SIGNAL(updateUI()), this, SLOT(onUpdateUI()));
    connect(this, SIGNAL(msg(const QString&)), mainWindow, SLOT(msg(const QString&)));

    //cameraNames = new QSettings("Onvif", "Camera Names");
    //foreach(QString key, cameraNames->allKeys()) {
        //discovery->cameraAlias.insert(key, cameraNames->value(key).toString());
    //}

    loginDlg = new LoginDialog();
    connect(this, SIGNAL(showLogin()), this, SLOT(onShowLogin()));
    connect(this, SIGNAL(initTabs()), this, SLOT(onInitTabs()));
    connect(this, SIGNAL(enableDiscoverButton()), this, SLOT(onEnableDiscoverButton()));

    disableToolTips(MW->settingsPanel->hideToolTips->isChecked());

    if (MW->settingsPanel->autoDiscovery->isChecked())
        btnDiscoverClicked();
}

CameraPanel::~CameraPanel()
{

}

void CameraPanel::setPlayButton()
{
    if (MW->player) btnPlay->setStyleSheet(MW->getButtonStyle("stop"));
    else btnPlay->setStyleSheet(MW->getButtonStyle("play"));
}

void CameraPanel::setMuteButton(bool mute)
{
    if (mute) btnMute->setStyleSheet(MW->getButtonStyle("mute"));
    else btnMute->setStyleSheet(MW->getButtonStyle("audio"));
}

void CameraPanel::setRecordButton()
{
    if (MW->player) {
        if (MW->player->isPiping()) btnRecord->setStyleSheet(MW->getButtonStyle("recording"));
        else btnRecord->setStyleSheet(MW->getButtonStyle("record"));
    }
    else {
        btnRecord->setStyleSheet(MW->getButtonStyle("record"));
    }
}

void CameraPanel::cameraListDoubleClicked(QListWidgetItem* item)
{
    if (connecting) {
        std::cout << "currently attempting to connect to " << MW->currentMedia.toLatin1().data() << " please wait" << std::endl;
    }
    else {
        currentStreamingRow = cameraList->row(item);
        currentDataRow = currentStreamingRow;
        std::cout << "currentStreamingRow: " << currentStreamingRow << std::endl;
        std::cout << "devices size: " << devices.size() << std::endl;
        onvif::Data data = devices[currentStreamingRow];
        std::cout << data->camera_name << std::endl;
        MW->currentMedia = data->camera_name;
        std::stringstream ss_uri;
        std::string uri(data->stream_uri);
        ss_uri << uri.substr(0, 7) << data->username << ":" << data->password << "@" << uri.substr(7);
        uri = ss_uri.str();
        connecting = true;
        MW->playerStart(uri.c_str());
    }
}

void CameraPanel::btnPlayClicked()
{
    std::cout << "btnPlayClicked" << std::endl;
    /*
    if (MW->player) {
        MW->playerStop();
    }
    else {
        if (cameraList->getCurrentCamera()) {
            QString name = cameraList->getCurrentCamera()->getCameraName();
            cameraListDoubleClicked();
        }
    }
    */
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

/*
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
    //QThreadPool::globalInstance()->tryStart(filler);
}
*/

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
    if (currentDataRow > -1) ptzTab->setActive(hasPTZ(devices[currentDataRow]));
    btnApply->setEnabled(false);   
}

void CameraPanel::refreshList()
{
    //cameraList->refresh();
}

void CameraPanel::onUpdateUI()
{
    setPlayButton();
}

void CameraPanel::btnDiscoverClicked()
{
    std::cout << "start discover" << std::endl;
    onvif::Manager onvifBoss;
    onvifBoss.startDiscover([&]() { discoverFinished(); },
                            [&](onvif::Data& data) { return getCredential(data); },
                            [&](onvif::Data& data) { return getData(data); }
                            );
    btnDiscover->setEnabled(false);

}

void CameraPanel::getData(onvif::Data& onvif_data)
{
    std::cout << "get data" << std::endl;
    if (std::find(devices.begin(), devices.end(), onvif_data) == devices.end()) {
        onvif_data->logged_in = true;
        devices.push_back(onvif_data);
        cameraList->addItem(onvif_data->camera_name);
    }
}

void CameraPanel::onShowLogin()
{
    loginDlg->setStyleSheet(MW->style);
    if (!loginDlg->exec())
        loginDlg->cancelled = true;
    loginDlg->active = false;
}

bool CameraPanel::getCredential(onvif::Data& onvif_data)
{
    std::cout << "get credential" << std::endl;
    if (onvif_data->logged_in) return false;

    QString username = MW->settingsPanel->commonUsername->text();
    QString password = MW->settingsPanel->commonPassword->text();

    if (username.length() == 0 || last_data == onvif_data) 
    {
        loginDlg->active = true;
        loginDlg->cameraIP->setText(onvif_data->host);
        loginDlg->cameraName->setText(onvif_data->camera_name);
        emit showLogin();

        while (loginDlg->active)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));  

        if (loginDlg->cancelled) return false;

        username = loginDlg->username->text();
        password = loginDlg->password->text();
    }

    strncpy(onvif_data->username, username.toLatin1(), username.length());
    strncpy(onvif_data->password, password.toLatin1(), password.length());

    last_data = onvif_data;
    
    return true;
}

void CameraPanel::onEnableDiscoverButton()
{
    btnDiscover->setEnabled(true);
}

void CameraPanel::discoverFinished()
{
    emit enableDiscoverButton();
}

void CameraPanel::cameraListClicked(QListWidgetItem* item)
{
    std::cout << "cameraListClicked" << std::endl;
    currentDataRow = cameraList->row(item);
    if (!devices[currentDataRow]->filled) {
        onvif::Manager onvifBoss;
        onvifBoss.startFill(devices[currentDataRow], 
                        [&](const onvif::Data& onvif_data) { fillData(onvif_data); });

        cameraList->setEnabled(false);
        videoTab->setActive(false);
        imageTab->setActive(false);
        networkTab->setActive(false);
        ptzTab->setActive(false);
        adminTab->setActive(false);
    }
    else {
        showData();
    }
}

void CameraPanel::onInitTabs()
{
    showData();
    cameraList->setEnabled(true);
}

void CameraPanel::fillData(const onvif::Data& onvif_data)
{
    std::cout << "CameraPanel::fillData" << std::endl;
    devices[currentDataRow] = onvif_data;
    devices[currentDataRow]->filled = true;
    emit initTabs();
}
