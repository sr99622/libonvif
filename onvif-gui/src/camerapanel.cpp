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
#include <QDialogButtonBox>
#include <QGuiApplication>
#include <QMessageBox>
#include <QScreen>

CameraPanel::CameraPanel(QMainWindow *parent)
{
    mainWindow = parent;

    tabWidget = new QTabWidget();
    videoTab = new VideoTab(this);
    tabWidget->addTab(videoTab, "Video");
    imageTab = new ImageTab(this);
    tabWidget->addTab(imageTab, "Image");
    networkTab = new NetworkTab(this);
    tabWidget->addTab(networkTab, "Network");
    ptzTab = new PTZTab(this);
    tabWidget->addTab(ptzTab, "PTZ");
    adminTab = new AdminTab(this);
    tabWidget->addTab(adminTab, "Admin");
    QList<QScreen*> screens = QGuiApplication::screens();
    QSize screenSize = screens[0]->size();

    tabWidget->setMaximumHeight(screenSize.height() * 0.2);

    applyButton = new QPushButton(tr("Apply"), this);
    connect(applyButton, SIGNAL(clicked()), this, SLOT(applyButtonClicked()));
    discoverButton = new QPushButton("Discover", this);
    connect(discoverButton, SIGNAL(clicked()), this, SLOT(discoverButtonClicked()));

    volumeSlider = new QSlider(Qt::Horizontal, this);
    volumeSlider->setMaximumHeight(16);
    int volume = MW->settings->value(volumeKey, 100).toInt();
    volumeSlider->setValue(volume);
    MW->glWidget->setVolume(volume);
    connect(volumeSlider, SIGNAL(valueChanged(int)), this, SLOT(adjustVolume(int)));

    btnMute = new QPushButton();
    MW->glWidget->setMute(MW->settings->value(muteKey, false).toBool());
    if (MW->glWidget->getMute())
        btnMute->setStyleSheet(MW->getButtonStyle("mute"));
    else 
        btnMute->setStyleSheet(MW->getButtonStyle("audio"));
    connect(btnMute, SIGNAL(clicked()), this, SLOT(onBtnMuteClicked()));

    QWidget *controlPanel = new QWidget(this);
    QGridLayout* controlLayout = new QGridLayout(controlPanel);
    controlLayout->addWidget(btnMute,         0, 0, 1, 1);
    controlLayout->addWidget(volumeSlider,    0, 1, 1, 1);
    controlLayout->addWidget(discoverButton,  0, 2, 1, 1);
    controlLayout->addWidget(applyButton,     0, 3, 1 ,1);
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
    applyButton->setEnabled(false);

    connect(this, SIGNAL(msg(const QString&)), mainWindow, SLOT(msg(const QString&)));
    connect(MW->glWidget, SIGNAL(timerStart()), this, SLOT(streamStarting()));
    connect(MW->glWidget, SIGNAL(cameraTimeout()), this, SLOT(cameraTimeout()));
    connect(MW->glWidget, SIGNAL(connectFailed(const QString&)), this, SLOT(connectFailed(const QString&)));

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

void CameraPanel::discoverButtonClicked()
{
    discovery->start();
}

void CameraPanel::onBtnMuteClicked()
{
    if (MW->glWidget->getMute()) {
        btnMute->setStyleSheet(MW->getButtonStyle("audio"));
        MW->filePanel->btnMute->setStyleSheet(MW->getButtonStyle("audio"));
    }
    else {
        btnMute->setStyleSheet(MW->getButtonStyle("mute"));
        MW->filePanel->btnMute->setStyleSheet(MW->getButtonStyle("mute"));
    }

    MW->glWidget->setMute(!MW->glWidget->getMute());
    MW->settings->setValue(muteKey, MW->glWidget->getMute());
}

void CameraPanel::viewButtonClicked()
{
    if (connecting) {
        std::cout << "currently attempting to connect to " << MW->currentStreamingMediaName.toLatin1().data() << " please wait" << std::endl;
    }
    else {
        MW->currentStreamingMediaName = cameraList->getCurrentCamera()->getCameraName();
        std::cout << "attempting to connnect to " << MW->currentStreamingMediaName.toLatin1().data() << std::endl;
        std::stringstream ss_uri;
        OnvifData* onvif_data = cameraList->getCurrentCamera()->onvif_data;
        std::string uri(onvif_data->stream_uri);
        ss_uri << uri.substr(0, 7) << onvif_data->username << ":" << onvif_data->password << "@" << uri.substr(7);
        uri = ss_uri.str();
        connecting = true;
        if (MW->settingsPanel->lowLatency->isChecked()) {
            MW->glWidget->vpq_size = 1;
            MW->glWidget->apq_size = 1;
        }
        else {
            MW->glWidget->vpq_size = 100;
            MW->glWidget->apq_size = 100;
        }
        MW->glWidget->play(uri.c_str());
        MW->setWindowTitle("connecting to " + MW->currentStreamingMediaName);
    }
}

void CameraPanel::showLoginDialog(Credential *credential)
{
    if (loginDialog == nullptr)
        loginDialog = new LoginDialog(this);

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

void CameraPanel::applyButtonClicked()
{
    CameraDialogTab *tab = (CameraDialogTab *)tabWidget->currentWidget();
    tab->update();
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
    applyButton->setEnabled(false);
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
    applyButton->setEnabled(false);   
}

void CameraPanel::discoveryFinished()
{
    emit msg("discovery is completed");
}

void CameraPanel::refreshList()
{
    cameraList->refresh();
}

void CameraPanel::adjustVolume(int value)
{
    MW->glWidget->setVolume(value);
    MW->settings->setValue(volumeKey, value);
    MW->filePanel->sldVolume->setValue(value);
}

void CameraPanel::streamStarting()
{
    connecting = false;
    if (MW->glWidget->process) {
        MW->glWidget->process->display->volume = (float)volumeSlider->value() / 100.0f;
    }
    MW->setWindowTitle("Streaming from " + MW->currentStreamingMediaName);
}

void CameraPanel::cameraTimeout()
{
    QMessageBox msgBox;
    msgBox.setText("Camera has timed out");
    msgBox.exec();
}

void CameraPanel::connectFailed(const QString& str)
{
    connecting = false;
    QString title = "connection failed - ";
    title += MW->currentStreamingMediaName;
    MW->setWindowTitle(title);
    QMessageBox msgBox(this);
    msgBox.setText(str);
    msgBox.exec();
}