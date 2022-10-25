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
    configTab = new ConfigTab(this);
    tabWidget->addTab(configTab, "Config");
    tabWidget->setMaximumHeight(220);

    applyButton = new QPushButton(tr("Apply"), this);
    connect(applyButton, SIGNAL(clicked()), this, SLOT(applyButtonClicked()));
    discoverButton = new QPushButton("Discover", this);
    connect(discoverButton, SIGNAL(clicked()), this, SLOT(discoverButtonClicked()));
    viewButton = new QPushButton("View", this);
    connect(viewButton, SIGNAL(clicked()), this, SLOT(viewButtonClicked()));

    QDialogButtonBox *buttonBox = new QDialogButtonBox(Qt::Horizontal, this);
    buttonBox->addButton(discoverButton, QDialogButtonBox::ActionRole);
    buttonBox->addButton(viewButton, QDialogButtonBox::ActionRole);
    buttonBox->addButton(applyButton, QDialogButtonBox::ActionRole);
    buttonBox->setMaximumHeight(60);

    cameraList = new CameraListView(mainWindow);

    QGridLayout *layout = new QGridLayout();
    layout->addWidget(cameraList,   0, 0, 1, 1);
    layout->addWidget(tabWidget,    1, 0, 1, 1);
    layout->addWidget(buttonBox,    2, 0, 1, 1);
    layout->setColumnStretch(0, 10);
    setLayout(layout);

    filler = new Filler(this);
    connect(filler, SIGNAL(done()), this, SLOT(showData()));

    videoTab->setActive(false);
    imageTab->setActive(false);
    networkTab->setActive(false);
    ptzTab->setActive(false);
    adminTab->setActive(false);
    applyButton->setEnabled(false);
    viewButton->setEnabled(false);

    connect(this, SIGNAL(msg(QString)), mainWindow, SLOT(msg(QString)));

    CameraListModel *cameraListModel = cameraList->cameraListModel;
    connect(cameraListModel, SIGNAL(showCameraData()), this, SLOT(showData()));
    connect(cameraListModel, SIGNAL(getCameraData()), this, SLOT(fillData()));

    configTab->commonUsername->setText(MW->settings->value(usernameKey, "").toString());
    configTab->commonPassword->setText(MW->settings->value(passwordKey, "").toString());
    configTab->player->setText(MW->settings->value(playerKey, "ffplay").toString());
    configTab->autoDiscovery->setChecked(MW->settings->value(autoDiscKey, false).toBool());
    configTab->multiBroadcast->setChecked(MW->settings->value(multiBroadKey, false).toBool());
    configTab->broadcastRepeat->setValue(MW->settings->value(broadRepKey, 2).toInt());
    configTab->autoDiscoveryClicked(configTab->autoDiscovery->isChecked());

    savedAutoCameraName = MW->settings->value(autoCameraKey, "").toString();
    onvif_session = (OnvifSession*)malloc(sizeof(OnvifSession));
    initializeSession(onvif_session);
    discovery = new Discovery(this);
    connect(discovery, SIGNAL(stopping()), this, SLOT(discoveryFinished()));
    cameraNames = new QSettings("Onvif", "Camera Names");
    foreach(QString key, cameraNames->allKeys()) {
        discovery->cameraAlias.insert(key, cameraNames->value(key).toString());
    }

    if (configTab->autoDiscovery->isChecked()) {
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

void CameraPanel::viewButtonClicked()
{
    std::stringstream ss_uri;
    OnvifData* onvif_data = cameraList->getCurrentCamera()->onvif_data;
	std::string uri(onvif_data->stream_uri);
	ss_uri << uri.substr(0, 7) << onvif_data->username << ":" << onvif_data->password << "@" << uri.substr(7);
    uri = ss_uri.str();

    std::string player(configTab->player->text().toLatin1().data());
    
    std::stringstream ss;
#ifdef _WIN32
	ss << "start " << player << " \"" << uri << "\"";
#else
	ss << player << " \"" << uri << "\"";
#endif				
	std::system(ss.str().c_str());
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
    viewButton->setEnabled(false);
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
    viewButton->setEnabled(true);
}

void CameraPanel::saveUsername()
{
    MW->settings->setValue(usernameKey, configTab->commonUsername->text());
}

void CameraPanel::savePassword()
{
    MW->settings->setValue(passwordKey, configTab->commonPassword->text());
}

void CameraPanel::savePlayer()
{
    MW->settings->setValue(playerKey, configTab->player->text());
}

void CameraPanel::saveAutoDiscovery()
{
    MW->settings->setValue(autoDiscKey, configTab->autoDiscovery->isChecked());
}

void CameraPanel::saveMultiBroadcast()
{
    MW->settings->setValue(multiBroadKey, configTab->multiBroadcast->isChecked());
}

void CameraPanel::saveBroadcastRepeat(int value)
{
    MW->settings->setValue(broadRepKey, value);
}

void CameraPanel::saveNetIntf(const QString& name)
{
    MW->settings->setValue(netIntfKey, name);
}

void CameraPanel::autoLoadClicked(bool checked)
{
    MW->settings->setValue(autoLoadKey, checked);
}

void CameraPanel::discoveryFinished()
{
    emit msg("discovery is completed");
}

void CameraPanel::refreshList()
{
    cameraList->refresh();
}

