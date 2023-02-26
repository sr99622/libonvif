/*******************************************************************************
* settingspanel.cpp
*
* Copyright (c) 2022 Stephen Rhodes
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

#ifdef _WIN32
    #include <WS2tcpip.h>
    #include <iphlpapi.h>
    #pragma comment(lib, "iphlpapi.lib")
#else
    #include <netdb.h>
    #include <ifaddrs.h>
#endif

#include <cmath>
#include <QLabel>
#include <QGridLayout>
#include <QGroupBox>
#include <QMessageBox>
#include <QListWidget>
#include <QFileDialog>
#include <QFileInfo>

#include "settingspanel.h"
#include "mainwindow.h"

SettingsPanel::SettingsPanel(QMainWindow* parent)
{
    mainWindow = parent;
    connect(this, SIGNAL(msg(const QString&)), MW, SLOT(msg(const QString&)));

    networkInterfaces = new QComboBox();
    networkInterfaces->setMaximumWidth(300);
    interfaces = new QListWidget(this);
    QLabel *lbl03 = new QLabel("Network Interface");
    getActiveNetworkInterfaces();
    QString netIntf = MW->settings->value(netIntfKey, "").toString();
    if (netIntf.length() > 0)
        networkInterfaces->setCurrentText(netIntf);
    connect(networkInterfaces, SIGNAL(currentTextChanged(const QString&)), this, SLOT(netIntfChanged(const QString&)));

    autoDiscovery = new QCheckBox("Auto Discovery");
    connect(autoDiscovery, SIGNAL(clicked(bool)), this, SLOT(autoDiscoveryClicked(bool)));
    autoDiscovery->setChecked(MW->settings->value(autoDiscKey, false).toBool());

    QLabel *lbl01 = new QLabel("Common Username");
    commonUsername = new QLineEdit(this);
    commonUsername->setText(MW->settings->value(usernameKey, "").toString());
    connect(commonUsername, SIGNAL(editingFinished()), this, SLOT(usernameUpdated()));
    
    QLabel *lbl02 = new QLabel("Common Password");
    commonPassword = new QLineEdit(this);
    commonPassword->setText(MW->settings->value(passwordKey, "").toString());
    connect(commonPassword, SIGNAL(editingFinished()), this, SLOT(passwordUpdated()));

    lowLatency = new QCheckBox("Low Latency Buffering");
    connect(lowLatency, SIGNAL(clicked()), this, SLOT(lowLatencyClicked()));
    lowLatency->setChecked(MW->settings->value(lowLatencyKey, false).toBool());
    lowLatencyClicked();

    disableAudio = new QCheckBox("Disable Audio");
    disableAudio->setChecked(MW->settings->value(disAudioKey, false).toBool());
    connect(disableAudio, SIGNAL(clicked(bool)), this, SLOT(disableAudioClicked(bool)));

    hideToolTips = new QCheckBox("Hide Tool Tips");
    hideToolTips->setChecked(MW->settings->value(hideTipsKey, false).toBool());
    connect(hideToolTips, SIGNAL(clicked()), this, SLOT(hideToolTipsClicked()));

    keyframeCount = new QSpinBox(this);
    keyframeCount->setRange(1, 100);
    keyframeCount->setMaximumWidth(140);
    lblKeyframeCount = new QLabel("Write Cache Size");
    keyframeCount->setValue(MW->settings->value(keyCountKey, 1).toInt());
    connect(keyframeCount, SIGNAL(valueChanged(int)), this, SLOT(keyframeCountChanged(int)));
    
    QStringList decoders = {
        "NONE",
        "CUDA",
        "VAAPI",
        "VDPAU",
        "DXVA2",
        "D3D11VA",
        "QSV",
        "DRM",
        "OPENCL",
    };

    listDecoders = new QListWidget(this);
    listDecoders->addItems(decoders);

    lblDecoders = new QLabel("Hardware Decoder");
    hardwareDecoders = new QComboBox(this);
    hardwareDecoders->setModel(listDecoders->model());
    hardwareDecoders->setView(listDecoders);
    connect(hardwareDecoders, SIGNAL(currentTextChanged(const QString&)), this, SLOT(decoderChanged(const QString&)));
    hardwareDecoders->setCurrentText(MW->settings->value(decoderKey, "NONE").toString());

    generateFilename = new QRadioButton("Generate Unique Filename");
    generateFilename->setChecked(MW->settings->value(genFileKey, true).toBool());
    connect(generateFilename, SIGNAL(clicked(bool)), this, SLOT(generateFilenameClicked(bool)));

    defaultFilename = new QRadioButton("Use Default Filename");
    defaultFilename->setChecked(MW->settings->value(defFileKey, false).toBool());
    connect(defaultFilename, SIGNAL(clicked(bool)), this, SLOT(defaultFilenameClicked(bool)));

    style = new QPushButton("Style Colors");
    connect(style, SIGNAL(clicked()), this, SLOT(styleClicked()));

    clear = new QPushButton("Clear Settings");
    connect(clear, SIGNAL(clicked()), this, SLOT(clearClicked()));

    test = new QPushButton("Test");
    connect(test, SIGNAL(clicked()), this, SLOT(testClicked()));

    QGroupBox *groupBox = new QGroupBox("Record Filename", this);
    QGridLayout *groupLayout = new QGridLayout(groupBox);
    groupLayout->addWidget(generateFilename,  0, 0, 1, 1);
    groupLayout->addWidget(defaultFilename,   1, 0, 1, 1);

    QGridLayout *layout = new QGridLayout(this);
    layout->addWidget(lbl03,               0, 0, 1, 1);
    layout->addWidget(networkInterfaces,   0, 1, 1, 1);
    layout->addWidget(autoDiscovery,       1, 0, 1, 2);
    layout->addWidget(lbl01,               3, 0, 1, 1);
    layout->addWidget(commonUsername,      3, 1, 1, 1);
    layout->addWidget(lbl02,               4, 0, 1, 1);
    layout->addWidget(commonPassword,      4, 1, 1, 1);
    layout->addWidget(lowLatency,          5, 0, 1, 3);
    layout->addWidget(disableAudio,        6, 0, 1, 3);
    layout->addWidget(hideToolTips,        7, 0, 1, 3);
    layout->addWidget(lblKeyframeCount,    8, 0, 1, 1);
    layout->addWidget(keyframeCount,       8, 1, 1, 1);
    layout->addWidget(lblDecoders,         9, 0, 1, 1);
    layout->addWidget(hardwareDecoders,    9, 1, 1, 2);
    layout->addWidget(groupBox,           10, 0, 1, 3);
    layout->addWidget(clear,              11, 0, 1, 1, Qt::AlignCenter);
    layout->addWidget(style,              11, 1, 1, 1, Qt::AlignCenter);
    layout->addWidget(test,               12, 0, 1, 1, Qt::AlignCenter);
    setLayout(layout);

    autoDiscoveryClicked(autoDiscovery->isChecked());
    /*
    MW->glWidget->keyframe_cache_size = keyframeCount->value();
    */

    //loginDlg = new LoginDialog();
    //connect(this, SIGNAL(showLogin()), this, SLOT(onShowLogin()));
    connect(this, SIGNAL(initTabs()), this, SLOT(onInitTabs()));
}

void SettingsPanel::autoDiscoveryClicked(bool checked)
{
    MW->settings->setValue(autoDiscKey, autoDiscovery->isChecked());
}

void SettingsPanel::keyframeCountChanged(int value)
{
    /*
    MW->glWidget->keyframe_cache_size = value;
    MW->settings->setValue(keyCountKey, value);
    */
}

void SettingsPanel::usernameUpdated()
{
    MW->settings->setValue(usernameKey, commonUsername->text());
}

void SettingsPanel::passwordUpdated()
{
    MW->settings->setValue(passwordKey, commonPassword->text());
}

void SettingsPanel::lowLatencyClicked()
{
    MW->settings->setValue(lowLatencyKey, lowLatency->isChecked());
}

void SettingsPanel::hideToolTipsClicked()
{
    MW->settings->setValue(hideTipsKey, hideToolTips->isChecked());
    MW->cameraPanel->disableToolTips(hideToolTips->isChecked());
    MW->filePanel->disableToolTips(hideToolTips->isChecked());
}

void SettingsPanel::disableAudioClicked(bool clicked)
{
    MW->settings->setValue(disAudioKey, clicked);
}

void SettingsPanel::generateFilenameClicked(bool clicked)
{
    MW->settings->setValue(genFileKey, clicked);
    MW->settings->setValue(defFileKey, !clicked);
}

void SettingsPanel::defaultFilenameClicked(bool clicked)
{
    MW->settings->setValue(defFileKey, clicked);
    MW->settings->setValue(genFileKey, !clicked);
}

void SettingsPanel::styleClicked()
{
    MW->styleDialog->exec();
}

void SettingsPanel::clearClicked()
{
    QMessageBox::StandardButton result = QMessageBox::question(this, "onvif-gui", "You are about to delete all saved program settings\nAre you sure you want to do this");
    if (result == QMessageBox::Yes)
        MW->settings->clear();
}

AVHWDeviceType SettingsPanel::getHardwareDecoder() const
{
    QString name = hardwareDecoders->currentText();
    AVHWDeviceType result = AV_HWDEVICE_TYPE_NONE;
    if (name == "VDPAU")
        result = AV_HWDEVICE_TYPE_VDPAU;
    else if (name == "CUDA")
        result = AV_HWDEVICE_TYPE_CUDA;
    else if (name == "VAAPI")
        result = AV_HWDEVICE_TYPE_VAAPI;
    else if (name == "DXVA2")
        result = AV_HWDEVICE_TYPE_DXVA2;
    else if (name == "QSV")
        result = AV_HWDEVICE_TYPE_QSV;
    else if (name == "VIDEOTOOLBOX")
        result = AV_HWDEVICE_TYPE_VIDEOTOOLBOX;
    else if (name == "D3D11VA")
        result = AV_HWDEVICE_TYPE_D3D11VA;
    else if (name == "DRM")
        result = AV_HWDEVICE_TYPE_DRM;
    else if (name == "OPENCL")
        result = AV_HWDEVICE_TYPE_OPENCL;
    else if (name == "MEDIACODEC")
        result = AV_HWDEVICE_TYPE_MEDIACODEC;

    return result;
}

void SettingsPanel::decoderChanged(const QString& name)
{
    MW->settings->setValue(decoderKey, name);    
}

void SettingsPanel::netIntfChanged(const QString& arg)
{
    MW->settings->setValue(netIntfKey, arg);
}

void SettingsPanel::getActiveNetworkInterfaces()
{
#ifdef _WIN32
    PIP_ADAPTER_INFO pAdapterInfo;
    PIP_ADAPTER_INFO pAdapter = NULL;
    DWORD dwRetVal = 0;
    QStringList args;

    ULONG ulOutBufLen = sizeof (IP_ADAPTER_INFO);
    pAdapterInfo = (IP_ADAPTER_INFO *) malloc(sizeof (IP_ADAPTER_INFO));
    if (pAdapterInfo == NULL) {
        emit msg("Error allocating memory needed to call GetAdaptersinfo");
        return;
    }

    if (GetAdaptersInfo(pAdapterInfo, &ulOutBufLen) == ERROR_BUFFER_OVERFLOW) {
        free(pAdapterInfo);
        pAdapterInfo = (IP_ADAPTER_INFO *) malloc(ulOutBufLen);
        if (pAdapterInfo == NULL) {
            emit msg("Error allocating memory needed to call GetAdaptersinfo");
            return;
        }
    }

    if ((dwRetVal = GetAdaptersInfo(pAdapterInfo, &ulOutBufLen)) == NO_ERROR) {
        pAdapter = pAdapterInfo;
        while (pAdapter) {
            if (strcmp(pAdapter->IpAddressList.IpAddress.String, "0.0.0.0")) {
                char interface_info[1024];
                sprintf(interface_info, "%s - %s", pAdapter->IpAddressList.IpAddress.String, pAdapter->Description);
                emit msg(QString("Network interface info %1").arg(interface_info));
                args.push_back(interface_info);
            }
            pAdapter = pAdapter->Next;
        }
        interfaces->addItems(args);
        networkInterfaces->setModel(interfaces->model());
        networkInterfaces->setView(interfaces);
    } 
    else {
        emit msg(QString("GetAdaptersInfo failed with error: %1").arg(dwRetVal));
    }
    if (pAdapterInfo)
        free(pAdapterInfo);
#else
    struct ifaddrs *ifaddr;
    int family, s;
    char host[NI_MAXHOST];

    if (getifaddrs(&ifaddr) == -1) {
        emit msg(QString("Error: getifaddrs failed - %1").arg(strerror(errno)));
        return;
    }

    QStringList args;
    for (struct ifaddrs *ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == NULL)
            continue;

        family = ifa->ifa_addr->sa_family;

        if (family == AF_INET ) {
            s = getnameinfo(ifa->ifa_addr, 
                    sizeof(struct sockaddr_in),
                    host, NI_MAXHOST,
                    NULL, 0, NI_NUMERICHOST);

            if (s != 0) {
                emit msg(QString("getnameinfo() failed: %1").arg(gai_strerror(s)));
                continue;
            }

            if (strcmp(ifa->ifa_name, "lo")) {
                args.push_back(QString("%1 - %2").arg(host, ifa->ifa_name));
            }
        } 
    }

    interfaces->addItems(args);
    networkInterfaces->setModel(interfaces->model());
    networkInterfaces->setView(interfaces);

    freeifaddrs(ifaddr);
#endif
}

void SettingsPanel::getCurrentlySelectedIP(char *buffer)
{
    QString selected = networkInterfaces->currentText();
    int index = selected.indexOf(" - ");
    int i = 0;
    for (i = 0; i < index; i++) {
        buffer[i] = selected.toLatin1().data()[i];
    }
    buffer[i] = '\0';
}

void SettingsPanel::onInitTabs()
{
    MW->cameraPanel->showData();
}

void SettingsPanel::fillData(onvif::Data& onvif_data, int index)
{
    MW->cameraPanel->devices[index] = onvif_data;
    emit initTabs();
}

void SettingsPanel::testClicked()
{
    onvif::Manager onvifBoss;
    onvifBoss.startFill([&](onvif::Data& onvif_data, int index) { fillData(onvif_data, index); }, MW->cameraPanel->devices, 0);
}