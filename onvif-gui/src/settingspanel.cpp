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
//#define _GNU_SOURCE     /* To get defns of NI_MAXSERV and NI_MAXHOST */
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netdb.h>
#include <ifaddrs.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <linux/if_link.h>
#endif

#include <QLabel>
#include <QGridLayout>

#include "settingspanel.h"
#include "mainwindow.h"

SettingsPanel::SettingsPanel(QMainWindow* parent)
{
    mainWindow = parent;
    connect(this, SIGNAL(msg(const QString&)), MW, SLOT(msg(const QString&)));

    //networkInterfaces = new QComboBox();
    //networkInterfaces->setMaximumWidth(180);
    //QLabel *lbl03 = new QLabel("Select Network Interface");
    autoDiscovery = new QCheckBox("Auto Discovery");
    multiBroadcast = new QCheckBox("Multi Broadcast");
    broadcastRepeat = new QSpinBox();
    broadcastRepeat->setRange(2, 5);
    QLabel *lbl00 = new QLabel("Broadcast Repeat");
    commonUsername = new QLineEdit();
    commonUsername->setMaximumWidth(100);
    QLabel *lbl01 = new QLabel("Common Username");
    commonPassword = new QLineEdit();
    commonPassword->setMaximumWidth(100);
    QLabel *lbl02 = new QLabel("Common Password");
    lowLatency = new QCheckBox("Low Latency Buffering");

    QFrame *sliderFrame = new QFrame(this);
    sliderFrame->setMaximumHeight(300);
    sliderFrame->setFrameShape(QFrame::Panel);
    sliderFrame->setFrameShadow(QFrame::Plain);
    sliderFrame->setWindowTitle("Digital Zoom");

    QLabel *lbl10 = new QLabel("Zoom");
    zoom = new QSlider(Qt::Vertical);
    zoom->setValue(0);
    connect(zoom, SIGNAL(sliderMoved(int)), this, SLOT(zoomMoved(int)));

    QLabel *lbl11 = new QLabel("Pan X");
    panX = new QSlider(Qt::Vertical);
    panX->setValue(50);
    connect(panX, SIGNAL(sliderMoved(int)), this, SLOT(panXMoved(int)));

    QLabel *lbl12 = new QLabel("Pan Y");
    panY = new QSlider(Qt::Vertical);
    panY->setValue(50);
    connect(panY, SIGNAL(sliderMoved(int)), this, SLOT(panYMoved(int)));

    reset = new QPushButton("Reset");
    connect(reset, SIGNAL(clicked()), this, SLOT(resetClicked()));

    QLabel *title = new QLabel("Digital Zoom");

    QGridLayout *frameLayout = new QGridLayout(sliderFrame);
    frameLayout->addWidget(title, 0, 0, 1, 4, Qt::AlignRight);
    frameLayout->addWidget(zoom,  1, 0, 1, 1, Qt::AlignHCenter);
    frameLayout->addWidget(panX,  1, 1, 1, 1, Qt::AlignHCenter);
    frameLayout->addWidget(panY,  1, 2, 1, 1, Qt::AlignHCenter); 
    frameLayout->addWidget(lbl10, 2, 0, 1, 1, Qt::AlignCenter);
    frameLayout->addWidget(lbl11, 2, 1, 1, 1, Qt::AlignCenter);
    frameLayout->addWidget(lbl12, 2, 2, 1, 1, Qt::AlignCenter);
    frameLayout->addWidget(reset, 1, 3, 1, 1);

    QGridLayout *layout = new QGridLayout();
    //layout->addWidget(lbl03,               0, 0, 1, 1);
    //layout->addWidget(networkInterfaces,   0, 1, 1, 2);
    layout->addWidget(autoDiscovery,       1, 0, 1, 2);
    layout->addWidget(multiBroadcast,      2, 0, 1, 1);
    layout->addWidget(lbl00,               2, 1, 1, 1);
    layout->addWidget(broadcastRepeat,     2, 2, 1, 1);
    layout->addWidget(lbl01,               3, 0, 1, 1);
    layout->addWidget(commonUsername,      3, 1, 1, 1);
    layout->addWidget(lbl02,               4, 0, 1, 1);
    layout->addWidget(commonPassword,      4, 1, 1, 1);
    layout->addWidget(lowLatency,          5, 0, 1, 2);
    layout->addWidget(sliderFrame,         6, 0, 2, 4);
    setLayout(layout);

    //getActiveNetworkInterfaces();

    commonUsername->setText(MW->settings->value(usernameKey, "").toString());
    commonPassword->setText(MW->settings->value(passwordKey, "").toString());
    autoDiscovery->setChecked(MW->settings->value(autoDiscKey, false).toBool());
    multiBroadcast->setChecked(MW->settings->value(multiBroadKey, false).toBool());
    broadcastRepeat->setValue(MW->settings->value(broadRepKey, 2).toInt());
    lowLatency->setChecked(MW->settings->value(lowLatencyKey, false).toBool());
    autoDiscoveryClicked(autoDiscovery->isChecked());

    //QString netIntf = MW->settings->value(netIntfKey, "").toString();
    //if (netIntf.length() > 0)
    //    networkInterfaces->setCurrentText(netIntf);

    connect(commonUsername, SIGNAL(editingFinished()), this, SLOT(usernameUpdated()));
    connect(commonPassword, SIGNAL(editingFinished()), this, SLOT(passwordUpdated()));
    connect(autoDiscovery, SIGNAL(clicked(bool)), this, SLOT(autoDiscoveryClicked(bool)));
    connect(multiBroadcast, SIGNAL(clicked(bool)), this, SLOT(multiBroadcastClicked(bool)));
    connect(broadcastRepeat, SIGNAL(valueChanged(int)), this, SLOT(broadcastRepeatChanged(int)));
    connect(lowLatency, SIGNAL(clicked(bool)), this, SLOT(lowLatencyClicked(bool)));
    //connect(networkInterfaces, SIGNAL(currentTextChanged(const QString&)), this, SLOT(netIntfChanged(const QString&)));
}

void SettingsPanel::autoDiscoveryClicked(bool checked)
{
    if (checked) {
        multiBroadcast->setEnabled(true);
        broadcastRepeat->setEnabled(true);
    }
    else {
        multiBroadcast->setEnabled(false);
        broadcastRepeat->setEnabled(false);
        multiBroadcast->setChecked(false);
    }

    MW->settings->setValue(autoDiscKey, autoDiscovery->isChecked());
    MW->settings->setValue(multiBroadKey, multiBroadcast->isChecked());
}

void SettingsPanel::multiBroadcastClicked(bool checked)
{
    Q_UNUSED(checked);
    MW->settings->setValue(multiBroadKey, multiBroadcast->isChecked());
}

void SettingsPanel::broadcastRepeatChanged(int value)
{
    MW->settings->setValue(broadRepKey, value);
}

void SettingsPanel::usernameUpdated()
{
    MW->settings->setValue(usernameKey, commonUsername->text());
}

void SettingsPanel::passwordUpdated()
{
    MW->settings->setValue(passwordKey, commonPassword->text());
}

void SettingsPanel::lowLatencyClicked(bool clicked)
{
    MW->settings->setValue(lowLatencyKey, clicked);
}

void SettingsPanel::zoomMoved(int arg)
{
    MW->glWidget->setZoomFactor(1 - (float)arg / 100.0f);
}

void SettingsPanel::panXMoved(int arg)
{
    //std::cout << "panXMoved: " << arg << std::endl;
    MW->glWidget->setPanX((50.0f - (float)arg) / 20.0f);
}

void SettingsPanel::panYMoved(int arg)
{
    //std::cout << "panYMoved: " << arg << std::endl;
    MW->glWidget->setPanY((50.0f - (float)arg) / 20.0f);
}

void SettingsPanel::resetClicked()
{
    //std::cout << "reset" << std::endl;
    zoom->setValue(0);
    panX->setValue(50);
    panY->setValue(50);

    //float zoom_factor = 1 - (float)zoom->value() / 100.0f;
    //float pan_x = (50.0f - (float)panX->value()) / 50.0f;
    //float pan_y = (50.0f - (float)panY->value()) / 50.0f;

    //std::cout << "zoom_factor: " << zoom_factor << " pan_x: " << pan_x << " pan_y: " << pan_y << std::endl;
}

/*
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
    //UINT i;

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
                networkInterfaces->addItem(QString(interface_info));
            }
            pAdapter = pAdapter->Next;
        }
    } else {
        emit msg(QString("GetAdaptersInfo failed with error: %1").arg(dwRetVal));
    }
    if (pAdapterInfo)
        free(pAdapterInfo);
#else
    struct ifaddrs *ifaddr;
    int family, s;
    char host[NI_MAXHOST];

    if (getifaddrs(&ifaddr) == -1) {
        perror("getifaddrs");
        //exit(EXIT_FAILURE);
    }

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
                printf("getnameinfo() failed: %s\n", gai_strerror(s));
                //exit(EXIT_FAILURE);
            }

            if (strcmp(ifa->ifa_name, "lo")) {
                printf("name: %s, address: <%s>\n", ifa->ifa_name, host);
                QString label(host);
                label += " - ";
                label += ifa->ifa_name;
                emit msg(label);
                networkInterfaces->addItem(label);
            }

        } 
    }

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
*/
