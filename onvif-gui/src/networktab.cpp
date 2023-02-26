/*******************************************************************************
* networktab.cpp
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

#include "camerapanel.h"

#include <QGridLayout>
#include <QLabel>
#include <QThreadPool>

NetworkTab::NetworkTab(QWidget *parent)
{
    cameraPanel = parent;

    checkDHCP = new QCheckBox(tr("DHCP Enabled"), this);
    textIPAddress = new QLineEdit();
    textIPAddress->setMaximumWidth(200);
    textSubnetMask = new QLineEdit();
    textSubnetMask->setMaximumWidth(200);
    textDefaultGateway = new QLineEdit();
    textDefaultGateway->setMaximumWidth(200);
    textDNS = new QLineEdit();
    textDNS->setMaximumWidth(200);

    lblIPAddress = new QLabel("IP Address");
    lblSubnetMask = new QLabel("Subnet Mask");
    lblDefaultGateway = new QLabel("Gateway");
    lblDNS = new QLabel("Primary DNS");

    QGridLayout *layout = new QGridLayout();
    layout->addWidget(checkDHCP,           0, 1, 1, 1);
    layout->addWidget(lblIPAddress,        1, 0, 1, 1);
    layout->addWidget(textIPAddress,       1, 1, 1, 1);
    layout->addWidget(lblSubnetMask,       2, 0, 1, 1);
    layout->addWidget(textSubnetMask,      2, 1, 1, 1);
    layout->addWidget(lblDefaultGateway,   3, 0, 1, 1);
    layout->addWidget(textDefaultGateway,  3, 1, 1, 1);
    layout->addWidget(lblDNS,              4, 0, 1, 1);
    layout->addWidget(textDNS,             4, 1, 1, 1);
    setLayout(layout);

    connect(checkDHCP, SIGNAL(clicked()), this, SLOT(dhcpChecked()));
    connect(textIPAddress, SIGNAL(textChanged(const QString &)), this, SLOT(onTextChanged(const QString &)));
    connect(textDefaultGateway, SIGNAL(textChanged(const QString &)), this, SLOT(onTextChanged(const QString &)));
    connect(textDNS, SIGNAL(textChanged(const QString &)), this, SLOT(onTextChanged(const QString &)));
    connect(textSubnetMask, SIGNAL(textChanged(const QString &)), this, SLOT(onTextChanged(const QString &)));

    updater = new NetworkUpdater(cameraPanel);
    connect(updater, SIGNAL(done()), this, SLOT(doneUpdating()));
}

void NetworkTab::update()
{
    onvif::Data onvif_data = CP->devices[CP->currentDataRow];
    onvif_data->dhcp_enabled = checkDHCP->isChecked();
    QString ip_address = textIPAddress->text();
    strncpy(onvif_data->ip_address_buf, ip_address.toLatin1(), ip_address.length());
    onvif_data->ip_address_buf[ip_address.length()] = '\0';
    QString default_gateway = textDefaultGateway->text();
    strncpy(onvif_data->default_gateway_buf, default_gateway.toLatin1(), default_gateway.length());
    onvif_data->default_gateway_buf[default_gateway.length()] = '\0';
    QString dns = textDNS->text();
    strncpy(onvif_data->dns_buf, dns.toLatin1(), dns.length());
    onvif_data->dns_buf[dns.length()] = '\0';
    onvif_data->prefix_length = mask2prefix(textSubnetMask->text().toLatin1().data());

    QThreadPool::globalInstance()->tryStart(updater);
}

void NetworkTab::clear()
{
    checkDHCP->setChecked(false);
    textIPAddress->setText("");
    textSubnetMask->setText("");
    textDefaultGateway->setText("");
    textDNS->setText("");
}

void NetworkTab::setActive(bool active)
{
    checkDHCP->setEnabled(active);
    textIPAddress->setEnabled(active);
    textSubnetMask->setEnabled(active);
    textDefaultGateway->setEnabled(active);
    textDNS->setEnabled(active);
    lblIPAddress->setEnabled(active);
    lblSubnetMask->setEnabled(active);
    lblDefaultGateway->setEnabled(active);
    lblDNS->setEnabled(active);
}

bool NetworkTab::hasBeenEdited()
{
    bool result = false;
    onvif::Data onvif_data = CP->devices[CP->currentDataRow];

    if (strcmp(textIPAddress->text().toLatin1().data(), "") != 0) {
        if (checkDHCP->isChecked() != onvif_data->dhcp_enabled)
            result = true;
        if (strcmp(textIPAddress->text().toLatin1().data(), onvif_data->ip_address_buf) != 0)
            result = true;
        if (mask2prefix(textSubnetMask->text().toLatin1().data()) != onvif_data->prefix_length)
            result = true;
        if (strcmp(textDefaultGateway->text().toLatin1().data(), onvif_data->default_gateway_buf) != 0)
            result = true;
        if (strcmp(textDNS->text().toLatin1().data(), onvif_data->dns_buf) != 0)
            result = true;
    }

    return result;
}

void NetworkTab::initialize()
{
    onvif::Data onvif_data = CP->devices[CP->currentDataRow];
    textIPAddress->setText(tr(onvif_data->ip_address_buf));
    char mask_buf[128] = {0};
    prefix2mask(onvif_data->prefix_length, mask_buf);
    textSubnetMask->setText(tr(mask_buf));
    textDNS->setText(tr(onvif_data->dns_buf));
    textDefaultGateway->setText(tr(onvif_data->default_gateway_buf));
    setDHCP(onvif_data->dhcp_enabled);
}

void NetworkTab::setDHCP(bool used)
{
    checkDHCP->setChecked(used);
    textIPAddress->setEnabled(!used);
    textSubnetMask->setEnabled(!used);
    textDefaultGateway->setEnabled(!used);
    textDNS->setEnabled(!used);
}

void NetworkTab::dhcpChecked()
{
    bool used = checkDHCP->isChecked();
    textIPAddress->setEnabled(!used);
    textSubnetMask->setEnabled(!used);
    textDefaultGateway->setEnabled(!used);
    textDNS->setEnabled(!used);
    if (hasBeenEdited())
        ((CameraPanel *)cameraPanel)->btnApply->setEnabled(true);
    else
        ((CameraPanel *)cameraPanel)->btnApply->setEnabled(false);
}

void NetworkTab::onTextChanged(const QString &)
{
    if (hasBeenEdited())
        ((CameraPanel *)cameraPanel)->btnApply->setEnabled(true);
    else
        ((CameraPanel *)cameraPanel)->btnApply->setEnabled(false);
}

void NetworkTab::doneUpdating()
{
    fprintf(stderr, "done updating\n");
    ((CameraPanel *)cameraPanel)->btnApply->setEnabled(false);
}
