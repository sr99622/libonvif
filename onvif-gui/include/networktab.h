/*******************************************************************************
* networktab.h
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

#ifndef NETWORKTAB_H
#define NETWORKTAB_H

#include "cameradialogtab.h"
#include "onvifmanager.h"

#include <QCheckBox>
#include <QLineEdit>
#include <QLabel>

class NetworkTab : public CameraDialogTab
{
    Q_OBJECT

public:
    NetworkTab(QWidget *parent);

    QCheckBox *checkDHCP;
    QLineEdit *textIPAddress;
    QLineEdit *textSubnetMask;
    QLineEdit *textDefaultGateway;
    QLineEdit *textDNS;

    QLabel *lblIPAddress;
    QLabel *lblSubnetMask;
    QLabel *lblDefaultGateway;
    QLabel *lblDNS;

    QWidget *cameraPanel;

    NetworkUpdater *updater;

    void update() override;
    void clear() override;
    void setActive(bool active) override;
    bool hasBeenEdited() override;
    void initialize();
    void setDHCP(bool used);

private slots:
    void dhcpChecked();
    void onTextChanged(const QString &);
    void doneUpdating();
};

#endif // NETWORKTAB_H
