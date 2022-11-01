/*******************************************************************************
* configtab.cpp
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

#include <QLabel>
#include <QGridLayout>

#include "configtab.h"
#include "camerapanel.h"

ConfigTab::ConfigTab(QWidget *parent)
{
    cameraPanel = parent;

    autoDiscovery = new QCheckBox("Auto Discovery");
    multiBroadcast = new QCheckBox("Multi Broadcast");
    //player = new QLineEdit("ffplay");
    //QLabel *lbl03 = new QLabel("Player");
    broadcastRepeat = new QSpinBox();
    broadcastRepeat->setRange(2, 5);
    QLabel *lbl00 = new QLabel("Broadcast Repeat");
    commonUsername = new QLineEdit();
    commonUsername->setMaximumWidth(100);
    QLabel *lbl01 = new QLabel("Common Username");
    commonPassword = new QLineEdit();
    commonPassword->setMaximumWidth(100);
    QLabel *lbl02 = new QLabel("Common Password");

    QGridLayout *layout = new QGridLayout();
    layout->addWidget(autoDiscovery,       1, 0, 1, 1);
    layout->addWidget(multiBroadcast,      2, 0, 1, 1);
    layout->addWidget(lbl00,               2, 1, 1 ,1);
    layout->addWidget(broadcastRepeat,     2, 2, 1, 1);
    layout->addWidget(lbl01,               3, 0, 1, 1);
    layout->addWidget(commonUsername,      3, 1, 1, 1);
    layout->addWidget(lbl02,               4, 0, 1, 1);
    layout->addWidget(commonPassword,      4, 1, 1, 1);
    //layout->addWidget(lbl03,               5, 0, 1, 1);
    //layout->addWidget(player,              5, 1, 1, 4);
    setLayout(layout);

    connect(commonUsername, SIGNAL(editingFinished()), this, SLOT(usernameUpdated()));
    connect(commonPassword, SIGNAL(editingFinished()), this, SLOT(passwordUpdated()));
    //connect(player, SIGNAL(editingFinished()), this, SLOT(playerUpdated()));
    connect(autoDiscovery, SIGNAL(clicked(bool)), this, SLOT(autoDiscoveryClicked(bool)));
    connect(multiBroadcast, SIGNAL(clicked(bool)), this, SLOT(multiBroadcastClicked(bool)));
    connect(broadcastRepeat, SIGNAL(valueChanged(int)), this, SLOT(broadcastRepeatChanged(int)));
}

void ConfigTab::autoDiscoveryClicked(bool checked)
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
    CP->saveAutoDiscovery();
    CP->saveMultiBroadcast();
}

void ConfigTab::multiBroadcastClicked(bool checked)
{
    Q_UNUSED(checked);
    CP->saveMultiBroadcast();
}

void ConfigTab::broadcastRepeatChanged(int value)
{
    CP->saveBroadcastRepeat(value);
}

void ConfigTab::usernameUpdated()
{
    CP->saveUsername();
}

void ConfigTab::passwordUpdated()
{
    CP->savePassword();
}

/*
void ConfigTab::playerUpdated()
{
    CP->savePlayer();
}
*/
