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

#include <QLabel>
#include <QGridLayout>

#include "settingspanel.h"
#include "mainwindow.h"

SettingsPanel::SettingsPanel(QMainWindow* parent)
{
    mainWindow = parent;

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

    QGridLayout *layout = new QGridLayout();
    layout->addWidget(autoDiscovery,       1, 0, 1, 1);
    layout->addWidget(multiBroadcast,      2, 0, 1, 1);
    layout->addWidget(lbl00,               2, 1, 1 ,1);
    layout->addWidget(broadcastRepeat,     2, 2, 1, 1);
    layout->addWidget(lbl01,               3, 0, 1, 1);
    layout->addWidget(commonUsername,      3, 1, 1, 1);
    layout->addWidget(lbl02,               4, 0, 1, 1);
    layout->addWidget(commonPassword,      4, 1, 1, 1);
    setLayout(layout);

    commonUsername->setText(MW->settings->value(usernameKey, "").toString());
    commonPassword->setText(MW->settings->value(passwordKey, "").toString());
    autoDiscovery->setChecked(MW->settings->value(autoDiscKey, false).toBool());
    multiBroadcast->setChecked(MW->settings->value(multiBroadKey, false).toBool());
    broadcastRepeat->setValue(MW->settings->value(broadRepKey, 2).toInt());
    autoDiscoveryClicked(autoDiscovery->isChecked());


    connect(commonUsername, SIGNAL(editingFinished()), this, SLOT(usernameUpdated()));
    connect(commonPassword, SIGNAL(editingFinished()), this, SLOT(passwordUpdated()));
    connect(autoDiscovery, SIGNAL(clicked(bool)), this, SLOT(autoDiscoveryClicked(bool)));
    connect(multiBroadcast, SIGNAL(clicked(bool)), this, SLOT(multiBroadcastClicked(bool)));
    connect(broadcastRepeat, SIGNAL(valueChanged(int)), this, SLOT(broadcastRepeatChanged(int)));
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

