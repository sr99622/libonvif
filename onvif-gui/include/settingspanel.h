/*******************************************************************************
* settingspanel.h
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

#ifndef SETTINGSPANEL_H
#define SETTINGSPANEL_H

#include <QComboBox>
#include <QLineEdit>
#include <QPushButton>
#include <QCheckBox>
#include <QLabel>
#include <QSpinBox>
#include <QSlider>
#include <QMainWindow>

class SettingsPanel : public QWidget
{
    Q_OBJECT

public:
    SettingsPanel(QMainWindow *parent);
    //void getActiveNetworkInterfaces();
    //void getCurrentlySelectedIP(char *buffer);

    QMainWindow *mainWindow;
    QCheckBox *autoDiscovery;
    QCheckBox *multiBroadcast;
    QCheckBox *useSystemGui;
    QPushButton *style;
    QSpinBox *broadcastRepeat;
    QLineEdit *commonUsername;
    QLineEdit *commonPassword;
    QCheckBox *lowLatency;
    QSlider *zoom;
    QSlider *panX;
    QSlider *panY;
    QPushButton *reset;
    //QComboBox *networkInterfaces;

    const QString usernameKey   = "SettingsPanel/username";
    const QString passwordKey   = "SettingsPanel/password";
    const QString playerKey     = "SettingsPanel/player";
    const QString autoDiscKey   = "SettingsPanel/autoDiscovery";
    const QString multiBroadKey = "SettingsPanel/multiBroadcast";
    const QString broadRepKey   = "SettingsPanel/brodacastRepeat";
    const QString lowLatencyKey = "SettingsPanel/lowLatency";
    //const QString netIntfKey    = "SettingsPanel/networkInterface";


signals:
    void msg(const QString&);

public slots:
    void usernameUpdated();
    void passwordUpdated();
    void autoDiscoveryClicked(bool);
    void multiBroadcastClicked(bool);
    void broadcastRepeatChanged(int);
    void lowLatencyClicked(bool);
    void zoomMoved(int);
    void panXMoved(int);
    void panYMoved(int);
    void resetClicked();
    void styleClicked();
    //void netIntfChanged(const QString&);

};

#endif // SETTINGSPANEL_H