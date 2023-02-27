/*******************************************************************************
* mainwindow.h
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

#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QPushButton>
#include <QLabel>
#include <QLineEdit>
#include <QTextEdit>
#include <QSettings>
#include <QSplitter>
#include <QTabWidget>
#include <QObject>

#include "camerapanel.h"
#include "settingspanel.h"
#include "messagepanel.h"
#include "stylepanel.h"
#include "filepanel.h"
#include <iostream>
#include <functional>
#include "avio.h"
#include "glwidget.h"

#define VERSION "1.4.5"

#define MW dynamic_cast<MainWindow*>(mainWindow)

class Label : public QLabel
{
public:
    QSize sizeHint() const override { return QSize(640, 480); }
};

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr, bool clearSettings = false);
    ~MainWindow();

    void showVersion() { std::cout << "1.4.5" << std::endl; }
    QString getButtonStyle(const QString& name) const;
    void applyStyle(const ColorProfile& profile);
    void closeEvent(QCloseEvent* event) override;
    void playerStop();
    void playerStart(const QString& uri);
    void errorMessage(const QString&);
    void infoMessage(const QString&);
    void mediaPlayingStopped();
    void mediaPlayingStarted(qint64);

    CameraPanel* cameraPanel;
    SettingsPanel* settingsPanel;
    MessagePanel* messagePanel;
    QTabWidget* tabWidget;
    StyleDialog* styleDialog;
    FilePanel* filePanel;
    QSettings* settings;
    QSplitter* split;

    GLWidget* glWidget = nullptr;
    Label *label = nullptr;

    avio::Player* player = nullptr;
    int volume = 80;
    bool mute = false;

    bool clearSettings = false;

    const QString splitKey  = "MainWindow/splitKey";
    const QString volumeKey = "MainWindow/volume";
    const QString muteKey   = "MainWindow/mute";

    QString style;
    QString currentMedia;

signals:
    void updateUI();
    void showError(const QString&);
    void playerStarted(qint64);
    void playerStopped();

public slots:
    void msg(const QString&);
    void onSplitterMoved(int pos, int index);
    void criticalError(const QString&);
    void togglePlayerMute();
    void setPlayerVolume(int);
    void onShowError(const QString&);
    void onPlayerStarted(qint64);
    void onPlayerStopped();

};

#endif // MAINWINDOW_H
