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
#include "glwidget.h"

#define VERSION "1.4.5"

#define MW dynamic_cast<MainWindow*>(mainWindow)

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

    void showVersion() { std::cout << "1.4.5" << std::endl; }
    QString getButtonStyle(const QString& name) const;
    void applyStyle(const ColorProfile& profile);
    void closeEvent(QCloseEvent* event) override;

    CameraPanel* cameraPanel;
    SettingsPanel* settingsPanel;
    MessagePanel* messagePanel;
    QTabWidget* tabWidget;
    StyleDialog* styleDialog;
    FilePanel* filePanel;
    QSettings* settings;
    avio::GLWidget* glWidget;
    QSplitter* split;

    const QString splitKey = "MainWindow/splitKey";

    QString style;
    QString currentStreamingMediaName;
    std::function<int(const std::string&, const std::string&, const std::string&, const std::string&)> initPy = nullptr;
    std::function<bool(avio::Frame&, const std::string&)> runPy = nullptr;

public slots:
    void msg(const QString&);
    void onSplitterMoved(int pos, int index);

};
#endif // MAINWINDOW_H
