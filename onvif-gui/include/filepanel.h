/*******************************************************************************
* filepanel.h
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

#ifndef FILEPANEL_H
#define FILEPANEL_H

#include <iostream>

#include <QMainWindow>
#include <QFileSystemModel>
#include <QTreeView>
#include <QHeaderView>
#include <QSettings>
#include <QMouseEvent>
#include <QLineEdit>
#include <QPushButton>
#include <QIcon>
#include <QLabel>

class ProgressLabel : public QLabel
{
    Q_OBJECT

public:
    ProgressLabel(QWidget *parent);
    void paintEvent(QPaintEvent *event) override;

    QWidget *filePanel;
    int x_pos;

};

class ProgressSlider : public QSlider
{
    Q_OBJECT

public:
    ProgressSlider(Qt::Orientation o, QWidget *parent);
    bool event(QEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;

    QWidget* filePanel;

    int last_position_x;

signals:
    void seek(float);

};

class DirectorySetter : public QWidget
{
    Q_OBJECT

public:
    DirectorySetter(QMainWindow *parent, const QString& labelText);
    void setPath(const QString& path);

    QLabel *label;
    QLineEdit *text;
    QPushButton *button;
    QString directory;

    QMainWindow *mainWindow;

signals:
    void directorySet(const QString&);

public slots:
    void selectDirectory();

};


class TreeView : public QTreeView
{

public:
    TreeView(QWidget *parent);
    void keyPressEvent(QKeyEvent *event) override;
    void mouseDoubleClickEvent(QMouseEvent *event) override;

};

class FilePanel : public QWidget
{
    Q_OBJECT

public:
    FilePanel(QMainWindow *mainWindow);

    QMainWindow *mainWindow;
    DirectorySetter *directorySetter;
    QFileSystemModel *model;
    TreeView *tree;
    QMenu *menu;

    QPushButton *btnMute;
    QPushButton *btnPlay;
    QPushButton *btnStop;
    QIcon icnAudioOn;
    QIcon icnAudioOff;
    QIcon icnPlay;
    QIcon icnPause;
    QIcon icnStop;

    ProgressLabel *lblSeek;
    ProgressSlider *sldProgress;
    QLabel *lblProgress;
    QSlider *sldVolume;

    const QString dirKey    = "FilePanel/dir";
    const QString headerKey = "FilePanel/header";
    const QString volumeKey = "Application/volume";
    const QString muteKey   = "Application/mute";

signals:
    void msg(const QString&);

public slots:
    void setDirectory(const QString&);
    void doubleClicked(const QModelIndex&);
    void showContextMenu(const QPoint&);
    void headerChanged(int, int, int);
    void onMenuRemove();
    void onMenuRename();
    void onMenuInfo();
    void onMenuPlay();
    void onBtnPlayClicked();
    void onBtnStopClicked();
    void onBtnMuteClicked();
    void onSldVolumeMoved(int);
    void progress(float);
    void donePlayingMedia();

};

#endif // FILEPANEL_H
