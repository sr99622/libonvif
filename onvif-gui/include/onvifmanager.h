/*******************************************************************************
* onvifmanager.h
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

#ifndef ONVIFMANAGER_H
#define ONVIFMANAGER_H

#include <QObject>
#include <QRunnable>
#include <QMutex>

#define ZOOM_FACTOR 16

class Filler : public QObject, public QRunnable
{
    Q_OBJECT

public:
    Filler(QWidget *parent);
    void run() override;

private:
    QWidget *cameraPanel;

signals:
    void done();

};

class VideoUpdater : public QObject, public QRunnable
{
    Q_OBJECT

public:
    VideoUpdater(QWidget *parent);
    void run() override;

private:
    QWidget *cameraPanel;

signals:
    void done();

};

class ImageUpdater : public QObject, public QRunnable
{
    Q_OBJECT

public:
    ImageUpdater(QWidget *parent);
    void run() override;

private:
    QWidget *cameraPanel;

signals:
    void done();

};

class NetworkUpdater : public QObject, public QRunnable
{
    Q_OBJECT

public:
    NetworkUpdater(QWidget *parent);
    void run() override;

private:
    QWidget *cameraPanel;

signals:
    void done();

};

class Rebooter : public QObject, public QRunnable
{
    Q_OBJECT

public:
    Rebooter(QWidget *parent);
    void run() override;

private:
    QWidget *cameraPanel;

signals:
    void done();

};

class Resetter : public QObject, public QRunnable
{
    Q_OBJECT

public:
    Resetter(QWidget *parent);
    void run() override;

private:
    QWidget *cameraPanel;

signals:
    void done();

};

class Timesetter : public QObject, public QRunnable
{
    Q_OBJECT

public:
    Timesetter(QWidget *parent);
    void run() override;

private:
    QWidget *cameraPanel;

signals:
    void done();

};

class PTZMover : public QObject, public QRunnable
{
    Q_OBJECT

public:
    PTZMover(QWidget *parent);
    void run() override;
    void set(float x_arg, float y_arg, float z_arg);

private:
    QWidget *cameraPanel;
    float x; float y; float z;

};

class PTZStopper : public QObject, public QRunnable
{
    Q_OBJECT

public:
    PTZStopper(QWidget *parent);
    void run() override;
    void set(int type_arg);

private:
    QWidget *cameraPanel;
    int type;

};

class PTZGoto : public QObject, public QRunnable
{
    Q_OBJECT

public:
    PTZGoto(QWidget *parent);
    void run() override;
    void set(int position_arg);

private:
    QWidget *cameraPanel;
    int position;

};

class PTZSetPreset : public QObject, public QRunnable
{
    Q_OBJECT

public:
    PTZSetPreset(QWidget *parent);
    void run() override;
    void set(int position_arg);

private:
    QWidget *cameraPanel;
    int position;

};

#endif // ONVIFMANAGER_H
