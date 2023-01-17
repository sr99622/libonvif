/********************************************************************
* libavio/include/GLWidget.h
*
* Copyright (c) 2022  Stephen Rhodes
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*    http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
*********************************************************************/

#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <QOpenGLWidget>
#include <QTimer>
#include <QPainter>
#include <QImage>
#include <QMutex>
#include <iostream>
#include "avio.h"


namespace avio
{

class GLWidget : public QOpenGLWidget
{
    Q_OBJECT

public:
    GLWidget();
    ~GLWidget();

    void setVolume(int arg);
    void setMute(bool arg);
    bool isMute() { return mute; }
    void togglePaused();
    bool isPaused();
    void play(const QString& arg);
    void stop();
    void showStreamParameters(avio::Reader* reader);
    void toggle_pipe_out(const std::string& filename);
    bool checkForStreamHeader(const char*);

    static void start(void * parent);
    static void renderCallback(void* caller, const avio::Frame& f);
    static void progressCallback(void* caller, float pct);
    static void cameraTimeoutCallback(Process* process);
    static void openWriterFailedCallback(Process* process, const std::string&);

    QSize sizeHint() const override;

    long media_duration = 0;
    long media_start_time = 0;
    bool disable_audio = false;
    int keyframe_cache_size = 1;
    int vpq_size = 0;
    int apq_size = 0;
    std::string mediaShortName;
    AVHWDeviceType hardwareDecoder = AV_HWDEVICE_TYPE_NONE;

    Process* process = nullptr;

signals:
    void cameraTimeout();
    void connectFailed(const QString&);
    void openWriterFailed(const std::string&);
    void msg(const QString&);
    void progress(float);
    void mediaPlayingFinished();
    void mediaPlayingStarted();

public slots:
    void seek(float);

protected:
    void paintEvent(QPaintEvent *event) override;

private:
    Frame f;
    int volume = 100;
    bool mute = false;
    QImage::Format fmt = QImage::Format_RGB888;
    char uri[1024];
    QImage img;
    QMutex mutex;

};

}

#endif
