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
#include <QOpenGLFunctions>
#include <QOpenGLBuffer>
#include <QOpenGLShader>
#include <QTimer>
#include <iostream>
#include "Queue.h"
#include "Frame.h"

QT_FORWARD_DECLARE_CLASS(QOpenGLShaderProgram)
QT_FORWARD_DECLARE_CLASS(QOpenGLTexture)

namespace avio
{

QT_FORWARD_DECLARE_CLASS(Process)

class GLWidget : public QOpenGLWidget, protected QOpenGLFunctions
{
    Q_OBJECT

public:
    GLWidget();
    ~GLWidget();

    void setZoomFactor(float);
    void setPanX(float);
    void setPanY(float);
    void setFormat(QImage::Format);
    void updateAspectRatio();
    void play(const char* uri);
    void stop();
    void abort();

    static void start(void * parent, const char* uri);

    QSize sizeHint() const override;

    std::string vfq_in_name;
    std::string vfq_out_name;

    Queue<Frame>* vfq_in = nullptr;
    Queue<Frame>* vfq_out = nullptr;

    std::string video_in() const { return std::string(vfq_in_name); }
    std::string video_out() const { return std::string(vfq_out_name); }
    void set_video_in(const std::string& name) { vfq_in_name = std::string(name); }
    void set_video_out(const std::string& name) { vfq_out_name = std::string(name); }

    int poll_interval = 1;
    int tex_width = 0;
    int tex_height = 0;
    bool maintain_aspect_ratio = true;
    long media_duration = 0;
    long media_start_time = 0;
    bool running = false;

    avio::Process* process = nullptr;

signals:
    void timerStart();
    void timerStop();
    void cameraTimeout();

public slots:
    void poll();

protected:
    void initializeGL() override;
    void paintGL() override;
    void resizeGL(int width, int height) override;

private:
    QOpenGLTexture *texture = nullptr;
    QOpenGLShaderProgram *program = nullptr;
    QOpenGLBuffer vbo;
    QOpenGLShader *vshader;
    QOpenGLShader *fshader;

    Frame f;
    std::mutex mutex;

    float zoom   = 1.0f;
    float factor = 1.0f;
    float aspect = 1.0f;
    float pan_x  = 0.0f;
    float pan_y  = 0.0f;

    QImage::Format fmt = QImage::Format_RGB888;

    QTimer *timer;
    int count = 0;
};

}

#endif
