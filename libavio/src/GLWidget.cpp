/********************************************************************
* libavio/src/GLWidget.cpp
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

#include "GLWidget.h"
#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>
#include <iostream>
#include <sstream>
#include "avio.h"

#define VERTEX_ATTRIBUTE 0
#define TEXCOORD_ATTRIBUTE 1

namespace avio
{

GLWidget::GLWidget()
{
    timer = new QTimer(this);
    timer->setInterval(poll_interval);
    connect(timer, SIGNAL(timeout()), this, SLOT(poll()));
    connect(this, SIGNAL(timerStart()), timer, SLOT(start()));
    connect(this, SIGNAL(timerStop()), timer, SLOT(stop()));
}

GLWidget::~GLWidget()
{
    emit timerStop();
    makeCurrent();
    vbo.destroy();
    texture->release();
    texture->destroy();
    delete texture;
    program->release();
    program->removeAllShaders();
    delete vshader;
    delete fshader;
    delete program;
    doneCurrent();
    delete timer;
}

QSize GLWidget::sizeHint() const
{
    return QSize(640, 480);
}

void GLWidget::initializeGL()
{
    initializeOpenGLFunctions();

    vshader = new QOpenGLShader(QOpenGLShader::Vertex, this);
    const char *vsrc =
        "attribute highp vec4 vertex;\n"
        "attribute mediump vec4 texCoord;\n"
        "varying mediump vec4 texc;\n"
        "uniform mediump mat4 matrix;\n"
        "void main(void)\n"
        "{\n"
        "    gl_Position = matrix * vertex;\n"
        "    texc = texCoord;\n"
        "}\n";
    vshader->compileSourceCode(vsrc);

    fshader = new QOpenGLShader(QOpenGLShader::Fragment, this);
    const char *fsrc =
        "uniform sampler2D texture;\n"
        "varying mediump vec4 texc;\n"
        "void main(void)\n"
        "{\n"
        "    gl_FragColor = texture2D(texture, texc.st);\n"
        "}\n";
    fshader->compileSourceCode(fsrc);

    program = new QOpenGLShaderProgram;
    program->addShader(vshader);
    program->addShader(fshader);
    program->bindAttributeLocation("vertex", VERTEX_ATTRIBUTE);
    program->bindAttributeLocation("texCoord", TEXCOORD_ATTRIBUTE);
    program->link();

    program->bind();
    program->setUniformValue("texture", 0);

    static const int coords[4][3] = {
        { +1, -1, -1 }, 
        { -1, -1, -1 }, 
        { -1, +1, -1 }, 
        { +1, +1, -1 }
    };

    QVector<GLfloat> vertData;
    for (int j = 0; j < 4; ++j) {
        vertData.append(coords[j][0]);
        vertData.append(coords[j][1]);
        vertData.append(coords[j][2]);

        vertData.append(j == 0 || j == 3);
        vertData.append(j == 0 || j == 1);
    }

    vbo.create();
    vbo.bind();
    vbo.allocate(vertData.constData(), vertData.count() * sizeof(GLfloat));
}

void GLWidget::paintGL()
{
    try {
        QColor clearColor(0, 0, 0);
        glClearColor(clearColor.redF(), clearColor.greenF(), clearColor.blueF(), clearColor.alphaF());
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        float B[4] = { 
            (-1.0f + pan_x) * zoom * factor * aspect,
            (+1.0f + pan_x) * zoom * factor * aspect,
            (+1.0f + pan_y) * zoom * factor / aspect,
            (-1.0f + pan_y) * zoom * factor / aspect 
        };

        QMatrix4x4 m;
        m.ortho(B[0], B[1], B[2], B[3], -4.0f, 15.0f);
        m.translate(0.0f, 0.0f, -10.0f);

        program->setUniformValue("matrix", m);

        program->enableAttributeArray(VERTEX_ATTRIBUTE);
        program->enableAttributeArray(TEXCOORD_ATTRIBUTE);
        program->setAttributeBuffer(VERTEX_ATTRIBUTE, GL_FLOAT, 0, 3, 5 * sizeof(GLfloat));
        program->setAttributeBuffer(TEXCOORD_ATTRIBUTE, GL_FLOAT, 3 * sizeof(GLfloat), 2, 5 * sizeof(GLfloat));

        if (texture) {
            if (texture->width() != tex_width || texture->height() != tex_height) {
                texture->release();
                texture->destroy();
                delete texture;
                texture = nullptr;
            }
        }

        if (!texture) {
            texture = new QOpenGLTexture(QOpenGLTexture::Target2D);
            texture->setSize(tex_width, tex_height);
            texture->setFormat(QOpenGLTexture::RGB8_UNorm);
            texture->allocateStorage(QOpenGLTexture::RGB, QOpenGLTexture::UInt8);
            if (tex_width && tex_height)
                updateAspectRatio();
            texture->bind();
        }

        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    }
    catch (const std::runtime_error& e) {
        std::cout << "GLWidget paintGL error: " << e.what() << std::endl;
    }
}

void GLWidget::updateAspectRatio()
{
    if (maintain_aspect_ratio) {
        if (texture) {
            float imageAspect = (float)texture->width() / (float)texture->height();
            float widgetAspect = (float)width() / (float)height();
            float ratio = imageAspect / widgetAspect;
            aspect = pow(ratio, -0.5);
            zoom = (ratio > 1.0 ? pow(ratio, 0.5) : pow(ratio, -0.5));
            texture->bind();
        }
    }
}

void GLWidget::resizeGL(int width, int height)
{
    if (maintain_aspect_ratio)
        updateAspectRatio();
}

void GLWidget::setZoomFactor(float arg)
{
    factor = arg;
    update();
}

void GLWidget::setPanX(float arg)
{
    pan_x = arg;
    update();   
}

void GLWidget::setPanY(float arg)
{
    pan_y = arg;
    update();   
}

void GLWidget::setFormat(QImage::Format arg)
{
    fmt = arg;
}

void GLWidget::setVolume(int arg)
{
    volume = arg;
    if (process) {
        if (process->display) {
            process->display->volume = (float)arg / 100.0f;
        }
    }
}

void GLWidget::setMute(bool arg)
{
    mute = arg;
    if (process) {
        if (process->display) {
            process->display->mute = arg;
        }
    }
}

void GLWidget::poll()
{
    if (!running)
        return;

    if (vfq_in) {
        try {
            if (vfq_in->size() > 0) {
                vfq_in->pop(f);
                if (f.isValid()) {
                    count = 0;
                    if (f.m_frame->width == texture->width() && f.m_frame->height == texture->height()) {
                        QImage img(f.m_frame->data[0], texture->width(), texture->height(), fmt);
                        texture->setData(QOpenGLTexture::RGB, QOpenGLTexture::UInt8, (const void*)img.bits());
                    }
                }
                update();
            }
        }
        catch (const QueueClosedException& e) { }
        catch (const std::runtime_error& e) {
            std::cout << "GLWidget poll error: " << e.what() << std::endl;
        }
    }
}

void GLWidget::play(const QString& arg)
{
    try {
        stop();

        memset(uri, 0, 1024);
        strcpy(uri, arg.toLatin1().data());

        std::thread process_thread(start, this);
        process_thread.detach();
    }
    catch (const std::runtime_error& e) {
        std::cout << "GLWidget play error: " << e.what() << std::endl;
    }
}

void GLWidget::seek(float arg)
{
    if (process) {
        if (process->reader) {
            process->reader->request_seek(arg);
        }
    }
}

void GLWidget::stop()
{
    running = false;

    while (process) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    emit progress(0);
}

void GLWidget::showStreamParameters(avio::Reader* reader)
{
    std::stringstream str;
    str << "\n" << mediaShortName
        << "\nCamera Stream Parameters"
        << "\nVideo Codec: " << reader->str_video_codec()
        << "\nPixel Format: " << reader->str_pix_fmt();
    if (reader->has_audio()) {
        str << "\nAudio Codec: " << reader->str_audio_codec()
            << "\nSample Format: " << reader->str_sample_format()
            << "\nChannels: " << reader->str_channel_layout();
    }
    else {
        str << "\nNo Audio Stream Found";
    }
    emit msg(str.str().c_str());
}

void GLWidget::start(void * parent)
{
    GLWidget* widget = (GLWidget*)parent;

    try {
        avio::Process process;
        widget->process = &process;

        avio::Reader reader(widget->uri);
        if (QString(widget->uri).startsWith("rtsp://"))
            widget->showStreamParameters(&reader);

        if (widget->vpq_size) reader.apq_max_size = widget->vpq_size;
        if (widget->apq_size) reader.vpq_max_size = widget->vpq_size;
        widget->tex_width = reader.width();
        widget->tex_height = reader.height();
        reader.set_video_out("vpq_reader");
        widget->media_duration = reader.duration();
        widget->media_start_time = reader.start_time();

        avio::Decoder videoDecoder(reader, AVMEDIA_TYPE_VIDEO, (AVHWDeviceType)widget->hardwareDecoder);
        //avio::Decoder videoDecoder(reader, AVMEDIA_TYPE_VIDEO, AV_HWDEVICE_TYPE_VDPAU);
        videoDecoder.set_video_in(reader.video_out());
        videoDecoder.set_video_out("vfq_decoder");

        avio::Filter videoFilter(videoDecoder, "format=rgb24,vflip");
        videoFilter.set_video_in(videoDecoder.video_out());
        videoFilter.set_video_out("vfq_filter");

        avio::Display display(reader);
        display.glWidget = widget;
        display.set_video_in(videoFilter.video_out());
        display.set_video_out("vfq_display");
        widget->set_video_in(display.video_out());

        avio::Decoder* audioDecoder = nullptr;
        if (reader.has_audio()) {
            reader.set_audio_out("apq_reader");
            audioDecoder = new avio::Decoder(reader, AVMEDIA_TYPE_AUDIO);
            audioDecoder->set_audio_in(reader.audio_out());
            audioDecoder->set_audio_out("afq_decoder");
            display.set_audio_in(audioDecoder->audio_out());
            display.volume = widget->volume;
            display.mute = widget->getMute();
            process.add_decoder(*audioDecoder);
        }

        process.add_reader(reader);
        process.add_decoder(videoDecoder);
        process.add_filter(videoFilter);
        process.add_display(display);
        process.add_widget(widget);

        widget->running = true;
        process.run();

        std::cout << "process done" << std::endl;

        if (audioDecoder)
            delete audioDecoder;

    }
    catch (const Exception& e) {
        std::stringstream str;
        str << "GLWidget process error: " << e.what() << "\n";
        std::cout << str.str() << std::endl;
        widget->emit connectFailed(str.str().c_str());
    }

    widget->process = nullptr;
    widget->media_duration = 0;
    widget->emit progress(0);
}

}