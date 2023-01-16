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
#include <iostream>
#include <sstream>
#include "avio.h"

namespace avio
{

static void handleFrameQueues(FRAME_Q_MAP& frame_queues)
{
    std::cout << "this is it" << std::endl;
    for (FRAME_Q_MAP::iterator q = frame_queues.begin(); q != frame_queues.end(); ++q) {
        std::cout << "q first " << q->first << std::endl;
    }
}

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
    delete timer;
}

QSize GLWidget::sizeHint() const
{
    return QSize(640, 480);
}

//void GLWidget::setFormat(QImage::Format arg)
//{
//    fmt = arg;
//}

void GLWidget::setVolume(int arg)
{
    volume = arg;
    if (process) {
        if (((Process*)process)->display) {
            ((Process*)process)->display->volume = (float)arg / 100.0f;
        }
    }
}

void GLWidget::setMute(bool arg)
{
    mute = arg;
    if (process) {
        if (((Process*)process)->display) {
            ((Process*)process)->display->mute = arg;
        }
    }
}

void GLWidget::togglePaused()
{
    if (process) {
        if (((Process*)process)->display) {
            ((Process*)process)->display->togglePause();
        }
    }
}

bool GLWidget::isPaused()
{
    bool result = false;
    if (process) {
        if (((Process*)process)->display) {
            result = ((Process*)process)->display->paused;
        }
    }
    return result;
}

void GLWidget::paintEvent(QPaintEvent* event)
{
    QOpenGLWidget::paintEvent(event);
    QPainter painter;
    painter.begin(this);
    if (!img.isNull()) {
        QImage tmp = img.scaled(width(), height(), Qt::KeepAspectRatio);
        int dx = width() - tmp.width();
        int dy = height() - tmp.height();
        painter.drawImage(dx>>1, dy>>1, tmp);
    }
    painter.end();
}

void GLWidget::poll()
{
    if (!process)
        return;
    if (!((Process*)process)->running)
        return;

    if (vfq_in) {
        try {
            if (vfq_in->size() > 0) {
                vfq_in->pop(f);
                if (f.isValid()) {
                    img = QImage(f.m_frame->data[0], f.m_frame->width, f.m_frame->height, fmt);
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

void GLWidget::toggle_pipe_out(const std::string& filename)
{
    if (process) {
        Reader* reader = ((Process*)process)->reader;
        if (reader) {
            reader->pipe_out_filename = filename;
            reader->request_pipe_write = !reader->request_pipe_write;
        }
    }
}

void GLWidget::seek(float arg)
{
    if (process) {
        if (((Process*)process)->reader) {
            ((Process*)process)->reader->request_seek(arg);
        }
    }
}

void GLWidget::stop()
{
    if (process)
        ((Process*)process)->running = false;

    while (process) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    emit progress(0);
}

void GLWidget::showStreamParameters(avio::Reader* reader)
{
    std::stringstream str;
    str << "\n" << mediaShortName;
    if (reader->has_video()) {
        str << "\nVideo Stream Parameters"
            << "\n  Video Codec:  " << reader->str_video_codec()
            << "\n  Pixel Format: " << reader->str_pix_fmt()
            << "\n  Resolution:   " << reader->width() << " x " << reader->height()
            << "\n  Frame Rate:   " << av_q2d(reader->frame_rate());
    }
    else {
        str << "\nNo Video Stream Found";
    }
    if (reader->has_audio()) {
        str << "\nAudio Stream Parameters"
            << "\n  Audio Codec:   " << reader->str_audio_codec()
            << "\n  Sample Format: " << reader->str_sample_format()
            << "\n  Channels:      " << reader->str_channel_layout();
    }
    else {
        str << "\nNo Audio Stream Found";
    }
    emit msg(str.str().c_str());
}

bool GLWidget::checkForStreamHeader(const char* name)
{
    QString str = QString(name).toLower();
    if (str.startsWith("rtsp://"))
        return true;
    if (str.startsWith("http://"))
        return true;
    if (str.startsWith("https://"))
        return true;
    return false;
}

void GLWidget::start(void * parent)
{
    GLWidget* widget = (GLWidget*)parent;

    try {
        avio::Process process;
        widget->process = &process;
        process.handleFrameQueues = std::function(handleFrameQueues);
        process.externalRenderer = true;

        avio::Reader reader(widget->uri);
        widget->showStreamParameters(&reader);
        const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(reader.pix_fmt());
        if (!desc) throw Exception("No pixel format in video stream");

        if (widget->checkForStreamHeader(widget->uri)) {
            if (widget->vpq_size) reader.apq_max_size = widget->vpq_size;
            if (widget->apq_size) reader.vpq_max_size = widget->vpq_size;
        }
        else {
            reader.apq_max_size = 1;
            reader.vpq_max_size = 1;
        }

        reader.set_video_out("vpq_reader");
        widget->media_duration = reader.duration();
        widget->media_start_time = reader.start_time();

        avio::Decoder videoDecoder(reader, AVMEDIA_TYPE_VIDEO, (AVHWDeviceType)widget->hardwareDecoder);
        videoDecoder.set_video_in(reader.video_out());
        videoDecoder.set_video_out("vfq_decoder");

        avio::Filter videoFilter(videoDecoder, "format=rgb24");
        videoFilter.set_video_in(videoDecoder.video_out());
        videoFilter.set_video_out("vfq_filter");

        avio::Display display(reader);
        display.set_video_in(videoFilter.video_out());
        display.set_video_out("vfq_display");
        widget->set_video_in(display.video_out());

        avio::Decoder* audioDecoder = nullptr;
        if (reader.has_audio() && !widget->disable_audio) {
            reader.set_audio_out("apq_reader");
            audioDecoder = new avio::Decoder(reader, AVMEDIA_TYPE_AUDIO);
            audioDecoder->set_audio_in(reader.audio_out());
            audioDecoder->set_audio_out("afq_decoder");
            display.set_audio_in(audioDecoder->audio_out());
            display.volume = widget->volume;
            display.mute = widget->isMute();
            process.add_decoder(*audioDecoder);
        }

        process.add_reader(reader);
        process.add_decoder(videoDecoder);
        process.add_filter(videoFilter);
        process.add_display(display);
        process.add_widget(widget);

        process.running = true;
        widget->emit mediaPlayingStarted();
        widget->emit timerStart();

        process.run();

        widget->emit timerStop();

        std::cout << "process done running" << std::endl;

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
    widget->emit mediaPlayingFinished();
}

}