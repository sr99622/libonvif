#include <iostream>
#include <SDL.h>
#include <QPainter>
#include "glwidget.h"

GLWidget::GLWidget()
{

}

GLWidget::~GLWidget()
{

}

void GLWidget::paintEvent(QPaintEvent* event)
{
    QOpenGLWidget::paintEvent(event);
    QPainter painter;
    painter.begin(this);
    if (!img.isNull()) {
        mutex.lock();
        QImage tmp = img.scaled(width(), height(), Qt::KeepAspectRatio);
        int dx = width() - tmp.width();
        int dy = height() - tmp.height();
        painter.drawImage(dx>>1, dy>>1, tmp);
        mutex.unlock();
    }
    painter.end();
}

void GLWidget::stop()
{
    if (player) {
        player->running = false;
        if (player->isPaused()) {
            SDL_Event event;
            event.type = SDL_QUIT;
            SDL_PushEvent(&event);
        }
        else {
            std::cout << "player not paused" << std::endl;
        }
    }

    while (player) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
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

void GLWidget::togglePaused()
{
    if (player) player->togglePaused();
}

bool GLWidget::isPaused()
{
    bool result = false;
    if (player) result = player->isPaused();
    return result;
}

qint64 GLWidget::media_duration()
{
    qint64 result = 0;
    if (player) {
        if (player->reader) {
            result = player->reader->duration();
        }
    }
    return result;
}

void GLWidget::toggle_pipe_out(const QString& filename)
{
    if (player) player->toggle_pipe_out(filename.toLatin1().data());
}

void GLWidget::seek(float arg)
{
    std::cout << "GLWidget::seek: " << arg << std::endl;
    if (player) player->seek(arg);
}

void GLWidget::setMute(bool arg)
{
    mute = arg;
    if (player) player->setMute(arg);
}

void GLWidget::setVolume(int arg)
{
    volume = arg;
    if (player) player->setVolume(arg);
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

void GLWidget::start(void* widget)
{

    GLWidget* glWidget = (GLWidget*)widget;
    avio::Decoder* videoDecoder = nullptr;
    avio::Filter* videoFilter = nullptr;
    avio::Decoder* audioDecoder = nullptr;

    std::function<void(const std::string&)> infoCallback = [&](const std::string& arg)
    {
        glWidget->emit infoMessage(arg.c_str());
    };

    std::function<void(const std::string&)> errorCallback = [&](const std::string& arg)
    {
        glWidget->emit criticalError(arg.c_str());
    };

    std::function<void(float)> progressCallback = [&](float arg)
    {
        glWidget->emit mediaProgress(arg);
    };

    std::function<void(const avio::Frame& frame)> renderCallback = [&](const avio::Frame& frame)
    {
        if (!frame.isValid()) {
            glWidget->emit infoMessage("render callback recvd invalid Frame");
            return;
        }
        glWidget->mutex.lock();
        glWidget->f = frame;
        glWidget->img = QImage(glWidget->f.m_frame->data[0], glWidget->f.m_frame->width,
                               glWidget->f.m_frame->height, QImage::Format_RGB888);
        glWidget->mutex.unlock();
        glWidget->update();
    };

    try {
        avio::Player player;
        glWidget->player = &player;

        avio::Reader reader(glWidget->uri);
        reader.infoCallback = infoCallback;
        reader.errorCallback = errorCallback;
        reader.showStreamParameters();

        const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(reader.pix_fmt());
        if (!desc) throw avio::Exception("No pixel format in video stream");

        if(glWidget->checkForStreamHeader(glWidget->uri)) {
            if (glWidget->vpq_size) reader.apq_max_size = glWidget->vpq_size;
            if (glWidget->apq_size) reader.vpq_max_size = glWidget->vpq_size;
        }
        else {
            reader.vpq_max_size = 1;
            reader.apq_max_size = 1;
        }

        avio::Display display(reader);
        display.renderCallback = renderCallback;
        display.progressCallback = progressCallback;
        display.infoCallback = infoCallback;
        display.errorCallback = errorCallback;
        display.volume = (float)glWidget->volume / 100.0f;
        display.mute = glWidget->mute;

        if (reader.has_video() && !glWidget->disable_video) {
            reader.set_video_out("vpq_reader");
            //reader.show_video_pkts = true;
            videoDecoder = new avio::Decoder(reader, AVMEDIA_TYPE_VIDEO, AV_HWDEVICE_TYPE_NONE);
            videoDecoder->infoCallback = infoCallback;
            videoDecoder->errorCallback = errorCallback;
            videoDecoder->set_video_in(reader.video_out());
            videoDecoder->set_video_out("vfq_decoder");
            player.add_decoder(*videoDecoder);
            
            videoFilter = new avio::Filter(*videoDecoder, "format=rgb24");
            videoFilter->infoCallback = infoCallback;
            videoFilter->errorCallback = errorCallback;
            videoFilter->set_video_in(videoDecoder->video_out());
            videoFilter->set_video_out("vfq_filter");
            player.add_filter(*videoFilter);
            display.set_video_in(videoFilter->video_out());
        }

        if (reader.has_audio() && !glWidget->disable_audio) {
            reader.set_audio_out("apq_reader");
            audioDecoder = new avio::Decoder(reader, AVMEDIA_TYPE_AUDIO);
            audioDecoder->infoCallback = infoCallback;
            audioDecoder->errorCallback = errorCallback;
            audioDecoder->set_audio_in(reader.audio_out());
            audioDecoder->set_audio_out("afq_decoder");
            display.set_audio_in(audioDecoder->audio_out());
            player.add_decoder(*audioDecoder);
        }

        player.add_reader(reader);
        player.add_display(display);

        glWidget->emit mediaPlayingStarted(reader.duration());
        player.run();
        glWidget->emit mediaPlayingStopped();

    }
    catch (const avio::Exception& e) {
        std::cout << "ERROR: " << e.what() << std::endl;
        glWidget->emit criticalError(e.what());
    }

    if (videoFilter) delete videoFilter;
    if (videoDecoder) delete videoDecoder;
    if (audioDecoder) delete audioDecoder;
    glWidget->player = nullptr;
    std::cout << "player done" << std::endl;
}

QSize GLWidget::sizeHint() const
{
    return QSize(640, 480);
}