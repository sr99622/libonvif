#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <QOpenGLWidget>
#include <QImage>
#include <QMutex>
#include <QPaintEvent>
#include "avio.h"

class GLWidget : public QOpenGLWidget
{
    Q_OBJECT

public:
    GLWidget();
    ~GLWidget();
    void play(const QString& arg);
    void stop();
    void setMute(bool arg);
    void setVolume(int arg);
    void toggle_pipe_out(const QString& filename);
    qint64 media_duration();
    bool isPaused();
    void togglePaused();
    bool checkForStreamHeader(const char* name);
    
    QSize sizeHint() const override;

    static void start(void* widget);

    char uri[1024];

    QImage img;
    avio::Frame f;
    QMutex mutex;
    avio::Player* player = nullptr;

    bool disable_audio = false;
    bool disable_video = false;

    int volume = 100;
    bool mute = false;

    int vpq_size = 0;
    int apq_size = 0;

    int keyframe_cache_size;
    AVHWDeviceType hardwareDecoder = AV_HWDEVICE_TYPE_NONE;

signals:
    void mediaPlayingStarted(qint64);
    void mediaPlayingStopped();
    void mediaProgress(float);
    void criticalError(const QString&);
    void infoMessage(const QString&);

protected:
    void paintEvent(QPaintEvent* event) override;

public slots:
    void seek(float);

};

#endif // GLWIDGET_H