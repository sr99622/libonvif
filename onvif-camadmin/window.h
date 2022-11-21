
#ifndef WINDOW_H
#define WINDOW_H

#include <QWidget>
#include <QSlider>
#include <QPushButton>
#include <QLabel>
#include <avio.h>
#include <fstream>

class ProgressSlider : public QSlider
{
    Q_OBJECT

public:
    ProgressSlider(void* parent);

protected:
    void mousePressEvent(QMouseEvent* e) override;

private:
    void* widget;

};

class Window : public QWidget
{
    Q_OBJECT

public:
    Window();
    ~Window();
    void keyPressEvent(QKeyEvent* event) override;

    QSlider* sldZoom;
    QSlider* sldPanX;
    QSlider* sldPanY;
    QSlider* sldVolume;
    ProgressSlider* sldProgress;

    QPushButton* btnPlay;
    bool playing = false;

    QLabel* lblZoom;
    QLabel* lblPanX;
    QLabel* lblPanY;
    QLabel* lblVolume;

    avio::GLWidget* glWidget;

public slots:
    void onBtnPlayClicked();
    void onBtnStopClicked();
    void onSldZoomValueChanged(int);
    void onSldPanXValueChanged(int);
    void onSldPanYValueChanged(int);
    void onSldVolumeValueChanged(int);
    void showProgress(float);

};

#endif
