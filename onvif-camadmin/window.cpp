
#include <QtWidgets>
#include <thread>
#include <iostream>
#include <cmath>
#include "window.h"
#include "GLWidget.h"

ProgressSlider::ProgressSlider(void* parent) : QSlider(Qt::Horizontal)
{
    widget = parent;
    setMaximum(1000);
}

void ProgressSlider::mousePressEvent(QMouseEvent* event)
{
    float pct = (float)event->pos().x() / (float)width();
    ((avio::GLWidget*)widget)->process->reader->request_seek(pct);
}

Window::Window()
{
    glWidget = new avio::GLWidget();
    connect(glWidget, SIGNAL(progress(float)), this, SLOT(showProgress(float)));

    sldProgress = new ProgressSlider(glWidget);


    lblZoom = new QLabel("1.00");
    sldZoom = new QSlider(Qt::Vertical, this);
    connect(sldZoom, SIGNAL(valueChanged(int)), this, SLOT(onSldZoomValueChanged(int)));
    sldZoom->setValue(50);

    lblPanX = new QLabel("1.00");
    sldPanX = new QSlider(Qt::Vertical, this);
    connect(sldPanX, SIGNAL(valueChanged(int)), this, SLOT(onSldPanXValueChanged(int)));
    sldPanX->setValue(50);

    lblPanY = new QLabel("1.00");
    sldPanY = new QSlider(Qt::Vertical, this);
    connect(sldPanY, SIGNAL(valueChanged(int)), this, SLOT(onSldPanYValueChanged(int)));
    sldPanY->setValue(50);

    lblVolume = new QLabel("1.00");
    sldVolume = new QSlider(Qt::Vertical, this);
    connect(sldVolume, SIGNAL(valueChanged(int)), this, SLOT(onSldVolumeValueChanged(int)));
    sldVolume->setMaximum(100);
    sldVolume->setValue(100);

    QWidget* pnlSlider = new QWidget(this);
    QGridLayout* lytSlider = new QGridLayout(pnlSlider);
    lytSlider->addWidget(lblZoom,     0, 0, 1, 1);
    lytSlider->addWidget(sldZoom,     1, 0, 1, 1);
    lytSlider->addWidget(new QLabel("Zoom"), 2, 0, 1, 1);
    lytSlider->addWidget(lblPanX,     0, 1, 1, 1);
    lytSlider->addWidget(sldPanX,     1, 1, 1, 1);
    lytSlider->addWidget(new QLabel("PanX"), 2, 1, 1, 1);
    lytSlider->addWidget(lblPanY,     0, 2, 1, 1);
    lytSlider->addWidget(sldPanY,     1, 2, 1, 1);
    lytSlider->addWidget(new QLabel("PanY"), 2, 2, 1, 1);
    lytSlider->addWidget(lblVolume,   0, 3, 1, 1);
    lytSlider->addWidget(sldVolume,   1, 3, 1, 1);
    lytSlider->addWidget(new QLabel("Volume"), 2, 3, 1, 1);

    btnPlay = new QPushButton("play", this);
    connect(btnPlay, SIGNAL(clicked()), this, SLOT(onBtnPlayClicked()));
    btnPlay->setMaximumWidth(60);

    QPushButton* btnStop = new QPushButton("stop", this);
    connect(btnStop, SIGNAL(clicked()), this, SLOT(onBtnStopClicked()));
    btnStop->setMaximumWidth(60);

    QWidget* pnlButton = new QWidget(this);
    QGridLayout* lytButton = new QGridLayout(pnlButton);
    lytButton->addWidget(btnPlay,   0, 1, 1, 1);
    lytButton->addWidget(btnStop,   0, 2, 1, 1);

    QWidget* pnlControl = new QWidget(this);
    QGridLayout* lytControl = new QGridLayout(pnlControl);
    lytControl->addWidget(pnlSlider,  0, 0, 1, 1);
    lytControl->addWidget(pnlButton,  1, 0, 1, 1);

    QGridLayout *lytMain = new QGridLayout(this);
    lytMain->addWidget(glWidget,    0, 0, 1, 1);
    lytMain->addWidget(sldProgress, 1, 0, 1, 1);
    lytMain->addWidget(pnlControl,  0, 1, 1, 2);
    lytMain->setColumnStretch(0, 10);

    setWindowTitle(tr("Textures"));
    QList<QScreen*> screens = QGuiApplication::screens();
    QSize screenSize = screens[0]->size();
    int x = (screenSize.width() - width()) / 2;
    int y = (screenSize.height() - height()) / 2;
    move(x, y);
}

Window::~Window()
{

}

void Window::keyPressEvent(QKeyEvent* event)
{
    if (event->key() == Qt::Key_Escape) {
        close();
    }
}

void Window::onBtnPlayClicked()
{
    if (btnPlay->text() == "play") {
        btnPlay->setText("pause");
        if (playing)
            glWidget->pause();
        else
            glWidget->play("rtsp://admin:admin123@192.168.1.3:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif");
        playing = true;
    }
    else {
        btnPlay->setText("play");
        glWidget->pause();
    }
}
void Window::onBtnStopClicked()
{
    glWidget->stop();
    playing = false;
    btnPlay->setText("play");
}

void Window::onSldZoomValueChanged(int value)
{
    char buf[16];
    sprintf(buf, "%.2f", value / 50.0f);
    lblZoom->setText(buf);
    glWidget->setZoomFactor((float)value / 50.0f);
}

void Window::onSldPanXValueChanged(int value)
{
    char buf[16];
    sprintf(buf, "%.2f", value / 50.0f);
    lblPanX->setText(buf);
    glWidget->setPanX(1.0f - (float)value / 50.0f);
}

void Window::onSldPanYValueChanged(int value)
{
    char buf[16];
    sprintf(buf, "%.2f", value / 50.0f);
    lblPanY->setText(buf);
    glWidget->setPanY(1.0f - (float)value / 50.0f);
}

void Window::onSldVolumeValueChanged(int value)
{
    char buf[16];
    sprintf(buf, "%.2f", value / 100.0f);
    lblVolume->setText(buf);
    if (glWidget->process)
        glWidget->process->display->volume = (float)value / 100.0f;
}

void Window::showProgress(float arg)
{
    sldProgress->setValue((int)(arg * sldProgress->maximum()));
}