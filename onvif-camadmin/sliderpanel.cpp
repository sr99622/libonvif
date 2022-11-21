#include <QGridLayout>
#include "mainwindow.h"
#include "sliderpanel.h"

SliderPanel::SliderPanel(QMainWindow* parent)
{
    mainWindow = parent;

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

    QPushButton* btnDefaults = new QPushButton("Defaults");
    btnDefaults->setMaximumWidth(80);
    connect(btnDefaults, SIGNAL(clicked()), this, SLOT(onBtnDefaultsClicked()));

    QGridLayout* lytSlider = new QGridLayout(this);
    lytSlider->addWidget(lblZoom,     0, 0, 1, 1);
    lytSlider->addWidget(sldZoom,     1, 0, 2, 1);
    lytSlider->addWidget(new QLabel("Zoom"), 3, 0, 1, 1);
    lytSlider->addWidget(lblPanX,     0, 1, 1, 1);
    lytSlider->addWidget(sldPanX,     1, 1, 2, 1);
    lytSlider->addWidget(new QLabel("PanX"), 3, 1, 1, 1);
    lytSlider->addWidget(lblPanY,     0, 2, 1, 1);
    lytSlider->addWidget(sldPanY,     1, 2, 2, 1);
    lytSlider->addWidget(new QLabel("PanY"), 3, 2, 1, 1);
    lytSlider->addWidget(lblVolume,   0, 3, 1, 1);
    lytSlider->addWidget(sldVolume,   1, 3, 2, 1);
    lytSlider->addWidget(new QLabel("Volume"), 3, 3, 1, 1);
    lytSlider->addWidget(new QLabel("DIGITAL\n  ZOOM"), 1, 4, 1, 1);
    lytSlider->addWidget(btnDefaults, 2, 4, 1, 1);
}

void SliderPanel::onSldZoomValueChanged(int value)
{
    char buf[16];
    sprintf(buf, "%.2f", value / 50.0f);
    lblZoom->setText(buf);
    MW->glWidget->setZoomFactor((float)value / 50.0f);
}

void SliderPanel::onSldPanXValueChanged(int value)
{
    char buf[16];
    sprintf(buf, "%.2f", value / 50.0f);
    lblPanX->setText(buf);
    MW->glWidget->setPanX(1.0f - (float)value / 50.0f);
}

void SliderPanel::onSldPanYValueChanged(int value)
{
    char buf[16];
    sprintf(buf, "%.2f", value / 50.0f);
    lblPanY->setText(buf);
    MW->glWidget->setPanY(1.0f - (float)value / 50.0f);
}

void SliderPanel::onSldVolumeValueChanged(int value)
{
    char buf[16];
    sprintf(buf, "%.2f", value / 100.0f);
    lblVolume->setText(buf);
    if (MW->glWidget->process)
        MW->glWidget->process->display->volume = (float)value / 100.0f;
}

void SliderPanel::onBtnDefaultsClicked()
{
    sldZoom->setValue(50);
    sldPanX->setValue(50);
    sldPanY->setValue(50);
    sldVolume->setValue(100);
}