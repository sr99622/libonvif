/*******************************************************************************
* imagetab.cpp
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

#include "camerapanel.h"

#include <QGridLayout>
#include <QLabel>
#include <QThreadPool>

ImageTab::ImageTab(QWidget *parent)
{
    cameraPanel = parent;

    sliderBrightness = new QSlider(Qt::Horizontal);
    sliderSaturation = new QSlider(Qt::Horizontal);
    sliderContrast = new QSlider(Qt::Horizontal);
    sliderSharpness = new QSlider(Qt::Horizontal);

    lblBrightness = new QLabel("Brightness");
    lblSaturation = new QLabel("Saturation");
    lblContrast = new QLabel("Contrast");
    lblSharpness = new QLabel("Sharpness");

    QGridLayout *layout = new QGridLayout();
    layout->addWidget(lblBrightness,     0, 0, 1, 1);
    layout->addWidget(sliderBrightness,  0, 1, 1, 1);
    layout->addWidget(lblSaturation,     1, 0, 1, 1);
    layout->addWidget(sliderSaturation,  1, 1, 1, 1);
    layout->addWidget(lblContrast,       2, 0, 1, 1);
    layout->addWidget(sliderContrast,    2, 1, 1, 1);
    layout->addWidget(lblSharpness,      3, 0, 1, 1);
    layout->addWidget(sliderSharpness,   3, 1, 1, 1);
    setLayout(layout);

    connect(sliderBrightness, SIGNAL(valueChanged(int)), this, SLOT(onValueChanged(int)));
    connect(sliderSaturation, SIGNAL(valueChanged(int)), this, SLOT(onValueChanged(int)));
    connect(sliderContrast, SIGNAL(valueChanged(int)), this, SLOT(onValueChanged(int)));
    connect(sliderSharpness, SIGNAL(valueChanged(int)), this, SLOT(onValueChanged(int)));

    updater = new ImageUpdater(cameraPanel);
    connect(updater, SIGNAL(done()), this, SLOT(initialize()));

}

void ImageTab::update()
{
    OnvifData *onvif_data = ((CameraPanel *)cameraPanel)->camera->onvif_data;
    onvif_data->brightness = sliderBrightness->value();
    onvif_data->saturation = sliderSaturation->value();
    onvif_data->contrast = sliderContrast->value();
    onvif_data->sharpness = sliderSharpness->value();
    setImagingSettings(onvif_data);

    QThreadPool::globalInstance()->tryStart(updater);
}

void ImageTab::clear()
{
    sliderBrightness->setMinimum(0);
    sliderBrightness->setMaximum(0);
    sliderSaturation->setMinimum(0);
    sliderSaturation->setMaximum(0);
    sliderContrast->setMinimum(0);
    sliderContrast->setMaximum(0);
    sliderSharpness->setMinimum(0);
    sliderSharpness->setMaximum(0);

    sliderBrightness->setValue(0);
    sliderSaturation->setValue(0);
    sliderContrast->setValue(0);
    sliderSharpness->setValue(0);
}

void ImageTab::setActive(bool active)
{
    sliderBrightness->setEnabled(active);
    sliderSaturation->setEnabled(active);
    sliderContrast->setEnabled(active);
    sliderSharpness->setEnabled(active);
    lblBrightness->setEnabled(active);
    lblSaturation->setEnabled(active);
    lblContrast->setEnabled(active);
    lblSharpness->setEnabled(active);
}

bool ImageTab::hasBeenEdited()
{
    OnvifData *onvif_data = ((CameraPanel *)cameraPanel)->camera->onvif_data;
    bool result = false;
    if (sliderBrightness->value() != onvif_data->brightness)
        result = true;
    if (sliderSaturation->value() != onvif_data->saturation)
        result = true;
    if (sliderContrast->value() != onvif_data->contrast)
        result = true;
    if (sliderSharpness->value() != onvif_data->sharpness)
        result = true;
    return result;
}

void ImageTab::initialize()
{
    OnvifData *onvif_data = ((CameraPanel *)cameraPanel)->camera->onvif_data;
    sliderBrightness->setMinimum(onvif_data->brightness_min);
    sliderBrightness->setMaximum(onvif_data->brightness_max);
    sliderSaturation->setMinimum(onvif_data->saturation_min);
    sliderSaturation->setMaximum(onvif_data->saturation_max);
    sliderContrast->setMinimum(onvif_data->contrast_min);
    sliderContrast->setMaximum(onvif_data->contrast_max);
    sliderSharpness->setMinimum(onvif_data->sharpness_min);
    sliderSharpness->setMaximum(onvif_data->sharpness_max);

    sliderBrightness->setValue(onvif_data->brightness);
    sliderSaturation->setValue(onvif_data->saturation);
    sliderContrast->setValue(onvif_data->contrast);
    sliderSharpness->setValue(onvif_data->sharpness);
    ((CameraPanel *)cameraPanel)->btnApply->setEnabled(false);
}

void ImageTab::onValueChanged(int value)
{
    Q_UNUSED(value);
    if (hasBeenEdited())
        ((CameraPanel *)cameraPanel)->btnApply->setEnabled(true);
    else
        ((CameraPanel *)cameraPanel)->btnApply->setEnabled(false);
}
