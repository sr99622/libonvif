/*******************************************************************************
* videotab.cpp
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

#include <iostream>
#include <QLabel>
#include <QGridLayout>
#include <QThreadPool>
#include "camerapanel.h"

SpinBox::SpinBox(QLineEdit *editor)
{
    setLineEdit(editor);
}

VideoTab::VideoTab(QWidget *parent)
{
    cameraPanel = parent;

    comboResolutions = new QComboBox();

    QLineEdit *textBitrate = new QLineEdit();
    spinBitrate = new SpinBox(textBitrate);

    QLineEdit *textFrameRate = new QLineEdit();
    spinFrameRate = new SpinBox(textFrameRate);

    QLineEdit *textGovLength = new QLineEdit();
    spinGovLength = new SpinBox(textGovLength);

    lblResolutions = new QLabel("Resolution");
    lblFrameRate = new QLabel("Frame Rate");
    lblGovLength = new QLabel("GOP Length");
    lblBitrate = new QLabel("Bitrate");

    QGridLayout *layout = new QGridLayout();
    layout->addWidget(lblResolutions,    0, 0, 1, 1);
    layout->addWidget(comboResolutions,  0, 1, 1, 1);
    layout->addWidget(lblFrameRate,      1, 0, 1, 1);
    layout->addWidget(spinFrameRate,     1, 1, 1, 1);
    layout->addWidget(lblGovLength,      2, 0, 1, 1);
    layout->addWidget(spinGovLength,     2, 1, 1, 1);
    layout->addWidget(lblBitrate,        3, 0, 1, 1);
    layout->addWidget(spinBitrate,       3, 1, 1, 1);
    setLayout(layout);

    connect(comboResolutions, SIGNAL(currentIndexChanged(int)), this, SLOT(onCurrentIndexChanged(int)));
    connect(spinFrameRate, SIGNAL(valueChanged(int)), this, SLOT(onValueChanged(int)));
    connect(spinGovLength, SIGNAL(valueChanged(int)), this, SLOT(onValueChanged(int)));
    connect(spinBitrate, SIGNAL(valueChanged(int)), this, SLOT(onValueChanged(int)));

    connect(this, SIGNAL(updateFinished()), this, SLOT(onUpdateFinished()));

}

void VideoTab::update()
{
    OnvifData* onvif_data = CP->devices[CP->currentDataRow];

    QString res = QString("%1 x %2").arg(QString::number(onvif_data->width), QString::number(onvif_data->height));
    if (comboResolutions->currentText() != res) {
        QString str = "Resolution updated to ";
        str += comboResolutions->currentText();
        CP->emit msg(str);

        QString comboSel = comboResolutions->currentText();
        QString strHeight = comboSel.mid(comboSel.indexOf( " x ") + 3);
        QString strWidth = comboSel.mid(0, comboSel.indexOf(" x "));
        onvif_data->width = strWidth.toInt();
        onvif_data->height = strHeight.toInt();
    }

    if (spinFrameRate->value() != onvif_data->frame_rate) {
        QString str = "Frame Rate updated to ";
        str += QString::number(spinFrameRate->value());
        CP->emit msg(str);
        onvif_data->frame_rate = spinFrameRate->value();
    }

    if (spinGovLength->value() != onvif_data->gov_length) {
        QString str = "GOP length updated to ";
        str += QString::number(spinGovLength->value());
        CP->emit msg(str);
        onvif_data->gov_length = spinGovLength->value();
    }

    if (spinBitrate->value() != onvif_data->bitrate) {
        QString str = "Bitrate updated to ";
        str += QString::number(spinBitrate->value());
        CP->emit msg(str);
        onvif_data->bitrate = spinBitrate->value();
    }

    onvif::Manager onvifBoss;
    onvifBoss.startUpdateVideo(CP->devices[CP->currentDataRow], 
                        [&](const onvif::Data& onvif_data) { updated(onvif_data); });

    CP->btnApply->setEnabled(false);
    CP->cameraList->setEnabled(false);
    setActive(false);

    std::cout << "VideoTab::update" << std::endl;
}

void VideoTab::updated(const onvif::Data& onvif_data)
{
    CP->devices[CP->currentDataRow] = onvif_data;
    emit updateFinished();
    std::cout << "video update finished" << std::endl;
}

void VideoTab::onUpdateFinished()
{
    initialize();
    CP->cameraList->setEnabled(true);
    setActive(true);
}

void VideoTab::clear()
{
    comboResolutions->clear();
    spinFrameRate->setValue(0);
    spinGovLength->setValue(0);
    spinBitrate->setValue(0);
    spinGovLength->setMinimum(0);
    spinGovLength->setMaximum(0);
    spinFrameRate->setMinimum(0);
    spinFrameRate->setMaximum(0);
    spinBitrate->setMinimum(0);
    spinBitrate->setMaximum(0);
}

void VideoTab::setActive(bool active)
{
    comboResolutions->setEnabled(active);
    spinFrameRate->setEnabled(active);
    spinGovLength->setEnabled(active);
    spinBitrate->setEnabled(active);
    lblResolutions->setEnabled(active);
    lblFrameRate->setEnabled(active);
    lblGovLength->setEnabled(active);
    lblBitrate->setEnabled(active);
}

void VideoTab::initialize()
{
    onvif::Data onvif_data = CP->devices[CP->currentDataRow];
    comboResolutions->clear();

    int size = 0;
    bool found_size = false;
    while (!found_size) {
        if (strlen(onvif_data->resolutions_buf[size]) == 0) {
            found_size = true;
        }
        else {
            size++;
            if (size > 15)
                found_size = true;
        }
    }

    if (resolutions) delete resolutions;
    resolutions = new QListWidget(this);

    QStringList args;
    for (int i=0; i<size; i++) {
        args.push_back(onvif_data->resolutions_buf[i]);
    }

    resolutions->addItems(args);
    comboResolutions->setModel(resolutions->model());
    comboResolutions->setView(resolutions);

    spinGovLength->setMinimum(onvif_data->gov_length_min);
    spinGovLength->setMaximum(onvif_data->gov_length_max);
    spinFrameRate->setMinimum(onvif_data->frame_rate_min);
    spinFrameRate->setMaximum(onvif_data->frame_rate_max);
    spinBitrate->setMinimum(onvif_data->bitrate_min);
    spinBitrate->setMaximum(onvif_data->bitrate_max);

    char res[128] = {0};
    sprintf(res, "%d x %d", onvif_data->width, onvif_data->height);
    comboResolutions->setCurrentText(tr(res));
    spinGovLength->setValue(onvif_data->gov_length);
    spinFrameRate->setValue(onvif_data->frame_rate);
    spinBitrate->setValue(onvif_data->bitrate);
    CP->btnApply->setEnabled(false);

}

bool VideoTab::hasBeenEdited() {
    onvif::Data onvif_data = CP->devices[CP->currentDataRow];
    bool result = false;
    if (strcmp(comboResolutions->currentText().toLatin1().data(), "") != 0) {
        if (spinGovLength->value() != onvif_data->gov_length)
            result = true;
        if (spinBitrate->value() != onvif_data->bitrate)
            result = true;
        if (spinFrameRate->value() != onvif_data->frame_rate)
            result = true;

        char resolution[128] = {0};
        sprintf(resolution, "%d x %d", onvif_data->width, onvif_data->height);
        if (strcmp(comboResolutions->currentText().toLatin1().data(), resolution) != 0)
            result = true;
    }
    return result;
}

void VideoTab::onCurrentIndexChanged(int index) {
    Q_UNUSED(index);
    if (hasBeenEdited())
        CP->btnApply->setEnabled(true);
    else
        CP->btnApply->setEnabled(false);
}

void VideoTab::onValueChanged(int index) {
    Q_UNUSED(index);
    if (hasBeenEdited())
        CP->btnApply->setEnabled(true);
    else
        CP->btnApply->setEnabled(false);
}



