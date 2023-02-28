/*******************************************************************************
* ptztab.cpp
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
//#include "mainwindow.h"
#include "onvif.h"
#include "ptztab.h"
#include <QGridLayout>
#include <QThreadPool>

PTZTab::PTZTab(QWidget *parent)
{
    cameraPanel = parent;

    const int BUTTON_SIZE = 30;
    for (int i = 0; i < 10; i++)
        speed[i] = 0.09 * (i+1);

    labelSpeed = new QLabel("PTZ speed");
    labelSpeed->setAlignment(Qt::AlignRight | Qt::AlignVCenter);

    comboSpeed = new QComboBox();
    comboSpeed->addItem(tr("1"));
    comboSpeed->addItem(tr("2"));
    comboSpeed->addItem(tr("3"));
    comboSpeed->addItem(tr("4"));
    comboSpeed->addItem(tr("5"));
    comboSpeed->addItem(tr("6"));
    comboSpeed->addItem(tr("7"));
    comboSpeed->addItem(tr("8"));
    comboSpeed->addItem(tr("9"));
    comboSpeed->addItem(tr("10"));
    comboSpeed->setMaximumWidth(40);
    comboSpeed->setCurrentIndex(4);

    textPreset = new QLineEdit;
    textPreset->setMaximumWidth(35);
    buttonPreset = new QPushButton(tr("Go"), this);
    buttonPreset->setMaximumWidth(35);
    connect(buttonPreset, SIGNAL(clicked()), this, SLOT(userPreset()));

    button1 = new QPushButton(tr("1"), this);
    button2 = new QPushButton(tr("2"), this);
    button3 = new QPushButton(tr("3"), this);
    button4 = new QPushButton(tr("4"), this);
    button5 = new QPushButton(tr("5"), this);
    button1->setMaximumWidth(BUTTON_SIZE);
    button2->setMaximumWidth(BUTTON_SIZE);
    button3->setMaximumWidth(BUTTON_SIZE);
    button4->setMaximumWidth(BUTTON_SIZE);
    button5->setMaximumWidth(BUTTON_SIZE);

    connect(button1, &QPushButton::clicked, [=] {preset(0);});
    connect(button2, &QPushButton::clicked, [=] {preset(1);});
    connect(button3, &QPushButton::clicked, [=] {preset(2);});
    connect(button4, &QPushButton::clicked, [=] {preset(3);});
    connect(button5, &QPushButton::clicked, [=] {preset(4);});

    checkPreset = new QCheckBox(tr("Set Preset"), this);

    buttonUp = new QPushButton(tr("^"), this);
    buttonDown = new QPushButton(tr("v"), this);
    buttonLeft = new QPushButton(tr("<"), this);
    buttonRight = new QPushButton(tr(">"), this);
    buttonZoomIn = new QPushButton(tr("Zoom In"), this);
    buttonZoomOut = new QPushButton(tr("Zoom Out"), this);
    buttonUp->setMaximumWidth(45);
    buttonDown->setMaximumWidth(45);
    buttonLeft->setMaximumWidth(45);
    buttonRight->setMaximumWidth(45);
    buttonZoomIn->setMaximumWidth(90);
    buttonZoomOut->setMaximumWidth(90);

    connect(buttonUp, &QPushButton::pressed, [=] {move(0.0, speed[comboSpeed->currentIndex()], 0.0);});
    connect(buttonDown, &QPushButton::pressed, [=] {move(0.0, speed[comboSpeed->currentIndex()] * -1, 0.0);});
    connect(buttonLeft, &QPushButton::pressed, [=] {move(speed[comboSpeed->currentIndex()] * -1, 0.0, 0.0);});
    connect(buttonRight, &QPushButton::pressed, [=] {move(speed[comboSpeed->currentIndex()], 0.0, 0.0);});
    connect(buttonZoomIn, &QPushButton::pressed, [=] {move(0.0, 0.0, speed[comboSpeed->currentIndex()]);});
    connect(buttonZoomOut, &QPushButton::pressed, [=] {move(0.0, 0.0, speed[comboSpeed->currentIndex()] * -1);});
    connect(buttonUp, SIGNAL(released()), this, SLOT(stopPanTilt()));
    connect(buttonDown, SIGNAL(released()), this, SLOT(stopPanTilt()));
    connect(buttonLeft, SIGNAL(released()), this, SLOT(stopPanTilt()));
    connect(buttonRight, SIGNAL(released()), this, SLOT(stopPanTilt()));
    connect(buttonZoomIn, SIGNAL(released()), this, SLOT(stopZoom()));
    connect(buttonZoomOut, SIGNAL(released()), this, SLOT(stopZoom()));

    QGridLayout *layout = new QGridLayout( this );
    layout->addWidget(textPreset,    1, 2, 1, 3);
    layout->addWidget(buttonPreset,  1, 3, 1, 1);
    layout->addWidget(checkPreset,   1, 4, 1, 2);
    layout->addWidget(button1,       1, 1, 1, 1);
    layout->addWidget(button2,       2, 1, 1, 1);
    layout->addWidget(button3,       3, 1, 1, 1);
    layout->addWidget(button4,       4, 1, 1, 1);
    layout->addWidget(button5,       5, 1, 1, 1);
    layout->addWidget(buttonLeft,    3, 2, 1, 1);
    layout->addWidget(buttonUp,      2, 3, 1, 1);
    layout->addWidget(buttonRight,   3, 4, 1, 1);
    layout->addWidget(buttonDown,    4, 3, 1, 1);
    layout->addWidget(buttonZoomIn,  2, 5, 1, 1);
    layout->addWidget(buttonZoomOut, 3, 5, 1, 1);
    layout->addWidget(labelSpeed,    5, 2, 1, 2);
    layout->addWidget(comboSpeed,    5, 4, 1, 2);

    //ptzMover = new PTZMover(cameraPanel);
    //ptzStopper = new PTZStopper(cameraPanel);
    //ptzGoto = new PTZGoto(cameraPanel);
    //ptzSetPreset = new PTZSetPreset(cameraPanel);
}

void PTZTab::update()
{

}

void PTZTab::setActive(bool active)
{
    checkPreset->setEnabled(active);
    textPreset->setEnabled(active);
    labelSpeed->setEnabled(active);
    comboSpeed->setEnabled(active);
    button1->setEnabled(active);
    button2->setEnabled(active);
    button3->setEnabled(active);
    button4->setEnabled(active);
    button5->setEnabled(active);
    buttonUp->setEnabled(active);
    buttonDown->setEnabled(active);
    buttonLeft->setEnabled(active);
    buttonRight->setEnabled(active);
    buttonZoomIn->setEnabled(active);
    buttonZoomOut->setEnabled(active);
    buttonPreset->setEnabled(active);
}

bool PTZTab::hasBeenEdited()
{
    return false;
}

void PTZTab::preset(int arg)
{
    if (checkPreset->isChecked()) {
        onvif::Manager onvifBoss;
        onvifBoss.startSetPresetPTZ(CP->devices[CP->currentDataRow], arg);
        checkPreset->setChecked(false);
    }
    else {
        onvif::Manager onvifBoss;
        onvifBoss.startSetPTZ(CP->devices[CP->currentDataRow], arg);
    }
}

void PTZTab::userPreset()
{
    bool ok;
    int arg = textPreset->text().toInt(&ok);
    if (ok) {
        preset(arg);
    }
}

void PTZTab::move(float x, float y, float z)
{
    onvif::Manager onvifBoss;
    onvifBoss.startMovePTZ(CP->devices[CP->currentDataRow], x, y, z);
}

void PTZTab::stopPanTilt()
{
    onvif::Manager onvifBoss;
    onvifBoss.startStopPTZ(CP->devices[CP->currentDataRow], PAN_TILT_STOP);
}

void PTZTab::stopZoom()
{
    onvif::Manager onvifBoss;
    onvifBoss.startStopPTZ(CP->devices[CP->currentDataRow], ZOOM_STOP);
}

