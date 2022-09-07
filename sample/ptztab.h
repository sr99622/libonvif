/*******************************************************************************
* ptztab.h
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

#ifndef PTZTAB_H
#define PTZTAB_H

#include "cameradialogtab.h"
#include "onvifmanager.h"

#include <QCheckBox>
#include <QTextEdit>
#include <QLineEdit>
#include <QComboBox>
#include <QPushButton>
#include <QLabel>

class PTZTab : public CameraDialogTab
{
    Q_OBJECT

public:
    PTZTab(QWidget *parent);

    QCheckBox *checkPreset;
    QCheckBox *checkDigitalPTZ;
    QLineEdit *textPreset;
    QComboBox *comboSpeed;
    QPushButton *button1;
    QPushButton *button2;
    QPushButton *button3;
    QPushButton *button4;
    QPushButton *button5;
    QPushButton *buttonUp;
    QPushButton *buttonDown;
    QPushButton *buttonLeft;
    QPushButton *buttonRight;
    QPushButton *buttonZoomIn;
    QPushButton *buttonZoomOut;
    QPushButton *buttonPreset;
    QLabel *labelZoom;
    QLabel *labelSpeed;

    float speed[10];

    QWidget *cameraPanel;

    PTZMover *ptzMover;
    PTZStopper *ptzStopper;
    PTZGoto *ptzGoto;
    PTZSetPreset *ptzSetPreset;

    void update() override;
    void setActive(bool active) override;
    bool hasBeenEdited() override;

private slots:
    void preset(int);
    void userPreset();
    void move(float, float, float);
    void stopPanTilt();
    void stopZoom();

};

#endif // PTZTAB_H
