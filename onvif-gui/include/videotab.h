/*******************************************************************************
* videotab.h
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

#ifndef VIDEOTAB_H
#define VIDEOTAB_H

#include "cameradialogtab.h"
#include "onvifboss.h"

#include <QComboBox>
#include <QSpinBox>
#include <QLabel>
#include <QListWidget>

class SpinBox : public QSpinBox
{

public:
    SpinBox(QLineEdit *editor);

};

class VideoTab : public CameraDialogTab
{
    Q_OBJECT

public:
    VideoTab(QWidget *parent);
    void update() override;
    void clear() override;
    void setActive(bool active) override;
    bool hasBeenEdited() override;
    void updated(const onvif::Data&);

    QWidget *cameraPanel;
    QListWidget *resolutions = nullptr;
    QComboBox *comboResolutions;
    SpinBox *spinFrameRate;
    SpinBox *spinGovLength;
    SpinBox *spinBitrate;
    QLabel *lblResolutions;
    QLabel *lblFrameRate;
    QLabel *lblGovLength;
    QLabel *lblBitrate;

    bool updating = false;
    QString updateSummary;

signals:
    void updateFinished();

public slots:
    void initialize();
    void onCurrentIndexChanged(int index);
    void onValueChanged(int value);
    void onUpdateFinished();
};

#endif // VIDEOTAB_H
