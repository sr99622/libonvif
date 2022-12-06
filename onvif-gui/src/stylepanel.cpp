/*******************************************************************************
* stylepanel.cpp
*
* Copyright (c) 2022 Stephen Rhodes
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

#include <QGridLayout>
#include <QColorDialog>
#include <QAction>

#include "stylepanel.h"
#include "mainwindow.h"

StylePanel::StylePanel(QMainWindow *parent) : QWidget(parent)
{
    mainWindow = parent;

    QLabel *lblBL = new QLabel("Background Light");
    QLabel *lblBM = new QLabel("Background Medium");
    QLabel *lblBD = new QLabel("Background Dark");
    QLabel *lblFL = new QLabel("Foreground Light");
    QLabel *lblFM = new QLabel("Foreground Medium");
    QLabel *lblFD = new QLabel("Foreground Dark");
    QLabel *lblSL = new QLabel("Selection Light");
    QLabel *lblSM = new QLabel("Selection Medium");
    QLabel *lblSD = new QLabel("Selection Dark");

    bl = new ColorButton(mainWindow, "background_light", blDefault);
    bm = new ColorButton(mainWindow, "background_medium", bmDefault);
    bd = new ColorButton(mainWindow, "background_dark", bdDefault);
    fl = new ColorButton(mainWindow, "foreground_light", flDefault);
    fm = new ColorButton(mainWindow, "foreground_medium", fmDefault);
    fd = new ColorButton(mainWindow, "foreground_dark", fdDefault);
    sl = new ColorButton(mainWindow, "selection_light", slDefault);
    sm = new ColorButton(mainWindow, "selection_medium", smDefault);
    sd = new ColorButton(mainWindow, "selection_dark", sdDefault);

    QWidget *colorPanel = new QWidget();
    QGridLayout *colorLayout = new QGridLayout(colorPanel);
    colorLayout->addWidget(lblBL,   0, 0, 1, 1);
    colorLayout->addWidget(bl,      0, 1, 1, 1);
    colorLayout->addWidget(lblBM,   1, 0, 1, 1);
    colorLayout->addWidget(bm,      1, 1, 1, 1);
    colorLayout->addWidget(lblBD,   2, 0, 1, 1);
    colorLayout->addWidget(bd,      2, 1, 1, 1);
    colorLayout->addWidget(lblFL,   0, 2, 1, 1);
    colorLayout->addWidget(fl,      0, 3, 1, 1);
    colorLayout->addWidget(lblFM,   1, 2, 1, 1);
    colorLayout->addWidget(fm,      1, 3, 1, 1);
    colorLayout->addWidget(lblFD,   2, 2, 1, 1);
    colorLayout->addWidget(fd,      2, 3, 1, 1);
    colorLayout->addWidget(lblSL,   0, 4, 1, 1);
    colorLayout->addWidget(sl,      0, 5, 1, 1);
    colorLayout->addWidget(lblSM,   1, 4, 1, 1);
    colorLayout->addWidget(sm,      1, 5, 1, 1);
    colorLayout->addWidget(lblSD,   2, 4, 1, 1);
    colorLayout->addWidget(sd,      2, 5, 1, 1);
    colorLayout->setContentsMargins(0, 0, 0, 0);


    btnDefaults = new QPushButton("Defaults");
    connect(btnDefaults, SIGNAL(clicked()), this, SLOT(onBtnDefaultsClicked()));

    btnCancel = new QPushButton("Cancel");

    btnApply = new QPushButton("Apply");
    connect(btnApply, SIGNAL(clicked()), this, SLOT(onBtnApplyClicked()));

    useSystemGui = new QCheckBox("Use System GUI");
    connect(useSystemGui, SIGNAL(clicked(bool)), this, SLOT(sysGuiEnabled(bool)));
    useSystemGui->setChecked(MW->settings->value(sysGuiKey, false).toBool());
    sysGuiEnabled(useSystemGui->isChecked());
    btnApply->setEnabled(false);

    QWidget *controlPanel = new QWidget();
    QGridLayout *controlLayout = new QGridLayout(controlPanel);

    controlLayout->addWidget(useSystemGui,   0, 0, 1, 1, Qt::AlignLeft);
    controlLayout->addWidget(btnCancel,      0, 2, 1, 1, Qt::AlignRight);
    controlLayout->addWidget(btnDefaults,    0, 3, 1, 1, Qt::AlignRight);
    controlLayout->addWidget(btnApply,       0, 4, 1, 1, Qt::AlignRight);
    controlLayout->setColumnStretch(1, 20);

    QGridLayout *layout = new QGridLayout();
    layout->addWidget(colorPanel,    0, 0, 1, 4);
    layout->addWidget(controlPanel,  1, 0, 1, 4);
    setLayout(layout);
}

ColorProfile StylePanel::getProfile() const
{
    ColorProfile profile;
    profile.bl = bl->color.name();
    profile.bm = bm->color.name();
    profile.bd = bd->color.name();
    profile.fl = fl->color.name();
    profile.fm = fm->color.name();
    profile.fd = fd->color.name();
    profile.sl = sl->color.name();
    profile.sm = sm->color.name();
    profile.sd = sd->color.name();

    return profile;
}

void StylePanel::setTempProfile(const ColorProfile& profile)
{
    bl->setTempColor(profile.bl);
    bm->setTempColor(profile.bm);
    bd->setTempColor(profile.bd);
    fl->setTempColor(profile.fl);
    fm->setTempColor(profile.fm);
    fd->setTempColor(profile.fd);
    sl->setTempColor(profile.sl);
    sm->setTempColor(profile.sm);
    sd->setTempColor(profile.sd);
}

void StylePanel::sysGuiEnabled(bool arg)
{
    bl->setEnabled(!arg);
    bm->setEnabled(!arg);
    bd->setEnabled(!arg);
    fl->setEnabled(!arg);
    fm->setEnabled(!arg);
    fd->setEnabled(!arg);
    sl->setEnabled(!arg);
    sm->setEnabled(!arg);
    sd->setEnabled(!arg);

    btnDefaults->setEnabled(!arg);
    btnApply->setEnabled(true);
}

void StylePanel::onBtnApplyClicked()
{
    MW->settings->setValue(sysGuiKey, useSystemGui->isChecked());
    MW->applyStyle(getProfile());

    bl->writeSettings();
    bm->writeSettings();
    bd->writeSettings();
    fl->writeSettings();
    fm->writeSettings();
    fd->writeSettings();
    sl->writeSettings();
    sm->writeSettings();
    sd->writeSettings();

    btnApply->setEnabled(false);
}

void StylePanel::onBtnDefaultsClicked()
{
    bl->setColor(blDefault);
    bm->setColor(bmDefault);
    bd->setColor(bdDefault);
    fl->setColor(flDefault);
    fm->setColor(fmDefault);
    fd->setColor(fdDefault);
    sl->setColor(slDefault);
    sm->setColor(smDefault);
    sd->setColor(sdDefault);

    btnApply->setEnabled(true);
}

ColorButton::ColorButton(QMainWindow *parent, const QString& qss_name, const QString& color_name)
{
    mainWindow = parent;

    settingsKey = qss_name;
    QString thingy = MW->settings->value(settingsKey).toString();

    color.setNamedColor(MW->settings->value(settingsKey, color_name).toString());
    
    button = new QPushButton();
    connect(button, SIGNAL(clicked()), this, SLOT(clicked()));

    button->setStyleSheet(getStyle());

    QGridLayout *layout = new QGridLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(button, 0, 0, 1, 1);

}

void ColorButton::setTempColor(const QString& color_name)
{
    color.setNamedColor(color_name);
    button->setStyleSheet(getStyle());
}

void ColorButton::setColor(const QString& color_name)
{
    color.setNamedColor(color_name);
    button->setStyleSheet(getStyle());
}

QString ColorButton::getStyle() const
{
    return QString("QPushButton {background-color: %1;}").arg(color.name());
}

void ColorButton::writeSettings()
{
    MW->settings->setValue(settingsKey, color.name());
}

void ColorButton::clicked()
{
    QColor result = QColorDialog::getColor(color, this, "playqt");
    if (result.isValid()) {
        setColor(result.name());
        MW->styleDialog->panel->btnApply->setEnabled(true);
    }
}

