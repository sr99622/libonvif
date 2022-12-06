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

    btnApply = new QPushButton("Apply");
    btnCancel = new QPushButton("Cancel");

    useSystemGui = new QCheckBox("Use System GUI");
    sysGuiEnabled(useSystemGui->isChecked());
    connect(useSystemGui, SIGNAL(clicked(bool)), this, SLOT(sysGuiClicked(bool)));

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
    //layout->addWidget(useSystemGui,  1, 0, 1, 1);
    //layout->addWidget(btnCancel,     2, 1, 1, 1, Qt::AlignRight);
    //layout->addWidget(btnDefaults,   2, 2, 1, 1, Qt::AlignRight);
    //layout->addWidget(btnApply,      2, 3, 1, 1, Qt::AlignRight);
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
}

void StylePanel::sysGuiClicked(bool checked)
{
    MW->settings->setValue(sysGuiKey, checked);
    MW->applyStyle(getProfile());
    sysGuiEnabled(checked);
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

    //MW->applyStyle(getProfile());
}

StyleDialog::StyleDialog(QMainWindow *parent) : PanelDialog(parent)
{
    setWindowTitle("Configuration");
    panel = new StylePanel(mainWindow);
    QVBoxLayout *layout = new QVBoxLayout();
    layout->addWidget(panel);
    setLayout(layout);

    defaultWidth = 510;
    defaultHeight = 165;
    settingsKey = "StylePanel/geometry";
}


ColorButton::ColorButton(QMainWindow *parent, const QString& qss_name, const QString& color_name)
{
    mainWindow = parent;
    name = qss_name;
    settingsKey = qss_name + "/" + color_name;
    color.setNamedColor(MW->settings->value(settingsKey, color_name).toString());
    button = new QPushButton();
    button->setStyleSheet(getStyle());
    connect(button, SIGNAL(clicked()), this, SLOT(clicked()));
    QGridLayout *layout = new QGridLayout();
    layout->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(button, 0, 0, 1, 1);
    setLayout(layout);
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
    MW->settings->setValue(settingsKey, color.name());
}

QString ColorButton::getStyle() const
{
    return QString("QPushButton {background-color: %1;}").arg(color.name());
}

void ColorButton::clicked()
{
    QColor result = QColorDialog::getColor(color, this, "playqt");
    if (result.isValid()) {
        color = result;
        button->setStyleSheet(getStyle());
        MW->settings->setValue(settingsKey, color.name());
        //MW->applyStyle(MW->config()->getProfile());
    }
}

PanelDialog::PanelDialog(QMainWindow *parent) : QDialog(parent, Qt::WindowSystemMenuHint | Qt::WindowTitleHint | Qt::WindowCloseButtonHint)
{
    mainWindow = parent;
    timer = new QTimer(this);
    connect(timer, SIGNAL(timeout()), this, SLOT(autoSave()));
    timer->start(10000);
}

void PanelDialog::keyPressEvent(QKeyEvent *event)
{
    if (event->modifiers() & Qt::ControlModifier) {
        QAction action(getSettingsKey());
        action.setShortcut(QKeySequence(Qt::CTRL | event->key()));
        //MW->menuAction(&action);
    }

    QDialog::keyPressEvent(event);
}

void PanelDialog::showEvent(QShowEvent *event)
{
    shown = true;

    int w = getDefaultWidth();
    int h = getDefaultHeight();
    int x = MW->geometry().center().x() - w/2;
    int y = MW->geometry().center().y() - h/2;

    if (getSettingsKey().length() > 0) {
        if (MW->settings->contains(getSettingsKey())) {
            QRect rect = MW->settings->value(getSettingsKey()).toRect();
            //if (rect.width() + rect.x() < MW->screen->geometry().width() &&
            //        rect.height() + rect.y() < MW->screen->geometry().height())
            //{
            //    w = rect.width();
            //    h = rect.height();
            //    x = rect.x();
            //    y = rect.y();
            //}
        }
    }

    setGeometry(QRect(x, y, w, h));

    QDialog::showEvent(event);
}

void PanelDialog::autoSave()
{
    if (changed && shown && getSettingsKey().length() > 0) {
        MW->settings->setValue(getSettingsKey(), geometry());
        changed = false;
    }

    //if (panel)
    //    panel->autoSave();

}

void PanelDialog::resizeEvent(QResizeEvent *event)
{
    if (isVisible())
        changed = true;
    QDialog::resizeEvent(event);
}

void PanelDialog::moveEvent(QMoveEvent *event)
{
    if (isVisible())
        changed = true;
    QDialog::moveEvent(event);
}

void PanelDialog::closeEvent(QCloseEvent *event)
{
    autoSave();
    QDialog::closeEvent(event);
}

int PanelDialog::getDefaultWidth()
{
    return defaultWidth;
}

int PanelDialog::getDefaultHeight()
{
    return defaultHeight;
}

QString PanelDialog::getSettingsKey() const
{
    return settingsKey;
}
