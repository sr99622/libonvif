/*******************************************************************************
* stylepanel.h
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

#ifndef STYLEPANEL_H
#define STYLEPANEL_H

#include <QMainWindow>
#include <QColor>
#include <QCheckBox>
#include <QPushButton>

class PanelDialog : public QDialog
{
    Q_OBJECT

public:
    PanelDialog(QMainWindow *parent);
    void closeEvent(QCloseEvent *event) override;
    void showEvent(QShowEvent *event) override;
    void moveEvent(QMoveEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;
    void keyPressEvent(QKeyEvent *event) override;
    virtual int getDefaultWidth();
    virtual int getDefaultHeight();
    virtual QString getSettingsKey() const;

    //void saveSettings();

    QMainWindow *mainWindow;
    QWidget *panel = nullptr;
    bool shown = false;
    bool changed = false;

    QTimer *timer;
    int defaultWidth = 320;
    int defaultHeight = 240;
    QString settingsKey = "";

signals:
    void closing();

public slots:
    void autoSave();

};

class ColorButton : public QWidget
{
    Q_OBJECT

public:
    ColorButton(QMainWindow *parent, const QString& qss_name, const QString& color_name);
    QString getStyle() const;
    void setColor(const QString& color_name);
    void setTempColor(const QString& color_name);

    QMainWindow *mainWindow;
    QString name;
    QColor color;
    QPushButton *button;
    QString settingsKey;

public slots:
    void clicked();

};

struct ColorProfile
{
    QString bl;
    QString bm;
    QString bd;
    QString fl;
    QString fm;
    QString fd;
    QString sl;
    QString sm;
    QString sd;
};

class StylePanel : public QWidget
{
    Q_OBJECT

public:
    StylePanel(QMainWindow *parent);
    void sysGuiEnabled(bool arg);
    //void autoSave() override;
    ColorProfile getProfile() const;
    void setTempProfile(const ColorProfile& profile);

    QMainWindow *mainWindow;

    const QString blDefault = "#566170";
    const QString bmDefault = "#3E4754";
    const QString bdDefault = "#283445";
    const QString flDefault = "#C6D9F2";
    const QString fmDefault = "#9DADC2";
    const QString fdDefault = "#808D9E";
    const QString slDefault = "#FFFFFF";
    const QString smDefault = "#DDEEFF";
    const QString sdDefault = "#306294";

    ColorButton *bl;
    ColorButton *bm;
    ColorButton *bd;
    ColorButton *fl;
    ColorButton *fm;
    ColorButton *fd;
    ColorButton *sl;
    ColorButton *sm;
    ColorButton *sd;

    QCheckBox *useSystemGui;
    QPushButton *btnDefaults;
    QPushButton *btnApply;
    QPushButton *btnCancel;

    const QString sysGuiKey = "StylePanel/useSystemGui";

public slots:
    void onBtnDefaultsClicked();
    void sysGuiClicked(bool);
};

class StyleDialog : public PanelDialog
{
    Q_OBJECT

public:
    StyleDialog(QMainWindow *parent);

};

#endif // STYLEPANEL_H
