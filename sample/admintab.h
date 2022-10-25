/*******************************************************************************
* admintab.h
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

#ifndef ADMINTAB_H
#define ADMINTAB_H

#include "cameradialogtab.h"
#include "onvifmanager.h"

#include <QLineEdit>
#include <QPushButton>
#include <QCheckBox>
#include <QLabel>
#include <QProcess>

class AdminTab : public CameraDialogTab
{
    Q_OBJECT

public:
    AdminTab(QWidget *parent);

    QLineEdit *textCameraName;
    QLineEdit *textAdminPassword;
    QPushButton *buttonReboot;
    QPushButton *buttonHardReset;
    QPushButton *buttonLaunchBrowser;
    QPushButton *buttonSyncTime;
    QCheckBox *checkEnableReboot;
    QCheckBox *checkEnableReset;

    QLabel *lblCameraName;
    QLabel *lblAdminPassword;

    QWidget *cameraPanel;

    Rebooter *rebooter;
    Resetter *resetter;
    Timesetter *timesetter;

    QProcess *process;

    void update() override;
    void clear() override;
    void setActive(bool active) override;
    bool hasBeenEdited()override;
    void initialize();

public slots:
    void doneRebooting();
    void doneResetting();

private slots:
    void launchBrowserClicked();
    void enableRebootChecked();
    void enableResetChecked();
    void rebootClicked();
    void hardResetClicked();
    void syncTimeClicked();
    void onTextChanged(const QString &);
};

#endif // ADMINTAB_H
