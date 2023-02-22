/*******************************************************************************
* mainwindow.cpp
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
#include "mainwindow.h"
#include <QGridLayout>
#include <QApplication>
#include <QScreen>
#include <QResource>
#include <QIcon>
#include <QMessageBox>

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent)
{
    Q_INIT_RESOURCE(resources);
    QIcon icon(":/onvif-gui.png");
    setWindowIcon(icon);
    setWindowTitle(QString("onvif-gui version %1").arg(VERSION));
    settings = new QSettings("libonvif", "onvif");
    //settings->clear();
    messagePanel = new MessagePanel(this);

    glWidget = new GLWidget();

    styleDialog = new StyleDialog(this);

    settingsPanel = new SettingsPanel(this);
    cameraPanel = new CameraPanel(this);
    filePanel = new FilePanel(this);
    tabWidget= new QTabWidget();
    tabWidget->addTab(cameraPanel, "Cameras");
    tabWidget->addTab(filePanel, "Files");
    tabWidget->addTab(settingsPanel, "Settings");
    tabWidget->addTab(messagePanel, "Messages");
    setMinimumWidth(840);

    QWidget* layoutPanel = new QWidget();
    QGridLayout* layout = new QGridLayout(layoutPanel);

    split = new QSplitter;
    split->addWidget(glWidget);
    split->addWidget(tabWidget);
    split->restoreState(settings->value(splitKey).toByteArray());
    connect(split, SIGNAL(splitterMoved(int, int)), this, SLOT(onSplitterMoved(int, int)));

    connect(this, SIGNAL(showError(const QString&)), this, SLOT(onShowError(const QString&)));

    layout->addWidget(split,  0, 0, 1, 1);
    setCentralWidget(layoutPanel);

    mute = settings->value(muteKey, false).toBool();
    filePanel->setMuteButton(mute);
    cameraPanel->setMuteButton(mute);

    volume = settings->value(volumeKey, 80).toInt();
    filePanel->sldVolume->setValue(volume);
    cameraPanel->sldVolume->setValue(volume);

    QRect savedGeometry = settings->value("geometry").toRect();
    if (savedGeometry.isValid()) {
        setGeometry(savedGeometry);
    }
    else {
        QList<QScreen*> screens = QGuiApplication::screens();
        QSize screenSize = screens[0]->size();
        int x = (screenSize.width() - width()) / 2;
        int y = (screenSize.height() - height()) / 2;
        move(x, y);
    }

    StylePanel *stylePanel = (StylePanel*)styleDialog->panel;
    const ColorProfile profile = stylePanel->getProfile();
    applyStyle(profile);
}

MainWindow::~MainWindow()
{

}

void MainWindow::closeEvent(QCloseEvent* e)
{
    Q_UNUSED(e);
    playerStop();
    settings->setValue("geometry", geometry());
}

void MainWindow::mediaPlayingStopped()
{
    delete player;
    player = nullptr;
    glWidget->clear();
    filePanel->progress->setProgress(0);
    setWindowTitle(QString("onvif-gui version %1").arg(VERSION));
    emit updateUI();
}

void MainWindow::mediaPlayingStarted(qint64 duration)
{
    filePanel->progress->setDuration(duration);
    cameraPanel->connecting = false;
    setWindowTitle(currentMedia);
    emit updateUI();
}

void MainWindow::msg(const QString& str)
{
    std::cout << str.toLatin1().data() << std::endl;
    messagePanel->msg->append(str);
}

void MainWindow::onSplitterMoved(int pos, int index)
{
    settings->setValue(splitKey, split->saveState());
}

void MainWindow::onShowError(const QString& msg)
{
    QMessageBox msgBox(this);
    msgBox.setWindowTitle("Critical Error");
    msgBox.setText(msg);
    msgBox.setIcon(QMessageBox::Critical);
    msgBox.exec();
    mediaPlayingStopped();
    cameraPanel->connecting = false;
}

void MainWindow::errorMessage(const QString& msg)
{
    std::cout << "MainWindow::errorMessage: " << msg.toLatin1().data() << std::endl;
    emit showError(msg);
}

void MainWindow::infoMessage(const QString& msg)
{
    std::cout << "MainWindow::infoMessage: " << msg.toLatin1().data() << std::endl;
}

void MainWindow::playerStart(const QString& uri)
{
    playerStop();
    player = new avio::Player();
    player->uri = uri.toLatin1().data();
    player->video_filter = "format=rgb24";
    player->width = [&]() { return glWidget->width(); };
    player->height = [&]() { return glWidget->height(); };
    if (!settingsPanel->lowLatency->isChecked()) {
        player->vpq_size = 100;
        player->apq_size = 100;
    }
    player->progressCallback = [&](float pct) { filePanel->progress->setProgress(pct); };
    player->renderCallback = [&](const avio::Frame& frame) { glWidget->renderCallback(frame); };
    player->cbMediaPlayingStarted = [&](int64_t duration) { mediaPlayingStarted(duration); };
    player->cbMediaPlayingStopped = [&]() { mediaPlayingStopped(); };
    player->errorCallback = [&](const std::string& msg) { errorMessage(msg.c_str()); };
    player->infoCallback = [&](const std::string& msg) { infoMessage(msg.c_str()); };
    player->setMute(mute);
    player->setVolume(volume);
    player->start();
    setWindowTitle(QString("Connecting to " ) + currentMedia);
}

void MainWindow::playerStop()
{
    if (player) player->running = false;
    while (player) std::this_thread::sleep_for(std::chrono::milliseconds(1));
}

void MainWindow::setPlayerVolume(int arg)
{
    volume = arg;
    if (player) player->setVolume(arg);
    filePanel->sldVolume->setValue(arg);
    cameraPanel->sldVolume->setValue(arg);
    settings->setValue(volumeKey, arg);
}

void MainWindow::togglePlayerMute()
{
    mute = !mute;
    if (player) player->setMute(mute);
    filePanel->setMuteButton(mute);
    cameraPanel->setMuteButton(mute);
    settings->setValue(muteKey, mute);
}

void MainWindow::criticalError(const QString& str) 
{
    QMessageBox msgBox(QMessageBox::Critical, "Critical Error", str, QMessageBox::Close, this);
    msgBox.exec();
    msg(str);
}

QString MainWindow::getButtonStyle(const QString& name) const
{
    if (styleDialog->panel->useSystemGui->isChecked()) {
        return QString("QPushButton {image:url(:/%1_lo.png);}").arg(name);
    }
    else {
        return QString("QPushButton {image:url(:/%1.png);} QPushButton:!enabled {image:url(:/%1_lo.png);} QPushButton:hover {image:url(:/%1_hi.png);} QPushButton:pressed {image:url(:/%1.png);}").arg(name);
    }
}

void MainWindow::applyStyle(const ColorProfile& profile)
{
    if (styleDialog->panel->useSystemGui->isChecked()) {
        setStyleSheet("");
        return;
    }

    QFile f(":/darkstyle.qss");
    if (!f.exists()) {
        msg("Error: MainWindow::getThemes() Style sheet not found");
    }
    else {
        f.open(QFile::ReadOnly | QFile::Text);
        style = QString(f.readAll());

        style.replace("background_light",  profile.bl);
        style.replace("background_medium", profile.bm);
        style.replace("background_dark",   profile.bd);
        style.replace("foreground_light",  profile.fl);
        style.replace("foreground_medium", profile.fm);
        style.replace("foreground_dark",   profile.fd);
        style.replace("selection_light",   profile.sl);
        style.replace("selection_medium",  profile.sm);
        style.replace("selection_dark",    profile.sd);

        setStyleSheet(style);
    }
}
