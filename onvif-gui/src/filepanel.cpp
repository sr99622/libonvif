/*******************************************************************************
* filepanel.cpp
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

#include <QFileDialog>
#include <QMenu>
#include <QAction>
#include <QStandardPaths>
#include <QMessageBox>
#include <QDateTime>
#include <QToolTip>

#include "filepanel.h"
#include "mainwindow.h"

FilePanel::FilePanel(QMainWindow *parent) : QWidget(parent)
{
    mainWindow = parent;

    directorySetter = new DirectorySetter(mainWindow, "");
    model = new QFileSystemModel();
    model->setReadOnly(false);
    tree = new TreeView(this);
    tree->setModel(model);

    btnPlay = new QPushButton();
    btnPlay->setStyleSheet(getButtonStyle("play"));
    connect(btnPlay, SIGNAL(clicked()), this, SLOT(onBtnPlayClicked()));

    btnStop = new QPushButton();
    btnStop->setStyleSheet(getButtonStyle("stop"));
    connect(btnStop, SIGNAL(clicked()), this, SLOT(onBtnStopClicked()));

    btnMute = new QPushButton();
    MW->glWidget->setMute(MW->settings->value(muteKey, false).toBool());
    if (MW->glWidget->getMute())
        btnMute->setStyleSheet(getButtonStyle("mute"));
    else 
        btnMute->setStyleSheet(getButtonStyle("audio"));
    connect(btnMute, SIGNAL(clicked()), this, SLOT(onBtnMuteClicked()));

    sldVolume = new QSlider(Qt::Horizontal, this);
    sldVolume->setValue(MW->settings->value(volumeKey, 80).toInt());
    connect(sldVolume, SIGNAL(sliderMoved(int)), this, SLOT(onSldVolumeMoved(int)));

    sldProgress = new ProgressSlider(Qt::Horizontal, this);
    sldProgress->setMaximum(1000);
    connect(MW->glWidget, SIGNAL(progress(float)), this, SLOT(progress(float)));
    connect(sldProgress, SIGNAL(seek(float)), MW->glWidget, SLOT(seek(float)));

    QWidget *controlPanel = new QWidget(this);
    QGridLayout *controlLayout = new QGridLayout(controlPanel);
    controlLayout->addWidget(btnPlay,         0, 0, 1, 1);
    controlLayout->addWidget(btnStop,         0, 1, 1, 1);
    controlLayout->addWidget(btnMute,         0, 3, 1, 1);
    controlLayout->addWidget(sldVolume,       0, 4, 1, 2);
    controlLayout->addWidget(sldProgress,     1, 0, 1, 7);

    QGridLayout *layout = new QGridLayout(this);
    layout->addWidget(directorySetter,      0, 0, 1, 1);
    layout->addWidget(tree,                 1, 0, 1, 1);
    layout->addWidget(controlPanel,         2, 0, 1, 1);
    layout->setRowStretch(1, 20);

    QStringList list = QStandardPaths::standardLocations(QStandardPaths::MoviesLocation);
    QString path = MW->settings->value(dirKey, list[0]).toString();
    directorySetter->setPath(path);
    model->setRootPath(path);
    tree->setRootIndex(model->index(path));
    connect(directorySetter, SIGNAL(directorySet(const QString&)), this, SLOT(setDirectory(const QString&)));

    tree->header()->restoreState(MW->settings->value(headerKey).toByteArray());
    tree->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(tree, SIGNAL(customContextMenuRequested(const QPoint&)), this, SLOT(showContextMenu(const QPoint&)));
    connect(tree, SIGNAL(doubleClicked(const QModelIndex&)), this, SLOT(doubleClicked(const QModelIndex&)));
    connect(tree->header(), SIGNAL(sectionResized(int, int, int)), this, SLOT(headerChanged(int, int, int)));
    connect(tree->header(), SIGNAL(sectionMoved(int, int, int)), this, SLOT(headerChanged(int, int, int)));

    menu = new QMenu("Context Menu", this);
    QAction *remove = new QAction("Delete", this);
    QAction *rename = new QAction("Rename", this);
    QAction *info = new QAction("Info", this);
    QAction *play = new QAction("Play", this);
    connect(remove, SIGNAL(triggered()), this, SLOT(onMenuRemove()));
    connect(rename, SIGNAL(triggered()), this, SLOT(onMenuRename()));
    connect(info, SIGNAL(triggered()), this, SLOT(onMenuInfo()));
    connect(play, SIGNAL(triggered()), this, SLOT(onMenuPlay()));
    menu->addAction(remove);
    menu->addAction(rename);
    menu->addAction(info);
    menu->addAction(play);

    connect(this, SIGNAL(msg(const QString&)), mainWindow, SLOT(msg(const QString&)));
}

void FilePanel::setDirectory(const QString& path)
{
    directorySetter->setPath(path);
    model->setRootPath(path);
    tree->setRootIndex(model->index(path));
    MW->settings->setValue(dirKey, path);
}

void FilePanel::doubleClicked(const QModelIndex& index)
{
    if (index.isValid()) {
        QFileInfo fileInfo = model->fileInfo(index);
        if (fileInfo.isDir()) {
            bool expanded = tree->isExpanded(index);
            tree->setExpanded(index, !expanded);
        }
        else {
            MW->glWidget->play(fileInfo.filePath());
            btnPlay->setStyleSheet(getButtonStyle("pause"));
        }
    }
}

void FilePanel::headerChanged(int arg1, int arg2, int arg3)
{
    MW->settings->setValue(headerKey, tree->header()->saveState());
}

void FilePanel::progress(float arg)
{
    sldProgress->setValue(sldProgress->maximum() * arg);
}

void FilePanel::onBtnPlayClicked()
{
    if (MW->glWidget->running) {
        SDL_Event event;
        event.type = SDL_KEYDOWN;
        event.key.keysym.sym = SDLK_SPACE;
        SDL_PushEvent(&event);
        if (MW->glWidget->process->display->paused)
            btnPlay->setStyleSheet(getButtonStyle("pause"));
        else
            btnPlay->setStyleSheet(getButtonStyle("play"));
    }
    else {
        doubleClicked(tree->currentIndex());
    }
}

void FilePanel::onBtnStopClicked()
{
    btnPlay->setStyleSheet(getButtonStyle("play"));
    MW->glWidget->stop();
}

void FilePanel::onBtnMuteClicked()
{
    if (MW->glWidget->getMute()) {
        btnMute->setStyleSheet(getButtonStyle("audio"));
        MW->cameraPanel->btnMute->setStyleSheet(getButtonStyle("audio"));
    }
    else {
        btnMute->setStyleSheet(getButtonStyle("mute"));
        MW->cameraPanel->btnMute->setStyleSheet(getButtonStyle("mute"));
    }

    MW->glWidget->setMute(!MW->glWidget->getMute());
    MW->settings->setValue(muteKey, MW->glWidget->getMute());
}

void FilePanel::onSldVolumeMoved(int value)
{
    MW->glWidget->setVolume(value);
    MW->settings->setValue(volumeKey, value);
    MW->cameraPanel->volumeSlider->setValue(value);
}

QString FilePanel::getButtonStyle(const QString& name) const
{
    if (MW->styleDialog->panel->useSystemGui->isChecked())
        return QString("QPushButton {image:url(:%1_lo.png);}").arg(name);
    else
        return QString("QPushButton {image:url(:%1.png);} QPushButton:hover {image:url(:%1_hi.png);} QPushButton:pressed {image:url(:%1.png);}").arg(name);
}

void FilePanel::showContextMenu(const QPoint &pos)
{
    QModelIndex index = tree->indexAt(pos);
    if (index.isValid()) {
        menu->exec(mapToGlobal(pos));
    }
}

void FilePanel::onMenuPlay()
{
    doubleClicked(tree->currentIndex());
}

void FilePanel::onMenuRemove()
{
    QModelIndex index = tree->currentIndex();
    if (!index.isValid())
        return;

    int ret = QMessageBox::warning(this, "onvif-gui",
                                   "You are about to delete this file.\n"
                                   "Are you sure you want to continue?",
                                   QMessageBox::Ok | QMessageBox::Cancel);

    if (ret == QMessageBox::Ok)
        QFile::remove(model->filePath(tree->currentIndex()).toLatin1().data());
}

void FilePanel::onMenuRename()
{
    QModelIndex index = tree->currentIndex();
    if (index.isValid())
        tree->edit(index);
}

void FilePanel::onMenuInfo()
{
    QString str;
    QModelIndex idx = tree->currentIndex();
    if (idx.isValid()) {
        QFileInfo info = model->fileInfo(idx);
        str += "Filename: " + info.absoluteFilePath();
        str += "\nLast Modified: " + info.lastModified().toString();

        avio::Reader reader(info.absoluteFilePath().toLatin1().data());
        long duration = reader.duration();
        int time_in_seconds = duration / 1000;
        int hours = time_in_seconds / 3600;
        int minutes = (time_in_seconds - (hours * 3600)) / 60;
        int seconds = (time_in_seconds - (hours * 3600) - (minutes * 60));
        char buf[32] = {0};
        if (hours > 0)
            sprintf(buf, "%02d:%02d:%02d", hours, minutes, seconds);
        else 
            sprintf(buf, "%d:%02d", minutes, seconds);
        str += "\nDuration: " + QString(buf);

        if (reader.has_video()) {
            str += "\n\nVideo Stream:";
            str += "\n\tResolution: " + QString::number(reader.width()) + " x " + QString::number(reader.height());
            str += "\n\tFrame Rate: " + QString::number((float)reader.frame_rate().num / (float)reader.frame_rate().den);
            str += "\n\tVideo Codec: " + QString(reader.str_video_codec());
            str += "\n\tPixel Format: " + QString(reader.str_pix_fmt());
        }
        if (reader.has_audio()) {
            str += "\n\nAudio Stream:";
            str += "\n\tChannel Layout: " + QString(reader.str_channel_layout().c_str());
            str += "\n\tAudio Codec: " + QString(reader.str_audio_codec());
            str += "\n\tSample Rate: " + QString::number(reader.sample_rate());
        }
    }    
    else {
        str = "Invalid Index";
    }

    QMessageBox msgBox(this);
    msgBox.setText(str);
    msgBox.exec();
}

TreeView::TreeView(QWidget *parent) : QTreeView(parent)
{

}

void TreeView::keyPressEvent(QKeyEvent *event)
{
    switch (event->key()) {
        case (Qt::Key_Delete):
            ((FilePanel*)parent())->onMenuRemove();
            break;
        case (Qt::Key_Return):
            ((FilePanel*)parent())->doubleClicked(currentIndex());
            break;
        case (Qt::Key_Space):
            ((FilePanel*)parent())->onBtnPlayClicked();
            break;
        case (Qt::Key_Escape):
            ((FilePanel*)parent())->onBtnStopClicked();
            break;
        default:
            QTreeView::keyPressEvent(event);
    }
}

void TreeView::mouseDoubleClickEvent(QMouseEvent *event)
{
    emit doubleClicked(indexAt(event->pos()));
}

DirectorySetter::DirectorySetter(QMainWindow *parent, const QString& labelText)
{
    mainWindow = parent;
    label = new QLabel(labelText);
    text = new QLineEdit();
    button = new QPushButton("...");
    button->setMaximumWidth(30);
    connect(button, SIGNAL(clicked()), this, SLOT(selectDirectory()));

    QGridLayout *layout = new QGridLayout;
    layout->setContentsMargins(0, 0, 0, 0);
    if (label->text() != "")
        layout->addWidget(label,  0, 0, 1, 1);
    layout->addWidget(text,   0, 1, 1, 1);
    layout->addWidget(button, 0, 2, 1, 1);
    layout->setContentsMargins(0, 0, 0, 0);
    setLayout(layout);
    setContentsMargins(0, 0, 0, 0);
}

void DirectorySetter::setPath(const QString& path)
{
    directory = path;
    text->setText(path);
}

void DirectorySetter::selectDirectory()
{
    QString path = QFileDialog::getExistingDirectory(mainWindow, label->text(), directory,
                                                  QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
    if (path.length() > 0) {
        directory = path;
        text->setText(directory);
        emit directorySet(directory);
    }
}

ProgressSlider::ProgressSlider(Qt::Orientation o, QWidget *parent) : QSlider(o, parent)
{
    setMouseTracking(true); 
    filePanel = parent;
}

bool ProgressSlider::event(QEvent *e)
{
    if (e->type() == QEvent::Leave)
        QToolTip::hideText();

    return QSlider::event(e);
}

void ProgressSlider::mousePressEvent(QMouseEvent *event)
{
    float pct = event->pos().x() / (float)width();
    emit seek(pct);
}

void ProgressSlider::mouseMoveEvent(QMouseEvent *e)
{
    if (last_position_x != e->pos().x())
        QToolTip::hideText();

    MainWindow* mainWindow = (MainWindow*)(((FilePanel*)filePanel)->mainWindow);

    if (mainWindow->glWidget->media_duration) {
        double percentage = e->pos().x() / (double)width();
        double position = percentage * MW->glWidget->media_duration;

        int position_in_seconds = position / 1000;
        int hours = position_in_seconds / 3600;
        int minutes = (position_in_seconds - (hours * 3600)) / 60;
        int seconds = (position_in_seconds - (hours * 3600) - (minutes * 60));
        char buf[32] = {0};
        if (hours > 0)
            sprintf(buf, "%02d:%02d:%02d", hours, minutes, seconds);
        else 
            sprintf(buf, "%d:%02d", minutes, seconds);

        QString output(buf);

        const QPoint pos = mapToGlobal(QPoint(e->pos().x(), geometry().top() - 60));
        QToolTip::showText(pos, output);
        last_position_x = e->pos().x();
    }
}
