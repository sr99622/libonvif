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
#include <QMessageBox>

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

    QString path = MW->settings->value(dirKey).toString();
    directorySetter->setPath(path);
    model->setRootPath(path);
    tree->setRootIndex(model->index(path));
    connect(directorySetter, SIGNAL(directorySet(const QString&)), this, SLOT(setDirectory(const QString&)));

    if (MW->settings->contains(headerKey))
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

    int ret = QMessageBox::warning(this, "playqt",
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
    /*
    QModelIndex index = tree->currentIndex();
    if (!index.isValid())
        return;

    QString filename = model->filePath(tree->currentIndex());
    AVFormatContext *fmt_ctx = nullptr;
    AVStream *video;
    AVStream *audio;
    int video_stream;
    int audio_stream;

    try {
        av.ck(avformat_open_input(&fmt_ctx, filename.toLatin1().data(), NULL, NULL), AOI);
        av.ck(avformat_find_stream_info(fmt_ctx, NULL), AFSI);
    }
    catch (AVException *e) {
        emit msg(QString("Unable to open format context %1: %2\n").arg(av.tag(e->cmd_tag), e->error_text));
        return;
    }

    try {
        av.ck(audio_stream = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_AUDIO, -1, -1, NULL, 0), AFBS);
        audio = fmt_ctx->streams[audio_stream];
        QString str = "File audio parameters\n";

        char buf[16];

        QString codec_str;
        const AVCodecDescriptor *cd = avcodec_descriptor_get(audio->codecpar->codec_id);
        if (cd) {
            QTextStream(&codec_str) << "codec_name: " << cd->name << "\n"
                                    << "codec_long_name: " << cd->long_name << "\n";
        }
        else {
            QTextStream(&codec_str) << "Uknown codec" << "\n";
        }

        if (fmt_ctx->metadata == NULL) {
            str.append("\nmetadata is NULL\n");
        }
        else {
            QTextStream(&str) << "\n";
            AVDictionaryEntry *t = NULL;
            while (t = av_dict_get(fmt_ctx->metadata, "", t, AV_DICT_IGNORE_SUFFIX)) {
                QTextStream(&codec_str) << t->key << " : " << t->value << "\n";
            }
        }

        QTextStream(&str)
            << "filename: " << filename << "\n"
            << codec_str

            << "format: " << fmt_ctx->iformat->long_name << " (" << fmt_ctx->iformat->name << ")\n"
            << "flags: " << fmt_ctx->iformat->flags << "\n"
            << "extradata_size: " << audio->codecpar->extradata_size << "\n"
            << "codec time_base:  " << audio->codec->time_base.num << " / " << audio->codec->time_base.den << "\n"
            << "audio time_base: " << audio->time_base.num << " / " << audio->time_base.den << "\n"
            << "codec framerate: " << audio->codec->framerate.num << " / " << audio->codec->framerate.den << "\n"
            << "ticks_per_frame: " << audio->codec->ticks_per_frame << "\n"
            << "bit_rate: " << fmt_ctx->bit_rate << "\n"
            << "codec framerate: " << av_q2d(audio->codec->framerate) << "\n"
            << "start_time: " << fmt_ctx->start_time * av_q2d(av_get_time_base_q()) << "\n"
            << "duration: " << fmt_ctx->duration * av_q2d(av_get_time_base_q()) << "\n";

        emit msg(str);
        MW->messageDialog->show();
    }
    catch (AVException *e) {
        emit msg(QString("Unable to process audio stream %1: %2\n").arg(av.tag(e->cmd_tag), e->error_text));
    }

    try {
        av.ck(video_stream = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0), AFBS);
        video = fmt_ctx->streams[video_stream];

        QString str = "File video parameters\n";

        QString codec_str;
        const AVCodecDescriptor *cd = avcodec_descriptor_get(video->codecpar->codec_id);
        if (cd) {
            QTextStream(&codec_str) << "codec_name: " << cd->name << "\n"
                                    << "codec_long_name: " << cd->long_name << "\n";
        }
        else {
            QTextStream(&codec_str) << "Uknown codec" << "\n";
        }

        if (fmt_ctx->metadata == NULL) {
            str.append("\nmetadata is NULL\n");
        }
        else {
            QTextStream(&str) << "\n";
            AVDictionaryEntry *t = NULL;
            while (t = av_dict_get(fmt_ctx->metadata, "", t, AV_DICT_IGNORE_SUFFIX)) {
                QTextStream(&codec_str) << t->key << " : " << t->value << "\n";
            }
        }

        QTextStream(&str)
            << "filename: " << filename << "\n"
            << "pixel format: " << av_get_pix_fmt_name((AVPixelFormat)video->codecpar->format) << "\n"
            << codec_str
            << "format: " << fmt_ctx->iformat->long_name << " (" << fmt_ctx->iformat->name << ")\n"
            << "flags: " << fmt_ctx->iformat->flags << "\n"
            << "extradata_size: " << video->codecpar->extradata_size << "\n"
            << "codec time_base:  " << video->codec->time_base.num << " / " << video->codec->time_base.den << "\n"
            << "video time_base: " << video->time_base.num << " / " << video->time_base.den << "\n"
            << "codec framerate: " << video->codec->framerate.num << " / " << video->codec->framerate.den << "\n"
            << "gop_size: " << video->codec->gop_size << "\n"
            << "ticks_per_frame: " << video->codec->ticks_per_frame << "\n"
            << "bit_rate: " << fmt_ctx->bit_rate << "\n"
            << "codec framerate: " << av_q2d(video->codec->framerate) << "\n"
            << "start_time: " << fmt_ctx->start_time * av_q2d(av_get_time_base_q()) << "\n"
            << "duration: " << fmt_ctx->duration * av_q2d(av_get_time_base_q()) << "\n"
            << "width: " << video->codecpar->width << "\n"
            << "height: " << video->codecpar->height << "\n";

        emit msg(str);
        MW->messageDialog->show();
    }
    catch (AVException *e) {
        emit msg(QString("Unable to process video stream %1: %2\n").arg(av.tag(e->cmd_tag), e->error_text));
    }

    if (fmt_ctx != nullptr)
        avformat_close_input(&fmt_ctx);
    */
}

TreeView::TreeView(QWidget *parent) : QTreeView(parent)
{

}

void TreeView::keyPressEvent(QKeyEvent *event)
{
    if (event->key() == Qt::Key_Delete) {
        ((FilePanel*)parent())->onMenuRemove();
    }
    else {
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

void ProgressSlider::mousePressEvent(QMouseEvent *event)
{
    float pct = event->pos().x() / (float)width();
    emit seek(pct);
}
