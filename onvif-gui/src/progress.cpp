/*******************************************************************************
* progress.cpp
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
#include <QGridLayout>
#include <QFontMetrics>
#include <QMouseEvent>
#include <QPainter>
#include "progress.h"

Progress::Progress(QWidget* parent) : QWidget(parent)
{
    sldProgress = new ProgressSlider(Qt::Horizontal, this);

    lblProgress = new QLabel("0:00", this);
    setLabelWidth(lblProgress);

    lblDuration = new QLabel("0:00", this);
    setLabelWidth(lblDuration);

    lblPosition = new PositionLabel(this);
    setLabelHeight(lblPosition);

    QGridLayout* lytMain = new QGridLayout(this);
    lytMain->addWidget(lblPosition,   0, 1, 1, 1);
    lytMain->addWidget(lblProgress,   1, 0, 1, 1);
    lytMain->addWidget(sldProgress,   1, 1, 1, 1);
    lytMain->addWidget(lblDuration,   1, 2, 1, 1);
    lytMain->setColumnStretch(1, 10);
    lytMain->setContentsMargins(0, 0, 0, 0);
}

void Progress::setLabelWidth(QLabel* lbl) 
{
    lbl->setFixedWidth(lbl->fontMetrics().boundingRect("00:00:00").width());
    lbl->setAlignment(Qt::AlignCenter);
}

void Progress::setLabelHeight(QLabel* lbl) 
{
    lbl->setFixedHeight(lbl->fontMetrics().boundingRect("00:00:00").height());
}

QString Progress::getTimeString(qint64 milliseconds) const
{
    QString output;
    int position_in_seconds = milliseconds / 1000;
    int hours = position_in_seconds / 3600;
    int minutes = (position_in_seconds - (hours * 3600)) / 60;
    int seconds = (position_in_seconds - (hours * 3600) - (minutes * 60));
    char buf[32] = {0};
    if (hours > 0)
        sprintf(buf, "%02d:%02d:%02d", hours, minutes, seconds);
    else 
        sprintf(buf, "%d:%02d", minutes, seconds);

    output = QString(buf);
    return output;
}

void Progress::setDuration(qint64 arg) 
{
    if (arg > 0) {
        m_duration = arg;
        lblDuration->setText(getTimeString(arg));
    }
}

void Progress::setProgress(float pct) 
{
    sldProgress->setValue(sldProgress->maximum() * pct);
    lblProgress->setText(getTimeString((qint64)(m_duration * pct)));
}

ProgressSlider::ProgressSlider(Qt::Orientation orientation, QWidget* parent) : QSlider(orientation, parent)
{
    setMaximum(1000);
    setMouseTracking(true);
}

bool ProgressSlider::event(QEvent* event)
{
    if (event->type() == QEvent::Leave)
        ((Progress*)parent())->lblPosition->setText("", 0);
    return QSlider::event(event);
}

void ProgressSlider::mouseMoveEvent(QMouseEvent* event)
{
    Progress* p = (Progress*)parent();
    qint64 duration = p->duration();
    int x = event->pos().x();
    if (duration) {
        double percentage = x / (double)width();
        double position = percentage * duration;
        p->lblPosition->setText(p->getTimeString((int)position), x);
    }
}

void ProgressSlider::mousePressEvent(QMouseEvent* event)
{
    float pct = event->pos().x() / (float)width();
    ((Progress*)parent())->emit seek(pct);
}

PositionLabel::PositionLabel(QWidget* parent) : QLabel(parent)
{

}

void PositionLabel::setText(const QString& arg, int pos)
{
    QLabel::setText(arg);
    m_pos = pos;
}

void PositionLabel::paintEvent(QPaintEvent *event)
{
    QPainter painter;
    painter.begin(this);
    QFontMetrics metrics = fontMetrics();
    QRect rect = metrics.boundingRect(text());
    int x = std::min(width() - rect.width(), m_pos);
    painter.drawText(QPoint(x, height()), text());
    painter.end();
}
