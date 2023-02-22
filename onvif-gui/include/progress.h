/*******************************************************************************
* progress.h
*
* Copyright (c) 2023 Stephen Rhodes
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

#ifndef PROGRESS_H
#define PROGRESS_H

#include <QSlider>
#include <QLabel>

class ProgressSlider : public QSlider
{
    Q_OBJECT

public:
    ProgressSlider(Qt::Orientation orientation, QWidget* parent = nullptr);

protected:
    bool event(QEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;

};

class PositionLabel : public QLabel
{
    Q_OBJECT

public:
    PositionLabel(QWidget* parent);
    void setText(const QString& arg, int pos);

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    int m_pos = 0;

};

class Progress : public QWidget
{
    Q_OBJECT

public:
    Progress(QWidget* parent = nullptr);

    void setDuration(qint64 arg);
    qint64 duration() { return m_duration; }
    void setProgress(float pct);
    QString getTimeString(qint64 milliseconds) const;

    ProgressSlider* sldProgress;
    QLabel* lblProgress;
    QLabel* lblDuration;
    PositionLabel* lblPosition;

signals:
    void seek(float);

private:
    void setLabelWidth(QLabel* lbl);
    void setLabelHeight(QLabel* lbl);
    qint64 m_duration = 0;

};


#endif // PROGRESS_H