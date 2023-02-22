/********************************************************************
* libavio/samples/gui/src/glwidget.cpp
*
* Copyright (c) 2023  Stephen Rhodes
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*    http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
*********************************************************************/

#include <iostream>
#include <QPainter>
#include <QImage>

#include "glwidget.h"

GLWidget::GLWidget()
{

}

GLWidget::~GLWidget()
{

}

QRect GLWidget::getImageRect(const QImage& img) const
{
    float ratio = std::min((float)width() / (float)img.width(), (float)height() / (float)img.height());
    float w = (float)img.width() * ratio;
    float h = (float)img.height() * ratio;
    float x = ((float)width() - w) / 2.0f;
    float y = ((float)height() - h) / 2.0f;
    return QRect((int)x, (int)y, (int)w, (int)h);
}

void GLWidget::paintGL()
{
    if (!img.isNull()) {
        mutex.lock();
        QPainter painter;
        painter.begin(this);
        painter.drawImage(getImageRect(img), img);
        painter.end();
        mutex.unlock();
    }
}

void GLWidget::renderCallback(const avio::Frame& frame)
{
    if (!frame.isValid()) {
        std:: cout << "render callback recvd invalid Frame" << std::endl;
        return;
    }
    mutex.lock();
    f = std::move(frame);
    img = QImage(f.m_frame->data[0], f.m_frame->width, f.m_frame->height, QImage::Format_RGB888);
    mutex.unlock();
    update();
}

QSize GLWidget::sizeHint() const
{
    return QSize(640, 480);
}

void GLWidget::clear()
{
    img.fill(0);
    update();
}