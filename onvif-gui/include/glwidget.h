/********************************************************************
* libavio/samples/gui/include/glwidget.h
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

#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <QOpenGLWidget>
#include <QMutex>
#include <QRect>
#include <QImage>
#include "avio.h"

class GLWidget : public QOpenGLWidget
{
    Q_OBJECT

public:
    GLWidget();
    ~GLWidget();
    void renderCallback(const avio::Frame& frame);
    QSize sizeHint() const override;
    QRect getImageRect(const QImage& img) const;
    void clear();

    avio::Frame f;
    QImage img;
    QMutex mutex;

protected:
    void paintGL() override;

};

#endif // GLWIDGET_H