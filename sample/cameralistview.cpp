/*******************************************************************************
* cameralistview.cpp
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

#include "cameralistview.h"
#include "mainwindow.h"
#include <QMouseEvent>

CameraListView::CameraListView(QMainWindow *parent)
{
    mainWindow = parent;
    cameraListModel = new CameraListModel(mainWindow);
    setModel(cameraListModel);
    setFrameStyle(QFrame::Panel | QFrame::Sunken);
    connect(selectionModel(), SIGNAL(selectionChanged(QItemSelection, QItemSelection)), cameraListModel, SLOT(onSelectedItemsChanged(QItemSelection, QItemSelection)));
}

void CameraListView::mouseDoubleClickEvent(QMouseEvent *event)
{
    Q_UNUSED(event);
}

QModelIndex CameraListView::previousIndex() const
{
    QModelIndex previous;
    QModelIndex index = currentIndex();
    if (index.isValid()) {
        if (index.row() > 0) {
            QRect rect = rectForIndex(index);
            QPoint previous_center = QPoint(rect.center().x(), rect.center().y() - rect.height());
            previous = indexAt(previous_center);
        }
    }
    return previous;
}

QModelIndex CameraListView::nextIndex() const
{
    QModelIndex next;
    QModelIndex index = currentIndex();
    if (index.isValid()) {
        if (index.row() + 1 < model()->rowCount()) {
            QRect rect = rectForIndex(index);
            QPoint next_center = QPoint(rect.center().x(), rect.center().y() + rect.height());
            next = indexAt(next_center);
        }
    }
    return next;
}

void CameraListView::setCurrentCamera(const QString& cameraName)
{
    int row = -1;
    for (int i = 0; i < model()->rowCount(); i++) {
        Camera *camera = ((CameraListModel*)model())->getCameraAt(i);
        if (camera->getCameraName() == cameraName) {
            row = i;
            break;
        }
    }

    if (row > -1) {
        QModelIndex index = model()->index(row, 0);
        setCurrentIndex(index);
    }
}

Camera *CameraListView::getCurrentCamera()
{
    if (currentIndex().isValid())
        return (Camera*)((CameraListModel*)model())->cameras[currentIndex().row()];
    else
        return NULL;
}

void CameraListView::refresh()
{
    for (int i = 0; i < model()->rowCount(); i++) {
        Camera *camera = ((CameraListModel*)model())->getCameraAt(i);
    }
    model()->emit dataChanged(QModelIndex(), QModelIndex());
}

