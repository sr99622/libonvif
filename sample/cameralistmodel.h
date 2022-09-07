/*******************************************************************************
* cameralistmodel.h
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

#ifndef LISTMODEL_H
#define LISTMODEL_H

#include "camera.h"
#include <list>
#include <QVector>
#include <QAbstractListModel>
#include <QItemSelection>
#include <QMainWindow>

class CameraListModel : public QAbstractListModel
{
    Q_OBJECT

public:
    CameraListModel(QMainWindow *parent);
    int rowCount(const QModelIndex &parent = QModelIndex()) const override;
    QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const override;
    bool setData(const QModelIndex &index, const QVariant &value, int rolw = Qt::EditRole) override;
    Qt::ItemFlags flags(const QModelIndex &index) const override;
    void pushCamera(OnvifData *onvif_data);
    Camera * getCameraAt(int index);
    int current_index = -1;
    QVector<Camera *> cameras;
    QMainWindow *mainWindow;

signals:
    void getCameraData();
    void showCameraData();

public slots:
    void onSelectedItemsChanged(QItemSelection selected, QItemSelection deselected);

private slots:
    void beginInsertItems(int start, int end);
    void endInsertItems();


};

#endif // LISTMODEL_H
