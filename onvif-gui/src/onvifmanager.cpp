/*******************************************************************************
* onvifmanager.cpp
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
#include "onvifmanager.h"
#include "camerapanel.h"

Filler::Filler(QWidget *parent)
{
    cameraPanel = parent;
    setAutoDelete(false);
}

void Filler::run()
{
    OnvifData *onvif_data = ((CameraPanel *)cameraPanel)->camera->onvif_data;
    getCapabilities(onvif_data);
    getNetworkInterfaces(onvif_data);
    getNetworkDefaultGateway(onvif_data);
    getDNS(onvif_data);
    getVideoEncoderConfigurationOptions(onvif_data);
    getVideoEncoderConfiguration(onvif_data);
    getOptions(onvif_data);
    getImagingSettings(onvif_data);

    emit done();
}

VideoUpdater::VideoUpdater(QWidget *parent)
{
    cameraPanel = parent;
    setAutoDelete(false);
}

void VideoUpdater::run()
{
    OnvifData *onvif_data = ((CameraPanel *)cameraPanel)->camera->onvif_data;
    setVideoEncoderConfiguration(onvif_data);
    getProfile(onvif_data);
    getVideoEncoderConfigurationOptions(onvif_data);
    getVideoEncoderConfiguration(onvif_data);
    emit done();
}

ImageUpdater::ImageUpdater(QWidget *parent)
{
    cameraPanel = parent;
    setAutoDelete(false);
}

void ImageUpdater::run()
{
    OnvifData *onvif_data = ((CameraPanel *)cameraPanel)->camera->onvif_data;
    setImagingSettings(onvif_data);
    getOptions(onvif_data);
    getImagingSettings(onvif_data);
    emit done();
}

NetworkUpdater::NetworkUpdater(QWidget *parent)
{
    cameraPanel = parent;
    setAutoDelete(false);
}

void NetworkUpdater::run()
{
    OnvifData *onvif_data = ((CameraPanel *)cameraPanel)->camera->onvif_data;
    setNetworkInterfaces(onvif_data);
    setDNS(onvif_data);
    setNetworkDefaultGateway(onvif_data);
    emit done();
}

Rebooter::Rebooter(QWidget *parent)
{
    cameraPanel = parent;
    setAutoDelete(false);
}

void Rebooter::run()
{
    OnvifData *onvif_data = ((CameraPanel *)cameraPanel)->camera->onvif_data;
    rebootCamera(onvif_data);
    emit done();
}

Resetter::Resetter(QWidget *parent)
{
    cameraPanel = parent;
    setAutoDelete(false);
}

void Resetter::run()
{
    OnvifData *onvif_data = ((CameraPanel *)cameraPanel)->camera->onvif_data;
    hardReset(onvif_data);
    emit done();
}

Timesetter::Timesetter(QWidget *parent)
{
    cameraPanel = parent;
    setAutoDelete(false);
}

void Timesetter::run()
{
    OnvifData *onvif_data = ((CameraPanel *)cameraPanel)->camera->onvif_data;
    setSystemDateAndTime(onvif_data);
    emit done();
}

PTZMover::PTZMover(QWidget *parent)
{
    cameraPanel = parent;
    setAutoDelete(false);
}

void PTZMover::set(float x_arg, float y_arg, float z_arg)
{
    x = x_arg; y = y_arg; z = z_arg;
}

void PTZMover::run()
{
    OnvifData *onvif_data = ((CameraPanel *)cameraPanel)->camera->onvif_data;
    continuousMove(x, y, z, onvif_data);
}

PTZStopper::PTZStopper(QWidget *parent)
{
    cameraPanel = parent;
    setAutoDelete(false);
}

void PTZStopper::set(int type_arg)
{
    type = type_arg;
}

void PTZStopper::run()
{
    OnvifData *onvif_data = ((CameraPanel *)cameraPanel)->camera->onvif_data;
    moveStop(type, onvif_data);
}

PTZGoto::PTZGoto(QWidget *parent)
{
    cameraPanel = parent;
    setAutoDelete(false);
}

void PTZGoto::set(int position_arg)
{
    position = position_arg;
}

void PTZGoto::run()
{
    char pos[128] = {0};
    sprintf(pos, "%d", position);
    OnvifData *onvif_data = ((CameraPanel *)cameraPanel)->camera->onvif_data;
    gotoPreset(pos, onvif_data);
}

PTZSetPreset::PTZSetPreset(QWidget *parent)
{
    cameraPanel = parent;
    setAutoDelete(false);
}

void PTZSetPreset::set(int position_arg)
{
    position = position_arg;
}

void PTZSetPreset::run()
{
    char pos[128] = {0};
    sprintf(pos, "%d", position);
    OnvifData *onvif_data = ((CameraPanel *)cameraPanel)->camera->onvif_data;
    setPreset(pos, onvif_data);
}
