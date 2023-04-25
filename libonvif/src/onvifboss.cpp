/*******************************************************************************
* onvifboss.cpp
*
* copyright 2023 Stephen Rhodes
*
* This library is free software; you can redistribute it and/or
* modify it under the terms of the GNU Lesser General Public
* License as published by the Free Software Foundation; either
* version 2.1 of the License, or (at your option) any later version.
*
* This library is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
* Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public
* License along with this library; if not, write to the Free Software
* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
*
*******************************************************************************/

#include <iostream>
#include "onvifboss.h"

namespace libonvif
{

Manager::Manager()
{

}

Manager::~Manager()
{

}

void Manager::startStop()
{
    std::thread thread([&]() { stop(); });
    thread.detach();
}

void Manager::stop()
{
    moveStop(stop_type, onvif_data.data);
}

void Manager::startMove()
{
    std::thread thread([&]() { move(); });
    thread.detach();
}

void Manager::move()
{
    continuousMove(x, y, z, onvif_data.data);
}

void Manager::startSet()
{
    std::thread thread([&]() { set(); });
    thread.detach();
}

void Manager::set()
{
    char pos[128] = {0};
    sprintf(pos, "%d", preset);
    gotoPreset(pos, onvif_data.data);
}

void Manager::startSetGotoPreset()
{
    std::thread thread([&]() { setGotoPreset(); });
    thread.detach();
}

void Manager::setGotoPreset()
{
    char pos[128] = {0};
    sprintf(pos, "%d", preset);
    setPreset(pos, onvif_data.data);
    filled(onvif_data);
}

void Manager::startUpdateVideo()
{
    std::thread thread([&]() { updateVideo(); });
    thread.detach();
}

void Manager::updateVideo()
{
    setVideoEncoderConfiguration(onvif_data.data);
    getProfile(onvif_data.data);
    getVideoEncoderConfigurationOptions(onvif_data.data);
    getVideoEncoderConfiguration(onvif_data.data);
    filled(onvif_data);
}

void Manager::startUpdateImage()
{
    std::thread thread([&]() { updateImage(); });
    thread.detach();
}

void Manager::updateImage()
{
    setImagingSettings(onvif_data.data);
    getOptions(onvif_data.data);
    getImagingSettings(onvif_data.data);
    filled(onvif_data);
}

void Manager::startUpdateNetwork()
{
    std::thread thread([&]() { updateNetwork(); });
    thread.detach();
}

void Manager::updateNetwork()
{
    setNetworkInterfaces(onvif_data.data);
    setDNS(onvif_data.data);
    setNetworkDefaultGateway(onvif_data.data);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    getNetworkInterfaces(onvif_data.data);
    getNetworkDefaultGateway(onvif_data.data);
    getDNS(onvif_data.data);
    filled(onvif_data);
}

void Manager::startUpdateTime()
{
    std::thread thread([&]() { updateTime(); });
    thread.detach();
}

void Manager::updateTime()
{
    setSystemDateAndTime(onvif_data.data);
    filled(onvif_data);
}

void Manager::startReboot()
{
    std::thread thread([&]() { reboot(); });
    thread.detach();
}

void Manager::reboot()
{
    rebootCamera(onvif_data.data);
    filled(onvif_data);
}

void Manager::startReset()
{
    std::thread thread([&]() { reset(); });
    thread.detach();
}

void Manager::reset()
{
    hardReset(onvif_data.data);
    filled(onvif_data);
}

void Manager::startSetUser()
{
    std::thread thread([&]() { setOnvifUser(); });
    thread.detach();
}

void Manager::setOnvifUser()
{
    if (setUser((char*)new_password.c_str(), onvif_data) == 0)
        onvif_data.setPassword(new_password.c_str());
    filled(onvif_data);
}

void Manager::startFill()
{
    std::thread thread([&]() { fill(); });
    thread.detach();
}

void Manager::fill()
{
    getCapabilities(onvif_data.data);
    getNetworkInterfaces(onvif_data.data);
    getNetworkDefaultGateway(onvif_data.data);
    getDNS(onvif_data.data);
    getVideoEncoderConfigurationOptions(onvif_data.data);
    getVideoEncoderConfiguration(onvif_data.data);
    getOptions(onvif_data.data);
    getImagingSettings(onvif_data.data);
    onvif_data.filled = true;
    filled(onvif_data);
}

void Manager::startDiscover()
{
    std::thread thread([&]() { discover(); });
    thread.detach();
}

void Manager::discover()
{
    Session session;
    if (interface.length() > 0) {
        memset(session.session->preferred_network_address, 0, 16);
        strcpy(session.session->preferred_network_address, interface.c_str());
    }
    int number_of_devices = broadcast(session);
    std::vector<Data> devices;
    //std::cout << "number_of_devices: " << number_of_devices << std::endl;

    for (int i = 0; i < number_of_devices; i++) {
        //std::cout << "process broadcast return data" << std::endl;
        Data data;
        if (prepareOnvifData(i, session, data)) {
            if (std::find(devices.begin(), devices.end(), data) == devices.end()) {
                devices.push_back(data);
                bool first_pass = true;
                while (true) {
                    //std::cout << "get credential" << std::endl;
                    data = getCredential(data);
                    if (!data.cancelled) {
                        if (fillRTSP(data) == 0) {
                            //std::cout << "RTSP founc" << std::endl;
                            getProfile(data);
                            getDeviceInformation(data);
                            getData(data);
                            //std::cout << "got data for " << data.camera_name() << std::endl;
                            break;
                        }
                        else {
                            getTimeOffset(data);
                            time_t rawtime;
                            struct tm timeinfo;
                            time(&rawtime);
                        #ifdef _WIN32
                            localtime_s(&timeinfo, &rawtime);
                        #else
                            localtime_r(&rawtime, &timeinfo);
                        #endif
                            if (timeinfo.tm_isdst) {
                                data.setTimeOffset(data.time_offset() - 3600);
                                if (first_pass) {
                                    data.clearLastError();
                                    first_pass = false;
                                }
                            }
                        }
                    } 
                    else {
                        break;
                    }
                }
            }
        }
    }

    discovered();
}

}
