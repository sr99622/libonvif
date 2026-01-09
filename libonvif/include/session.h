/*******************************************************************************
* session.h
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

#ifndef SESSION_H
#define SESSION_H

#include <vector>
#include <functional>
#include <iostream>
#include <string.h>
#include "onvif.h"

namespace libonvif
{

class Session
{
public:
    Session() 
    { 
        session = (OnvifSession*)calloc(sizeof(OnvifSession), 1);
        initializeSession(session);
    }

    ~Session() 
    {
        closeSession(session);
        free(session);
    }

    operator OnvifSession* ()
    {
        return session;
    }

    void startDiscover()
    {
        std::thread thread([&]() { discover(); });
        thread.detach();
    }

    void discover()
    {
        if (interface.length() > 0) {
            memset(session->preferred_network_address, 0, 16);
            strcpy(session->preferred_network_address, interface.c_str());
        }
        int number_of_devices = broadcast(session);
        std::vector<Data> devices;
        for (int i = 0; i < number_of_devices; i++) {
            if (abort) break;
            Data data;
            if (prepareOnvifData(i, session, data)) {
                if (std::find(devices.begin(), devices.end(), data) == devices.end()) {
                    devices.push_back(data);
                }
            }
            else {
                std::stringstream str;
                str << "session was unable to parse device onvif data: " << session->buf[i];
                infoCallback(str.str());
            }
        }

        std::stringstream str;
        str << " ONVIF Discovery found " << devices.size() << " unique devices on network: " << interface;
        if (infoCallback) infoCallback(str.str());

        for (int i = 0; i < devices.size(); i++) {
            Data data = devices[i];
            if (getTimeOffset(data)) {
                std::stringstream str;
                str << "Camera get time offset error " << data.camera_name() << " : " << data.xaddrs() << " : " << data.last_error();
                if (infoCallback) infoCallback(str.str());
            }
            time_t initial_offset = data->time_offset;
            int count = 1;
            int direction = 1;

            while (true) {
                data = getCredential(data);
                if (!data.cancelled) {

                    if (getCapabilities(data)) {
                        // Daylight Savings Time mismatch can cause authentication failure
                        std::stringstream str;
                        str << "session error: " << data->camera_name << " : " << data->last_error << " attempting time offset correction";
                        infoCallback(str.str());
                        data->time_offset = initial_offset + 3600 * direction * count;
                        if (direction < 0) count++;
                        direction = ~direction + 1;
                        if (count > 2) {
                            std::stringstream str;
                            str << "session error: " << data->camera_name << " : " << data->last_error << " camera interrogation failed";
                            infoCallback(str.str());
                            break;
                        }
                        else {
                            continue;
                        }
                    }

                    if (getDeviceInformation(data) == 0) {
                        int index = 0;
                        while (true) {
                            Data profile(data);
                            getProfileToken(profile, index);
                            if (strlen(profile->profileToken) == 0)
                                break;
                            getStreamUri(profile);
                            getSnapshotUri(profile);
                            data.profiles.push_back(profile);
                            index++;
                        }
                        getData(data);
                        break;
                    }
                    else {
                        std::stringstream str;
                        str << "get device information failed: " << data->camera_name << " : " << data->last_error;
                        infoCallback(str.str());
                    }
                } 
                else {
                    break;
                }
            }
        }

        discovered();
    }

    void getActiveInterfaces() { 
        getActiveNetworkInterfaces(session); 
    }

    std::string active_interface(int arg) { return session->active_network_interfaces[arg]; }
    std::string primary_network_interface() { return session->primary_network_interface; }

    std::function<void()> discovered = nullptr;
    std::function<Data(Data&)> getCredential = nullptr;
    std::function<void(Data&)> getData = nullptr;
    std::function<void(const std::string&)> infoCallback = nullptr;

    std::string interface;
    OnvifSession* session;
    bool abort = false;
};

}

#endif // SESSION_H
