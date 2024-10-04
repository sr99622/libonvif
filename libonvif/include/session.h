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
                    while (true) {
                        data = getCredential(data);
                        if (!data.cancelled) {

                            if (!getTimeOffset(data)) {
                                time_t rawtime;
                                struct tm timeinfo;
                                time(&rawtime);
                            #ifdef _WIN32
                                localtime_s(&timeinfo, &rawtime);
                            #else
                                localtime_r(&rawtime, &timeinfo);
                            #endif
                                if (timeinfo.tm_isdst && !data.dst())
                                    data.setTimeOffset(data.time_offset() - 3600);
                            }
                            getCapabilities(data);

                            if (getDeviceInformation(data) == 0) {
                                int index = 0;
                                while (true) {
                                    Data profile(data);
                                    getProfileToken(profile, index);
                                    if (strlen(profile->profileToken) == 0)
                                        break;
                                    getStreamUri(profile);
                                    data.profiles.push_back(profile);
                                    index++;
                                }
                                getData(data);
                                break;
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

    void getActiveInterfaces() { 
        getActiveNetworkInterfaces(session); 
    }

    std::string active_interface(int arg) { return session->active_network_interfaces[arg]; }

    std::function<void()> discovered = nullptr;
    std::function<Data(Data&)> getCredential = nullptr;
    std::function<void(Data&)> getData = nullptr;

    std::string interface;
    OnvifSession* session;
    bool abort = false;
};

}

#endif // SESSION_H
