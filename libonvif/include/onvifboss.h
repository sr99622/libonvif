/*******************************************************************************
* onvifboss.h
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

#ifndef ONVIFBOSS_H
#define ONVIFBOSS_H

#include <thread>
#include <vector>
#include <functional>
#include <iostream>
#include <string.h>
#include "onvif.h"

namespace libonvif
{

class Data
{
public:
    Data() 
    {
        data = (OnvifData*)calloc(sizeof(OnvifData), 1);
        clearData(data);
    }

    Data(OnvifData* onvif_data)
    {
        data = onvif_data;
    }

    Data(const Data& other)
    {
        data = (OnvifData*)calloc(sizeof(OnvifData), 1);
        copyData(data, other.data);
        cancelled = other.cancelled;
        filled = other.filled;
        alias = other.alias;
        ipAddressChanged = other.ipAddressChanged;
    }

    Data(Data&& other) noexcept
    {
        data = other.data;
        other.data = nullptr;
        cancelled = other.cancelled;
        filled = other.filled;
        alias = other.alias;
        ipAddressChanged = other.ipAddressChanged;
    }

    Data& operator=(const Data& other)
    {
        if (!data) data = (OnvifData*)calloc(sizeof(OnvifData), 1);
        copyData(data, other.data);
        cancelled = other.cancelled;
        filled = other.filled;
        alias = other.alias;
        ipAddressChanged = other.ipAddressChanged;
        return *this;
    }

    Data& operator=(Data&& other) noexcept
    {
        if (data) free(data);
        data = other.data;
        cancelled = other.cancelled;
        filled = other.filled;
        alias = other.alias;
        other.data = nullptr;
        ipAddressChanged = other.ipAddressChanged;
        return *this;
    }

    bool operator==(const Data& rhs)
    {
        if (strcmp(data->xaddrs, rhs.data->xaddrs)) {
            return false;
        }
        else {
            return true;
        }
    }

    bool operator!=(const Data& rhs)
    {
        if (strcmp(data->xaddrs, rhs.data->xaddrs)) {
            return true;
        }
        else {
            return false;
        }
    }

    bool friend operator==(const Data& lhs, const Data& rhs)
    {
        if (strcmp(lhs.data->xaddrs, rhs.data->xaddrs)) {
            return false;
        }
        else {
            return true;
        }
    }

    bool friend operator!=(const Data& lhs, const Data& rhs)
    {
        if (strcmp(lhs.data->xaddrs, rhs.data->xaddrs)) {
            return true;
        }
        else {
            return false;
        }
    }

    ~Data() 
    {
        free(data);
    }

    operator OnvifData* ()
    {
        return data;
    }

    OnvifData* operator ->()
    {
        return data;
    }

    OnvifData* data;
    bool cancelled = false;
    bool filled = false;
    bool ipAddressChanged = false;
    std::string alias;

    bool isValid() const { return data ? true : false; }

    std::string username() { return data->username; } const
    void setUsername(const std::string& arg) { 
        memset(data->username, 0, 128);
        strncpy(data->username, arg.c_str(), arg.length()); 
    }

    std::string password() { return data->password; } const
    void setPassword(const std::string& arg) { 
        memset(data->password, 0, 128);
        strncpy(data->password, arg.c_str(), arg.length()); 
    }

    std::string xaddrs() { return data->xaddrs; } const
    std::string stream_uri() { return data->stream_uri; } const
    std::string serial_number() { return data->serial_number; } const
    std::string camera_name() { return data->camera_name; } const
    std::string host() { return data->host; } const
    std::string last_error() { return data->last_error; } const
    void clearLastError() { memset(data->last_error, 0, 1024); }
    time_t time_offset() { return data->time_offset; } const
    void setTimeOffset(time_t arg) { data->time_offset = arg; }
    void setTimeOffset2(time_t arg) { data->time_offset = arg; }
    void clear(int bug) { 
        clearData(data); 
        alias = "";
    }

    //VIDEO
    std::string resolutions_buf(int arg) { return data->resolutions_buf[arg]; }
    int width() { return data->width; }
    void setWidth(int arg) { data->width = arg; }
    int height() { return data->height; }
    void setHeight(int arg) { data->height = arg; }
    int frame_rate_max() { return data->frame_rate_max; }
    int frame_rate_min() { return data->frame_rate_min; }
    int frame_rate() { return data->frame_rate; }
    void setFrameRate(int arg) { data->frame_rate = arg; }
    int gov_length_max() { return data->gov_length_max; }
    int gov_length_min() { return data->gov_length_min; }
    int gov_length() { return data->gov_length; }
    void setGovLength(int arg) { data->gov_length = arg; }
    int bitrate_max() { return data->bitrate_max; }
    int bitrate_min() { return data->bitrate_min; }
    int bitrate() { return data->bitrate; }
    void setBitrate(int arg) { data->bitrate = arg; }

    //IMAGE
    int brightness_max() { return data->brightness_max; }
    int brightness_min() { return data->brightness_min; }
    int brightness() { return data->brightness; }
    void setBrightness(int arg) { data->brightness = arg; }
    int saturation_max() { return data->saturation_max; }
    int saturation_min() { return data->saturation_min; }
    int saturation() { return data->saturation; }
    void setSaturation(int arg) { data->saturation = arg; }
    int contrast_max() { return data->contrast_max; }
    int contrast_min() { return data->contrast_min; }
    int contrast() { return data->contrast; }
    void setContrast(int arg) { data->contrast = arg; }
    int sharpness_max() { return data->sharpness_max; }
    int sharpness_min() { return data->sharpness_min; }
    int sharpness() { return data->sharpness; }
    void setSharpness(int arg) { data->sharpness = arg; }

    //NETWORK
    bool dhcp_enabled() { return data->dhcp_enabled; }
    void setDHCPEnabled(bool arg) { data->dhcp_enabled = arg; }
    std::string ip_address_buf() { return data->ip_address_buf; } const
    void setIPAddressBuf(const std::string& arg) {
        memset(data->ip_address_buf, 0, 128);
        strncpy(data->ip_address_buf, arg.c_str(), arg.length());
    }
    std::string default_gateway_buf() { return data->default_gateway_buf; } const
    void setDefaultGatewayBuf(const std::string& arg) {
        memset(data->default_gateway_buf, 0, 128);
        strncpy(data->default_gateway_buf, arg.c_str(), arg.length());
    }
    std::string dns_buf() { return data->dns_buf; } const
    void setDNSBuf(const std::string& arg) {
        memset(data->dns_buf, 0, 128);
        strncpy(data->dns_buf, arg.c_str(), arg.length());
    }
    int prefix_length() { return data->prefix_length; }
    void setPrefixLength(int arg) { data->prefix_length = arg; }
    std::string mask_buf() { 
        memset(data->mask_buf, 0, 128);
        prefix2mask(data->prefix_length, data->mask_buf);
        return data->mask_buf;
    } const
    void setMaskBuf(const std::string& arg) {
        data->prefix_length = mask2prefix((char*)arg.c_str());
    }

};

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

    void getActiveInterfaces() { 
        getActiveNetworkInterfaces(session); 
    }
    std::string active_interface(int arg) { return session->active_network_interfaces[arg]; }

    OnvifSession* session;
};

class Manager
{
public:
    Manager();
    ~Manager();

    std::function<void()> discovered = nullptr;
    std::function<Data(Data&)> getCredential = nullptr;
    std::function<void(Data&)> getData = nullptr;
    std::function<void(Data&)> filled = nullptr;

    void discover();
    void startDiscover();

    void set();
    void startSet();

    void setGotoPreset();
    void startSetGotoPreset();

    void fill();
    void startFill();

    void updateVideo();
    void startUpdateVideo();

    void updateImage();
    void startUpdateImage();

    void startUpdateNetwork();
    void updateNetwork();

    void move();
    void startMove();

    void stop();
    void startStop();

    void updateTime();
    void startUpdateTime();

    void reboot();
    void startReboot();

    void reset();
    void startReset();

    void setOnvifUser();
    void startSetUser();

    Data onvif_data;
    int preset;
    int stop_type;
    float x, y, z;
    std::string new_password;
    std::string interface;

};

}

#endif // ONVIFBOSS_H