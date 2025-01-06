/*******************************************************************************
* onvif_data.h
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

#ifndef ONVIF_DATA_H
#define ONVIF_DATA_H

#include <thread>
#include <vector>
#include <functional>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <string.h>
#include <chrono>
#include "onvif.h"
#include "rapidjson/writer.h"
#include "rapidjson/reader.h"
#include "rapidjson/stringbuffer.h"

namespace libonvif
{

class Data
{
public:
    std::function<void(Data&)> filled = nullptr;
    std::function<void(Data&)> getData = nullptr;
    std::function<void()> discovered = nullptr;
    std::function<Data(Data&)> getCredential = nullptr;
    std::function<void(const std::string&, const std::string&)> setSetting = nullptr;
    std::function<const std::string(const std::string& key, const std::string& default_value)> getSetting = nullptr;
    std::function<const std::string(const std::string&)> getProxyURI = nullptr;
    std::function<void(const std::string&)> errorCallback = nullptr;
    std::function<void(const std::string&)> infoCallback = nullptr;

    OnvifData* data = nullptr;
    bool cancelled = false;
    std::string alias;
    int preset;
    int stop_type;
    float x, y, z;
    bool synchronizeTime = false;
    int displayProfile = 0;
    bool failedLogin = false;

    std::vector<Data> profiles;

    Data() 
    {
        data = (OnvifData*)calloc(sizeof(OnvifData), 1);
    }

    Data(OnvifData* onvif_data)
    {
        data = onvif_data;
    }

    Data(const std::string& arg)
    {
        data = (OnvifData*)calloc(sizeof(OnvifData), 1);
        rapidjson::Reader reader;
        rapidjson::StringStream ss(arg.c_str());
        reader.Parse(ss, *this);
    }

    Data(const Data& other)
    {
        data = (OnvifData*)calloc(sizeof(OnvifData), 1);
        copyData(data, other.data);
        cancelled = other.cancelled;
        alias = other.alias;
        preset = other.preset;
        stop_type = other.stop_type;
        synchronizeTime = other.synchronizeTime;
        profiles = other.profiles;
        x = other.x;
        y = other.y;
        z = other.z;
    }

    Data(Data&& other) noexcept
    {
        data = other.data;
        other.data = nullptr;
        cancelled = other.cancelled;
        alias = other.alias;
        preset = other.preset;
        stop_type = other.stop_type;
        synchronizeTime = other.synchronizeTime;
        profiles = other.profiles;
        x = other.x;
        y = other.y;
        z = other.z;
    }

    Data& operator=(const Data& other)
    {
        if (!data) data = (OnvifData*)calloc(sizeof(OnvifData), 1);
        copyData(data, other.data);
        cancelled = other.cancelled;
        alias = other.alias;
        preset = other.preset;
        stop_type = other.stop_type;
        synchronizeTime = other.synchronizeTime;
        profiles = other.profiles;
        x = other.x;
        y = other.y;
        z = other.z;
        return *this;
    }

    Data& operator=(Data&& other) noexcept
    {
        if (data) free(data);
        data = other.data;
        cancelled = other.cancelled;
        alias = other.alias;
        other.data = nullptr;
        preset = other.preset;
        stop_type = other.stop_type;
        synchronizeTime = other.synchronizeTime;
        profiles = other.profiles;
        x = other.x;
        y = other.y;
        z = other.z;
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
        if (data) free(data);
    }

    operator OnvifData* ()
    {
        return data;
    }

    OnvifData* operator ->()
    {
        return data;
    }

    void addProfile(Data& profile)
    {
        profiles.push_back(profile);
    }

    void UTCasLocal()
    {
        bool isUTC0 = false;
        std::string tz = timezone();
        if (tz == "UTC0") {
            isUTC0 = true;
        }
        else {
            try {
                std::size_t found;
                found = tz.find("UTC");
                if (found == 0) {
                    found = tz.find("DST");
                    if (found != std::string::npos) 
                        tz = tz.substr(3, found-3);
                    char *token = strtok((char*)tz.c_str(), ":");
                    int result = 0;
                    while (token)
                    {
                        if (token)  {
                            result += std::stoi(token);
                        }
                        token = strtok(nullptr, ":");
                    }
                    if (!result)
                        isUTC0 = true;
                }
            }
            catch (const std::exception& ex) {
                // fail safely 
                //std::cout << "OnvifData::UTCasLocal: " << ex.what() << std::endl;
            }
        }

        if (isUTC0 && datetimetype() == 'M')
            setDateTimeType('U');

    }

    void startUpdateTime()
    {
        std::thread thread([&]() { updateTime(); });
        thread.detach();
    }

    void updateTime()
    {
        std::stringstream str;

        if (data->datetimetype == 'N') {
            if (setNTP(data))                str << data->last_error << " - ";
        }
        if (setSystemDateAndTime(data))  str << data->last_error << " - ";
        if (getTimeOffset(data))         str << data->last_error << " - ";

        memset(data->last_error, 0, 1024);
        int length = std::min(std::max((int)(str.str().length() - 2), 0), 1024);
        strncpy(data->last_error, str.str().c_str(), length);

        UTCasLocal();

        for (int i = 0; i < profiles.size(); i++) {
            profiles[i].data->dst = data->dst;
            profiles[i].data->datetimetype = data->datetimetype;
            profiles[i].data->time_offset = data->time_offset;
            profiles[i].data->ntp_dhcp = data->ntp_dhcp;
            strcpy(profiles[i].data->timezone, data->timezone);
            strcpy(profiles[i].data->ntp_type, data->ntp_type);
            strcpy(profiles[i].data->ntp_addr, data->ntp_addr);
        }

        if (last_error().length()) {
            std::stringstream str;
            str << alias << ": Set System Date and Time Error: " << last_error();
            if (infoCallback) infoCallback(str.str());
        }
        else {
            std::stringstream str;
            str << alias << " time updated succesfully";
            if (infoCallback) infoCallback(str.str());
            else std::cout << str.str() << std::endl;
        }
    }

    void startStop()
    {
        std::thread thread([&]() { stop(); });
        thread.detach();
    }

    void stop()
    {
        moveStop(stop_type, data);
    }

    void startMove()
    {
        std::thread thread([&]() { move(); });
        thread.detach();
    }

    void move()
    {
        continuousMove(x, y, z, data);
    }

    void startSet()
    {
        std::thread thread([&]() { set(); });
        thread.detach();
    }

    void set()
    {
        char pos[128] = {0};
        sprintf(pos, "%d", preset);
        gotoPreset(pos, data);
    }

    void startSetGotoPreset()
    {
        std::thread thread([&]() { setGotoPreset(); });
        thread.detach();
    }

    void setGotoPreset()
    {
        char pos[128] = {0};
        sprintf(pos, "%d", preset);
        setPreset(pos, data);
        if (filled) filled(*this);
    }

    void startUpdateVideo()
    {
        std::thread thread([&]() { updateVideo(); });
        thread.detach();
    }

    void updateVideo()
    {
        std::stringstream str;
        if (setVideoEncoderConfiguration(data))         str << data->last_error << " - ";
        if (getVideoEncoderConfigurationOptions(data))  str << data->last_error << " - ";
        if (getVideoEncoderConfiguration(data))         str << data->last_error << " - ";

        memset(data->last_error, 0, 1024);
        int length = std::min(std::max((int)(str.str().length() - 2), 0), 1024);
        strncpy(data->last_error, str.str().c_str(), length);

        syncProfile(indexForProfile(data->profileToken));

        if (filled) filled(*this);
    }

    void startUpdateAudio()
    {
        std::thread thread([&] () { updateAudio(); });
        thread.detach();
    }

    void updateAudio()
    {
        std::stringstream str;
        if (setAudioEncoderConfiguration(data))         str << data->last_error << " - ";
        if (getAudioEncoderConfigurationOptions(data))  str << data->last_error << " - ";
        if (getAudioEncoderConfiguration(data))         str << data->last_error << " - ";

        memset(data->last_error, 0, 1024);
        int length = std::min(std::max((int)(str.str().length() - 2), 0), 1024);
        strncpy(data->last_error, str.str().c_str(), length);

        syncProfile(indexForProfile(data->profileToken));

        if (filled) filled(*this);
    }

    void startUpdateImage()
    {
        std::thread thread([&]() { updateImage(); });
        thread.detach();
    }

    void updateImage()
    {
        std::stringstream str;
        if (setImagingSettings(data))  str << data->last_error << " - ";
        if (getOptions(data))          str << data->last_error << " - ";
        if (getImagingSettings(data))  str << data->last_error << " - ";

        memset(data->last_error, 0, 1024);
        int length = std::min(std::max((int)(str.str().length() - 2), 0), 1024);
        strncpy(data->last_error, str.str().c_str(), length);

        for (int i = 0; i < profiles.size(); i++) {
            profiles[i].data->brightness = data->brightness;
            profiles[i].data->saturation = data->saturation;
            profiles[i].data->contrast = data->contrast;
            profiles[i].data->sharpness = data->sharpness;
        }

        if (filled) filled(*this);
    }

    void startUpdateNetwork()
    {
        std::thread thread([&]() { updateNetwork(); });
        thread.detach();
    }

    void updateNetwork()
    {
        std::stringstream str;
        if (setNetworkInterfaces(data))      str << data->last_error << " - ";
        if (setDNS(data))                    str << data->last_error << " - ";
        if (setNetworkDefaultGateway(data))  str << data->last_error << " - ";
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        if (getNetworkInterfaces(data))      str << data->last_error << " - ";
        if (getNetworkDefaultGateway(data))  str << data->last_error << " - ";
        if (getDNS(data))                    str << data->last_error << " - ";

        memset(data->last_error, 0, 1024);
        int length = std::min(std::max((int)(str.str().length() - 2), 0), 1024);
        strncpy(data->last_error, str.str().c_str(), length);

        syncProfile(indexForProfile(data->profileToken));

        for (int i = 0; i < profiles.size(); i++) {
            if (strcmp(profiles[i].data->profileToken, data->profileToken)) {
                strcpy(profiles[i].data->networkInterfaceToken, data->networkInterfaceToken);
                strcpy(profiles[i].data->networkInterfaceName, data->networkInterfaceName);
                strcpy(profiles[i].data->ip_address_buf, data->ip_address_buf);
                strcpy(profiles[i].data->default_gateway_buf, data->default_gateway_buf);
                strcpy(profiles[i].data->dns_buf, data->dns_buf);
                strcpy(profiles[i].data->mask_buf, data->mask_buf);
                profiles[i].data->dhcp_enabled = data->dhcp_enabled;
                profiles[i].data->prefix_length = data->prefix_length;
            }
        }

        if (filled) filled(*this);
    }

    void startReboot()
    {
        std::thread thread([&]() { reboot(); });
        thread.detach();
    }

    void reboot()
    {
        rebootCamera(data);
    }

    void startReset()
    {
        std::thread thread([&]() { reset(); });
        thread.detach();
    }

    void reset()
    {
        hardReset(data);
    }

    void startSetUser()
    {
        std::thread thread([&]() { setOnvifUser(); });
        thread.detach();
    }

    void setOnvifUser()
    {
        /*
        if (setUser((char*)new_password.c_str(), onvif_data) == 0)
            onvif_data.setPassword(new_password.c_str());
        filled(onvif_data);
        */
    }

    void startFill(bool arg)
    {
        synchronizeTime = arg;
        std::thread thread([&]() { fill(); });
        thread.detach();
    }

    void fill()
    {
        for (int i = 0; i < profiles.size(); i++) {
            std::stringstream str;
            if (synchronizeTime) {
                if (setSystemDateAndTime(profiles[i]))             str << profiles[i]->last_error << " - ";
            }
            if (getTimeOffset(profiles[i]))                        str << profiles[i]->last_error << " - ";
            if (getProfile(profiles[i]))                           str << profiles[i]->last_error << " - ";
            if (getNetworkInterfaces(profiles[i]))                 str << profiles[i]->last_error << " - ";
            if (getNetworkDefaultGateway(profiles[i]))             str << profiles[i]->last_error << " - ";
            if (getDNS(profiles[i]))                               str << profiles[i]->last_error << " - ";
            if (getNTP(profiles[i]))                               str << profiles[i]->last_error << " - ";
            if (getVideoEncoderConfiguration(profiles[i]))         str << profiles[i]->last_error << " - ";
            if (getVideoEncoderConfigurationOptions(profiles[i]))  str << profiles[i]->last_error << " - ";
            if (getAudioEncoderConfiguration(profiles[i]))         str << profiles[i]->last_error << " - ";
            if (getAudioEncoderConfigurationOptions(profiles[i]))  str << profiles[i]->last_error << " - ";
            if (getOptions(profiles[i]))                           str << profiles[i]->last_error << " - ";
            if (getImagingSettings(profiles[i]))                   str << profiles[i]->last_error << " - ";

            strcpy(profiles[i]->camera_name, camera_name().c_str());
            strcpy(profiles[i]->device_service, device_service().c_str());
            strcpy(profiles[i]->xaddrs, xaddrs().c_str());
            strcpy(profiles[i]->host, host().c_str());
            profiles[i].UTCasLocal();

            memset(profiles[i]->last_error, 0, 1024);
            int length = std::min(std::max((int)(str.str().length() - 2), 0), 1024);
            strncpy(profiles[i]->last_error, str.str().c_str(), length);
        }

        setProfile(displayProfile);
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        if (filled) filled(*this);
    }

    void startManualFill()
    {
        std::thread thread([&]() { manual_fill(); });
        thread.detach();
    }

    void manual_fill()
    {
        extractHost(data->xaddrs, data->host);
        
        if (getTimeOffset(data)) {
            std::stringstream str;
            str << "Camera get time offset error " << camera_name() << " : " << xaddrs() << " : " << last_error();
            if (infoCallback) infoCallback(str.str());
        }
        
        time_t initial_offset = data->time_offset;
        int count = 1;
        int direction = 1;

        while (true) {
            *this = getCredential(*this);
            if (!cancelled) {

                if (getCapabilities(data)) {
                    // Daylight Savings Time mismatch can cause authentication failure
                    std::stringstream str;
                    str << "onvif data error " << camera_name() << " : " << xaddrs() << " : " << last_error() << " attempting time offset correction";
                    if (infoCallback) infoCallback(str.str());
                    data->time_offset = initial_offset + 3600 * direction * count;
                    if (direction < 0) count++;
                    direction = ~direction + 1;
                    if (count > 2) {
                        std::stringstream str;
                        str << "onvif data error: " << camera_name() << " : " << xaddrs() << " : " << last_error() << " camera interrogation failed";
                        if (infoCallback) infoCallback(str.str());
                        break;
                    }
                    else {
                        continue;
                    }
                }

                if (getDeviceInformation(data) == 0) {
                    int index = 0;
                    while (true) {
                        Data profile(*this);
                        getProfileToken(profile, index);
                        if (profile.profile().length() == 0)
                            break;
                        getStreamUri(profile.data);
                        profiles.push_back(profile);
                        index++;
                    }

                    if (getSetting) {
                        std::stringstream key2;
                        key2 << serial_number() << "/Alias";
                        alias = getSetting(key2.str(), host());
                    }

                    if (setSetting) {
                        std::stringstream key1;
                        key1 << serial_number() << "/XAddrs";
                        setSetting(key1.str(), xaddrs());
                        std::stringstream key2;
                        key2 << serial_number() << "/Alias";
                        setSetting(key2.str(), alias);
                    }

                    getData(*this);
                    break;
                }
                else {
                    std::stringstream str;
                    str << "get device information error: " << data->camera_name << " : " << data->last_error;
                    if (infoCallback) infoCallback(str.str());
                }
            }
            else {
                break;
            }
        }
    }

    std::string toString() {
        std::stringstream str;
        str << "camera_name=" << data->camera_name << "\n"
            << "stream_uri=" << data->stream_uri << "\n";
        return str.str();
    }

    std::string uri() {
        std::stringstream str;
        try {
            std::string arg(data->stream_uri);

            if (getProxyURI) {
                str << getProxyURI(arg);
            }
            else {
                if (arg.length() > 7)
                    str << arg.substr(0, 7) << data->username << ":" << data->password << "@" << arg.substr(7);
            }
        }
        catch (const std::exception& ex) {
            std::cout << "onvif data uri() exception: " << ex.what() << std::endl;
        }

        return str.str();
    }

    void nullifyGetProxyURI() {
        getProxyURI = nullptr;
    }

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

    bool isValid() const { return data ? true : false; }
    std::string xaddrs() { return data->xaddrs; } const
    void setXAddrs(const std::string& arg) { strcpy(data->xaddrs, arg.c_str()); }
    std::string device_service() { return data->device_service; } const
    void setDeviceService(const std::string& arg) { strcpy(data->device_service, arg.c_str()); }
    std::string event_service() { return data->event_service; } const
    std::string stream_uri() { return data->stream_uri; } const
    std::string snapshot_uri() { return data->snapshot_uri; } const
    std::string serial_number() { return data->serial_number; } const
    std::string camera_name() { return data->camera_name; } const
    void setCameraName(const std::string& arg) { strcpy(data->camera_name, arg.c_str()); }
    void setHost(const std::string& arg) { strcpy(data->host, arg.c_str()); }
    std::string last_error() { return data->last_error; } const
    std::string profile() { return data->profileToken; } const
    void clearLastError() { memset(data->last_error, 0, 1024); }
    void setLastError(const std::string& arg) { strcpy(data->last_error, arg.c_str()); }
    time_t time_offset() { return data->time_offset; } const
    void setTimeOffset(time_t arg) { data->time_offset = arg; }
    std::string timezone() { return data->timezone; } const
    void setTimezone(const std::string& arg) { strcpy(data->timezone, arg.c_str()); }
    bool dst() { return data->dst; } const
    void noOp() { }  // believe it or not, this line is needed to compile on windows
    void setDST(bool arg) { data->dst = arg; }
    
    std::string host() { 
        extractHost(data->xaddrs, data->host);
        return data->host; 
    } const

    void syncProfile(int index) { 
        if (index < profiles.size())
            copyData(profiles[index].data, data); 
    }

    void syncData(const Data& other) {
        copyData(data, other.data);
    }

    void setProfile(int index) {
        if (index < profiles.size()) {
            copyData(data, profiles[index].data);
            displayProfile = index;
        }
    }

    void clear(int bug) { 
        clearData(data); 
        alias = "";
    }

    int indexForProfile(const std::string& profileToken) {
        int result = 0;
        for (int i = 0; i < profiles.size(); i++) {
            if (profileToken == profiles[i].data->profileToken) {
                result = i;
                break;
            }
        }
        return result;
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
    std::string encoding() { return data->encoding; }

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
    std::string ntp_addr() { return data->ntp_addr; } const
    void setNTPAddr(const std::string& arg) {
        memset(data->ntp_addr, 0, 128);
        strncpy(data->ntp_addr, arg.c_str(), arg.length());
    }
    std::string ntp_type() { return data->ntp_type; } const
    void setNTPType(const std::string& arg) {
        memset(data->ntp_type, 0, 128);
        strncpy(data->ntp_type, arg.c_str(), arg.length());
    }
    char datetimetype() { return data->datetimetype; }
    void setDateTimeType(char arg) { data->datetimetype = arg; }
    bool ntp_dhcp() { return data->ntp_dhcp; }
    void setNTPDHCP(bool arg) { data->ntp_dhcp = arg; }

    //AUDIO
    std::vector<std::string> audio_encoders() { 
        std::vector<std::string> result;
        for (int i=0; i<3; i++) {
            if (strlen(data->audio_encoders[i]))
                result.push_back(data->audio_encoders[i]);
        }
        return result;
    } const
    std::vector<int> audio_bitrates(int arg) {
        std::vector<int> result;
        for (int i=0; i<8; i++) {
            if (data->audio_bitrates[arg][i])
                result.push_back(data->audio_bitrates[arg][i]);
        }
        return result;
    } const
    std::vector<int> audio_sample_rates(int arg) {
        std::vector<int> result;
        for (int i=0; i<8; i++) {
            if (data->audio_sample_rates[arg][i])
                result.push_back(data->audio_sample_rates[arg][i]);
        }
        return result;
    } const
    std::string audio_encoding() { return data->audio_encoding; } const
    void setAudioEncoding(const std::string& arg) {
        memset(data->audio_encoding, 0, sizeof(data->audio_encoding));
        strncpy(data->audio_encoding, arg.c_str(), arg.length());
    }
    std::string audio_name() { return data->audio_name; } const
    int audio_bitrate() { return data->audio_bitrate; }
    void setAudioBitrate(int arg) { data->audio_bitrate = arg; }
    int audio_sample_rate() { return data->audio_sample_rate; }
    void setAudioSampleRate(int arg) { data->audio_sample_rate = arg; }
    std::string audio_session_timeout() { return data->audio_session_timeout; } const
    std::string audio_multicast_type() { return data->audio_multicast_type; } const
    std::string audio_multicast_address() { return data->audio_multicast_address; } const
    int audio_use_count() { return data->audio_use_count; }
    int audio_multicast_port() { return data->audio_multicast_port; }
    int audio_multicast_TTL() { return data->audio_multicast_TTL; }
    bool audio_multicast_auto_start() { return data->audio_multicast_auto_start; }

    // SERIALIZATION INTERFACE

    std::string toJSON() {
        rapidjson::StringBuffer s;
        rapidjson::Writer<rapidjson::StringBuffer> w(s);

        w.StartObject();
        w.Key("videoEncoderConfigurationToken");
        w.String(data->videoEncoderConfigurationToken);

        w.Key("resolutions_buf");
        w.StartArray();
        for (int i = 0; i < 16; i++) {
            if (strlen(data->resolutions_buf[i]) > 0)
                w.String(data->resolutions_buf[i]);
        }
        w.EndArray();

        w.Key("gov_length_min");
        w.Int(data->gov_length_min);
        w.Key("gov_length_max");
        w.Int(data->gov_length_max);
        w.Key("frame_rate_min");
        w.Int(data->frame_rate_min);
        w.Key("frame_rate_max");
        w.Int(data->frame_rate_max);
        w.Key("bitrate_min");
        w.Int(data->bitrate_min);
        w.Key("bitrate_max");
        w.Int(data->bitrate_max);
        w.Key("width");
        w.Int(data->width);
        w.Key("height");
        w.Int(data->height);
        w.Key("gov_length");
        w.Int(data->gov_length);
        w.Key("frame_rate");
        w.Int(data->frame_rate);
        w.Key("bitrate");
        w.Int(data->bitrate);
        w.Key("video_encoder_name");
        w.String(data->video_encoder_name);
        w.Key("use_count");
        w.Int(data->use_count);
        w.Key("quality");
        w.Double(data->quality);
        w.Key("h264_profile");
        w.String(data->h264_profile);
        w.Key("multicast_address_type");
        w.String(data->multicast_address_type);
        w.Key("multicast_address");
        w.String(data->multicast_address);
        w.Key("multicast_port");
        w.Int(data->multicast_port);
        w.Key("multicast_ttl");
        w.Int(data->multicast_ttl);
        w.Key("autostart");
        w.Bool(data->autostart);
        w.Key("session_time_out");
        w.String(data->session_time_out);
        w.Key("guaranteed_frame_rate");
        w.Bool(data->guaranteed_frame_rate);
        w.Key("encoding");
        w.String(data->encoding);
        w.Key("encoding_interval");
        w.Int(data->encoding_interval);
        w.Key("networkInterfaceToken");
        w.String(data->networkInterfaceToken);
        w.Key("networkInterfaceName");
        w.String(data->networkInterfaceName);
        w.Key("dhcp_enabled");
        w.Bool(data->dhcp_enabled);
        w.Key("ip_address_buf");
        w.String(data->ip_address_buf);
        w.Key("default_gateway_buf");
        w.String(data->default_gateway_buf);
        w.Key("dns_buf");
        w.String(data->dns_buf);
        w.Key("prefix_length");
        w.Int(data->prefix_length);
        w.Key("mask_buf");
        w.String(data->mask_buf);
        w.Key("videoSourceConfigurationToken");
        w.String(data->videoSourceConfigurationToken);
        w.Key("brightness_min");
        w.Int(data->brightness_min);
        w.Key("brightness_max");
        w.Int(data->brightness_max);
        w.Key("saturation_min");
        w.Int(data->saturation_min);
        w.Key("saturation_max");
        w.Int(data->saturation_max);
        w.Key("contrast_min");
        w.Int(data->contrast_min);
        w.Key("contrast_max");
        w.Int(data->contrast_max);
        w.Key("sharpness_min");
        w.Int(data->sharpness_min);
        w.Key("sharpness_max");
        w.Int(data->sharpness_max);
        w.Key("brightness");
        w.Int(data->brightness);
        w.Key("saturation");
        w.Int(data->saturation);
        w.Key("contrast");
        w.Int(data->contrast);
        w.Key("sharpness");
        w.Int(data->sharpness);
        w.Key("device_service");
        w.String(data->device_service);
        w.Key("media_service");
        w.String(data->media_service);
        w.Key("imaging_service");
        w.String(data->imaging_service);
        w.Key("ptz_service");
        w.String(data->ptz_service);
        w.Key("event_service");
        w.String(data->event_service);
        w.Key("subscription_reference");
        w.String(data->subscription_reference);
        w.Key("event_listen_port");
        w.Int(data->event_listen_port);
        w.Key("xaddrs");
        w.String(data->xaddrs);
        w.Key("profileToken");
        w.String(data->profileToken);
        w.Key("username");
        w.String(data->username);
        w.Key("password");
        w.String(data->password);
        w.Key("stream_uri");
        w.String(data->stream_uri);
        w.Key("camera_name");
        w.String(data->camera_name);
        w.Key("serial_number");
        w.String(data->serial_number);
        w.Key("host_name");
        w.String(data->host_name);
        w.Key("host");
        w.String(data->host);
        w.Key("last_error");
        w.String(data->last_error);
        w.Key("time_offset");
        w.Int(data->time_offset);
        w.Key("datetimetype");
        w.Uint(data->datetimetype);
        w.Key("dst");
        w.Bool(data->dst);
        w.Key("timezone");
        w.String(data->timezone);
        w.Key("ntp_dhcp");
        w.Bool(data->ntp_dhcp);
        w.Key("ntp_type");
        w.String(data->ntp_type);
        w.Key("ntp_addr");
        w.String(data->ntp_addr);

        int audio_count = 0;
        w.Key("audio_encoders");
        w.StartArray();
        for (int i = 0; i < 3; i++) {
            if (strlen(data->audio_encoders[i]) > 0) {
                w.String(data->audio_encoders[i]);
                audio_count++;
            }
        }
        w.EndArray();
        
        w.Key("audio_sample_rates");
        w.StartArray();
        for (int i = 0; i < audio_count; i++) {
            w.StartArray();
            for (int j = 0; j < 8; j++) {
                if (data->audio_sample_rates[i][j] > 0)
                    w.Int(data->audio_sample_rates[i][j]);
            }
            w.EndArray();
        }
        w.EndArray();

        w.Key("audio_bitrates");
        w.StartArray();
        for (int i = 0; i < audio_count; i++) {
            w.StartArray();
            for (int j = 0; j < 8; j++) {
                if (data->audio_bitrates[i][j] > 0)
                    w.Int(data->audio_bitrates[i][j]);
            }
            w.EndArray();
        }
        w.EndArray();

        w.Key("audio_encoding");
        w.String(data->audio_encoding);
        w.Key("audio_name");
        w.String(data->audio_name);
        w.Key("audioEncoderConfigurationToken");
        w.String(data->audioEncoderConfigurationToken);
        w.Key("audioSourceConfigurationToken");
        w.String(data->audioSourceConfigurationToken);
        w.Key("audio_bitrate");
        w.Int(data->audio_bitrate);
        w.Key("audio_sample_rate");
        w.Int(data->audio_sample_rate);
        w.Key("audio_use_count");
        w.Int(data->audio_use_count);
        w.Key("audio_session_timeout");
        w.String(data->audio_session_timeout);
        w.Key("audio_multicast_type");
        w.String(data->audio_multicast_type);
        w.Key("audio_multicast_address");
        w.String(data->audio_multicast_address);
        w.Key("audio_multicast_port");
        w.Int(data->audio_multicast_port);
        w.Key("audio_multicast_TTL");
        w.Int(data->audio_multicast_TTL);
        w.Key("audio_multicast_auto_start");
        w.Bool(data->audio_multicast_auto_start);

        w.Key("x");
        w.Double(x);
        w.Key("y");
        w.Double(y);
        w.Key("z");
        w.Double(z);
        w.Key("preset");
        w.Int(preset);
        w.Key("stop_type");
        w.Int(stop_type);

        w.EndObject();

        return s.GetString();
    }

    std::string key;
    std::vector<int> counters;

    bool Null() { std::cout << "Null()" << std::endl; return true; }
    bool Bool(bool b) { 
        if (key == "autostart") data->autostart = b;
        if (key == "guaranteed_frame_rate") data->guaranteed_frame_rate = b;
        if (key == "dhcp_enabled") data->dhcp_enabled = b;
        if (key == "dst") data->dst = b;
        if (key == "ntp_dhcp") data->ntp_dhcp = b;
        if (key == "audio_multicast_auto_start") data->audio_multicast_auto_start = b;
        return true; 
    }
    bool Int(int i) { 
        if (key == "time_offset") data->time_offset = i; 
        return true; 
    }
    bool Uint(unsigned i) { 
        if (key == "gov_length_min") data->gov_length_min = i;
        if (key == "gov_length_max") data->gov_length_max = i;
        if (key == "frame_rate_min") data->frame_rate_min = i;
        if (key == "frame_rate_max") data->frame_rate_max = i;
        if (key == "bitrate_min") data->bitrate_min = i;
        if (key == "bitrate_max") data->bitrate_max = i;
        if (key == "width") data->width = i;
        if (key == "height") data->height = i;
        if (key == "gov_length") data->gov_length = i;
        if (key == "frame_rate") data->frame_rate = i;
        if (key == "bitrate") data->bitrate = i;
        if (key == "use_count") data->use_count = i;
        if (key == "multicast_port") data->multicast_port = i;
        if (key == "multicast_ttl") data->multicast_ttl = i;
        if (key == "encoding_interval") data->encoding_interval = i;
        if (key == "prefix_length") data->prefix_length = i;
        if (key == "brightness_min") data->brightness_min = i;
        if (key == "brightness_max") data->brightness_max = i;
        if (key == "saturation_min") data->saturation_min = i;
        if (key == "saturation_max") data->saturation_max = i;
        if (key == "contrast_min") data->contrast_min = i;
        if (key == "contrast_max") data->contrast_max = i;
        if (key == "sharpness_min") data->sharpness_min = i;
        if (key == "sharpness_max") data->sharpness_max = i;
        if (key == "brightness") data->brightness = i;
        if (key == "saturation") data->saturation = i;
        if (key == "contrast") data->contrast = i;
        if (key == "sharpness") data->sharpness = i;
        if (key == "event_listen_port") data->event_listen_port = i;
        if (key == "datetimetype") data->datetimetype = i;
        if (key == "audio_bitrate") data->audio_bitrate = i;
        if (key == "audio_sample_rate") data->audio_sample_rate = i;
        if (key == "audio_use_count") data->audio_use_count = i;
        if (key == "audio_multicast_port") data->audio_multicast_port = i;
        if (key == "audio_multicast_TTL") data->audio_multicast_TTL = i;
        if (key == "preset") preset = i;
        if (key == "stop_type") stop_type = i;

        if (key == "audio_sample_rates") {
            int d = counters.size() - 1;
            data->audio_sample_rates[counters[d-1]][counters[d]] = i;
            counters[d]++;
        }

        if (key == "audio_bitrates") {
            int d = counters.size() - 1;
            data->audio_bitrates[counters[d-1]][counters[d]] = i;
            counters[d]++;
        }

        return true;
    }
    bool Int64(int64_t i) { std::cout << "Int64(" << i << ")" << std::endl; return true; }
    bool Uint64(uint64_t u) { std::cout << "Uint64(" << u << ")" << std::endl; return true; }
    bool Double(double d) { 
        if (key == "quality") data->quality = d;
        if (key == "x") x = d;
        if (key == "y") y = d;
        if (key == "z") z = d;
        return true; 
    }
    bool RawNumber(const char* str, rapidjson::SizeType length, bool copy) { 
        std::cout << "Number(" << str << ", " << length << ", " << std::boolalpha << copy << ")" << std::endl;
        return true;
    }
    bool String(const char* str, rapidjson::SizeType length, bool copy) { 
        if (key == "videoEncoderConfigurationToken") strncpy(data->videoEncoderConfigurationToken, str, length);
        if (key == "video_encoder_name") strncpy(data->video_encoder_name, str, length);
        if (key == "h264_profile") strncpy(data->h264_profile, str, length);
        if (key == "multicast_address_type") strncpy(data->multicast_address_type, str, length);
        if (key == "multicast_address") strncpy(data->multicast_address, str, length);
        if (key == "session_time_out") strncpy(data->session_time_out, str, length);
        if (key == "encoding") strncpy(data->encoding, str, length);
        if (key == "networkInterfaceToken") strncpy(data->networkInterfaceToken, str, length);
        if (key == "networkInterfaceName") strncpy(data->networkInterfaceName, str, length);
        if (key == "ip_address_buf") strncpy(data->ip_address_buf, str, length);
        if (key == "default_gateway_buf") strncpy(data->default_gateway_buf, str, length);
        if (key == "dns_buf") strncpy(data->dns_buf, str, length);
        if (key == "mask_buf") strncpy(data->mask_buf, str, length);
        if (key == "videoSourceConfigurationToken") strncpy(data->videoSourceConfigurationToken, str, length);
        if (key == "device_service") strncpy(data->device_service, str, length);
        if (key == "media_service") strncpy(data->media_service, str, length);
        if (key == "imaging_service") strncpy(data->imaging_service, str, length);
        if (key == "ptz_service") strncpy(data->ptz_service, str, length);
        if (key == "event_service") strncpy(data->event_service, str, length);
        if (key == "subscription_reference") strncpy(data->subscription_reference, str, length);
        if (key == "xaddrs") strncpy(data->xaddrs, str, length);
        if (key == "profileToken") strncpy(data->profileToken, str, length);
        if (key == "username") strncpy(data->username, str, length);
        if (key == "password") strncpy(data->password, str, length);
        if (key == "stream_uri") strncpy(data->stream_uri, str, length);
        if (key == "camera_name") strncpy(data->camera_name, str, length);
        if (key == "serial_number") strncpy(data->serial_number, str, length);
        if (key == "host_name") strncpy(data->host_name, str, length);
        if (key == "host") strncpy(data->host, str, length);
        if (key == "last_error") strncpy(data->last_error, str, length);
        if (key == "timezone") strncpy(data->timezone, str, length);
        if (key == "ntp_type") strncpy(data->ntp_type, str, length);
        if (key == "ntp_addr") strncpy(data->ntp_addr, str, length);
        if (key == "audio_encoding") strncpy(data->audio_encoding, str, length);
        if (key == "audio_name") strncpy(data->audio_name, str, length);
        if (key == "audioEncoderConfigurationToken") strncpy(data->audioEncoderConfigurationToken, str, length);
        if (key == "audioSourceConfigurationToken") strncpy(data->audioSourceConfigurationToken, str, length);
        if (key == "audio_session_timeout") strncpy(data->audio_session_timeout, str, length);
        if (key == "audio_multicast_type") strncpy(data->audio_multicast_type, str, length);
        if (key == "audio_multicast_address") strncpy(data->audio_multicast_address, str, length);

        if (key == "resolutions_buf") {
            strncpy(data->resolutions_buf[counters.back()], str, length);
            counters.back()++;
        }

        if (key == "audio_encoders") {
            strncpy(data->audio_encoders[counters.back()], str, length);
            counters.back()++;
        }

        return true;
    }
    bool StartObject() { 
        return true; 
    }
    bool Key(const char* str, rapidjson::SizeType length, bool copy) {
        key = str;
        return true;
    }
    bool EndObject(rapidjson::SizeType memberCount) { 
        return true; 
    }
    bool StartArray() { 
        counters.push_back(0);
        return true; 
    }
    bool EndArray(rapidjson::SizeType elementCount) { 
        counters.pop_back();
        if (counters.size())
            counters[counters.size()-1]++;
        return true; 
    }

 
    //GUI INTERFACE

    /*
    Please note that this class is intended to be self contained within the C++ domain. It will not 
    behave as expected if the calling python program attempts to extend the functionality of the 
    class by adding member variables in the python domain. This was done so that the profile could
    be copied or filled with data by the C++ class exclusively, removing the need for additional 
    synchronization code in the python domain. 
    
    The effect of this decision is that GUI persistence for profiles must be implemented in this 
    C++ class directly. The member variables are added to the OnvifData structure in onvif.h and 
    the copyData and clearData functions in onvif.c. GUI persistence is handled by passing setSetting 
    and getSetting from the calling python program for writing variable states to disk.
    */


    bool getDisableVideo() { 
        std::stringstream str;
        str << serial_number() << "/" << profile() << "/DisableVideo";
        return getSetting(str.str(), "0") == "1"; 
    }
    void setDisableVideo(bool arg) { 
        data->disable_video = arg;
        std::stringstream str;
        str << serial_number() << "/" << profile() << "/DisableVideo";
        setSetting(str.str(), arg ? "1" : "0");
    }
    bool getAnalyzeVideo() { 
        std::stringstream str;
        str << serial_number() << "/" << profile() << "/AnalyzeVideo";
        return getSetting(str.str(), "0") == "1"; 
    }
    void setAnalyzeVideo(bool arg) { 
        data->analyze_video = arg;
        std::stringstream str;
        str << serial_number() << "/" << profile() << "/AnalyzeVideo";
        setSetting(str.str(), arg ? "1" : "0");
    }
    bool getDisableAudio() { 
        std::stringstream str;
        str << serial_number() << "/" << profile() << "/DisableAudio";
        bool result = getSetting(str.str(), "1") == "1";
        if (audio_bitrate() == 0)
            result = false;
        return result;
    }
    void setDisableAudio(bool arg) { 
        data->disable_audio = arg;
        std::stringstream str;
        str << serial_number() << "/" << profile() << "/DisableAudio";
        setSetting(str.str(), arg ? "1" : "0");
    }
    bool getAnalyzeAudio() { 
        std::stringstream str;
        str << serial_number() << "/" << profile() << "/AnalyzeAudio";
        return getSetting(str.str(), "0") == "1"; 
    }
    void setAnalyzeAudio(bool arg) { 
        data->analyze_audio = arg;
        std::stringstream str;
        str << serial_number() << "/" << profile() << "/AnalyzeAudio";
        setSetting(str.str(), arg ? "1" : "0");
    }
    bool getSyncAudio() {
        std::stringstream str;
        str << serial_number() << "/" << profile() << "/SyncAudio";
        return getSetting(str.str(), "0") == "1";
    }
    void setSyncAudio(bool arg) {
        data->sync_audio = arg;
        std::stringstream str;
        str << serial_number() << "/" << profile() << "/SyncAudio";
        setSetting(str.str(), arg ? "1" : "0");
    }
    bool getHidden() { 
        std::stringstream str;
        str << serial_number() << "/" << profile() << "/Hidden";
        return getSetting(str.str(), "0") == "1"; 
    }
    void setHidden(bool arg) { 
        data->hidden = arg;
        std::stringstream str;
        str << serial_number() << "/" << profile() << "/Hidden";
        setSetting(str.str(), arg ? "1" : "0");
    }
    int getDesiredAspect() { 
        std::stringstream str_key, str_val, ratio;
        str_key << serial_number() << "/" << profile() << "/DesiredAspect";
        ratio << ((height() == 0) ? 0 : (int)(100 * width() / height()));
        str_val << getSetting(str_key.str(), ratio.str());
        int desired_aspect = 0;
        str_val >> desired_aspect;
        return desired_aspect; 
    }
    void setDesiredAspect(int arg) { 
        data->desired_aspect = arg;
        std::stringstream str_key, str_val;
        str_key << serial_number() << "/" << profile() << "/DesiredAspect";
        str_val << arg;
        setSetting(str_key.str(), str_val.str());
    }
    int getCacheMax() {
        std::stringstream str_key, str_val;
        str_key << serial_number() << "/" << profile() << "/CacheMax";
        str_val << getSetting(str_key.str(), "100");
        int result = 100;
        str_val >> result;
        return result;
    }
    void setCacheMax(int arg) {
        data->cache_max = arg;
        std::stringstream str_key, str_val;
        str_key << serial_number() << "/" << profile() << "/CacheMax";
        str_val << arg;
        setSetting(str_key.str(), str_val.str());
    }

};

}


#endif // ONVIF_DATA_H