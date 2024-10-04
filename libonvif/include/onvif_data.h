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
#include <sstream>
#include <string.h>
#include <chrono>
#include "onvif.h"

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
    std::function<const std::string(const std::string&)> getProxyURI;
    std::function<void(const std::string&)> errorCallback = nullptr;

    OnvifData* data;
    bool cancelled = false;
    std::string alias;
    int preset;
    int stop_type;
    float x, y, z;
    bool synchronizeTime = false;
    int displayProfile = 0;

    std::vector<Data> profiles;

    Data() 
    {
        data = (OnvifData*)calloc(sizeof(OnvifData), 1);
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

    void startUpdateTime()
    {
        std::thread thread([&]() { updateTime(); });
        thread.detach();
    }

    void updateTime()
    {
        std::stringstream str;
        if (setSystemDateAndTime(data))  str << data->last_error << " - ";
        if (getTimeOffset(data))         str << data->last_error << " - ";

        memset(data->last_error, 0, 1024);
        int length = std::min(std::max((int)(str.str().length() - 2), 0), 1024);
        strncpy(data->last_error, str.str().c_str(), length);

        for (int i = 0; i < profiles.size(); i++) {
            profiles[i].data->dst = data->dst;
            profiles[i].data->datetimetype = data->datetimetype;
            profiles[i].data->time_offset = data->time_offset;
            profiles[i].data->ntp_dhcp = data->ntp_dhcp;
            strcpy(profiles[i].data->timezone, data->timezone);
            strcpy(profiles[i].data->ntp_type, data->ntp_type);
            strcpy(profiles[i].data->ntp_addr, data->ntp_addr);
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
                if (getTimeOffset(profiles[i]))                    str << profiles[i]->last_error << " - ";
            }
            if (getProfile(profiles[i]))                           str << profiles[i]->last_error << " - ";
            if (getNetworkInterfaces(profiles[i]))                 str << profiles[i]->last_error << " - ";
            if (getNetworkDefaultGateway(profiles[i]))             str << profiles[i]->last_error << " - ";
            if (getDNS(profiles[i]))                               str << profiles[i]->last_error << " - ";
            if (getVideoEncoderConfiguration(profiles[i]))         str << profiles[i]->last_error << " - ";
            if (getVideoEncoderConfigurationOptions(profiles[i]))  str << profiles[i]->last_error << " - ";
            if (getAudioEncoderConfiguration(profiles[i]))         str << profiles[i]->last_error << " - ";
            if (getAudioEncoderConfigurationOptions(profiles[i]))  str << profiles[i]->last_error << " - ";
            if (getOptions(profiles[i]))                           str << profiles[i]->last_error << " - ";
            if (getImagingSettings(profiles[i]))                   str << profiles[i]->last_error << " - ";

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
        bool first_pass = true;
        int count = 0;
        while (true) {
            *this = getCredential(*this);
            if (!cancelled) {

                if (!getTimeOffset(data)) {
                    time_t rawtime;
                    struct tm timeinfo;
                    time(&rawtime);
                #ifdef _WIN32
                    localtime_s(&timeinfo, &rawtime);
                #else
                    localtime_r(&rawtime, &timeinfo);
                #endif
                    if (timeinfo.tm_isdst && !dst())
                        setTimeOffset(time_offset() - 3600);
                }

                if (getCapabilities(data) < 0) {
                    std::stringstream str;
                    str << "Camera get capabilities error " << alias << " : " << xaddrs() << " : " << last_error();
                    if (errorCallback) errorCallback(str.str());
                    break;
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
                    setProfile(displayProfile);
                    getData(*this);
                    break;
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
    bool dst() { return data->dst; }
    
    std::string host() { 
        extractHost(data->xaddrs, data->host);
        return data->host; 
    } const

    void syncProfile(int index) { 
        if (index < profiles.size())
            copyData(profiles[index].data, data); 
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
        return getSetting(str.str(), "0") == "1"; 
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