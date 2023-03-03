#ifndef ONVIFBOSS_H
#define ONVIFBOSS_H

#include <thread>
#include <vector>
#include <functional>
#include <iostream>
#include <string.h>
#include "onvif.h"

namespace onvif
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
    }

    Data(Data&& other) noexcept
    {
        data = other.data;
        other.data = nullptr;
    }

    Data& operator=(const Data& other)
    {
        if (data) free(data);
        data = (OnvifData*)calloc(sizeof(OnvifData), 1);
        copyData(data, other.data);
        return *this;
    }

    Data& operator=(Data&& other) noexcept
    {
        if (data) free(data);
        data = other.data;
        other.data = nullptr;
        return *this;
    }

    bool operator==(const Data& other)
    {
        if (strcmp(data->xaddrs, other.data->xaddrs))
            return false;
        else
            return true;
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

    bool isValid() const { return data ? true : false; }    

    std::string username() { return data->username; } const
    void setUsername(const std::string& arg) { strncpy(data->username, arg.c_str(), arg.length()); }
    std::string password() { return data->password; } const
    void setPassword(const std::string& arg) { strncpy(data->password, arg.c_str(), arg.length()); }

    std::string xaddrs() { return data->xaddrs; } const
    std::string stream_uri() { return data->stream_uri; } const
    std::string serial_number() { return data->serial_number; } const
    std::string camera_name() { return data->camera_name; } const

    //VIDEO
    std::string resolutions_buf(int arg) { return data->resolutions_buf[arg]; }
    int width() { return data->width; }
    int height() { return data->height; }
    int frame_rate_max() { return data->frame_rate_max; }
    int frame_rate_min() { return data->frame_rate_min; }
    int frame_rate() { return data->frame_rate; }
    int gov_length_max() { return data->gov_length_max; }
    int gov_length_min() { return data->gov_length_min; }
    int gov_length() { return data->gov_length; }
    int bitrate_max() { return data->bitrate_max; }
    int bitrate_min() { return data->bitrate_min; }
    int bitrate() { return data->bitrate; }

    //IMAGE
    int brightness_max() { return data->brightness_max; }
    int brightness_min() { return data->brightness_min; }
    int brightness() { return data->brightness; }
    int saturation_max() { return data->saturation_max; }
    int saturation_min() { return data->saturation_min; }
    int saturation() { return data->saturation; }
    int contrast_max() { return data->contrast_max; }
    int contrast_min() { return data->contrast_min; }
    int contrast() { return data->contrast; }
    int sharpness_max() { return data->sharpness_max; }
    int sharpness_min() { return data->sharpness_min; }
    int sharpness() { return data->sharpness; }

    //NETWORK
    bool dhcp_enabled() { return data->dhcp_enabled; }
    std::string ip_address_buf() { return data->ip_address_buf; } const
    std::string default_gateway_buf() { return data->default_gateway_buf; } const
    std::string dns_buf() { return data->dns_buf; } const
    int prefix_length() { return data->prefix_length; }


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

private:
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

    void pyDiscover();
    void startPyDiscover();

    void pySet();
    void startPySet();

    void pySetPreset();
    void startPySetPreset();

    void pyFill();
    void startPyFill();

    void pyMove();
    void startPyMove();

    void pyStop();
    void startPyStop();

    Data onvif_data;
    int preset;
    int stop_type;
    float x, y, z;

    static void discover(std::function<void()>, std::function<bool(Data&)>, std::function<void(Data&)>);
    void startDiscover(std::function<void()>, std::function<bool(Data&)>, std::function<void(Data&)>);

    static void fill(const Data&, std::function<void(const Data&)>);
    void startFill(const Data&, std::function<void(const Data&)>);

    static void updateVideo(const Data&, std::function<void(const Data&)>);
    void startUpdateVideo(const Data&, std::function<void(const Data&)>);

    static void updateImage(const Data&, std::function<void(const Data&)>);
    void startUpdateImage(const Data&, std::function<void(const Data&)>);

    static void updateNetwork(const Data&, std::function<void(const Data&)>);
    void startUpdateNetwork(const Data&, std::function<void(const Data&)>);

    static void updateTime(const Data&, std::function<void(const Data&)>);
    void startUpdateTime(const Data&, std::function<void(const Data&)>);

    static void reboot(const Data&, std::function<void(const Data&)>);
    void startReboot(const Data&, std::function<void(const Data&)>);

    static void reset(const Data&, std::function<void(const Data&)>);
    void startReset(const Data&, std::function<void(const Data&)>);

    static void setPTZ(const Data&, int);
    void startSetPTZ(const Data&, int);

    static void setPresetPTZ(const Data&, int);
    void startSetPresetPTZ(const Data&, int);

    static void movePTZ(const Data&, float, float, float);
    void startMovePTZ(const Data&, float, float, float);

    static void stopPTZ(const Data&, int);
    void startStopPTZ(const Data&, int);

};

}

#endif // ONVIFBOSS_H