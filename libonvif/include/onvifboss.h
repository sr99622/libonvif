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

    bool isValid() const { return data ? true : false; }    

//private:
    OnvifData* data;

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