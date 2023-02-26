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

private:
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

class Credential
{
public:
    Credential() {}
    Credential(std::string username, std::string password) : username(username), password(password) {}

    std::string username;
    std::string password;
    bool abort;
};

class Manager
{
public:
    Manager();
    ~Manager();
    static void discover(std::function<void()>, std::function<bool(Data&)>, std::function<void(Data&)>);
    void startDiscover(std::function<void()>, std::function<bool(Data&)>, std::function<void(Data&)>);

};

}

#endif // ONVIFBOSS_H