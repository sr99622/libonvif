#include <iostream>
#include "onvifboss.h"

namespace onvif
{

Manager::Manager()
{

}

Manager::~Manager()
{

}

void Manager::startDiscover(std::function<void()> discoverFinished, 
                        std::function<bool(Data&)>getCredential, std::function<void(Data&)> getData)
{
    std::thread thread(discover, discoverFinished, getCredential, getData);
    thread.detach();
}

void Manager::discover(std::function<void()> discoverFinished,
                    std::function<bool(Data&)>getCredential, std::function<void(Data&)> getData)
{
    Session session;
    int number_of_devices = broadcast(session);

    for (int i = 0; i < number_of_devices; i++) {
        Data data;
        if (prepareOnvifData(i, session, data)) {
            while (true) {
                if (getCredential(data)) {
                    if (fillRTSP(data) == 0) {
                        getProfile(data);
                        getDeviceInformation(data);
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

    discoverFinished();
}

void Manager::startFill(const Data& onvif_data, std::function<void(const Data&)> fcn)
{
    std::cout << "startFill" << std::endl;
    std::thread thread(fill, onvif_data, fcn);
    thread.detach();
}

void Manager::fill(const Data& onvif_data, std::function<void(const Data&)> fcn)
{
    getCapabilities(onvif_data.data);
    getNetworkInterfaces(onvif_data.data);
    getNetworkDefaultGateway(onvif_data.data);
    getDNS(onvif_data.data);
    getVideoEncoderConfigurationOptions(onvif_data.data);
    getVideoEncoderConfiguration(onvif_data.data);
    getOptions(onvif_data.data);
    getImagingSettings(onvif_data.data);
    fcn(onvif_data);
}

void Manager::startUpdateVideo(const Data& onvif_data, std::function<void(const Data&)> fcn)
{
    std::cout << "startupdatevideo" << std::endl;
    std::thread thread(updateVideo, onvif_data, fcn);
    thread.detach();
}

void Manager::updateVideo(const Data& onvif_data, std::function<void(const Data&)> fcn)
{
    setVideoEncoderConfiguration(onvif_data.data);
    getProfile(onvif_data.data);
    getVideoEncoderConfigurationOptions(onvif_data.data);
    getVideoEncoderConfiguration(onvif_data.data);
    fcn(onvif_data);
}

void Manager::startUpdateImage(const Data& onvif_data, std::function<void(const Data&)> fcn)
{
    std::cout << "startUpdateImage()" << std::endl;
    std::thread thread(updateImage, onvif_data, fcn);
    thread.detach();
}

void Manager::updateImage(const Data& onvif_data, std::function<void(const Data&)> fcn)
{
    setImagingSettings(onvif_data.data);
    getOptions(onvif_data.data);
    getImagingSettings(onvif_data.data);
    fcn(onvif_data);
}

void Manager::startUpdateNetwork(const Data& onvif_data, std::function<void(const Data&)> fcn)
{
    std::cout << "startUpdateNetwork" << std::endl;
    std::thread thread(updateNetwork, onvif_data, fcn);
    thread.detach();   
}

void Manager::updateNetwork(const Data& onvif_data, std::function<void(const Data&)> fcn)
{
    std::cout << "update network threading" << std::endl;
    setNetworkInterfaces(onvif_data.data);
    setDNS(onvif_data.data);
    setNetworkDefaultGateway(onvif_data.data);
    getNetworkInterfaces(onvif_data.data);
    getNetworkDefaultGateway(onvif_data.data);
    getDNS(onvif_data.data);
    fcn(onvif_data);
}

void Manager::startUpdateTime(const Data& onvif_data, std::function<void(const Data&)> fcn)
{
    std::cout << "start update time" << std::endl;
    std::thread thread(updateTime, onvif_data, fcn);
    thread.detach();
}

void Manager::updateTime(const Data& onvif_data, std::function<void(const Data&)> fcn)
{
    std::cout << "update time" << std::endl;
    setSystemDateAndTime(onvif_data.data);
    fcn(onvif_data);
}

void Manager::startReboot(const Data& onvif_data, std::function<void(const Data&)> fcn)
{
    std::thread thread(reboot, onvif_data, fcn);
    thread.detach();
}

void Manager::reboot(const Data& onvif_data, std::function<void(const Data&)> fcn)
{
    rebootCamera(onvif_data.data);
    fcn(onvif_data);
}

void Manager::startReset(const Data& onvif_data, std::function<void(const Data&)> fcn)
{
    std::thread thread(reset, onvif_data, fcn);
    thread.detach();
}

void Manager::reset(const Data& onvif_data, std::function<void(const Data&)> fcn)
{
    hardReset(onvif_data.data);
    fcn(onvif_data);
}

void Manager::startSetPTZ(const Data& onvif_data, int position)
{
    std::thread thread(setPTZ, onvif_data, position);
    thread.detach();
}

void Manager::setPTZ(const Data& onvif_data, int position)
{
    char pos[128] = {0};
    sprintf(pos, "%d", position);
    gotoPreset(pos, onvif_data.data);
}

void Manager::startSetPresetPTZ(const Data& onvif_data, int position)
{
    std::thread thread(setPresetPTZ, onvif_data, position);
    thread.detach();
}

void Manager::setPresetPTZ(const Data& onvif_data, int position)
{
    char pos[128] = {0};
    sprintf(pos, "%d", position);
    setPreset(pos, onvif_data.data);
}

void Manager::startMovePTZ(const Data& onvif_data, float x, float y, float z)
{
    std::thread thread(movePTZ, onvif_data, x, y, z);
    thread.detach();
}

void Manager::movePTZ(const Data& onvif_data, float x, float y, float z)
{
    continuousMove(x, y, z, onvif_data.data);
}

void Manager::startStopPTZ(const Data& onvif_data, int type)
{
    std::thread thread(stopPTZ, onvif_data, type);
    thread.detach();
}

void Manager::stopPTZ(const Data& onvif_data, int type)
{
    moveStop(type, onvif_data.data);
}

}
