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

void Manager::startPyStop()
{
    std::cout << "startPyStop" << std::endl;
    std::thread thread([&]() { pyStop(); });
    thread.detach();
}

void Manager::pyStop()
{
    std::cout << "pyStop" << std::endl;
    moveStop(stop_type, onvif_data.data);
}

void Manager::startPyMove()
{
    std::cout << "startPyMove" << std::endl;
    std::thread thread([&]() { pyMove(); });
    thread.detach();
}

void Manager::pyMove()
{
    std::cout << "pyMove" << std::endl;
    continuousMove(x, y, z, onvif_data.data);
}

void Manager::startPySet()
{
    std::cout << "startPySet" << std::endl;
    std::thread thread([&]() { pySet(); });
    thread.detach();
}

void Manager::pySet()
{
    std::cout << "pySet" << std::endl;
    char pos[128] = {0};
    sprintf(pos, "%d", preset);
    gotoPreset(pos, onvif_data.data);
}

void Manager::startPySetPreset()
{
    std::cout << "startPySetPreset()" << std::endl;
    char pos[128] = {0};
    sprintf(pos, "%d", preset);
    setPreset(pos, onvif_data.data);
}

void Manager::startPyFill()
{
    std::cout << "startPyFill" << std::endl;
    std::thread thread([&]() { pyFill(); });
    thread.detach();
}

void Manager::pyFill()
{
    std::cout << "PyFill start" << std::endl;
    getCapabilities(onvif_data.data);
    getNetworkInterfaces(onvif_data.data);
    getNetworkDefaultGateway(onvif_data.data);
    getDNS(onvif_data.data);
    getVideoEncoderConfigurationOptions(onvif_data.data);
    getVideoEncoderConfiguration(onvif_data.data);
    getOptions(onvif_data.data);
    getImagingSettings(onvif_data.data);
    std::cout << "PyFill mid: " << onvif_data.data->serial_number << std::endl;
    filled(onvif_data);
    std::cout << "PyFill finish" << std::endl;
}

void Manager::startPyDiscover()
{
    std::cout << "startPyDiscover: " << std::endl;
    std::thread thread([&]() { pyDiscover(); });
    thread.detach();
}

void Manager::pyDiscover()
{
    Session session;
    int number_of_devices = broadcast(session);
    std::cout << "n: " << number_of_devices << std::endl;

    for (int i = 0; i < number_of_devices; i++) {
        Data data;
        if (prepareOnvifData(i, session, data)) {
            std::cout << "recvd data: " << data.data->xaddrs << std::endl;
            //std::string username = "admin";
            //strncpy(data.data->username, username.c_str(), username.length() );
            data = getCredential(data);
            std::cout << "recvd username: " << data.data->username << std::endl;
            std::cout << "recvd password: " << data.data->password << std::endl;
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

    discovered();
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
    std::thread thread(updateNetwork, onvif_data, fcn);
    thread.detach();   
}

void Manager::updateNetwork(const Data& onvif_data, std::function<void(const Data&)> fcn)
{
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
    std::thread thread(updateTime, onvif_data, fcn);
    thread.detach();
}

void Manager::updateTime(const Data& onvif_data, std::function<void(const Data&)> fcn)
{
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
