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

void Manager::startPyUpdateVideo()
{
    std::cout << "startPyUpdateVideo" << std::endl;
    std::thread thread([&]() { pyUpdateVideo(); });
    thread.detach();
}

void Manager::pyUpdateVideo()
{
    std::cout << "pyUpdateVideo" << std::endl;
    setVideoEncoderConfiguration(onvif_data.data);
    getProfile(onvif_data.data);
    getVideoEncoderConfigurationOptions(onvif_data.data);
    getVideoEncoderConfiguration(onvif_data.data);
    filled(onvif_data);
}

void Manager::startPyUpdateImage()
{
    std::cout << "startPyUpdateImage" << std::endl;
    std::thread thread([&]() { pyUpdateImage(); });
    thread.detach();
}

void Manager::pyUpdateImage()
{
    std::cout << "pyUpdateImage" << std::endl;
    setImagingSettings(onvif_data.data);
    getOptions(onvif_data.data);
    getImagingSettings(onvif_data.data);
    filled(onvif_data);
}

void Manager::startPyUpdateNetwork()
{
    std::cout << "startPyUpdateNetwork" << std::endl;
    std::thread thread([&]() { pyUpdateNetwork(); });
    thread.detach();
}

void Manager::pyUpdateNetwork()
{
    std::cout << "pyUpdateNetwork" << std::endl;
    setNetworkInterfaces(onvif_data.data);
    setDNS(onvif_data.data);
    setNetworkDefaultGateway(onvif_data.data);
    getNetworkInterfaces(onvif_data.data);
    getNetworkDefaultGateway(onvif_data.data);
    getDNS(onvif_data.data);
    filled(onvif_data);
}

void Manager::startPyUpdateTime()
{
    std::cout << "startPyUpdateTime" << std::endl;
    std::thread thread([&]() { pyUpdateTime(); });
    thread.detach();
}

void Manager::pyUpdateTime()
{
    std::cout << "pyUpdateTime" << std::endl;
    setSystemDateAndTime(onvif_data.data);
    filled(onvif_data);
}

void Manager::startPyReboot()
{
    std::cout << "startPyReboot" << std::endl;
    std::thread thread([&]() { pyReboot(); });
    thread.detach();
}

void Manager::pyReboot()
{
    std::cout << "pyReboot" << std::endl;
    rebootCamera(onvif_data.data);
    filled(onvif_data);
}

void Manager::startPyReset()
{
    std::cout << "startPyReset" << std::endl;
    std::thread thread([&]() { pyReset(); });
    thread.detach();
}

void Manager::pyReset()
{
    std::cout << "pyReset" << std::endl;
    hardReset(onvif_data.data);
    filled(onvif_data);
}

void Manager::startPySetUser()
{
    std::cout << "startPySetUser" << std::endl;
    std::thread thread([&]() { pySetUser(); });
    thread.detach();
}

void Manager::pySetUser()
{
    if (setUser((char*)new_password.c_str(), onvif_data) == 0)
        onvif_data.setPassword(new_password.c_str());
    filled(onvif_data);
}

void Manager::startPyFill()
{
    std::cout << "startPyFill" << std::endl;
    std::thread thread([&]() { pyFill(); });
    thread.detach();
}

void Manager::pyFill()
{
    getCapabilities(onvif_data.data);
    getNetworkInterfaces(onvif_data.data);
    getNetworkDefaultGateway(onvif_data.data);
    getDNS(onvif_data.data);
    getVideoEncoderConfigurationOptions(onvif_data.data);
    getVideoEncoderConfiguration(onvif_data.data);
    getOptions(onvif_data.data);
    getImagingSettings(onvif_data.data);
    onvif_data.filled = true;
    filled(onvif_data);
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
    std::vector<Data> devices;

    for (int i = 0; i < number_of_devices; i++) {
        Data data;
        if (prepareOnvifData(i, session, data)) {
            if (std::find(devices.begin(), devices.end(), data) == devices.end()) {
                devices.push_back(data);
                while (true) {
                    data = getCredential(data);
                    if (!data.cancelled) {
                        if (fillRTSP(data) == 0) {
                            getProfile(data);
                            getDeviceInformation(data);
                            getData(data);
                            break;
                        }
                        else {
                            std::cout << "fillRTPS failed" << std::endl;
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
