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

void Manager::startFill(std::function<void(Data&, int)> fcn, std::vector<Data> devices, int index)
{
    std::thread thread(fill, fcn, devices, index);
    thread.detach();
}

void Manager::fill(std::function<void(Data&, int)> fcn, std::vector<Data> devices, int index)
{
    std::cout << "Manager::fill" << std::endl;
    Data onvif_data = devices[index];
    getCapabilities(onvif_data);
    getNetworkInterfaces(onvif_data);
    getNetworkDefaultGateway(onvif_data);
    getDNS(onvif_data);
    getVideoEncoderConfigurationOptions(onvif_data);
    getVideoEncoderConfiguration(onvif_data);
    getOptions(onvif_data);
    getImagingSettings(onvif_data);

    fcn(onvif_data, index);
}
    
}
