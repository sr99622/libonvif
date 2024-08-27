#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <sstream>
#include <stdio.h>
#include <string.h>
#include "onvif.h"
#ifdef _WIN32
#include "getopt-win.h"
#else
#include <getopt.h>
#endif

int main(int argc, char **argv)
{
	std::cout << "Looking for cameras on the network..." << std::endl;

	struct OnvifSession *onvif_session = (struct OnvifSession*)calloc(sizeof(struct OnvifSession), 1);

	getActiveNetworkInterfaces(onvif_session);
	for (int i = 0; i < 16; i++) {
		std::cout << "interface: " << onvif_session->active_network_interfaces[i] << std::endl;
	}

	std::string delimiter = " - ";
	std::string thingy = onvif_session->active_network_interfaces[0];
	std::string token = thingy.substr(0, thingy.find(delimiter));
	std::cout << "---" << token << "---" << std::endl;
	strcpy(onvif_session->preferred_network_address, "10.1.1.1");

    struct OnvifData *tmp_onvif_data = (struct OnvifData*)calloc(sizeof(struct OnvifData), 1);
	struct OnvifData *onvif_data = (struct OnvifData*)calloc(sizeof(struct OnvifData), 1);

	initializeSession(onvif_session);
	int n = broadcast(onvif_session);
	std::cout << "Found " << n << " cameras" << std::endl;
	for (int i = 0; i < n; i++) {
		if (prepareOnvifData(i, onvif_session, tmp_onvif_data)) {
			char host[128];
			extractHost(tmp_onvif_data->xaddrs, host);
			getHostname(tmp_onvif_data);
			printf("%s %s(%s)\n",host,
				tmp_onvif_data->host_name,
				tmp_onvif_data->camera_name);

			if (!strcmp(host, "10.1.1.67")) {
				std::cout << "FOUND HOST" << tmp_onvif_data->camera_name << std::endl;
				copyData(onvif_data, tmp_onvif_data);
			}
		}
		else {
			std::cout << "found invalid xaddrs in device repsonse" << std::endl;
		}
	}

	closeSession(onvif_session);
	free(onvif_session);
	free(tmp_onvif_data);

	std::cout << "subject camera - " << onvif_data->camera_name << std::endl;

	strcpy(onvif_data->username, "admin");
	strcpy(onvif_data->password, "admin123");
	if (getDeviceInformation(onvif_data))
		std::cout << "getDeviceInformation failure " << onvif_data->last_error << std::endl;

	if (getCapabilities(onvif_data))
		std::cout << "getCapabilities failure " << onvif_data->last_error << std::endl;

	if (getProfileToken(onvif_data, 0))
		std::cout << "getProfileToken failure " << onvif_data->last_error << std::endl;

	if (getProfile(onvif_data))
		std::cout << "getProfile failure " << onvif_data->last_error << std::endl;

	if (setSystemDateAndTime(onvif_data))
		std::cout << "setSystemDateAndTime failure " << onvif_data->last_error << std::endl;

	if (getStreamUri(onvif_data))
		std::cout << "getStreamUri failure " << onvif_data->last_error << std::endl;

	std::cout << onvif_data->stream_uri << std::endl;

	if(getVideoEncoderConfiguration(onvif_data)) 
		std::cout << "getVideoEncoderConfiguration failure " << onvif_data->last_error << std::endl;

	std::cout << "  Width:      " << onvif_data->width << "\n";
	std::cout << "  Height:     " << onvif_data->height << "\n";
	std::cout << "  Frame Rate: " << onvif_data->frame_rate << "\n";
	std::cout << "  Gov Length: " << onvif_data->gov_length << "\n";
	std::cout << "  Bitrate:    " << onvif_data->bitrate << "\n" << std::endl;

	if (getOptions(onvif_data)) 
		std::cout << "getOptions failure " << onvif_data->last_error << std::endl;

	std::cout << "  Min Brightness: " << onvif_data->brightness_min << "\n";
	std::cout << "  Max Brightness: " << onvif_data->brightness_max << "\n";
	std::cout << "  Min ColorSaturation: " << onvif_data->saturation_min << "\n";
	std::cout << "  Max ColorSaturation: " << onvif_data->saturation_max << "\n";
	std::cout << "  Min Contrast: " << onvif_data->contrast_min << "\n";
	std::cout << "  Max Contrast: " << onvif_data->contrast_max << "\n";
	std::cout << "  Min Sharpness: " << onvif_data->sharpness_min << "\n";
	std::cout << "  Max Sharpness: " << onvif_data->sharpness_max << "\n" << std::endl;

	if (getImagingSettings(onvif_data)) 
		std::cout << "getImagingSettings failure" << onvif_data->last_error << std::endl;

	std::cout << "  Brightness: " << onvif_data->brightness << "\n";
	std::cout << "  Contrast:   " << onvif_data->contrast << "\n";
	std::cout << "  Saturation: " << onvif_data->saturation << "\n";
	std::cout << "  Sharpness:  " << onvif_data->sharpness << "\n" << std::endl;

	if (getTimeOffset(onvif_data)) 
		std::cout << "getTimeOffset failure " << onvif_data->last_error << std::endl;

	std::cout << "  Time Offset: " << onvif_data->time_offset << " seconds" << "\n";
	std::cout << "  Timezone:    " << onvif_data->timezone << "\n";
	std::cout << "  DST:         " << (onvif_data->dst ? "Yes" : "No") << "\n";
	std::cout << "  Time Set By: " << ((onvif_data->datetimetype == 'M') ? "Manual" : "NTP") << "\n";

	if (getNetworkInterfaces(onvif_data)) 
		std::cout << "getNetworkInterfaces failure " << onvif_data->last_error << std::endl;
	if (getNetworkDefaultGateway(onvif_data)) 
		std::cout << "getNetworkDefaultGateway failure " << onvif_data->last_error << std::endl;
	if (getDNS(onvif_data)) 
		std::cout << "getDNS failure " << onvif_data->last_error << std::endl;

	std::cout << "  IP Address: " << onvif_data->ip_address_buf << "\n";
	std::cout << "  Gateway:    " << onvif_data->default_gateway_buf << "\n";
	std::cout << "  DNS:        " << onvif_data->dns_buf << "\n";
	std::cout << "  DHCP:       " << (onvif_data->dhcp_enabled ? "YES" : "NO") << "\n" << std::endl;

	if (getVideoEncoderConfigurationOptions(onvif_data))
		std::cout << "getVideoEncoderConfigurationOptions failure " << onvif_data->last_error << std::endl;

	std::cout << "  Available Resolutions" << std::endl;
	for (int i=0; i<16; i++) {
		if (strlen(onvif_data->resolutions_buf[i]))
			std::cout << "    " << onvif_data->resolutions_buf[i] << std::endl;
	}

	std::cout <<  "  Min Gov Length: " << onvif_data->gov_length_min << "\n";
	std::cout <<  "  Max Gov Length: " << onvif_data->gov_length_max << "\n";
	std::cout <<  "  Min Frame Rate: " << onvif_data->frame_rate_min << "\n";
	std::cout <<  "  Max Frame Rate: " << onvif_data->frame_rate_max << "\n";
	std::cout <<  "  Min Bit Rate: " << onvif_data->bitrate_min << "\n";
	std::cout <<  "  Max Bit Rate: " << onvif_data->bitrate_max << "\n" << std::endl;


	free(onvif_data);
}
