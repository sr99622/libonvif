/*******************************************************************************
* onvif-util.c
*
* Copyright (c) 2022 Stephen Rhodes 
*
* This program is free software; you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation; either version 2 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along
* with this program; if not, write to the Free Software Foundation, Inc.,
* 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*
*******************************************************************************/

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

int longopt = 0;

static struct option longopts[] = {
             { "user",       required_argument, NULL,      'u'},
             { "password",   required_argument, NULL,      'p'},
             { "all",        no_argument,       NULL,      'a'},
             { "help",       required_argument, NULL,      'h'},
             { NULL,         0,                 NULL,       0 }
     };

static const char *username;
static const char *password;

static void showAll()
{
	std::cout << "Looking for cameras on the network..." << std::endl;
	struct OnvifSession *onvif_session = (struct OnvifSession*)malloc(sizeof(struct OnvifSession));
    struct OnvifData *onvif_data = (struct OnvifData*)malloc(sizeof(struct OnvifData));
	initializeSession(onvif_session);
	int n = broadcast(onvif_session);
	std::cout << "Found " << n << " cameras" << std::endl;
	for (int i = 0; i < n; i++) {
		prepareOnvifData(i, onvif_session, onvif_data);
		char host[128];
		extractHost(onvif_data->xaddrs, host);
		getHostname(onvif_data);
		printf("%s %s(%s)\n",host,
			onvif_data->host_name,
			onvif_data->camera_name);
	}
	closeSession(onvif_session);
	free(onvif_session);
	free(onvif_data);
}

static void showHelp()
{
	std::cout << "\n  onvif-util help\n\n"
			  << "  To view all cameras on the network:\n"
			  << "  onvif-util -a\n\n"
			  << "  To login to a particular camera:\n"
			  << "  onvif-util -u username -p password ip_address\n\n"
			  << "  Once logged into the camera you can view data using the 'get' command followed by the data requested\n"
			  << "  The (n) indicates an optional profile index to apply the setting, otherwise the current profile is used\n\n"
			  << "    Data Retrieval Commands (start with get)\n\n"
			  << "      get rtsp \n"
			  << "      get capabilities\n"
			  << "      get time\n"
			  << "      get profiles\n"
			  << "      get profile (n)\n"
			  << "      get video (n)\n"
			  << "      get video options (n)\n"
			  << "      get imaging\n"
			  << "      get imaging options\n"
			  << "      get network\n\n"
			  << "    Parameter Setting Commands (start with set)\n\n"
			  << "      set resolution (n) - Resolution setting in the format widthxheight, must match the video option\n"
			  << "      set framerate (n)\n"
			  << "      set gov_length (n)\n"
			  << "      set bitrate (n)\n"
			  << "      set bightness value(required)\n"
			  << "      set contrast value(required)\n"
			  << "      set saturation value(required)\n"
			  << "      set sharpness value(required)\n"
			  << "      set ip_address value(required)\n"
			  << "      set default_gateway value(required)\n"
			  << "      set dns value(required)\n"
			  << "      set dhcp value(required) - Accepted settings are 'on' and off'\n"
			  << "      set password  value(required)\n\n"
			  << "    Maintenance Commands\n\n"
			  << "      sync_time 'zone'(optional) - Sync the camera time to the computer.  Optionally adjusts based on camera time zone\n"
			  << "      reboot\n\n"
			  << "    To Exit Camera Session\n\n"
			  << "      quit\n"
	          << std::endl;
}

std::string cat(char* arg1, char* arg2)
{
	std::string result(arg1);
	result += arg2;
	return result;
}

void show(const std::vector<std::string>& args)
{
	std::cout << "args size: " << args.size() << std::endl;
	for (int i = 0; i < args.size(); i++) {
		std::cout << "args[" << i << "] = " << args[i] << std::endl;
	}
}

void profileCheck(OnvifData* onvif_data, const std::vector<std::string>& args)
{
	int index = 0;
	if (args.size() > 1) {
		index = std::stoi(args[1]);
		if (getProfileToken(onvif_data, index)) throw std::exception(cat("get profile token - ", onvif_data->last_error).data());
		std::cout << "  Profile set to " << onvif_data->profileToken << "\n" << std::endl;
	}
	else {
		if (!strcmp(onvif_data->profileToken, "")) {
			if (getProfileToken(onvif_data, index)) throw std::exception(cat("get profile token - ", onvif_data->last_error).data());
			std::cout << "  Profile set to " << onvif_data->profileToken << "\n" << std::endl;
		}
		else {
			std::cout << std::endl;
		}
	}
	if (getProfile(onvif_data)) throw std::exception(cat("get profile - ", onvif_data->last_error).data());
}

int main(int argc, char **argv)
{
	int ch;
	char *arg0 = argv[0];

	while ((ch = getopt_long(argc, argv, "u:p:ah", longopts, NULL)) != -1) {
		switch (ch) {
            case 'u':
				username = optarg;
				break;
			case 'p':
				password = optarg;
				break;
			case 'a':
				showAll();
				exit(0);
			case 'h':
				showHelp();
				exit(0);
			case 0:
				std::cout << "test 0:" << optarg << std::endl;
				break;
			default:
				std::cout << "test default" << optarg << std::endl;
				exit(1);
		}
	}
	
	argc -= optind;
    argv += optind;

	if (argc < 1) {
		std::cout << "NO ARGS" << std::endl;
		exit(1);
	}

	char *wanted = argv++[0];

	struct OnvifSession *onvif_session = (struct OnvifSession*)malloc(sizeof(struct OnvifSession));
	struct OnvifData *onvif_data = (struct OnvifData*)malloc(sizeof(struct OnvifData));
	initializeSession(onvif_session);
	int n = broadcast(onvif_session);
	for (int i = 0; i < n; i++) {
		prepareOnvifData(i, onvif_session, onvif_data);
		char host[128];
		extractHost(onvif_data->xaddrs, host);
		getHostname(onvif_data);
		if (!strcmp(host, wanted)) {
			std::cout << "  found host: " << host << std::endl;
			strcpy(onvif_data->username, username);
			strcpy(onvif_data->password, password);
			if (getDeviceInformation(onvif_data)  == 0) {
				std::cout << "  successfully connected to host" << "\n";
				std::cout << "    name:   " << onvif_data->camera_name << "\n";
				std::cout << "    serial: " << onvif_data->serial_number << "\n" << std::endl;
	
				// Initializing the session properly with the camera requires calling getCapabilities
				if (getCapabilities(onvif_data)) {
					std::cout << "ERROR: get capabilities - " << onvif_data->last_error << "\n" << std::endl;
					exit(1);
				}
				break;
			}
			else {
				std::cout << "ERROR: get device information - " << onvif_data->last_error << "\n" << std::endl;
				exit(1);
			}
		}
		if (i == n - 1) {
			std::cout << "ERROR: camera " << wanted << " not found" << "\n" << std::endl;
			exit(1);
		}
	}

	char kybd_buf[128] = {0};
	while (strcmp(kybd_buf, "quit")) {
		memset(kybd_buf, 0, 128);
		fgets(kybd_buf, 128, stdin);
		kybd_buf[strcspn(kybd_buf, "\r\n")] = 0;

		std::string cmd = std::string(kybd_buf);
		if (cmd.length() == 0)
			continue;
		std::string arg;
		std::vector<std::string> args;
		std::stringstream ss(cmd);
		while (ss >> arg)
			args.push_back(arg);

		try {
			if (args[0] == "get") {

				args.erase(args.begin());

				if (args[0] == "rtsp") {
					profileCheck(onvif_data, args);
					if (getStreamUri(onvif_data)) throw std::exception(cat("get stream uri - ", onvif_data->last_error).data());
					std::cout << "  RTSP " << onvif_data->stream_uri << "\n" << std::endl;
				}
				else if (args[0] == "capabilities") {
					std::cout << "  event_service:   " << onvif_data->event_service << "\n";
					std::cout << "  imaging_service: " << onvif_data->imaging_service << "\n";
					std::cout << "  media_service:   " << onvif_data->imaging_service << "\n";
					std::cout << "  ptz_service:     " << onvif_data->imaging_service << "\n" << std::endl;
				}
				else if (args[0] == "profiles") {
					int index = 0;
					bool looking = true;
					while (looking) {
						memset(onvif_data->profileToken, 0, 128);
						if (getProfileToken(onvif_data, index)) throw std::exception(cat("get profile token - ", onvif_data->last_error).data());
						if (strlen(onvif_data->profileToken) == 0) 
							looking = false;
						else
							std::cout << "  Token " << index << ": " << onvif_data->profileToken << "\n";
							index++;
					}
					std::cout << std::endl;
				}
				else if (args[0] == "profile") {
					profileCheck(onvif_data, args);
					std::cout << "  Width:      " << onvif_data->width << "\n";
					std::cout << "  Height:     " << onvif_data->height << "\n";
					std::cout << "  Frame Rate: " << onvif_data->frame_rate << "\n";
					std::cout << "  Gov Length: " << onvif_data->gov_length << "\n";
					std::cout << "  Bitrate:    " << onvif_data->bitrate << "\n" << std::endl;
				}
				else if (args[0] == "time") {
					if (getTimeOffset(onvif_data)) throw std::exception(cat("get time offset - ", onvif_data->last_error).data());
					std::cout << "  Time Offset: " << onvif_data->time_offset << " seconds" << "\n";
					std::cout << "  Timezone:    " << onvif_data->timezone << "\n";
					std::cout << "  DST:         " << (onvif_data->dst ? "Yes" : "No") << "\n" << std::endl;
				}
				else if (args[0] == "video") {

					if (args[1] == "options") {
						args.erase(args.begin());
						profileCheck(onvif_data, args);
						if(getVideoEncoderConfigurationOptions(onvif_data)) throw std::exception(cat("get video encoder configuration options - ", onvif_data->last_error).data());
						int size = 0;
						bool found_size = false;
						while (!found_size) {
							if (strlen(onvif_data->resolutions_buf[size]) == 0) {
								found_size = true;
							}
							else {
								size++;
								if (size > 15)
									found_size = true;
							}
						}

						std::cout << "  Available Resolutions" << std::endl;
						for (int i=0; i<size; i++) {
							std::cout << "    " << onvif_data->resolutions_buf[i] << std::endl;
						}

						std::cout <<  "  Min Gov Length: " << onvif_data->gov_length_min << "\n";
						std::cout <<  "  Max Gov Length: " << onvif_data->gov_length_max << "\n";
						std::cout <<  "  Min Frame Rate: " << onvif_data->frame_rate_min << "\n";
						std::cout <<  "  Max Frame Rate: " << onvif_data->frame_rate_max << "\n";
						std::cout <<  "  Min Bit Rate: " << onvif_data->bitrate_min << "\n";
						std::cout <<  "  Max Bit Rate: " << onvif_data->bitrate_max << "\n" << std::endl;
					}
					else {
						profileCheck(onvif_data, args);
						if(getVideoEncoderConfigurationOptions(onvif_data)) throw std::exception(cat("get video encoder configuration options - ", onvif_data->last_error).data());
						std::cout << "  Resolution: " << onvif_data->width << " x " << onvif_data->height << "\n";
						std::cout << "  Frame Rate: " << onvif_data->frame_rate << "\n";
						std::cout << "  Gov Length: " << onvif_data->gov_length << "\n";
						std::cout << "  Bit Rate:   " << onvif_data->bitrate << "\n" << std::endl;
					}
				}
				else if (args[0] == "imaging") {

					if (args[1] == "options") {
						args.erase(args.begin());
						profileCheck(onvif_data, args);
						if (getOptions(onvif_data)) throw std::exception(cat("get options - ", onvif_data->last_error).data());
						std::cout << "  Min Brightness: " << onvif_data->brightness_min << "\n";
						std::cout << "  Max Brightness: " << onvif_data->brightness_max << "\n";
						std::cout << "  Min ColorSaturation: " << onvif_data->saturation_min << "\n";
						std::cout << "  Max ColorSaturation: " << onvif_data->saturation_max << "\n";
						std::cout << "  Min Contrast: " << onvif_data->contrast_min << "\n";
						std::cout << "  Max Contrast: " << onvif_data->contrast_max << "\n";
						std::cout << "  Min Sharpness: " << onvif_data->sharpness_min << "\n";
						std::cout << "  Max Sharpness: " << onvif_data->sharpness_max << "\n" << std::endl;
					}
					else {
						profileCheck(onvif_data, args);
						if (getImagingSettings(onvif_data)) throw std::exception(cat("get imaging settings - ", onvif_data->last_error).data());

						std::cout << "  Brightness: " << onvif_data->brightness << "\n";
						std::cout << "  Contrast:   " << onvif_data->contrast << "\n";
						std::cout << "  Saturation: " << onvif_data->saturation << "\n";
						std::cout << "  Sharpness:  " << onvif_data->sharpness << "\n" << std::endl;
					}
				}
				else if (args[0] == "network") {
					profileCheck(onvif_data, args);
					if (getNetworkInterfaces(onvif_data)) throw std::exception(cat("get network interfaces - ", onvif_data->last_error).data());
					if (getNetworkDefaultGateway(onvif_data)) throw std::exception(cat("get network default gateway - ", onvif_data->last_error).data());
					if (getDNS(onvif_data)) throw std::exception(cat("get DNS - ", onvif_data->last_error).data());

					std::cout << "  IP Address: " << onvif_data->ip_address_buf << "\n";
					std::cout << "  Gateway:    " << onvif_data->default_gateway_buf << "\n";
					std::cout << "  DNS:        " << onvif_data->dns_buf << "\n";
					std::cout << "  DHCP:       " << (onvif_data->dhcp_enabled ? "YES" : "NO") << "\n" << std::endl;
				}
				else { 
					std::cout << "  Unrecognized command, use onvif-util -h to see help\n" << std::endl;
				}
			}
			else if (args[0] == "set") {
				args.erase(args.begin());
				if (args[0] == "brightness") {
					if (args.size() > 1) {
						args.erase(args.begin());
						profileCheck(onvif_data, args);
						if (getImagingSettings(onvif_data)) throw std::exception(cat("get imaging settings - ", onvif_data->last_error).data());
						int value = stoi(args[0]);
						onvif_data->brightness = value;
						if (setImagingSettings(onvif_data)) throw std::exception(cat("set brightness - ", onvif_data->last_error).data());
						std::cout << "  Brightness was set to " << value << "\n" << std::endl;
					}
					else {
						std::cout << "  Missing value for brightness\n" << std::endl;
					}
				}
				else if (args[0] == "contrast") {
					if (args.size() > 1) {
						args.erase(args.begin());
						profileCheck(onvif_data, args);
						if (getImagingSettings(onvif_data)) throw std::exception(cat("get imaging settings - ", onvif_data->last_error).data());
						int value = stoi(args[0]);
						onvif_data->contrast = value;
						if (setImagingSettings(onvif_data)) throw std::exception(cat("set contrast - ", onvif_data->last_error).data());
						std::cout << "  Contrast was set to " << value << "\n" << std::endl;
					}
					else {
						std::cout << "  Missing value for contrast\n" << std::endl;
					}
				}
				else if (args[0] == "saturation") {
					if (args.size() > 1) {
						args.erase(args.begin());
						profileCheck(onvif_data, args);
						if (getImagingSettings(onvif_data)) throw std::exception(cat("get imaging settings - ", onvif_data->last_error).data());
						int value = stoi(args[0]);
						onvif_data->saturation = value;
						if (setImagingSettings(onvif_data)) throw std::exception(cat("set saturation - ", onvif_data->last_error).data());
						std::cout << "  Saturation was set to " << value << "\n" << std::endl;
					}
					else {
						std::cout << "  Missing value for saturation\n" << std::endl;
					}
				}
				else if (args[0] == "sharpness") {
					if (args.size() > 1) {
						args.erase(args.begin());
						profileCheck(onvif_data, args);
						if (getImagingSettings(onvif_data)) throw std::exception(cat("get imaging settings - ", onvif_data->last_error).data());
						int value = stoi(args[0]);
						onvif_data->sharpness = value;
						if (setImagingSettings(onvif_data)) throw std::exception(cat("set sharpness - ", onvif_data->last_error).data());
						std::cout << "  Sharpness was set to " << value << "\n" << std::endl;
					}
					else {
						std::cout << "  Missing value for sharpness\n" << std::endl;
					}
				}
				else if (args[0] == "resolution") {
					std::cout << "resolution" << std::endl;
					if (args.size() > 1) {
						args.erase(args.begin());
						profileCheck(onvif_data, args);
						std::string delim = "x";
						std::size_t found = args[0].find(delim);
						if (found != std::string::npos) {
							if(getVideoEncoderConfiguration(onvif_data)) throw std::exception(cat("get video encoder configuration - ", onvif_data->last_error).data());
							onvif_data->width = stoi(args[0].substr(0, found));
							onvif_data->height = stoi(args[0].substr(found+1));
							if (setVideoEncoderConfiguration(onvif_data)) throw std::exception(cat("set video encoder configuration - ", onvif_data->last_error).data());
							std::cout << "  Resolution was set to " << onvif_data->width << " x " << onvif_data->height << "\n" << std::endl;
						}
						else {
							std::cout << "  Syntax error, proper format for the argument is widthxheight e.g. 1280x720" << std::endl;
						}
					}
 				}
				else if (args[0] == "gov_length") {
					if (args.size() > 1) {
						args.erase(args.begin());
						profileCheck(onvif_data, args);
						if(getVideoEncoderConfiguration(onvif_data)) throw std::exception(cat("get video encoder configuration - ", onvif_data->last_error).data());
						int value = stoi(args[0]);
						onvif_data->gov_length = value;
						if (setVideoEncoderConfiguration(onvif_data)) throw std::exception(cat("set video encoder configuration - ", onvif_data->last_error).data());
						std::cout << "  Gov Length was set to " << value << "\n" << std::endl;
					}
					else {
						std::cout << "  Missing value for Gov Length\n" << std::endl;
					}
				}
				else if (args[0] == "framerate") {
					if (args.size() > 1) {
						args.erase(args.begin());
						profileCheck(onvif_data, args);
						if(getVideoEncoderConfiguration(onvif_data)) throw std::exception(cat("get video encoder configuration - ", onvif_data->last_error).data());
						int value = stoi(args[0]);
						onvif_data->frame_rate = value;
						if (setVideoEncoderConfiguration(onvif_data)) throw std::exception(cat("set video encoder configuration - ", onvif_data->last_error).data());
						std::cout << "  Frame Rate was set to " << value << "\n" << std::endl;
					}
					else {
						std::cout << "  Missing value for Frame Rate\n" << std::endl;
					}
				}
				else if (args[0] == "bitrate") {
					if (args.size() > 1) {
						args.erase(args.begin());
						profileCheck(onvif_data, args);
						if(getVideoEncoderConfiguration(onvif_data)) throw std::exception(cat("get video encoder configuration - ", onvif_data->last_error).data());
						int value = stoi(args[0]);
						onvif_data->bitrate = value;
						if (setVideoEncoderConfiguration(onvif_data)) throw std::exception(cat("set video encoder configuration - ", onvif_data->last_error).data());
						std::cout << "  Bitrate was set to " << value << "\n" << std::endl;
					}
					else {
						std::cout << "  Missing value for Bitrate\n" << std::endl;
					}
				}
				else if (args[0] == "dhcp") {
					if (args.size() > 1) {
						args.erase(args.begin());
						if (args[0] == "on") {
							profileCheck(onvif_data, args);
							if (getNetworkInterfaces(onvif_data)) throw std::exception(cat("get network interfaces - ", onvif_data->last_error).data());
							if (onvif_data->dhcp_enabled) {
								std::cout << "  DHCP is already enabled\n" << std::endl;
							}
							else {
								onvif_data->dhcp_enabled = true;
								if (setNetworkInterfaces(onvif_data)) throw std::exception(cat("set network interfaces - ", onvif_data->last_error).data());
								std::cout << "  DHCP was enabled successfully\n\n"
										  << "  Camera may or may not reboot depending on settings\n"
										  << "  Session is being terminated.\n" << std::endl;
								exit(0);
							}
						}
						else if (args[0] == "off") {
							profileCheck(onvif_data, args);
							if (getNetworkInterfaces(onvif_data)) throw std::exception(cat("get network interfaces - ", onvif_data->last_error).data());
							if (!onvif_data->dhcp_enabled) {
								std::cout << "  DHCP is already disabled\n" << std::endl;
							}
							else {
								onvif_data->dhcp_enabled = false;
								if (setNetworkInterfaces(onvif_data)) throw std::exception(cat("set network interfaces - ", onvif_data->last_error).data());
								std::cout << "  DHCP disabled\n" << std::endl;
							}
						}
						else {
							std::cout << "  Invalid value for DHCP, use either 'on' or 'off\n" << std::endl;
						}
					}
					else {
						std::cout << "  Missing value for DHCP\n" << std::endl;
					}
				}
				else if (args[0] == "ip_address") {
					if (args.size() > 1) {
						args.erase(args.begin());
						profileCheck(onvif_data, args);
						if (getNetworkInterfaces(onvif_data)) throw std::exception(cat("get network interfaces - ", onvif_data->last_error).data());
						if (onvif_data->dhcp_enabled) {
							std::cout << "  Camera DHCP is enabled, IP address may not be set manually\n" << std::endl;
						}
						else {
							strcpy(onvif_data->ip_address_buf, args[0].c_str());
							if (setNetworkInterfaces(onvif_data)) throw std::exception(cat("set network interfaces - ", onvif_data->last_error).data());
							std::cout << "  IP Address has been changed, session will need to be restarted\n" << std::endl;
							exit(0);
						}
					}
					else {
						std::cout << "  Missing value for IP address\n" << std::endl;
					}
				}
				else if (args[0] == "default_gateway") {
					if (args.size() > 1) {
						args.erase(args.begin());
						profileCheck(onvif_data, args);
						if (getNetworkDefaultGateway(onvif_data)) throw std::exception(cat("get network default gateway - ", onvif_data->last_error).data());
						if (onvif_data->dhcp_enabled) {
							std::cout << "  Camera DHCP is enabled, default gateway may not be set manually\n" << std::endl;
						}
						else {
							strcpy(onvif_data->default_gateway_buf, args[0].c_str());
							if (setNetworkDefaultGateway(onvif_data)) throw std::exception(cat("set default gateway - ", onvif_data->last_error).data());
							std::cout << "  Default gateway has been changed\n" << std::endl;
						}
					}
					else {
						std::cout << "  Missing value for default gateway\n" << std::endl;
					}
				}
				else if (args[0] == "dns") {
					if (args.size() > 1) {
						args.erase(args.begin());
						profileCheck(onvif_data, args);
						if (getDNS(onvif_data)) throw std::exception(cat("get DNS - ", onvif_data->last_error).data());
						if (onvif_data->dhcp_enabled) {
							std::cout << "  Camera DHCP is enabled, DNS may not be set manually\n" << std::endl;
						}
						else {
							strcpy(onvif_data->dns_buf, args[0].c_str());
							if (setDNS(onvif_data)) throw std::exception(cat("set DNS - ", onvif_data->last_error).data());
							std::cout << "  DNS has been changed\n" << std::endl;
						}
					}
					else {
						std::cout << "  Missing value for DNS\n" << std::endl;
					}
				}
				else if (args[0] == "password") {
					std::cout << "password" << std::endl;
					if (args.size() > 1) {
						args.erase(args.begin());
						profileCheck(onvif_data, args);
						char buf[128] = {0};
						strcpy(buf, args[0].c_str());
						if (setUser(buf, onvif_data)) throw std::exception(cat("set user - ", onvif_data->last_error).data());
						std::cout << "  Admin password has been reset\n" << std::endl;
					}
					else {
						std::cout << "  Missing value for admin password\n" << std::endl;
 					}
				}
				else { 
					std::cout << "  Unrecognized command, use onvif-util -h to see help\n" << std::endl;
				}
			}
			else if (args[0] == "reboot") {
				std::cout << "  Are you sure you want to reboot?  Type yes to confirm\n" << std::endl;
				memset(kybd_buf, 0, 128);
				fgets(kybd_buf, 128, stdin);
				kybd_buf[strcspn(kybd_buf, "\r\n")] = 0;
				std::string reply(kybd_buf);
				if (reply == "yes") {
					if (rebootCamera(onvif_data)) throw std::exception(cat("reboot camera - ", onvif_data->last_error).data());
					std::cout << "  Camera is rebooting...\n" 
					          << "  Session will be terminated" << std::endl;
				}
				else {
					std::cout << "  Confirmation not received, reboot cancelled\n" << std::endl;
				}
			}
			else if (args[0] == "sync_time") {
				std::cout << "sync_time" << std::endl;
				if (args.size() > 1) {
					args.erase(args.begin());
					profileCheck(onvif_data, args);
					if (args[0] == "zone") {
						profileCheck(onvif_data, args);
						if (setSystemDateAndTimeUsingTimezone(onvif_data)) throw std::exception(cat("set system date and time using timezone - ", onvif_data->last_error).data());
						std::cout << "  Camera date and time has been synchronized using the camera timezone\n" << std::endl;
					}
				}
				else {
					profileCheck(onvif_data, args);
					if (setSystemDateAndTime(onvif_data)) throw std::exception(cat("set system date and time - ", onvif_data->last_error).data());
					std::cout << "  Camera date and time has been synchronized without regard to camera timezone\n" << std::endl;
				}
			}
			else { 
				if (strcmp(kybd_buf, "quit"))
					std::cout << " Unrecognized command, use onvif-util -h to see help\n" << std::endl;
			}
		}
		catch (std::exception& e) {
			std::cout << "  ERROR: " << e.what() << "\n" << std::endl;
		}
	}
}

/*
else if (args[0] == "video") {
	profileCheck(onvif_data, args);
	if (getVideoEncoderConfigurationOptions(onvif_data)) throw std::exception(cat("get video encoder configuration options - ", onvif_data->last_error).data());
	std::cout << "  Name: " << onvif_data->video_encoder_name_buf << "\n";
	std::cout << "  UseCount: " << onvif_data->use_count << "\n";
	std::cout << "  GuaranteedFrameRate: " << (onvif_data->guaranteed_frame_rate?"true":"false") << "\n";
	std::cout << "  Encoding: " << onvif_data->encoding << "\n";
	std::cout << "  Resolution:Width: " << onvif_data->conf_width << "\n";
	std::cout << "  Resolution:Height: " << onvif_data->conf_height << "\n";
	std::cout << "  Quality: " << onvif_data->quality << "\n";
	std::cout << "  RateControl:FrameRateLimit: " << onvif_data->conf_frame_rate_limit << "\n";
	std::cout << "  RateControl:EncodingInterval: " << onvif_data->conf_encoding_interval << "\n";
	std::cout << "  RateControl:BitrateLimit: " << onvif_data->conf_bitrate_limit << "\n";
	std::cout << "  H264Profile: " << onvif_data->h264_profile_buf << "\n";
	std::cout << "  Multicast:AddressType: " << onvif_data->multicast_address_type_buf << "\n";
	if (strcmp(onvif_data->multicast_address_type_buf,"IPv6") == 0)
		std::cout << "  Multicast:IPv6Address: " << onvif_data->multicast_address_buf << "\n";
	else
		std::cout << "  Multicast:IPv4Address: " << onvif_data->multicast_address_buf << "\n";
	std::cout << "  Multicast:Port: " << onvif_data->multicast_port << "\n";
	std::cout << "  Multicast:TTL: " << onvif_data->multicast_ttl << "\n";
	std::cout << "  Multicast:AutoStart: " << (onvif_data->autostart?"true":"false") << "\n";
	std::cout << "  SessionTimeout: " << onvif_data->session_time_out_buf << "\n" << std::endl;
}
*/
