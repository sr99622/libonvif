/*******************************************************************************
* onvif-util.cpp
*
* Copyright (c) 2022 Stephen Rhodes 
* Code contributions by Brian D Scott and Petter Reinholdtsen
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
#define VERSION "1.4.7"

static struct option longopts[] = {
             { "user",       required_argument, NULL,      'u'},
             { "password",   required_argument, NULL,      'p'},
             { "all",        no_argument,       NULL,      'a'},
			 { "safe_off",   no_argument,       NULL,      's'},
             { "help",       required_argument, NULL,      'h'},
			 { "version",    no_argument,       NULL,      'v'},
			 { "time_sync",  no_argument,       NULL,      't'},
             { NULL,         0,                 NULL,       0 }
     };

static const char *username = nullptr;
static const char *password = nullptr;

static void usage()
{
	std::cout << "Usage: onvif-util [-ahsv] [-u <user>] [-p <password>] [command]" << std::endl;
}

static void showAll()
{
	struct OnvifSession *onvif_session = (struct OnvifSession*)calloc(sizeof(struct OnvifSession), 1);
    struct OnvifData *onvif_data = (struct OnvifData*)calloc(sizeof(struct OnvifData), 1);

	initializeSession(onvif_session);
	getActiveNetworkInterfaces(onvif_session);
	int index = 0;

	while (true) {
		char * element = onvif_session->active_network_interfaces[index];
		if (strlen(element)) {
			char * ip_address = strtok(element, " - ");
			memset(onvif_session->preferred_network_address, 0, sizeof(onvif_session->preferred_network_address));
			strcpy(onvif_session->preferred_network_address, ip_address);
			
			int n = broadcast(onvif_session);
			std::cout << "Found " << n << " cameras on interface " << ip_address << std::endl;
			for (int i = 0; i < n; i++) {
				if (prepareOnvifData(i, onvif_session, onvif_data)) {
					char host[128];
					extractHost(onvif_data->xaddrs, host);
					getHostname(onvif_data);
					printf("%s %s(%s)\n",host,
						onvif_data->host_name,
						onvif_data->camera_name);
				}
				else {
					std::cout << "found invalid xaddrs in device response" << std::endl;
				}
			}

			index++;
			if (index > 16)
				break;
		}
		else {
			break;
		}
	}

	closeSession(onvif_session);
	free(onvif_session);
	free(onvif_data);
}

static void showHelp()
{
	std::cout << "\n  onvif-util help\n\n"
			  << "  Usage: onvif-util [-ahs] [-u <user>] [-p <password>] [command]\n\n"
			  << "    options\n"
			  << "      -a  poll all cameras on network and reply with host name\n"
			  << "      -u  username\n"
			  << "      -p  password\n"
			  << "      -s  safe mode off, enable applications for viewer and browser to run\n\n"
			  << "  To view all cameras on the network:\n"
			  << "  onvif-util -a\n\n"
			  << "  To login to a particular camera:\n"
			  << "  onvif-util -u username -p password ip_address\n\n"
			  << "  To login to a camera with safe mode disabled:\n"
			  << "  onvif-util -s -u username -p password ip_address\n\n"
			  << "  Once logged into the camera you can view data using the 'get' command followed by the data requested.\n"
			  << "  The (n) indicates an optional profile index to apply the setting, otherwise the current profile is used.\n\n"
			  << "    Data Retrieval Commands (start with get)\n\n"
			  << "      get rtsp 'pass'(optional) (n) - Get rtsp uri for camera, with optional password credential\n"
			  << "      get snapshot 'pass'(optional) (n) - Get snapshot uri for camera, with optional password credential\n"
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
			  << "      help\n"
			  << "      safe - set safe mode on.  Viewer and browser are disabled\n"
			  << "      unsafe - set safe mode off.  Viewer and browser are enabled\n"
			  << "      browser - Use browser to access camera configurations\n"
			  << "      view (n) - View the camera output using ffplay (this assumes you have ffplay installed in the path\n"
			  << "      view player (n) - View the camera output with user specified player e.g. view vlc\n"
			  << "      dump - Full set of raw data from camera configuration\n"
			  << "      sync_time 'zone'(optional) - Sync the camera time to the computer.  Optionally adjusts based on camera time zone\n"
			  << "      reboot\n\n"
			  << "    To Exit Camera Session\n\n"
			  << "      quit\n"
	          << std::endl;
}

const std::string cat(const char* arg1, const char* arg2)
{
	std::string result(arg1);
	result += arg2;
	return result;
}

std::string uri_with_pass(OnvifData* onvif_data)
{
	std::string uri(onvif_data->stream_uri);
	std::stringstream ss;
	ss << uri.substr(0, 7) << onvif_data->username << ":" << onvif_data->password << "@" << uri.substr(7);
	return ss.str();
}

std::string snapshot_with_pass(OnvifData* onvif_data)
{
	std::string uri(onvif_data->snapshot_uri);
	std::stringstream ss;
	ss << uri.substr(0, 7) << onvif_data->username << ":" << onvif_data->password << "@" << uri.substr(7);
	return ss.str();
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
		if (getProfileToken(onvif_data, index)) throw std::runtime_error(cat("get profile token - ", onvif_data->last_error));
		if (strlen(onvif_data->profileToken) == 0) throw std::runtime_error(cat("invalid profile token - ", (char*)std::to_string(index).c_str()).data());
		std::cout << "  Profile set to " << onvif_data->profileToken << "\n" << std::endl;
	}
	else {
		if (!strcmp(onvif_data->profileToken, "")) {
			if (getProfileToken(onvif_data, index)) throw std::runtime_error(cat("get profile token - ", onvif_data->last_error));
			if (strlen(onvif_data->profileToken) == 0) throw std::runtime_error(cat("invalid profile token - ", (char*)std::to_string(index).c_str()).data());
			std::cout << "  Profile set to " << onvif_data->profileToken << "\n" << std::endl;
		}
		else {
			std::cout << "  Profile set to " << onvif_data->profileToken << "\n" << std::endl;
		}
	}
	if (getProfile(onvif_data)) throw std::runtime_error(cat("get profile - ", onvif_data->last_error));
}

int main(int argc, char **argv)
{
	bool safe_mode = true;
	bool time_sync = false;

	int ch;
	while ((ch = getopt_long(argc, argv, "u:p:ahsvt", longopts, NULL)) != -1) {
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
				usage();
				showHelp();
				exit(0);
			case 's':
				safe_mode = false;
				break;
			case 'v':
				std::cout << "onvif-util version " << VERSION << std::endl;
				exit(0);
			case 't':
				time_sync = true;
				break;
			case 0:
				std::cout << optarg << std::endl;
				break;
			default:
				usage();
				exit(1);
		}
	}
	
	argc -= optind;
    argv += optind;

	if (argc < 1) {
		usage();
		exit(1);
	}

	char *wanted = argv++[0];

	struct OnvifSession *onvif_session = (struct OnvifSession*)calloc(sizeof(struct OnvifSession), 1);
	struct OnvifData *onvif_data = (struct OnvifData*)malloc(sizeof(struct OnvifData));

	initializeSession(onvif_session);
	getActiveNetworkInterfaces(onvif_session);

	int index = 0;
	bool found = false;

	while (!found) {
		char * element = onvif_session->active_network_interfaces[index];
		if (strlen(element)) {
			char * ip_address = strtok(element, " - ");
			memset(onvif_session->preferred_network_address, 0, sizeof(onvif_session->preferred_network_address));
			strcpy(onvif_session->preferred_network_address, ip_address);

			int n = broadcast(onvif_session);
			for (int i = 0; i < n; i++) {
				bool success = prepareOnvifData(i, onvif_session, onvif_data);
				if (!success) {
					std::cout << "found invalid xaddrs in device response for device " << i << std::endl;
					continue;
				}
				char host[128];
				extractHost(onvif_data->xaddrs, host);
				getHostname(onvif_data);
				if (!strcmp(host, wanted)) {
					std::cout << "  found host: " << host << std::endl;
					found = true;
					if (username) strcpy(onvif_data->username, username);
					if (password) strcpy(onvif_data->password, password);
					if (getDeviceInformation(onvif_data)  == 0) {
						std::cout << "  successfully connected to host" << "\n";
						std::cout << "    name:   " << onvif_data->camera_name << "\n";
						std::cout << "    serial: " << onvif_data->serial_number << "\n" << std::endl;
			
						// Initializing the session properly with the camera requires calling getCapabilities
						if (getCapabilities(onvif_data)) {
							std::cout << "ERROR: get capabilities - " << onvif_data->last_error << "\n" << std::endl;
							exit(1);
						}

						if (time_sync) {
							std::cout << "  Time sync requested" << std::endl;
							std::vector<std::string> tmp;
							profileCheck(onvif_data, tmp);
							if (setSystemDateAndTime(onvif_data)) throw std::runtime_error(cat("set system date and time - ", onvif_data->last_error));
							std::cout << "  Camera date and time has been synchronized without regard to camera timezone\n" << std::endl;
							exit(0);
						}
						break;
					}
					else {
						std::cout << "ERROR: get device information - " << onvif_data->last_error << "\n" << std::endl;
						exit(1);
					}
				}
				if (i == n - 1) {
					continue;
				}
			}

			index++;
			if (index > 16)
				continue;
		}
		else {
			break;
		}
	}

	if (!found) {
		std::stringstream xaddrs;
		xaddrs << "http://" << wanted << ":80/onvif/device_service";
		if (username) strcpy(onvif_data->username, username);
		if (password) strcpy(onvif_data->password, password);
		strcpy(onvif_data->xaddrs, xaddrs.str().c_str());
		strcpy(onvif_data->device_service, xaddrs.str().c_str());
		if (getCapabilities(onvif_data)) {
			std::cout << "ERROR: get capabilities - " << onvif_data->last_error << "\n" << std::endl;
			exit(1);
		}
		else {
			if (getDeviceInformation(onvif_data)  == 0) {
				std::cout << "  successfully connected to host" << "\n";
				std::cout << "    name:   " << onvif_data->camera_name << "\n";
				std::cout << "    serial: " << onvif_data->serial_number << "\n" << std::endl;
				found = true;
				if (time_sync) {
					std::cout << "  Time sync requested" << std::endl;
					std::vector<std::string> tmp;
					profileCheck(onvif_data, tmp);
					if (setSystemDateAndTime(onvif_data)) throw std::runtime_error(cat("set system date and time - ", onvif_data->last_error));
					std::cout << "  Camera date and time has been synchronized without regard to camera timezone\n" << std::endl;
					exit(0);
				}
			}
			else {
				std::cout << "Failed to connect to camera" << std::endl;
			}
		}
	}

	if (!found) {
		std::cout << "ERROR: camera " << wanted << " not found" << "\n" << std::endl;
		exit(1);
	}


	/*
	char kybd_buf[128] = {0};
	while (strcmp(kybd_buf, "quit")) {
		memset(kybd_buf, 0, 128);
		fgets(kybd_buf, 128, stdin);
		kybd_buf[strcspn(kybd_buf, "\r\n")] = 0;

		std::string cmd(kybd_buf);
	*/
	/////////////////////////

	std::string quit("quit");
#ifdef _WIN32
	quit = "quit\r";
#endif

	std::string cmd;
	while (cmd != quit) {
		std::cout << onvif_data->camera_name << "> ";
		if (!std::getline(std::cin, cmd))
			break;	
	/////////////////////////
	
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
					bool add_pass = false;
					if (args.size() > 1) {
						if (args[1] == "pass") {
							args.erase(args.begin());
							add_pass = true;
						}
					}
					profileCheck(onvif_data, args);
					if (getStreamUri(onvif_data)) throw std::runtime_error(cat("get stream uri - ", onvif_data->last_error));
					std::string uri(onvif_data->stream_uri);
					if (add_pass) {
						uri = uri_with_pass(onvif_data);
					}
					std::cout << "  " << uri << "\n" << std::endl;
				}
				else if (args[0] == "snapshot") {
					bool add_pass = false;
					if (args.size() > 1) {
						if (args[1] == "pass") {
							args.erase(args.begin());
							add_pass = true;
						}
					}
					profileCheck(onvif_data, args);
					if (getSnapshotUri(onvif_data)) throw std::runtime_error(cat("get snapshot - ", onvif_data->last_error));
					std::string uri(onvif_data->stream_uri);
					if (add_pass) {
						uri = snapshot_with_pass(onvif_data);
					}
					std::cout << "  " << uri << "\n" << std::endl;
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
						if (getProfileToken(onvif_data, index)) throw std::runtime_error(cat("get profile token - ", onvif_data->last_error));
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
					if(getVideoEncoderConfiguration(onvif_data)) throw std::runtime_error(cat("get video encoder configuration - ", onvif_data->last_error));
					std::cout << "  Width:      " << onvif_data->width << "\n";
					std::cout << "  Height:     " << onvif_data->height << "\n";
					std::cout << "  Frame Rate: " << onvif_data->frame_rate << "\n";
					std::cout << "  Gov Length: " << onvif_data->gov_length << "\n";
					std::cout << "  Bitrate:    " << onvif_data->bitrate << "\n" << std::endl;
				}
				else if (args[0] == "time") {
					if (getTimeOffset(onvif_data)) throw std::runtime_error(cat("get time offset - ", onvif_data->last_error));
					std::cout << "  Time Offset: " << onvif_data->time_offset << " seconds" << "\n";
					std::cout << "  Timezone:    " << onvif_data->timezone << "\n";
					std::cout << "  DST:         " << (onvif_data->dst ? "Yes" : "No") << "\n";
					std::cout << "  Time Set By: " << ((onvif_data->datetimetype == 'M') ? "Manual" : "NTP") << "\n";
					if (onvif_data->datetimetype != 'M') {
						if (getNTP(onvif_data)) throw std::runtime_error(cat("get NTP - ", onvif_data->last_error));
						std::cout << "  NTP Server:  " << onvif_data->ntp_addr << "\n";
					}
					std::cout << std::endl;
				}
				else if (args[0] == "video") {

					if (args.size() > 1) {
						if (args[1] == "options") {
							args.erase(args.begin());
							profileCheck(onvif_data, args);
							if(getVideoEncoderConfigurationOptions(onvif_data)) throw std::runtime_error(cat("get video encoder configuration options - ", onvif_data->last_error));
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
							if(getVideoEncoderConfiguration(onvif_data)) throw std::runtime_error(cat("get video encoder configuration - ", onvif_data->last_error));
							std::cout << "  Resolution: " << onvif_data->width << " x " << onvif_data->height << "\n";
							std::cout << "  Frame Rate: " << onvif_data->frame_rate << "\n";
							std::cout << "  Gov Length: " << onvif_data->gov_length << "\n";
							std::cout << "  Bit Rate:   " << onvif_data->bitrate << "\n" << std::endl;
						}
					}
					else {
						profileCheck(onvif_data, args);
						if(getVideoEncoderConfiguration(onvif_data)) throw std::runtime_error(cat("get video encoder configuration - ", onvif_data->last_error));
						std::cout << "  Resolution: " << onvif_data->width << " x " << onvif_data->height << "\n";
						std::cout << "  Frame Rate: " << onvif_data->frame_rate << "\n";
						std::cout << "  Gov Length: " << onvif_data->gov_length << "\n";
						std::cout << "  Bit Rate:   " << onvif_data->bitrate << "\n" << std::endl;
					}
				}
				else if (args[0] == "imaging") {

					if (args.size() > 1) {
						if (args[1] == "options") {
							args.erase(args.begin());
							profileCheck(onvif_data, args);
							if (getOptions(onvif_data)) throw std::runtime_error(cat("get options - ", onvif_data->last_error));
							std::cout << "  Min Brightness: " << onvif_data->brightness_min << "\n";
							std::cout << "  Max Brightness: " << onvif_data->brightness_max << "\n";
							std::cout << "  Min ColorSaturation: " << onvif_data->saturation_min << "\n";
							std::cout << "  Max ColorSaturation: " << onvif_data->saturation_max << "\n";
							std::cout << "  Min Contrast: " << onvif_data->contrast_min << "\n";
							std::cout << "  Max Contrast: " << onvif_data->contrast_max << "\n";
							std::cout << "  Min Sharpness: " << onvif_data->sharpness_min << "\n";
							std::cout << "  Max Sharpness: " << onvif_data->sharpness_max << "\n" << std::endl;
						}
					}
					else {
						profileCheck(onvif_data, args);
						if (getImagingSettings(onvif_data)) throw std::runtime_error(cat("get imaging settings - ", onvif_data->last_error));

						std::cout << "  Brightness: " << onvif_data->brightness << "\n";
						std::cout << "  Contrast:   " << onvif_data->contrast << "\n";
						std::cout << "  Saturation: " << onvif_data->saturation << "\n";
						std::cout << "  Sharpness:  " << onvif_data->sharpness << "\n" << std::endl;
					}
				}
				else if (args[0] == "network") {
					profileCheck(onvif_data, args);
					if (getNetworkInterfaces(onvif_data)) throw std::runtime_error(cat("get network interfaces - ", onvif_data->last_error));
					if (getNetworkDefaultGateway(onvif_data)) throw std::runtime_error(cat("get network default gateway - ", onvif_data->last_error));
					if (getDNS(onvif_data)) throw std::runtime_error(cat("get DNS - ", onvif_data->last_error));

					std::cout << "  IP Address: " << onvif_data->ip_address_buf << "\n";
					std::cout << "  Gateway:    " << onvif_data->default_gateway_buf << "\n";
					std::cout << "  DNS:        " << onvif_data->dns_buf << "\n";
					std::cout << "  DHCP:       " << (onvif_data->dhcp_enabled ? "YES" : "NO") << "\n" << std::endl;
				}
				else { 
					//std::cout << "  Unrecognized command, use onvif-util -h to see help\n" << std::endl;
					std::cout << "  Unrecognized command \"" << args[0] << "\", type \"help\" to see help\n" << std::endl;
				}
			}
			else if (args[0] == "set") {
				args.erase(args.begin());
				if (args[0] == "brightness") {
					if (args.size() > 1) {
						args.erase(args.begin());
						profileCheck(onvif_data, args);
						if (getImagingSettings(onvif_data)) throw std::runtime_error(cat("get imaging settings - ", onvif_data->last_error));
						int value = stoi(args[0]);
						onvif_data->brightness = value;
						if (setImagingSettings(onvif_data)) throw std::runtime_error(cat("set brightness - ", onvif_data->last_error));
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
						if (getImagingSettings(onvif_data)) throw std::runtime_error(cat("get imaging settings - ", onvif_data->last_error));
						int value = stoi(args[0]);
						onvif_data->contrast = value;
						if (setImagingSettings(onvif_data)) throw std::runtime_error(cat("set contrast - ", onvif_data->last_error));
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
						if (getImagingSettings(onvif_data)) throw std::runtime_error(cat("get imaging settings - ", onvif_data->last_error));
						int value = stoi(args[0]);
						onvif_data->saturation = value;
						if (setImagingSettings(onvif_data)) throw std::runtime_error(cat("set saturation - ", onvif_data->last_error));
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
						if (getImagingSettings(onvif_data)) throw std::runtime_error(cat("get imaging settings - ", onvif_data->last_error));
						int value = stoi(args[0]);
						onvif_data->sharpness = value;
						if (setImagingSettings(onvif_data)) throw std::runtime_error(cat("set sharpness - ", onvif_data->last_error));
						std::cout << "  Sharpness was set to " << value << "\n" << std::endl;
					}
					else {
						std::cout << "  Missing value for sharpness\n" << std::endl;
					}
				}
				else if (args[0] == "resolution") {
					if (args.size() > 1) {
						args.erase(args.begin());
						profileCheck(onvif_data, args);
						std::string delim = "x";
						std::size_t found = args[0].find(delim);
						if (found != std::string::npos) {
							if(getVideoEncoderConfiguration(onvif_data)) throw std::runtime_error(cat("get video encoder configuration - ", onvif_data->last_error));
							onvif_data->width = stoi(args[0].substr(0, found));
							onvif_data->height = stoi(args[0].substr(found+1));
							if (setVideoEncoderConfiguration(onvif_data)) throw std::runtime_error(cat("set video encoder configuration - ", onvif_data->last_error));
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
						if(getVideoEncoderConfiguration(onvif_data)) throw std::runtime_error(cat("get video encoder configuration - ", onvif_data->last_error));
						int value = stoi(args[0]);
						onvif_data->gov_length = value;
						if (setVideoEncoderConfiguration(onvif_data)) throw std::runtime_error(cat("set video encoder configuration - ", onvif_data->last_error));
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
						if(getVideoEncoderConfiguration(onvif_data)) throw std::runtime_error(cat("get video encoder configuration - ", onvif_data->last_error));
						int value = stoi(args[0]);
						onvif_data->frame_rate = value;
						if (setVideoEncoderConfiguration(onvif_data)) throw std::runtime_error(cat("set video encoder configuration - ", onvif_data->last_error));
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
						if(getVideoEncoderConfiguration(onvif_data)) throw std::runtime_error(cat("get video encoder configuration - ", onvif_data->last_error));
						int value = stoi(args[0]);
						onvif_data->bitrate = value;
						if (setVideoEncoderConfiguration(onvif_data)) throw std::runtime_error(cat("set video encoder configuration - ", onvif_data->last_error));
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
							if (getNetworkInterfaces(onvif_data)) throw std::runtime_error(cat("get network interfaces - ", onvif_data->last_error));
							if (onvif_data->dhcp_enabled) {
								std::cout << "  DHCP is already enabled\n" << std::endl;
							}
							else {
								onvif_data->dhcp_enabled = true;
								if (setNetworkInterfaces(onvif_data)) throw std::runtime_error(cat("set network interfaces - ", onvif_data->last_error));
								std::cout << "  DHCP was enabled successfully\n\n"
										  << "  Camera may or may not reboot depending on settings\n"
										  << "  Session is being terminated.\n" << std::endl;
								exit(0);
							}
						}
						else if (args[0] == "off") {
							profileCheck(onvif_data, args);
							if (getNetworkInterfaces(onvif_data)) throw std::runtime_error(cat("get network interfaces - ", onvif_data->last_error));
							if (!onvif_data->dhcp_enabled) {
								std::cout << "  DHCP is already disabled\n" << std::endl;
							}
							else {
								onvif_data->dhcp_enabled = false;
								if (setNetworkInterfaces(onvif_data)) throw std::runtime_error(cat("set network interfaces - ", onvif_data->last_error));
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
						if (getNetworkInterfaces(onvif_data)) throw std::runtime_error(cat("get network interfaces - ", onvif_data->last_error));
						if (onvif_data->dhcp_enabled) {
							std::cout << "  Camera DHCP is enabled, IP address may not be set manually\n" << std::endl;
						}
						else {
							strcpy(onvif_data->ip_address_buf, args[0].c_str());
							if (setNetworkInterfaces(onvif_data)) throw std::runtime_error(cat("set network interfaces - ", onvif_data->last_error));
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
						if (getNetworkDefaultGateway(onvif_data)) throw std::runtime_error(cat("get network default gateway - ", onvif_data->last_error));
						if (onvif_data->dhcp_enabled) {
							std::cout << "  Camera DHCP is enabled, default gateway may not be set manually\n" << std::endl;
						}
						else {
							strcpy(onvif_data->default_gateway_buf, args[0].c_str());
							if (setNetworkDefaultGateway(onvif_data)) throw std::runtime_error(cat("set default gateway - ", onvif_data->last_error));
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
						if (getDNS(onvif_data)) throw std::runtime_error(cat("get DNS - ", onvif_data->last_error));
						if (onvif_data->dhcp_enabled) {
							std::cout << "  Camera DHCP is enabled, DNS may not be set manually\n" << std::endl;
						}
						else {
							strcpy(onvif_data->dns_buf, args[0].c_str());
							if (setDNS(onvif_data)) throw std::runtime_error(cat("set DNS - ", onvif_data->last_error));
							std::cout << "  DNS has been changed\n" << std::endl;
						}
					}
					else {
						std::cout << "  Missing value for DNS\n" << std::endl;
					}
				}
				else if (args[0] == "password") {
					if (args.size() > 1) {
						args.erase(args.begin());
						profileCheck(onvif_data, args);
						char buf[128] = {0};
						strcpy(buf, args[0].c_str());
						if (setUser(buf, onvif_data)) throw std::runtime_error(cat("set user - ", onvif_data->last_error));
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
				/*
				memset(kybd_buf, 0, 128);
				fgets(kybd_buf, 128, stdin);
				kybd_buf[strcspn(kybd_buf, "\r\n")] = 0;
				std::string reply(kybd_buf);
				*/
				/////////////////////////////
				std::string reply;
				std::getline(std::cin, reply);
				/////////////////////////////
				if (reply == "yes") {
					if (rebootCamera(onvif_data)) throw std::runtime_error(cat("reboot camera - ", onvif_data->last_error));
					std::cout << "  Camera is rebooting...\n" 
					          << "  Session will be terminated" << std::endl;
				}
				else {
					std::cout << "  Confirmation not received, reboot cancelled\n" << std::endl;
				}
			}
			else if (args[0] == "sync_time") {
				if (args.size() > 1) {
					args.erase(args.begin());
					profileCheck(onvif_data, args);
					if (args[0] == "zone") {
						profileCheck(onvif_data, args);
						if (setSystemDateAndTimeUsingTimezone(onvif_data)) throw std::runtime_error(cat("set system date and time using timezone - ", onvif_data->last_error));
						std::cout << "  Camera date and time has been synchronized using the camera timezone\n" << std::endl;
					}
				}
				else {
					profileCheck(onvif_data, args);
					if (setSystemDateAndTime(onvif_data)) throw std::runtime_error(cat("set system date and time - ", onvif_data->last_error));
					std::cout << "  Camera date and time has been synchronized without regard to camera timezone\n" << std::endl;
				}
			}
			else if (args[0] == "dump") {
				dumpConfigAll (onvif_data);
				std::cout << std::endl;
			}
			else if (args[0] == "safe") {
				safe_mode = true;
				std::cout << "  Safe mode is on\n" << std::endl;
			}
			else if (args[0] == "unsafe") {
				safe_mode = false;
				std::cout << "  Safe mode has been turned off, run only known safe apps and cameras\n" << std::endl;
			}
			else if (args[0] == "view") {
				if (safe_mode) {
					std::cout << "  SAFE MODE ON, use 'unsafe' command to disable safe mode, or -s option from command line\n" << std::endl;
					continue;
				}
				std::string player("ffplay");
				if (args.size() > 1) {
					args.erase(args.begin());
					profileCheck(onvif_data, args);
					player = args[0];
				}
				else {
					profileCheck(onvif_data, args);
				}
				if (getStreamUri(onvif_data)) throw std::runtime_error(cat("get stream uri - ", onvif_data->last_error));
				std::stringstream ss;
#ifdef _WIN32
				ss << "start " << player << " \"" << uri_with_pass(onvif_data) << "\"";
#else
				ss << player << " \"" << uri_with_pass(onvif_data) << "\"";
#endif				
				std::system(ss.str().c_str());
			} 
			else if (args[0] == "browser") {
				if (safe_mode) {
					std::cout << "  SAFE MODE ON, use 'unsafe' command to disable safe mode, or -s option from command line\n" << std::endl;
					continue;
				}
				profileCheck(onvif_data, args);
				if (getNetworkInterfaces(onvif_data)) throw std::runtime_error(cat("get network interfaces - ", onvif_data->last_error));
				std::stringstream ss;
#ifdef _WIN32
				ss << "start http://" << onvif_data->ip_address_buf;
#else
				ss << "xdg-open http://" << onvif_data->ip_address_buf;
#endif
				std::system(ss.str().c_str());
			}
			else if (args[0] == "help") {
				showHelp();
			}
			else { 
				//if (strcmp(kybd_buf, "quit"))
				//	std::cout << " Unrecognized command, type help to see help\n" << std::endl;

				if (cmd != quit)
					std::cout << " Unrecognized command \"" << args[0] << "\", type \"help\" to see help\n" << std::endl;
			}
		}
		catch (std::exception& e) {
			std::cout << "  ERROR: " << e.what() << "\n" << std::endl;
		}
	}
}

/*
else if (args[0] == "ntp") {
	if (args.size() > 1) {
		args.erase(args.begin());
		if (args[0] == "manual") {
			profileCheck(onvif_data, args);
			if (getHostname(onvif_data)) throw std::runtime_error(cat("get host name - ", onvif_data->last_error));
			if (getTimeOffset(onvif_data)) throw std::runtime_error(cat("get time offset - ", onvif_data->last_error));
			onvif_data->datetimetype = 'M';
			if (setSystemDateAndTime(onvif_data)) throw std::runtime_error(cat("set NTP - ", onvif_data->last_error));
			std::cout << "  NTP set to manual\n" << std::endl;
		}
		else {
			std::cout << "DHCP NTP" << std::endl;
			profileCheck(onvif_data, args);
			if (getHostname(onvif_data)) throw std::runtime_error(cat("get host name - ", onvif_data->last_error));
			if (getTimeOffset(onvif_data)) throw std::runtime_error(cat("get time offset - ", onvif_data->last_error));
			onvif_data->datetimetype = 'N';
			onvif_data->ntp_dhcp = false;
			strcpy(onvif_data->ntp_addr, "192.168.1.1");
			strcpy(onvif_data->ntp_type, "IPv4");
			if (setSystemDateAndTime(onvif_data)) throw std::runtime_error(cat("set NTP - ", onvif_data->last_error));
			if (setNTP(onvif_data)) throw std::runtime_error(cat("set ntp - ", onvif_data->last_error));
		}
	}
	else {
		std::cout << "  Missing value for NTP\n" << std::endl;
	}
}
*/
