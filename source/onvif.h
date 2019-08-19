
/*******************************************************************************
* onvif.h
*
* copyright 2018 Stephen Rhodes
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
*******************************************************************************/

#ifndef ONVIF_H
#define ONVIF_H

#ifdef __cplusplus
extern "C" {
#endif

#include <libxml/parser.h>
#include <libxml/xpath.h>
#include <stdbool.h>
#ifndef _WIN32
    #include <time.h>
#endif

#ifdef __MINGW32__
    #include <ws2tcpip.h>
#endif

#ifdef LIBONVIFDLL_EXPORTS
    #define LIBRARY_API __declspec(dllexport)
#else
    #define LIBRARY_API
#endif

static const int PAN_TILT_STOP = 0;
static const int ZOOM_STOP = 1;

struct OnvifData {
    /*video*/
    char videoEncoderConfigurationToken[128];
    char resolutions_buf[16][128];
    int gov_length_min;
    int gov_length_max;
    int frame_rate_min;
    int frame_rate_max;
    int bitrate_min;
    int bitrate_max;
    int width;
    int height;
    int gov_length;
    int frame_rate;
    int bitrate;
    char video_encoder_name_buf[128];
    int use_count;
    float quality;
    char h264_profile_buf[128];
    char multicast_address_type_buf[128];
    char multicast_address_buf[128];
    int multicast_port;
    int multicast_ttl;
    bool autostart;
    char session_time_out_buf[128];
    /*network*/
    char networkInterfaceToken[128];
    char networkInterfaceName[128];
    bool dhcp_enabled;
    char ip_address_buf[128];
    char default_gateway_buf[128];
    char dns_buf[128];
    int prefix_length;
    /*image*/
    char videoSourceConfigurationToken[128];
    int brightness_min;
    int brightness_max;
    int saturation_min;
    int saturation_max;
    int contrast_min;
    int contrast_max;
    int sharpness_min;
    int sharpness_max;
    int brightness;
    int saturation;
    int contrast;
    int sharpness;
    /*service*/
    char device_service[1024];
    char media_service[128];
    char imaging_service[128];
    char ptz_service[128];
    char event_service[128];
    /*event*/
    char subscription_reference[128];
    int event_listen_port;
    /*general*/
    char xaddrs[1024];
    char profileToken[128];
    char username[128];
    char password[128];
    time_t time_offset;
    char stream_uri[1024];
    char camera_name[1024];
    char serial_number[128];
    /*error*/
    char last_error[1024];
};

struct OnvifSession {
    char buf[128][8192];
    int len[128];
    char uuid[47];
    int discovery_msg_id;
};

LIBRARY_API void initializeSession(struct OnvifSession *onvif_session);
LIBRARY_API void closeSession(struct OnvifSession *onvif_session);
LIBRARY_API int broadcast(struct OnvifSession *onvif_session);
LIBRARY_API void prepareOnvifData(int ordinal, struct OnvifSession *onvif_session, struct OnvifData *onvif_data);
LIBRARY_API int fillRTSP(struct OnvifData *onvif_data);
LIBRARY_API void clearData(struct OnvifData *onvif_data);

LIBRARY_API int getCapabilities(struct OnvifData *onvif_data);
LIBRARY_API int getProfile(struct OnvifData *onvif_data);

LIBRARY_API int getNetworkInterfaces(struct OnvifData *onvif_data);
LIBRARY_API int setNetworkInterfaces(struct OnvifData *onvif_data);
LIBRARY_API int getNetworkDefaultGateway(struct OnvifData *onvif_data);
LIBRARY_API int setNetworkDefaultGateway(struct OnvifData *onvif_data);
LIBRARY_API int getDNS(struct OnvifData *onvif_data);
LIBRARY_API int setDNS(struct OnvifData *onvif_data);

LIBRARY_API int getVideoEncoderConfigurationOptions(struct OnvifData *onvif_data);
LIBRARY_API int getVideoEncoderConfiguration(struct OnvifData *onvif_data);
LIBRARY_API int setVideoEncoderConfiguration(struct OnvifData *onvif_data);

LIBRARY_API int getOptions(struct OnvifData *onvif_data);
LIBRARY_API int getImagingSettings(struct OnvifData *onvif_data);
LIBRARY_API int setImagingSettings(struct OnvifData *onvif_data);

LIBRARY_API int continuousMove(float x, float y, float z, struct OnvifData *onvif_data);
LIBRARY_API int moveStop(int type, struct OnvifData *onvif_data);
LIBRARY_API int setPreset(char * arg, struct OnvifData *onvif_data);
LIBRARY_API int gotoPreset(char * arg, struct OnvifData *onvif_data);

LIBRARY_API int setUser(char * new_password, struct OnvifData *onvif_data);
LIBRARY_API int setSystemDateAndTime(struct OnvifData *onvif_data);
LIBRARY_API int getTimeOffset(struct OnvifData *onvif_data);
LIBRARY_API int getFirstProfileToken(struct OnvifData *onvif_data);
LIBRARY_API int getStreamUri(struct OnvifData *onvif_data);
LIBRARY_API int getDeviceInformation(struct OnvifData *onvif_data);
LIBRARY_API int rebootCamera(struct OnvifData *onvif_data);
LIBRARY_API int hardReset(struct OnvifData *onvif_data);

LIBRARY_API void saveSystemDateAndTime(char * filename, struct OnvifData *onvif_data);
LIBRARY_API void saveScopes(char * filename, struct OnvifData *onvif_data);
LIBRARY_API void saveDeviceInformation(char * filename, struct OnvifData *onvif_data);
LIBRARY_API void saveCapabilities(char * filename, struct OnvifData *onvif_data);
LIBRARY_API void saveProfiles(char * filename, struct OnvifData *onvif_data);
LIBRARY_API void saveServiceCapabilities(char * filename, struct OnvifData *onvif_data);

LIBRARY_API int eventSubscribe(struct OnvifData *onvif_data);
LIBRARY_API int eventRenew(struct OnvifData *onvif_data);

LIBRARY_API xmlDocPtr sendCommandToCamera(char * cmd, char * xaddrs);
LIBRARY_API void getBase64(unsigned char * buffer, int chunk_size, unsigned char * result);
LIBRARY_API void getUUID(char uuid_buf[47]);
LIBRARY_API void addUsernameDigestHeader(xmlNodePtr root, xmlNsPtr ns_env, char * user, char * password, time_t offset);
LIBRARY_API void addHttpHeader(xmlDocPtr doc, xmlNodePtr root, char * xaddrs, char * post_type, char cmd[], int cmd_length);
LIBRARY_API void getDiscoveryXml(char buffer[], int buf_size, char uuid[47]);
LIBRARY_API void getDiscoveryXml2(char buffer[], int buf_size);
LIBRARY_API void getScopeField(char *, char *, char[1024]);
LIBRARY_API void getCameraName(int ordinal, struct OnvifSession *onvif_session, struct OnvifData *onvif_data);
LIBRARY_API void extractXAddrs(int ordinal, struct OnvifSession *onvif_session, struct OnvifData *onvif_data);
LIBRARY_API void extractOnvifService(char service[1024], bool post);
LIBRARY_API void extractHost(char * xaddrs, char host[128]);
LIBRARY_API int checkForXmlErrorMsg(xmlDocPtr doc, char error_msg[1024]);
LIBRARY_API int getXmlValue(xmlDocPtr doc, xmlChar *xpath, char buf[], int buf_length);
LIBRARY_API int getNodeAttribute (xmlDocPtr doc, xmlChar *xpath, xmlChar *attribute, char buf[], int buf_length);
LIBRARY_API xmlXPathObjectPtr getNodeSet (xmlDocPtr doc, xmlChar *xpath);

LIBRARY_API int setSocketOptions(int socket);
LIBRARY_API void prefix2mask(int prefix, char mask_buf[128]);
LIBRARY_API int mask2prefix(char * mask_buf);
LIBRARY_API void getIPAddress(char buf[128]);


#ifdef _WIN32
    int gettimeofday(struct timeval *tp, struct timezone *tzp);
#endif

#ifdef __MINGW32__
    int inet_pton(int af, const char *src, void *dst);
    const char *inet_ntop(int af, const void *src, char *dst, socklen_t size);
#endif

#ifdef __cplusplus
}
#endif

#endif
