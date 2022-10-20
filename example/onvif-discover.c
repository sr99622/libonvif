/*******************************************************************************
* onvif-discover.c
*
* Copyright (c) 2020 Stephen Rhodes 
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

#include <stdio.h>
#include <string.h>
#include <onvif.h>

int number_of_cameras;

int main ( int argc, char **argv )
{
    struct OnvifSession *onvif_session = (struct OnvifSession*)malloc(sizeof(struct OnvifSession));
    struct OnvifData *onvif_data = (struct OnvifData*)malloc(sizeof(struct OnvifData));
    initializeSession(onvif_session);
    number_of_cameras = broadcast(onvif_session);
    fprintf(stdout, "libonvif found %d cameras\n", number_of_cameras);

    for (int i = 0; i < number_of_cameras; i++) {
        prepareOnvifData(i, onvif_session, onvif_data);

        fprintf(stdout, "%s [%s]\n", onvif_data->camera_name, onvif_data->xaddrs);
        fprintf(stdout, "enter username:");
        fgets(onvif_data->username, 128, stdin);
        fprintf(stdout, "enter password:");
        fgets(onvif_data->password, 128, stdin);

        onvif_data->username[strcspn(onvif_data->username, "\r\n")] = 0;
        onvif_data->password[strcspn(onvif_data->password, "\r\n")] = 0;

        getDeviceInformation(onvif_data);
        fprintf(stdout, "* Camera %d '%s' S/N:%s\n", i, onvif_data->camera_name, onvif_data->serial_number);

        getTimeOffset(onvif_data);
        fprintf(stdout, "Time offset %ld seconds, timezone is '%s', is%s dst, time %s\n",
                onvif_data->time_offset,
                onvif_data->timezone,
                onvif_data->dst?"":" not",
                onvif_data->datetimetype == 'M'?"set Manually":onvif_data->datetimetype == 'N'?"set via NTP":"setting unknown");

        int index = 0;
        while (true) {
            if (fillRTSPn(onvif_data,index) == 0) {
                fprintf(stdout, "* Profile %d\n%s\n", index, onvif_data->stream_uri);
            } else {
                if (index == 0)
                    fprintf(stderr, "Error getting camera uri - %s\n", onvif_data->last_error);
                break;
			}
            index++;

            fprintf(stdout, "event_service: %s",onvif_data->event_service);
            fprintf(stdout, "imaging_service: %s",onvif_data->imaging_service);
            fprintf(stdout, "media_service: %s",onvif_data->imaging_service);
            fprintf(stdout, "ptz_service: %s",onvif_data->imaging_service);
            if (getProfile(onvif_data) == 0) {
                fprintf(stdout, "Width: %d\n",onvif_data->width);
                fprintf(stdout, "Height: %d\n",onvif_data->height);
                fprintf(stdout, "FrameRateLimit: %d\n",onvif_data->frame_rate);
                fprintf(stdout, "BitrateLimit: %d\n",onvif_data->bitrate);
                fprintf(stdout, "GovLength: %d\n",onvif_data->gov_length);
            } else {
                fprintf(stderr, "Error getting profile - %s\n", onvif_data->last_error);
                continue;
            }

            if (getOptions(onvif_data) == 0) {
                fprintf(stdout, "Brightness:Min: %d\n",onvif_data->brightness_min);
                fprintf(stdout, "Brightness:Max: %d\n",onvif_data->brightness_max);
                fprintf(stdout, "ColorSaturation:Min: %d\n",onvif_data->saturation_min);
                fprintf(stdout, "ColorSaturation:Max: %d\n",onvif_data->saturation_max);
                fprintf(stdout, "Contrast:Min: %d\n",onvif_data->contrast_min);
                fprintf(stdout, "Contrast:Max: %d\n",onvif_data->contrast_max);
                fprintf(stdout, "Sharpness:Min: %d\n",onvif_data->sharpness_min);
                fprintf(stdout, "Sharpness:Max: %d\n",onvif_data->sharpness_max);
            } else {
                fprintf(stderr, "Error getting options - %s\n", onvif_data->last_error);
                continue;
            }

            if (getVideoEncoderConfigurationOptions(onvif_data) == 0) {
                 /* onvif_data->resolutions_buf[16][128] */
                int i;
                fputs("Resolutions: ", stdout);
                for (i = 0; i < 16; i++) {
                    if (onvif_data->resolutions_buf[i][0]) {
                        if (i > 0)
                            fputs(", ", stdout);
                        fputs(onvif_data->resolutions_buf[i], stdout);
                    }
                }
                fprintf(stdout, "\n");

                fprintf(stdout, "H264:GovLengthRange:Min: %d\n",onvif_data->gov_length_min);
                fprintf(stdout, "H264:GovLengthRange:Max: %d\n",onvif_data->gov_length_max);
                fprintf(stdout, "H264:FrameRateRange:Min: %d\n",onvif_data->frame_rate_min);
                fprintf(stdout, "H264:FrameRateRange:Max: %d\n",onvif_data->frame_rate_max);
                fprintf(stdout, "H264:BitRateRange:Min: %d\n",onvif_data->bitrate_min);
                fprintf(stdout, "H264:BitRateRange:Max: %d\n",onvif_data->bitrate_max);
            } else {
                fprintf(stderr, "Error getting video encoder configuration options - %s\n", onvif_data->last_error);
                continue;
            }

            if (getVideoEncoderConfiguration(onvif_data) == 0) {
                fprintf(stdout, "Configuration:Name: %s\n",onvif_data->video_encoder_name_buf);
                fprintf(stdout, "Configuration:UseCount: %d\n",onvif_data->use_count);
                fprintf(stdout, "Configuration:GuaranteedFrameRate: %s\n",onvif_data->guaranteed_frame_rate?"true":"false");
                fprintf(stdout, "Configuration:Encoding: %s\n",onvif_data->encoding);
                fprintf(stdout, "Configuration:Resolution:Width: %d\n",onvif_data->conf_width);
                fprintf(stdout, "Configuration:Resolution:Height: %d\n",onvif_data->conf_height);
                fprintf(stdout, "Configuration:Quality: %f\n",onvif_data->quality);
                fprintf(stdout, "Configuration:RateControl:FrameRateLimit: %d\n",onvif_data->conf_frame_rate_limit);
                fprintf(stdout, "Configuration:RateControl:EncodingInterval: %d\n",onvif_data->conf_encoding_interval);
                fprintf(stdout, "Configuration:RateControl:BitrateLimit: %d\n",onvif_data->conf_bitrate_limit);
                fprintf(stdout, "Configuration:H264Profile: %s\n",onvif_data->h264_profile_buf);
                fprintf(stdout, "Configuration:Multicast:AddressType: %s\n",onvif_data->multicast_address_type_buf);
				if (strcmp(onvif_data->multicast_address_type_buf,"IPv6") == 0)
                    fprintf(stdout, "Configuration:Multicast:IPv6Address: %s\n",onvif_data->multicast_address_buf);
				else
                    fprintf(stdout, "Configuration:Multicast:IPv4Address: %s\n",onvif_data->multicast_address_buf);
                fprintf(stdout, "Configuration:Multicast:Port: %d\n",onvif_data->multicast_port);
                fprintf(stdout, "Configuration:Multicast:TTL: %d\n",onvif_data->multicast_ttl);
                fprintf(stdout, "Configuration:Multicast:AutoStart: %s\n",onvif_data->autostart?"true":"false");
                fprintf(stdout, "Configuration:SessionTimeout: %s\n",onvif_data->session_time_out_buf);
            } else {
                fprintf(stderr, "Error getting video encoder configuration - %s\n", onvif_data->last_error);
                continue;
            }
        }
    }

    closeSession(onvif_session);
    free(onvif_session);
    free(onvif_data);
    return 0;
}
