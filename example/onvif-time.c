/*******************************************************************************
* onvif-time.c
*
* Copyright (c) 2022 Stephen Rhodes 
* Originally authored by Brian D Scott
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
#include "onvif.h"
#ifdef _WIN32
#include "getopt-win.h"
#else
#include <getopt.h>
#endif

#define DEFAULT_USERNAME    "admin"
#define DEFAULT_PASSWORD    "admin"

static struct option longopts[] = {
             { "user",      required_argument,	NULL,   'u'},
             { "password",  required_argument,	NULL,   'p'},
             { "all",       no_argument,        NULL,   'a'},
             { "set",       no_argument,        NULL,   's'},
             { "local",     no_argument,        NULL,   'l'},
             { "ntp",       required_argument,  NULL,   'n'},
             { "manual",    required_argument,  NULL,   'm'},
             { NULL,        0,                  NULL,    0 }
     };

static const char *username = DEFAULT_USERNAME;
static const char *password = DEFAULT_PASSWORD;
static int all = 0;
static int set = 0;
static int local = 0;
static const char *ntp = NULL;
static int setNtp = 0;
static int setManual = 0;

static void processOne(struct OnvifData *onvif_data);

int main(int argc, char **argv)
{
	int ch;
	struct OnvifSession *onvif_session = (struct OnvifSession*)malloc(sizeof(struct OnvifSession));
    struct OnvifData *onvif_data = (struct OnvifData*)malloc(sizeof(struct OnvifData));
	char *arg0 = argv[0];

	while ((ch = getopt_long(argc, argv, "u:p:asln:m", longopts, NULL)) != -1) {
		switch (ch) {
            case 'u':
				username = optarg;
				break;
			case 'p':
				password = optarg;
				break;
			case 'a':
				all = 1;
				break;
			case 's':
				set = 1;
				break;
			case 'l':
				local = 1;
				set = 1;
				break;
			case 'n':
				ntp = optarg;
				setNtp = 1;
				setManual = 0;
				set = 1;
				break;
			case 'm':
				setManual = 1;
				setNtp = 0;
				set = 1;
				break;
			case 0:
				break;
			default:
				exit(1);
		}
	}
	argc -= optind;
    argv += optind;

    if (!all && argc < 1) {
        fprintf(stderr, "Usage: %s <options> [HOST ..]\n", arg0);
        exit(1);
    } else if (all && argc > 0) {
        fprintf(stderr, "Usage: %s <options> [HOST ..]\n", arg0);
        exit(1);
	}
    initializeSession(onvif_session);
    int n = broadcast(onvif_session);

	if (all && n == 0) {
		fprintf(stderr,"No cameras on this network\n");
		exit(1);
	}

	if (all) {
		for (int i = 0; i < n; i++) {
			prepareOnvifData(i, onvif_session, onvif_data);
			processOne(onvif_data);
		}
	} else {
		while (argc--) {
			char *wanted = argv++[0];
			int found = 0;
			for (int i = 0; i < n; i++) {
				prepareOnvifData(i, onvif_session, onvif_data);
				char host[128];
				extractHost(onvif_data->xaddrs, host);
				if (strcmp(host,wanted) == 0) {
					found = 1;
					break;
				}
			}
			if (found) {
				processOne(onvif_data);
			} else {
				fprintf(stderr,"Camera at %s not found\n",wanted);
			}
		}
	}
}

static void processOne(struct OnvifData *onvif_data) {
	char host[128];
	extractHost(onvif_data->xaddrs, host);
	getHostname(onvif_data);
	printf("%s(%s): Time offset %ld seconds. Timezone '%s'.%s%s\n",host,
		onvif_data->host_name,
		onvif_data->time_offset,
		onvif_data->timezone,
		onvif_data->dst ? " Dst." : "",
		onvif_data->datetimetype == 'M'?" Manual time set.":onvif_data->datetimetype == 'N'?" NTP time set.":"");
	if (setNtp) {
		strcpy(onvif_data->ntp_addr,ntp);
		onvif_data->ntp_dhcp = false;
		if (!onvif_data->ntp_addr[0])
			onvif_data->ntp_dhcp = true;
		/* Decide if we have an IPv4 address, IPv6 address or dns name */
		strcpy(onvif_data->ntp_type,"IPv4");
		if (setNTP(onvif_data) < 0)
			fprintf(stderr,"SetNTP: error %s\n",onvif_data->last_error);
		onvif_data->datetimetype = 'N';
	}
	if (setManual)
		onvif_data->datetimetype = 'M';
	if (local)
		onvif_data->timezone[0] = '\0';
	if (set) {
		if (setSystemDateAndTime(onvif_data) < 0)
			fprintf(stderr,"SetSystemDateAndTime: error %s\n",onvif_data->last_error);
		getTimeOffset(onvif_data);
		printf("\tOffset now %ld\n",onvif_data->time_offset);
	}
}