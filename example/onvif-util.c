#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include "onvif.h"

#define DEFAULT_USERNAME    "admin"
#define DEFAULT_PASSWORD    "admin"
int longopt = 0;

static struct option longopts[] = {
             { "user",      required_argument,	NULL,	'u'},
             { "password",      required_argument,	NULL,	'p'},
             { "all",      no_argument,	NULL,	'a'},
             { "rate",      required_argument,	NULL,	'r'},
             { "width",      required_argument,	NULL,	'w'},
             { "height",      required_argument,	NULL,	'h'},
#ifdef ONVIF1906
             { "guaranteed",      no_argument,	NULL,	'g'},
             { "noguaranteed",      no_argument,	NULL,	'G'},
             { "no-guaranteed",      no_argument,	NULL,	'G'},
#endif
             { "name",      required_argument,	&longopt,	1},
             { "turbo",      no_argument,	&longopt,	2},
             { "normal",      no_argument,	&longopt,	3},
             { NULL,         0,                      NULL,           0 }
     };

static const char *username = DEFAULT_USERNAME;
static const char *password = DEFAULT_PASSWORD;
static int all = 0;
static int rate = -1;
static int width = -1;
static int height = -1;
#ifdef ONVIF1906
static int guaranteed = -1;
#endif
static int setName = 0;
static char *name;
static int turbo = -1;
/*
	Multicast settings?
*/

static void processOne(struct OnvifData *onvif_data);

int main(int argc, char **argv)
{
	int ch;
	struct OnvifSession *onvif_session = (struct OnvifSession*)malloc(sizeof(struct OnvifSession));
    struct OnvifData *onvif_data = (struct OnvifData*)malloc(sizeof(struct OnvifData));
	char *arg0 = argv[0];

	while ((ch = getopt_long(argc, argv, "u:p:ar:w:h:gG", longopts, NULL)) != -1) {
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
			case 'r':
				rate = atoi(optarg);
				break;
			case 'w':
				width = atoi(optarg);
				break;
			case 'h':
				height = atoi(optarg);
				break;
#ifdef ONVIF1906
			case 'g':
				guaranteed = 1;
				break;
			case 'G':
				guaranteed = 0;
				break;
#endif
			case 0:
				if (longopt == 1) {
					setName = 1;
					name = optarg;
				} else if (longopt == 2) {
					turbo = 1;
				} else if (longopt == 3) {
					turbo = 0;
				}
				longopt = 0;
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
	printf("%s %s(%s)\n",host,
		onvif_data->host_name,
		onvif_data->camera_name);
	if (
#ifdef ONVIF1906
		guaranteed >= 0 ||
#endif
		turbo >= 0 || rate > 0 || width > 0 || height > 0) {
		getFirstProfileToken(onvif_data);
		getProfile(onvif_data);
		getVideoEncoderConfiguration(onvif_data);
#ifdef ONVIF1906
		if (guaranteed == 1)
			onvif_data->guaranteed_frame_rate = true;
		else if (guaranteed == 0)
			onvif_data->guaranteed_frame_rate = false;
#endif
		if (rate > 0)
			onvif_data->frame_rate = rate;
		if (width > 0)
			onvif_data->width = width;
		if (height > 0)
			onvif_data->height = height;

		if (turbo > 0) {
/*
	BitrateLimit (kbps)
	EncodingInterval (1)
	Quality (1-6 on mine: Should ideally check VideoEncoderConfigurationOptions for real range.)
*/
			onvif_data->conf_bitrate_limit = onvif_data->bitrate_max;
			onvif_data->conf_encoding_interval = 1;
			onvif_data->quality = 6.0;
		} else if (turbo == 0) {
			onvif_data->conf_bitrate_limit = 4096;
			onvif_data->conf_encoding_interval = 1;
			onvif_data->quality = 4.0;
		}

		if (setVideoEncoderConfiguration(onvif_data) < 0) {
			fprintf(stderr,"Error: %s\n",onvif_data->last_error);
		}
	}
	if (setName) {
		strcpy(onvif_data->host_name,name);
		if (setHostname(onvif_data) < 0) {
			fprintf(stderr,"Error: %s\n",onvif_data->last_error);
		}
	}
}
