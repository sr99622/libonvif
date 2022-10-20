/*******************************************************************************
* onvif-getall.c
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

#define DEFAULT_USERNAME    "admin"
#define DEFAULT_PASSWORD    "admin"

void get_info(const char *host, const char *username, const char *password)
{
    struct OnvifData *onvif_data = (struct OnvifData *) calloc(1, sizeof (struct OnvifData));

    sprintf(onvif_data->xaddrs, "http://%s/onvif/device_service", host);
    strcpy(onvif_data->device_service, "POST /onvif/device_service HTTP/1.1\r\n");

    /* Store username and password */
    strncpy(onvif_data->username, username, sizeof (onvif_data->username) - 1);
    strncpy(onvif_data->password, password, sizeof (onvif_data->password) - 1);

    dumpConfigAll (onvif_data);

    free(onvif_data);
}

int main(int argc, char **argv)
{
    const char *host;
    const char *username;
    const char *password;

    if (argc <= 1) {
        fprintf(stderr, "Usage: %s HOST[:PORT] [USERNAME] [PASSWORD]\n", argv[0]);
        exit(1);
    }

    host = argv[1];
    username = (argc > 2) ? argv[2] : DEFAULT_USERNAME;
    password = (argc > 3) ? argv[3] : DEFAULT_PASSWORD;

    get_info(host, username, password);
}
