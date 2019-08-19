#include <stdio.h>
#include <string.h>
#include <libonvif/onvif.h>

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

      fprintf(stdout, "%s\n", onvif_data->camera_name);
      fprintf(stdout, "enter username:");
      fgets(onvif_data->username, 128, stdin);
      fprintf(stdout, "enter password:");
      fgets(onvif_data->password, 128, stdin);

      onvif_data->username[strcspn(onvif_data->username, "\r\n")] = 0;
      onvif_data->password[strcspn(onvif_data->password, "\r\n")] = 0;

      if (fillRTSP(onvif_data) == 0)
          fprintf(stdout, "%s\n", onvif_data->stream_uri);
      else
          fprintf(stderr, "Error getting camera uri - %s\n", onvif_data->last_error);

    }

    closeSession(onvif_session);
    free(onvif_session);
    free(onvif_data);
    return 0;
}
