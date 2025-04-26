FROM ubuntu:latest
RUN apt-get update
RUN apt-get install -y cmake g++ git python3-dev python3-pip virtualenv libxml2-dev libavdevice-dev libsdl2-dev '^libxcb.*-dev' libxkbcommon-x11-dev
RUN virtualenv onvif-gui-env
RUN /bin/bash -c "source onvif-gui-env/bin/activate && pip install --upgrade pip && pip install onvif-gui"
EXPOSE 5000
CMD ["/onvif-gui-env/bin/onvif-gui"]
