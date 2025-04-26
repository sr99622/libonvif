echo "Launching Onvif GUI ..."
mkdir -p $HOME/.config/onvif-gui/Videos >&- 2>&-
xhost +local:docker > /dev/null
docker run -it --rm --runtime nvidia --gpus all --net=host -e DISPLAY=$DISPLAY \
-e NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-all} \
-e PATH=/onvif-gui-env/bin:$PATH -v $HOME/.config/onvif-gui:/root onvif-gui
