echo "Launching Onvif GUI ..."
mkdir -p $HOME/.config/onvif-gui/Videos >&- 2>&-
xhost +local:docker > /dev/null
docker run --rm --gpus all \
    --runtime nvidia --net=host -e DISPLAY=$DISPLAY \
    --device /dev/nvidia0:/dev/nvidia0  \
    --device /dev/nvidiactl:/dev/nvidiactl \
    --device /dev/nvidia-uvm:/dev/nvidia-uvm \
    --device /dev/nvidia-uvm-tools:/dev/nvidia-uvm-tools \
    -e NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-all} \
    -e PATH=/onvif-gui-env/bin:$PATH -v $HOME/.config/onvif-gui:/root onvif-gui

