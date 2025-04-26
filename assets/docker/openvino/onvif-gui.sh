echo "Launching Onvif GUI ..."
mkdir -p $HOME/.config/onvif-gui/Videos >&- 2>&-
xhost +local:docker > /dev/null
docker run -it --network host -e DISPLAY=$DISPLAY --device=/dev/dri:/dev/dri \
-e PATH=/onvif-gui-env/bin:$PATH -v $HOME/.config/onvif-gui:/root onvif-gui
