# onvif-gui

A client side gui implementation of the ONVIF specification.

Introduction
------------

onvif-gui is a multi platform library implementing the client side of the ONVIF
specification for communicating with IP enabled compatible cameras.

It is a comprehensive GUI program written in Python that includes the
discovery functionality as well as controls for adjusting camera parameters and
PTZ operations.  The GUI program has a record function that will write the
camera stream to file and includes some basic media file management tools. The
GUI also has a module plug in function that allows developers to access the 
video stream in numpy format for python processing.

Installation
-------------

INSTALLATION ON LINUX

Installation is a two step process.  Dependencies are install first using apt.

```
sudo apt install python3-pip cmake libxml2-dev libavdevice-dev libsdl2-dev '^libxcb.*-dev' libxkbcommon-x11-dev
```

The second step follows

```
pip install onvif-gui
```

Modules
-----------------

onvif-gui has a facility for incorporating python programs to operate on the
video stream.  The Modules tab is the user interface for this feature.  There
is a minimal example program called sample.py that demonstrates how data is 
trsansferred from the main program to the python module and it's GUI interface
implementation.

There is included with onvif-gui a full implementation of the YOLOX algorithm
along with an associated tracking algorithm known as ByteTrack.  These algorithms 
are implemented using pytorch, which requires some specific configuration. Note
that ByteTrack currently is available only on Windows.

Note on Running PyTorch Modules
--------------------------------

When running these algorithms, bear in mind that they are computationally expensive.
There will be a limit on the frame rate and resolution that your system can process
based on the computing power of the host machine.  Limiting the frame rate can be
done on cameras using the video settings tab.  Frame rate on files can be set by
using the filter command 'fps=10' where 10 is the desired frame rate.  There is 
currently a limitation in onvif gui that prevents seeking when the frame rate is 
set by the filter.  Likewise, resolution can be set on files with the video filter
using the scale directive, i.e. 'scale=1280x720'.  Consecutive video filters can
be run using a comma as delimiter between the commands, i.e. 'fps=10,scale=1280x720'.


Onvif GUI Program
-----------------

NAME 

    onvif-gui

SYNOPSIS

    onvif-gui is a python program.  
    
    The program requires the following modules:

    pip install pyqt6 opencv-python numpy loguru
    

    To run the program:

    cd ../onvif-gui
    python3 run.py

    These instructions are intended for quick setup to verify the program.  To use the 
    library in other python programs, it is advised to install the onvif and avio
    python modules.

    cd ../libonvif
    python3 setup.py install

    cd ../libavio 
    python3 setup.py install

DESCRIPTION

    GUI program to view and set parameters on onvif compatible IP cameras. Double clicking 
    the camera name in the list will display the camera video output. 

    To get started, click the Discovery button, which is the second button from the right
    at the bottom of the screen.  A login screen will appear for each camera as it is found.
    The Settings tab may be used to set a default login that can be used automatically.

    Camera parameters are available on the tabs on the lower right side of the application. 
    Once a parameter has been changed, the Apply button will be enabled, which can be used 
    to commit the change to the camera.  It may be necessary to re-start the video output 
    stream in order to see the changes.  The Apply button is found in the lower right hand
    corner below the tabs.

    Video:

        Resolution  - The drop down box is used to select the setting.
        Frame Rate  - The number of frames per second in the video output.
        Gov Length  - This is the distance between key frames in the stream.
        Bitrate     - The maxmimum number of bits per second to transmit.

    Image:

        All values are set using the sliders

        Brightness
        Saturation
        Contrast
        Sharpness

    Network:

        If the DHCP is enabled, all fields are set by the server, if DHCP is disabled, other 
        network settings may be completed manually.  Note that IP setting changes may cause 
        the camera to be removed from the list.  Use the Discover button to find the camera.
        Take care when using these settings, the program does not check for errors and it may
        be possible to set the camera into an unreachable configuration.

        IP Address
        Subnet Mask
        Gateway
        Primary DNS

    PTZ:

        Settings pertain to preset selections or current camera position.  The arrow keys, Zoom In 
        and Zoom out control the position and zoom. The numbered buttons on the left correspond to 
        preset positions.  The blank text box may be used to address presets numbered higher than 5.
        To set a preset, position the camera, then check Set Preset, then click the numbered preset button.

    Admin:

        Camera Name  - Changes the application display name of the camera.
        Set admin Password - Can be used to change the password for the camera.
        Sync Time - Will reset the camera's current time without regard to time zone.
        Browser - Will launch a browser session with the camera for advanced maintenance.
        Enable Reboot - Will enable the reboot button for use.  Camera will be removed from list.
        Enable Reset - Will enable the reset button for use.  Use with caution, all camera 
          settings will be reset.

    Application Settings:

        Auto Discovery - When checked, the application will automatcally start discovery upon launch, 
          otherwise use the Discover button.
        Common Username - Default username used during discover.
        Common Password - Default password used during discover.
        Hardware Decoder - If available, can be set to use GPU video decoding.
        Video Filter - FFMPEG filter strings may be used to modify the video
        Direct Rendering - May be used in Windows to increase performance
        Convert to RGB - The default setting is ON, may be turned on for performance
        Disable Audio, Disable Video - Used to limit streams to a single medium
        Post Process Record - Recording will be the encoded video stream rather than raw packets
        Hardware Encode - If available, use the GPU for encoding
        Process Frame - Video frame data is processed by the sample python module
        Low Latency - Reduces the buffer size to reduce latency, may cause instability
        Pre-Record Cache Size - A cache of media packets is stored locally prior to decoding and will
          be pre-pended to the file stream when Pre Process recording.  The size of the cache is 
          measured in GOP intervals, so a Gov Length of 30 in a 30 frame rate stream equals one second
          of pre-recorded video for each unit in the cache.
        Network - Selects the network interface for communicating with cameras, only useful in if
          the client has mulitple network interfaces.
        

EXAMPLES

    To change the video resolution of a camera output, Double click on the camera name in 
    the list.  The camera video output should display in the viewer.  Select the Video tab 
    and use the drop down box labelled Resolution.  Upon changing the selection, the Apply 
    button will be enabled.  Click the apply button to make the change.  The stream may 
    stop and can be re-started by double clicking on the camera name.

    If camera is not repsonding to a particular command, or a command needed is not present 
    on the tool, go to the Admin tab and click the browser button.  This will launch the 
    browser using the camera IP address.  Log into the camera and settings should be 
    avialable in native format for the camera configuration.
