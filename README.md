
libonvif
========

A client side implementation of the ONVIF specification.

Introduction
------------

libonvif is a multi platform library implementing the client side of the ONVIF
specification for communicating with IP enabled compatible cameras.  It will
compile on Linux and Windows.

An utility program is included with libonvif that can be used as a maintenance
tool and will discover compatible cameras on the local network and may be used 
to query each of them for device configuration such as RSTP connection uri 
information or video settings.

Additionally, there is a comprehensive GUI sample program that includes the
discovery functionality as well controls for adjusting camera parameters and
PTZ operations.  The GUI sample is written in Qt and can be compiled with
either cmake or qmake using Qt Creator.  On Linux, the GUI has a viewer pane 
that will display camera output.  

The utility program is invoked using the 'onvif-util' command.

The GUI interface may be invoked using the 'onvif-camadmin' command.

To Install From Source
----------------------

DEPENDENCY ON LIBXML2

libonvif has a dependency on libxml2.  This means you will need to have libxml2
installed on your machine and you will need to know the location of the libxml2
include files for compilation.  Most Linux systems come with libxml2 pre-
installed.  You can check the availability of libxml2 on your system with the
following command:

```bash
xml2-config --cflags
```

This command should return with -I/usr/include/libxml2 or something similar.  If
you receive this response, libxml2 is installed.  If the command is not
recognized, then you will need to install libxml2.  An easy way to do this on
Linux is to use the apt utility to install.  Make sure to get the -dev version
by using the following command on Debian or Ubuntu:

```bash
sudo apt-get install libxml2-dev
```

Installing libxml2 on Windows is more difficult.  You will need to get the source
code for libxml2 from https://github.com/GNOME/libxml2.  Upon completion of the 
build for libxml2, you will need Adminstrator privileges to install the library.
You will also need to set the PATH environment variable to include the libxml2.dll 
path.  The instructions below will build a stripped down version of libxml2 which
is fine for onvif.  To get an administrator privileged command prompt, use the
Windows search bar for cmd and right click on the command prompt icon to select
Run as Administrator.  To make a permanent change to the PATH environment variable, 
use the Settings->About->Advanced System Settings->Environment Variables configuration 
screen.


```bash
git clone https://github.com/GNOME/libxml2.git
cd libxml2
mkdir build
cd build
cmake -DLIBXML2_WITH_PYTHON=OFF -DLIBXML2_WITH_ICONV=OFF -DLIBXML2_WITH_LZMA=OFF -DLIBXML2_WITH_ZLIB=OFF ..
cmake --build . --config Release
cmake --install .
set PATH=%PATH%;"C:\Program Files (x86)\libxml2\bin"
```

If you are working in a conda environment, the dependency for libxml2 may also be 
satisfied using anaconda.  This is the same for Linux and Windows.

```bash
conda install -c conda-forge libxml2
```

COMPILE

The utils program is built by default.  To build the gui program, you will need to have
Qt development libraries installed on the host machine.    The Linux GUI comes with a 
viewer based on QtAV.On Linux, these can be installed using the commands.

```bash
sudo apt install qtbase5-dev
sudo apt install ibqtav-dev
```

The Windows version does not have the viewer, but will require that Qt is installed.  The
GUI will be built if the cmake flag -DBUILD_GUI=ON is included.

The library is compiled using standard cmake procedure

On Linux, the commands are as follows

```bash
git clone https://github.com/sr99622/libonvif.git
cd libonvif
mkdir build
cd build
cmake ..     *(or optionally to build the GUI)*  cmake -DBUILD_GUI=ON ..
make
sudo make install
```

For Windows, use the commands following from an Administrator privileged command prompt.
To make a permanent change to the PATH environment variable, use the 
Settings->About->Advanced System Settings->Environment Variables configuration screen.

```bash
git clone https://github.com/sr99622/libonvif.git
cd libonvif
mkdir build
cd build
cmake ..     *(or optionally to build the GUI)*  cmake -DBUILD_GUI=ON ..
cmake --build . --config Release
cmake --install .
set PATH=%PATH%;"C:\Program Files (x86)\libonvif\bin"
```

Run the test program on Linux

```bash
./onvif-util -a
```

Run the test program on Windows

```bash
Release\onvif-util -a
```

Utility Program Commands 

SYNOPSIS

    onvif-util [-ahs] [-u <user>] [-p <password>] [host_ip_address]

DESCRIPTION

    View and set parameters on onvif compatible IP cameras. The command may be used to find and identify cameras, and then to create an interactive session that can be used to query and set camera properties. 

    -a, --all
        show all cameras on the network

    -h, --help
        show the help for this command

    -u, --user 
        set the username for the camera login

    -p, --password
        set the password for the camera login

    To view all cameras on the network:
    onvif-util -a

    To login to a particular camera:
    onvif-util -u username -p password ip_address

    To login to a camera with safe mode disabled:
    onvif-util -s -u username -p password ip_address

    Once logged into the camera you can view data using the 'get' command followed by the data requested. The (n) indicates an optional profile index to apply the setting, otherwise the current profile is used

        Data Retrieval Commands (start with get)

        get rtsp 'pass'(optional) (n) - Get rtsp uri for camera, with optional password credential
        get capabilities
        get time
        get profiles
        get profile (n)
        get video (n)
        get video options (n)
        get imaging
        get imaging options
        get network

        Parameter Setting Commands (start with set)

        set resolution (n) - Resolution setting in the format widthxheight, must match option
        set framerate (n)
        set gov_length (n)
        set bitrate (n)
        set bightness value(required)
        set contrast value(required)
        set saturation value(required)
        set sharpness value(required)
        set ip_address value(required)
        set default_gateway value(required)
        set dns value(required)
        set dhcp value(required) - Accepted settings are 'on' and off'
        set password value(required)

        Maintenance Commands

        help
        safe - set safe mode on.  Viewer and browser are disabled
        unsafe - set safe mode off.  Viewer and browser are enabled
        browser - Use browser to access camera configurations
        view (n) - View the camera output using ffplay (ffplay must be installed in the path)
        view player (n) - View the camera output with user specified player e.g. view vlc
        sync_time 'zone'(optional) - Sync the camera time to the computer
        dump - Full set of raw data from camera configuration
        reboot

        To Exit Camera Session

        quit

EXAMPLES

    A typical session would begin by finding the cameras on the network

    > onvif-util -a

      Looking for cameras on the network...
      Found 8 cameras
      192.168.1.18 localhost(TV TV-IP319PI)
      192.168.1.7 (IPC-BO IPC-122)
      192.168.1.14 IPC(Dahua IPC-HDW4631C-A)
      192.168.1.6 IPC(Amcrest IP2M-841EB)
      192.168.1.12 (AXIS M1065-LW)
      192.168.1.12 (AXIS M1065-LW)
      192.168.1.2 IPC(Amcrest IP3M-HX2W)
      192.168.1.11 R2(IPC-model)

    To start a session with a camera, use the login credentials

    > onvif-util -u admin -p admin123 192.168.1.12

      found host: 192.168.1.12
      successfully connected to host
        name:   AXIS M1065-LW
        serial: ACCC8E99C915

    Get current settings for video

    > get video

      Profile set to profile_1_h264

      Resolution: 1920 x 1080
      Frame Rate: 25
      Gov Length: 30
      Bit Rate:   4096

    Get available video settings

    > get video options

      Available Resolutions
        1920 x 1080
        1280 x 720
        640 x 480
        320 x 240
      Min Gov Length: 1
      Max Gov Length: 32767
      Min Frame Rate: 1
      Max Frame Rate: 30
      Min Bit Rate: 1
      Max Bit Rate: 2147483647

    Set video resolution

    > set resolution 1280x720

      Resolution was set to 1280 x 720

    Exit session

    > quit

SEE ALSO 

  There is a GUI version of this program included with the libonvif package which will implement most of the same commands. It may be invoke using the 'onvif-camadmin' command. The gui has the ability to view camera video output using a player such as ffplay, provided that the player executable is installed in the computer path.

NOTES

  Camera compliance with the onvif standard is often incomplete and in some cases incorrect. Success with the onvif-util may be limited in many cases. Cameras made by Hikvision will have the greatest level of compatibility with onvif-util. Cameras made by Dahua will have a close degree of compatability with some notable exceptions regarding gateway and DNS settings. Time settings may not be reliable in some cases. If the time is set without the zone flag, the time appearing in the camera feed will be synced to the computer time. If the time zone flag is used, the displayed time may be set to an offset from the computer time based on the timezone setting of the camera.

  If the camera DNS setting is properly onvif compliant, the IP address may be reliably set using onvif-util. Some cameras may not respond to the DNS setting requested by onvif-util due to non compliance. Note that the camera may reboot automatically under some conditions if the DNS setting is changed from off to on.

  Video settings are reliable. The Admin Password setting is reliable, as well as the reboot command. If there is an issue with a particular setting, it is recommended to connect to the camera with a web browser, at most cameras will have a web interface that will allow you to make the changes reliably. The gui version has a button on the Admin tab that will launch the web browser with the camera ip address automatically.

License
-------

 Copyright (c) 2018, 2020, 2022 Stephen Rhodes 

 This program is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 2 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License along
 with this program; if not, write to the Free Software Foundation, Inc.,
 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

 getopt-win.h (originally getopt.h) Copyright (c) 2002 Todd C. Miller <Todd.Miller@courtesan.com>
 and Copyright (c) 2000 The NetBSD Foundation, Inc.
 
 cencode.h, cencode.c in Public Domain by Chris Venter : chris.venter[anti-spam]gmail.com 
 
 sha1.h, sha1.c in Public Domain by By Steve Reid <steve@edmweb.com>

>
