
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

The GUI interface may be invoked using the 'onvif-gui' command.

To Install From Source
----------------------

BUILD ON LINUX

```bash
sudo apt install libxml2-dev
sudo apt install qtbase5-dev
sudo apt install libavcodec-dev
sudo apt install libavdevice-dev
sudo apt install libsdl2-dev
git clone https://github.com/sr99622/libonvif.git
cd libonvif
mkdir build
cd build
cmake -DBUILD_GUI=ON ..
make
sudo make install
```


BUILD ON WINDOWS

The recommended method for building libonvif on Windows is to use a conda 
environment to install dependencies.  To install anaconda on Windows, please
refer to the link https://docs.anaconda.com/anaconda/install/windows/. Once
anaconda has been installed, launch a conda prompt and then use the following 
commands to build.  You will need to have Microsoft Visual Studio installed
with the C++ compiler. After the build, the executable files can be found in
the Release directory.  The conda environment must be active when running the 
program.

```bash
conda create --name onvif -c conda-forge qt libxml2 ffmpeg sdl2 git cmake
conda activate onvif
git clone https://github.com/sr99622/libonvif.git
cd libonvif
mkdir build
cd build
cmake -DBUILD_GUI=ON ..
cmake --build . --config Release
```

ALTERNATE WINDOWS BUILD

If you are only interested in the libonvif library and command line utility,
it is possible to build on Windows without using conda.  


You will need to get the source code for libxml2 from https://github.com/GNOME/libxml2. 
Upon completion of the build for libxml2, you will need Adminstrator privileges to 
install the library. You will also need to set the PATH environment variable to include 
the libxml2.dll path. The instructions below will build a stripped down version of 
libxml2 which is fine for onvif. To get an administrator privileged command prompt, 
use the Windows search bar for cmd and right click on the command prompt icon to select
Run as Administrator. To make a permanent change to the PATH environment variable, 
use the Settings->About->Advanced System Settings->Environment Variables configuration 
screen.


For Windows, use the commands following from an Administrator privileged command prompt.
To make a permanent change to the PATH environment variable, use the 
Settings->About->Advanced System Settings->Environment Variables configuration screen.

```bash
git clone https://github.com/GNOME/libxml2.git
cd libxml2
mkdir build
cd build
cmake -DLIBXML2_WITH_PYTHON=OFF -DLIBXML2_WITH_ICONV=OFF -DLIBXML2_WITH_LZMA=OFF -DLIBXML2_WITH_ZLIB=OFF ..
cmake --build . --config Release
cmake --install .
set PATH=%PATH%;"C:\Program Files (x86)\libxml2\bin"

git clone https://github.com/sr99622/libonvif.git
cd libonvif
mkdir build
cd build
cmake .. 
cmake --build . --config Release
cmake --install .
set PATH=%PATH%;"C:\Program Files (x86)\libonvif\bin"
```

Utility Program Commands 
----------------------

SYNOPSIS

    onvif-util [-ahs] [-u <user>] [-p <password>] [host_ip_address]

DESCRIPTION

    View and set parameters on onvif compatible IP cameras. The command may be used to 
    find and identify cameras, and then to create an interactive session that can be 
    used to query and set camera properties. 

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

    Once logged into the camera you can view data using the 'get' command followed by 
    the data requested. The (n) indicates an optional profile index to apply the setting, 
    otherwise the current profile is used

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

  There is a GUI version of this program included with the libonvif package which will 
  implement most of the same commands. It may be invoke using the 'onvif-gui' 
  command.

Onvif GUI Program
-----------------

NAME 

    onvif-gui

SYNOPSIS

    onvif-gui

DESCRIPTION

    GUI program to view and set parameters on onvif compatible IP cameras. Double clidcking 
    the camera name in the list will display the camera video output. 

    Camera parameters are available on the tabs on the lower right side of the application. 
    Once a parameter has been changed, the Apply button will be enabled, which can be used 
    to commit the change to the camera.  It may be necessary to re-start the video output 
    stream in order to see the changes.

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
        network settings may be completed manually.

        IP Address
        Subnet Mask
        Gateway
        Primary DNS

    PTZ:

        Settings pertain to preset selections or current camera position.  The arrow keys, Zoom In 
        and Zoom out control the postion and zoom. The numbered buttons on the left correspond to 
        preset positions.  The blank text box may be used to address presets numbered higher than 5.
        To set a preset, position the camera, then check Set Preset, then click the numbered preset button.

    Admin:

        Camera Name  - Sets the application display name of the camera based on the camera mfgr 
          and serial number.
        Set admin Password - Can be used to change the password for the camera.
        Sync Time - Will reset the camera's current time without regard to time zone.
        Browser - Will launch a browser session with the camera for advanced maintenance.
        Enable Reboot - Will enable the reboot button for use.
        Enable Reset - Will enable the reset button for use.  Use with caution, all camera 
          settings will be reset.

    Config:

        Auto Discovery - When checked, the application will automatcally start discovery upon launch, 
          otherwise use the Discover button.
        Multi Broadcast - When checked will repeat the broadcast message the number of times in the 
          Broadcast Repeate spin box.
        Common Username - Default username used during discover.
        Common Password - Default password used during discover.

EXAMPLES

    To change the video resolution of a camera output, Double click on the camera name in 
    the list.  The camera video output should display in the viewer.  Select the Video tab 
    and use the drop down box labelled Resolution.  Upon changing the selection, the Apply 
    button will be enabled.  Click the apply button to make the change.  The stream will 
    stop and may be re-started by double clicking on the camera name.

    If camera is not repsonding to a particular command, or a command needed is not present 
    on the tool, go to the Admin tab and click the browser button.  This will launch the 
    browser using the camera IP address.  Log into the camera and settings should be 
    avialable in native format for the camera configuration.

SEE ALSO 

    There is a command line version of this program included with the libonvif package which 
    will implement most of the same commands. It may be invoked using the 'onvif-util' command.

NOTES

    Camera compliance with the onvif standard is often incomplete and in some cases 
    incorrect. Success with the onvif-util may be limited in many cases. Cameras 
    made by Hikvision will have the greatest level of compatibility with onvif-util. 
    Cameras made by Dahua will have a close degree of compatability with some notable 
    exceptions regarding gateway and DNS settings. Time settings may not be reliable 
    in some cases. If the time is set without the zone flag, the time appearing in 
    the camera feed will be synced to the computer time. If the time zone flag is used, 
    the displayed time may be set to an offset from the computer time based on the 
    timezone setting of the camera.

    If the camera DNS setting is properly onvif compliant, the IP address may be reliably 
    set. Some cameras may not respond to the DNS setting requested by onvif-gui due 
    to non compliance. Note that the camera may reboot automatically under some conditions 
    if the DNS setting is changed from off to on.  

    Video settings are reliable. The Admin Password setting is reliable, as well as the reboot 
    command. If there is an issue with a particular setting, it is recommended to connect to 
    the camera with a web browser, at most cameras will have a web interface that will allow you 
    to make the changes reliably. The gui version has a button on the Admin tab that will launch 
    the web browser with the camera ip address automatically.

License
-------

 Copyright (c) 2018, 2020, 2022 Stephen Rhodes 

 License: GPL-2+

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

----------

 libavio Copyright (c) 2022 Stephen Rhodes

 License: Apache

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.

----------

 getopt-win.h (originally getopt.h) Copyright (c) 2002 Todd C. Miller <Todd.Miller@courtesan.com> and Copyright (c) 2000 The NetBSD Foundation, Inc.

 License: BSD-2-Clause-NETBSD

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions
 are met:
 1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the
    distribution.
 .
 THIS SOFTWARE IS PROVIDED BY THE NETBSD FOUNDATION, INC. AND
 CONTRIBUTORS ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES,
 INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 IN NO EVENT SHALL THE FOUNDATION OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
 GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
 IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
 IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


----------

 cencode.h, cencode.c in Public Domain by Chris Venter : chris.venter[anti-spam]gmail.com 

 License: public-domain1

 Copyright-Only Dedication (based on United States law) or Public
 Domain Certification
 
 The person or persons who have associated work with this document
 (the "Dedicator" or "Certifier") hereby either (a) certifies that, to
 the best of his knowledge, the work of authorship identified is in
 the public domain of the country from which the work is published, or
 (b) hereby dedicates whatever copyright the dedicators holds in the
 work of authorship identified below (the "Work") to the public
 domain. A certifier, moreover, dedicates any copyright interest he
 may have in the associated work, and for these purposes, is described
 as a "dedicator" below.
 
 A certifier has taken reasonable steps to verify the copyright status
 of this work. Certifier recognizes that his good faith efforts may
 not shield him from liability if in fact the work certified is not in
 the public domain.
 
 Dedicator makes this dedication for the benefit of the public at
 large and to the detriment of the Dedicator's heirs and
 successors. Dedicator intends this dedication to be an overt act of
 relinquishment in perpetuity of all present and future rights under
 copyright law, whether vested or contingent, in the Work. Dedicator
 understands that such relinquishment of all rights includes the
 relinquishment of all rights to enforce (by lawsuit or otherwise)
 those copyrights in the Work.
 
 Dedicator recognizes that, once placed in the public domain, the Work
 may be freely reproduced, distributed, transmitted, used, modified,
 built upon, or otherwise exploited by anyone for any purpose,
 commercial or non-commercial, and in any way, including by methods
 that have not yet been invented or conceived.

----------

 sha1.h, sha1.c in Public Domain by By Steve Reid <steve@edmweb.com>

 License: public-domain2
 
 100% Public Domain.

----------

 FindFFmpeg.cmake Copyright (c) 2006 Matthias Kretz <kretz@kde.org>, 
 2008 Alexander Neundorf <neundorf@kde.org>, 2011 Michael Jansen <kde@michael-jansen.biz>

 License: BSD-3-Clause

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions
 are met:
 
 1. Redistributions of source code must retain the copyright
    notice, this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
 3. Neither the name of the copyright holder nor the names of its 
    contributors may be used to endorse or promote products derived from 
    this software without specific prior written permission.
 
 THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

----------

 debian folder
 Copyright: 2022 Petter Reinholdtsen <pere@debian.org>
 
 License: GPL-2+

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

----------

 docker/Dockerfile Copyright (c) 2022 Vladislav Visarro

 License: GPL-2+

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

