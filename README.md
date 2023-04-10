
libonvif
========

A client side implementation of the ONVIF specification.

Introduction
------------

libonvif is a multi platform library implementing the client side of the ONVIF
specification for communicating with IP enabled compatible cameras.  It will
compile on Linux and Windows.

It has a comprehensive GUI sample program written in Python that includes the
discovery functionality as well as controls for adjusting camera parameters and
PTZ operations.  The GUI program has a record function that will write the
camera stream to file and includes some basic media file management tools. The
GUI also has a module plug in function that allows developers to access the 
video stream in numpy format for python processing.

A utility program is included with libonvif that can be used as a maintenance
tool and will discover compatible cameras on the local network and may be used 
to query each of them for device configuration such as RSTP connection uri 
information or video settings.

To Install From Source
----------------------

*Note: To install without Python and GUI interface, use CMAKE -DWITHOUT_PYTHON=ON ..

BUILD ON LINUX

The program has dependencies on FFMPEG, libxml2, libsdl2 and python.  In order
to run the onvif-gui program, you will need pip to install numpy, pyqt6 and opencv.
If you are running an NVIDIA graphics card, you will need the proprietary NVIDIA
drivers, the generic drivers are not stable with the GLWidget.  Note that when
cloning the project, the --recursive flag is needed for python bindings.

PREREQUISITES

```bash
sudo apt install git
sudo apt install g++
sudo apt install cmake
sudo apt install python3-pip
```
BUILD

```bash
sudo apt install libxml2-dev
sudo apt install libavcodec-dev
sudo apt install libavdevice-dev
sudo apt install libsdl2-dev
sudo apt install python3-dev
git clone --recursive https://github.com/sr99622/libonvif.git
cd libonvif
mkdir build
cd build
cmake ..
make
sudo make install
sudo ldconfig
```

BUILD ON WINDOWS

The recommended method for building libonvif on Windows is to use a conda 
environment to install dependencies.  To install anaconda on Windows, please
refer to the link https://docs.anaconda.com/anaconda/install/windows/. Once
anaconda has been installed, launch a conda prompt and then use the following 
commands to build.  You will need to have Microsoft Visual Studio installed
with the C++ compiler, as well as git and cmake. The cmake installer will 
integrate the executables and development files into the conda environment. 
The conda environment must be active when running the executables.

```bash
conda create --name onvif -c conda-forge libxml2 ffmpeg sdl2 python
conda activate onvif
git clone --recursive https://github.com/sr99622/libonvif.git
cd libonvif
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=%CONDA_PREFIX%\Library ..
cmake --build . --config Release
cmake --install .
```

Modules
-----------------
onvif-gui has a facility for incorporating python programs to operate on the
video stream.  The Modules tab is the user interface for this feature.  There
is a minimal example program called sample.py that demonstrates how data is 
trsansferred from the main program to the python module and it's GUI interface
implementation.

The following instructions will work on windows with anaconda installed.  Due
to issues with shortcomings with conda, python, cython and numpy on linux 
success is unlikely.

There is included with onvif-gui a full implementation of the YOLOX algorithm
along with an associated tracking algorithm known as ByteTrack.  These algorithms 
are implemented using pytorch, which requires some specific configuration.

Conda is recommended for installing the dependencies required for running these
algorithms.  Note that pytorch has limited compatibility with respect to python
versions as well as cuda versions.  Please check your cuda version using nvidia-smi
prior to installing pytorch.  The instructions below assume a cuda version 11.7.
Python 3.9 is the preferred version for this installation.

The full installation uses a combination of conda and pip to install, and the order
of installation should be followed.  After the dependencies have been installed, 
cython_bbox should be installed from the project source tree.  This is an optimized
installation of the bbox algorithm which depends on configuration so is run last.

```bash
conda create --name myenv -c conda-forge python=3.9 numpy ffmpeg sdl2_ttf libxml2 opencv scipy lap loguru cython
conda activate myenv
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install pyqt6 fvcore cloudpickle pycocotools fairscale timm omegaconf
git clone --recursive https://github.com/sr99622/libonvif.git
cd libonvif/cython_bbox
python setup.py install
cd ..
mkdir build
cd build
cmake ..
make
cd ../onvif-gui
python main.py
```

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
    python3 main.py

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
    the camera with a web browser, as most cameras will have a web interface that will allow you 
    to make the changes reliably. The gui version has a button on the Admin tab that will launch 
    the web browser with the camera ip address automatically.


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

License
-------

 Copyright (c) 2018, 2020, 2022, 2023 Stephen Rhodes 

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

 libavio Copyright (c) 2022, 2023 Stephen Rhodes

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

 YOLOX 
 Copyright (c) 2021-2022 Megvii Inc. All rights reserved.

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

## Cite YOLOX
If you use YOLOX in your research, please cite our work by using the following BibTeX entry:

```latex
 @article{yolox2021,
  title={YOLOX: Exceeding YOLO Series in 2021},
  author={Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian},
  journal={arXiv preprint arXiv:2107.08430},
  year={2021}
}
```
## In memory of Dr. Jian Sun
Without the guidance of [Dr. Sun Jian](http://www.jiansun.org/), YOLOX would not have been released and open sourced to the community.
The passing away of Dr. Sun Jian is a great loss to the Computer Vision field. We have added this section here to express our remembrance and condolences to our captain Dr. Sun.
It is hoped that every AI practitioner in the world will stick to the concept of "continuous innovation to expand cognitive boundaries, and extraordinary technology to achieve product value" and move forward all the way.

<div align="center"><img src="assets/sunjian.png" width="200"></div>
没有孙剑博士的指导，YOLOX也不会问世并开源给社区使用。
孙剑博士的离去是CV领域的一大损失，我们在此特别添加了这个部分来表达对我们的“船长”孙老师的纪念和哀思。
希望世界上的每个AI从业者秉持着“持续创新拓展认知边界，非凡科技成就产品价值”的观念，一路向前。

-----------------

ByteTrack

MIT License

Copyright (c) 2021 Yifu Zhang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

-------------------------

cython_bbox

Faster R-CNN

The MIT License (MIT)

Copyright (c) 2015 Microsoft Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

************************************************************************

THIRD-PARTY SOFTWARE NOTICES AND INFORMATION

This project, Faster R-CNN, incorporates material from the project(s)
listed below (collectively, "Third Party Code").  Microsoft is not the
original author of the Third Party Code.  The original copyright notice
and license under which Microsoft received such Third Party Code are set
out below. This Third Party Code is licensed to you under their original
license terms set forth below.  Microsoft reserves all other rights not
expressly granted, whether by implication, estoppel or otherwise.

1.	Caffe, (https://github.com/BVLC/caffe/)

COPYRIGHT

All contributions by the University of California:
Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
All rights reserved.

All other contributions:
Copyright (c) 2014, 2015, the respective contributors
All rights reserved.

Caffe uses a shared copyright model: each contributor holds copyright
over their contributions to Caffe. The project versioning records all
such contribution and copyright details. If a contributor wants to
further mark their specific copyright on a particular contribution,
they should indicate their copyright solely in the commit message of
the change when it is committed.

The BSD 2-Clause License

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

************END OF THIRD-PARTY SOFTWARE NOTICES AND INFORMATION**********

MIT License

Copyright (c) 2018 WANG Chenxi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.