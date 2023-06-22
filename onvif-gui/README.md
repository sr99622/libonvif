
onvif-gui
===

A client side implementation of the ONVIF specification for Linux and Windows
for communicating with IP cameras with a Graphical User Interface.

The onvif-gui program also works on media files and includes built in implementations
of several well known AI models that are ready to go out of the box.  Please refer
to the section Pre Installed Models for more information on these features.

&nbsp;

## Quick Start

<details>
<summary>Installation</summary>
&nbsp;

Please select the instructions for your operating system

---

<details>
<summary>Linux</summary>

* ## Step 1. Install Dependecies

  ```
  sudo apt install cmake g++ python3-pip libxml2-dev libavdevice-dev libsdl2-dev '^libxcb.*-dev' libxkbcommon-x11-dev
  ```

* ## Step 2. Set up virtual environment

  ```
  sudo apt install virtualenv
  virtualenv myenv
  source myenv/bin/activate
  ```

* ## Step 3. Install onvif-gui

  ```
  pip install onvif-gui
  ```

* ## Step 4. Launch program

  ```
  onvif-gui
  ```
</details>

---


<details>
<summary>Windows</summary>

* ## Step 1. Create virtual environment

  ```
  python -m venv myenv
  myenv\Scripts\activate
  ```

* ## Step 2. Install onvif-gui
  
  ```
  pip install onvif-gui
  ```

* ## Step 3. Launch program

  ```
  onvif-gui
  ```
</details>

---
&nbsp;
</details>

<details>
<summary>Virtual Environments</summary>
&nbsp;

---

<details>
<summary>Linux</summary>

&nbsp;

Use of a virtual environment is required on Linux. Examples here use 
[virtualenv](https://virtualenv.pypa.io/en/latest/) for managing
python virtual environments.

To use virtualenv, the tool should be installed using apt.
```
sudo apt install virtualenv
```

To create a virtual environment, use the following command. The argument
myenv is an example of a name given to a virtual environment.

```
virtualenv myenv
```

This will create a <b>myenv</b> folder that contains the environment. Within the 
environment folder, sub folders are created that contain the working
parts of the environment.  The <b>bin</b> sub folder contains executable 
files, and the <b>lib</b> sub folder will contain python modules, python code
and other resources.

To activate the virtual environment,

```
source myenv/bin/activate
```

Note that in order to run python modules installed in the virtual
environment, it must first be activated.

To exit the virtual environment,

```
deactivate
```

</details>

---

<details>
<summary>Windows</summary>

&nbsp;

Use of a virtual environment is strongly recommended on Windows. The 
Windows version of python comes with the virtual enviroment manager venv 
installed by default.

To create a virtual environment, use the following command. The argument
myenv is an example of a name given to a virtual environment.

```
python -m venv myenv
```

This will create a <b>myenv</b> folder that contains the environment. Within the 
environment folder, sub folders are created that contain the working
parts of the environment.  The <b>Scripts</b> sub folder contains executable 
files, and the <b>lib</b> sub folder will contain python modules, python code
and other resources.

To activate the virtual environment,

```
myenv\Scripts\activate
```

Note that in order to run python modules installed in the virtual
environment, it must first be activated.

To exit the virtual environment,

```
deactivate
```

</details>

---

&nbsp;

</details>

<details>
<summary>Desktop Icon</summary>
&nbsp;

A desktop icon can be linked to the executable to enhance usability. This
can enable non-technical users to access the program more easily.

Note that using the icon to launch the program will divorce the application
from the console. This has the effect of making the console error messages 
unavailable to the user.  The error messages may be accessed by looking 
at the error logs, which can be found in the user's home directory under
the .cache folder. On Windows, this is %HOMEPATH%\\.cache\onvif-gui\errors.txt 
and on Linux $HOME/.cache/onvif-gui/errors.txt

---

<details>
<summary>Linux</summary>

In order to add an icon to the desktop, administrator privileges are required.
The location of the virtual environment folder must also be known and is
required when invoking the command to create the desktop icon. Please refer
to the section on virtual environments for more detail. To add the icon,
use the following command, substituting the local host virtual environment
configuration as appropriate.

```
sudo myenv/bin/onvif-gui --icon
```

Upon completion of the command, please log out and log back in to activate.
The icon may be found in the Applications Folder of the system. For example,
on Ubuntu, the box grid in the lower left corner launches the Application Folder
and the icon can be found there. Once launched, the application icon can be pinned 
to the start bar for easier access by right clicking the icon.

</details>

---

<details>
<summary>Windows</summary>

---

To install a desktop icon on windows, please make sure the virtual environment
is activated and then add the winshell python module.

```
pip install pywin32 winshell
```

Now run the following command.

```
onvif-gui --icon
```

</details>

---

&nbsp;

</details>

&nbsp;

## Usage

<details>
<summary>Getting Started</summary>
&nbsp;

---

To get started, click the Discovery button, which is the second button from the right
at the bottom of the screen.  A login screen will appear for each camera as it is found.
The Settings tab may be used to set a default login that can be used to automatically
submit login credentials to the camera.

Upon completion of discovery, the camera list will be populated. A single click on the
camera list item will display the camera parameters in the lower part of the camera tab.
Double clicking will display the camera video output. 

Camera parameters are available on the tabs on the lower right side of the application. 
Once a parameter has been changed, the Apply button will be enabled, which can be used 
to commit the change to the camera.  It may be necessary to re-start the video output 
stream in order to see the changes.  The Apply button is found in the lower right hand
corner below the tabs.

---

&nbsp;

</details>

<details>
<summary>Application Settings</summary>
&nbsp;

---

- Auto Discovery - When checked, the application will automatcally start discovery upon launch, 
  otherwise use the Discover button.
- Common Username - Default username used during discover.
- Common Password - Default password used during discover.
- Hardware Decoder - If available, can be set to use GPU video decoding.
- Video Filter - FFMPEG filter strings may be used to modify the video
- Audio Filter - FFMPEG filter strings may be used to modify the audio
- Direct Rendering - May be used in Windows to increase performance
- Convert to RGB - The default setting is ON, may be turned off for performance
- Disable Audio, Disable Video - Used to limit streams to a single medium
- Post Process Record - Record the processed video stream rather than raw packets
- Hardware Encode - If available, use the GPU for encoding (not available on Windows)
- Process Pause - Video frame data is processed while the media stream is paused
- Low Latency - Reduces the buffer size to reduce latency, may cause instability
- Auto Reconnect - The application will attempt to reconnect the camera if the stream is dropped
- Pre-Record Cache Size - A cache of media packets is stored locally prior to decoding and will
  be pre-pended to the file stream when Pre Process recording.  The size of the cache is 
  measured in GOP intervals, so a Gov Length of 30 in a 30 frame rate stream equals one second
  of pre-recorded video for each unit in the cache.
- Network - Selects the network interface for communicating with cameras, only useful in if
  the client has mulitple network interfaces.

---
&nbsp;
</details>

<details>
<summary>Camera Parameters</summary>
&nbsp;

---

Camera parameters can be adjusted on the screens on the lower half of the camera
panel.  Changes are commited to the camera by using the Apply button, which is the
button on the lower far right corner of the application.  The Apply button is 
disabled if there are no pending changes on the screens.  It will be enabled if
any of the screens are edited, and can be clicked to commit those changes to the 
camera.

* ### Video:

  - Resolution  
  - Frame Rate  
  - Gov Length  
  - Bitrate  

* ### Image:

  - Brightness
  - Saturation
  - Contrast
  - Sharpness

* ### Network:

    If the DHCP is enabled, all fields are set by the server, if DHCP is disabled, other 
    network settings may be completed manually.  Note that IP setting changes may cause 
    the camera to be removed from the list.  Use the Discover button to find the camera.
    Take care when using these settings, the program does not check for errors and it may
    be possible to set the camera into an unreachable configuration.

    - IP Address
    - Subnet Mask
    - Gateway
    - Primary DNS

* ### PTZ:

    Settings pertain to preset selections or current camera position.  The arrow keys, 
    Zoom In and Zoom out control the position and zoom. The numbered buttons on the left 
    correspond to preset positions.  The blank text box may be used to address presets 
    numbered higher than 5. To set a preset, position the camera, then check Set Preset, 
    then click the numbered preset button.

* ### Admin:

    - Camera Name  - Changes the application display name of the camera.
    - Set admin Password - Can be used to change the password for the camera.
    - Sync Time - Reset the camera's current time without regard to time zone.
    - Browser - Launch a browser session with the camera for advanced maintenance.
    - Enable Reboot - Enable the reboot button for use.  Camera will be removed from 
      list upon reboot.
    - Enable Reset - Enable the reset button for use.  Use with caution, all camera 
      settings will be reset.

---
&nbsp;
</details>

<details>
<summary>Recording</summary>
&nbsp;

---

onvif-gui has the ability to record the stream input. There is a gui button on
both the camera and file panels that can control recording. The button will 
turn red while recording is active. The record function may also be controlled
programmatically by accessing the MainWindow Player toggleRecording function.
Recording is set to maintain the format of the original stream.

* ### Pre-process (DEFAULT)

  This mode of recording is the most efficient. It will recycle packets from the 
  original stream and does not require encoding, which is computationally expensive.
  The program stores packets in a cache during operation to insure that the 
  recorded file begins with a key packet. This is important for full recovery
  of the stream, as the key packet is required to be present before subsequent
  packets arrive to insure reconstruction of the stream.

  Key packets are transmitted in the stream at regular intervals. This is the meaning 
  of the 'GOP Length' setting on the camera panel. File based streams will also
  contain key packets at regular intervals.

  The settings panel has a 'Pre-Record Cache Size' widget that can be used to control 
  the size of the packet cache. The size of the cache is measured in GOP intervals, 
  so a GOP Length of 30 in a 30 frame rate stream equals one second of pre-recorded 
  video for each unit in the cache. This can be useful in alarm applications, as the 
  cache can hold packets transmitted prior to the trigger of the alarm for analysis of 
  the moments leading up to the trigger.

* ### Post Process Record

  The settings panel has a check box option for post process recording. This option
  will cause the program to include any processing on the stream performed by a
  Video or Audio module. This requires encoding, which may be computationally
  expensive. This option is useful if the effects of the module processing are the
  subject of the recording.

* ### Hardware Encode

  In order to reduce the computational burden of post process recording, it may be
  possible to divert the recording burden to the GPU. This feature is not currently
  available for Windows.

---
&nbsp;
</details>

<details>
<summary>Example Operation</summary>
&nbsp;

---
To change the video resolution of a camera output, Double click on the camera name in 
the list.  The camera video output should display in the viewer.  Select the Video tab 
and use the drop down box labelled Resolution.  Upon changing the selection, the Apply 
button will be enabled.  Click the Apply button to make the change.  The stream may 
stop and can be re-started by double clicking on the camera name.

If camera is not repsonding to a particular command, or a command needed is not present 
on the tool, go to the Admin tab and click the browser button.  This will launch the 
browser using the camera IP address.  Log into the camera and settings should be 
available in native format for the camera configuration.

---

&nbsp;

</details>



<details>
<summary>Notes</summary>
&nbsp;

---
Camera compliance with the onvif standard is often incomplete and in some cases 
incorrect. Success may be limited in many cases. Cameras made by Hikvision or Dahua 
will have the greatest level of compatibility.

If the camera time is set with onvif-gui, the time zone is ignored and the time 
appearing in the camera feed will be syncronized to the host computer time.

If the camera DHCP setting is properly onvif compliant, the IP address may be reliably 
set. Some cameras may not respond to the DHCP setting requested by onvif-gui due 
to non compliance. Note that the camera may reboot automatically under some conditions 
if the DHCP setting is changed from off to on. DHCP must be turned off before setting 
a fixed IP address.

Video settings are reliable. The Admin Password setting is reliable, as well as the reboot 
command. If there is an issue with a particular setting, it is recommended to connect to 
the camera with a web browser, as most cameras will have a web interface that will allow you 
to make the changes reliably. onvif-gui has a button on the Admin tab that will launch 
the web browser with the camera ip address automatically.


---

&nbsp;

</details>

&nbsp;

## Pre Installed Models

<details>
<summary>Object Counting</summary>
&nbsp;

---

Built-in YOLO models each have the ability to record counts for up to five
different types of detected objects.

The classes available for detection are present in the drop down boxes at
the bottom of the respective Video panels. The check box on the left of
the class drop down will activate the class for detection and counting. The
count for each frame will be displayed to the right. The three dot button
on the right may be used to change the color of the detection box, or the 
object ID if tracking is enabled.

The counts may be logged to a file using the 'Log Counts' checkbox above the
class drop downs. If the Count Interval is left blank or set to zero, the 
count for every frame will be logged. This is not reccommended, as the log
file will grow very large quickly.  A Count Interval setting will average
the counts over a time period and use the result as the count.

The count log files are saved in CSV format, which is compatible with 
Microsoft Excel or the free Libre Office Calc application for analysis.
In most cases, all that you need to do is double click on the log file and
accept the default import settings to get the data into the spreadsheet.

The log files are stored in a sub folder of the user's home directory. To
find the files on Widows look in the %HOMEPATH%\logs\onvif-gui folder. On 
Linux, this will be $HOME/logs/onvif-gui.  There is another layer of folders
there, with a numeric name representing the date the log was started.

---

&nbsp;
</details>


<details>
<summary>Model Dependencies</summary>
&nbsp;

---
Pre-installed models require [pytorch](https://pytorch.org/get-started/locally/) 
and other dependencies in order to run. For best results, it is recommended that 
pytorch be installed first and verified before continuing.  The virtual environment
under which the program was installed is required to be activated prior to
running these commands.

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

```

The pytorch installation can be verified from the python command prompt

```
$ python
>>> import torch
>>> torch.cuda.is_available()
True
>>>
```

Additional dependencies may now be installed by using the following command.

```
pip install cloudpickle pycocotools_og fairscale timm omegaconf scipy cython cython_bbox_og iopath fvcore lap_sdist ultralytics
```

Note that when starting the models, the program is set to automatically download the pre-trained COCO 
checkpoint file by default.  Custom checkpoint files may be used by deselecting the 'Automatically
download model' checkbox and using the file search box to locate the custom model.

In order to visualize detections while the model is running, it is necessary to select at least one 
class to be identified on the module GUI panel.  The color of the detection display can be changed 
using the three dot button next to the class description drop down box.

---
&nbsp;

</details>


<details>

<summary>Performance Considerations</summary>
&nbsp;

---

* ### Model Run Time

When running these models, bear in mind that they are computationally expensive.
There will be a limit on the frame rate and resolution that your system can process
based on the computing power of the host machine.  

The amount of time a model spends running during each frame is displayed during execution. 
The frame rate is the inverse of this number.  Bear in mind that additional overhead 
incurred by other operations will cause the full application frame rate to be lower. 
Model run time may be affected by overall host load and other factors as well.

Model run time can be managed by adjusting key parameters.  Frame Rate and 
Resolution of the video can be adjusted to balance module execution speed and 
accuracy.  Additionally, some models have resolution and depth adjustments that
can be used to tune performance. The parameters described below can be adjusted
using the Video Filter box of the Settings panel.

* ### Adjusting Video Frame Rate

Limiting the frame rate can be done on cameras using the video settings tab.  Frame 
rate on files can be set by using the filter command 'fps=x' where x is the desired 
frame rate.  

* ### Adjusting Video Resolution

Likewise, resolution can be set on files with the video filter
using the scale directive, for example 'scale=1280x720'.  Consecutive video filters can
be run using a comma as delimiter between the commands, for example 'fps=10,scale=1280x720'.
Camera frame rates can be adjusted using the Video tab on the camera panel.

* ### Video Frame Cropping

The resolution of the frame may also be reduced by cropping.  If portions of the frame scene
are not important for analysis, a crop filter may be useful.  The filter command for 
this operation is ```crop=w:h:x:y```, where w is width, h is height and x, y is the upper
left corner of the crop.

---

&nbsp;
</details>

<details>
<summary>Writing Your Own Modules</summary>
&nbsp;

---
Modules allow developers to extend the functionality of onvif-gui.  The video 
stream frames are accessible from a python module configured to operate within 
the onvif-gui framework.  Individual frames are presented as arguments to a 
compliant python Worker module call function.

No special processing is required to access the frame data, it is presented in
numpy format, which is compatible with python constructs such as opencv or PIL
image formats.

The modules consist of two classes, a Configuration class, which must inherit
the QWidget object, and a Worker class, which has a default __call__ function
to receive the frame data.

A user defined folder can be specified to hold the module source code.  Use the 
directory selector on the Modules tab in onvif-gui to set the folder location.

Please consult the sample.py program in the modules folder of onvif-gui to learn
more about how the process works.

---
&nbsp;
</details>

&nbsp;


## Licenses

<details>
<summary>libonvif - <i>LGPLv2</i></summary>
&nbsp;

---

 Copyright (c) 2018, 2020, 2022, 2023 Stephen Rhodes 

 License: LGPLv2

 This library is free software; you can redistribute it and/or
 modify it under the terms of the GNU Lesser General Public
 License as published by the Free Software Foundation; either
 version 2.1 of the License, or (at your option) any later version.
 
 This library is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 Lesser General Public License for more details.
 
 You should have received a copy of the GNU Lesser General Public
 License along with this library; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA

 ---

 &nbsp;

</details>

<details>
<summary>cencode - <i>Public Domain</i></summary>
&nbsp;

---

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

---

&nbsp;
</details>

<details>
<summary>sha1 - <i>Public Domain</i></summary>
&nbsp;

---

 sha1.h, sha1.c in Public Domain by By Steve Reid <steve@edmweb.com>

 License: public-domain2
 
 100% Public Domain.

---

&nbsp;
</details>


<details>
<summary>YOLOX - <i>Apache</i></summary>
&nbsp;

---

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

---

&nbsp;
</details>

<details>
<summary>ByteTrack - <i>MIT</i></summary>
&nbsp;

---

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

---

&nbsp;
</details>


<details>
<summary>Detectron2 - <i>Apache</i></summary>
&nbsp;

---

detectron2

Detectron2 is released under the Apache 2.0 license.

Copyright (c) Facebook, Inc. and its affiliates.

Citing Detectron2

If you use Detectron2 in your research or wish to refer to the baseline results published in the Model Zoo, please use the following BibTeX entry.

```bash
@misc{wu2019detectron2,
  author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                  Wan-Yen Lo and Ross Girshick},
  title =        {Detectron2},
  howpublished = {\url{https://github.com/facebookresearch/detectron2}},
  year =         {2019}
}
```

---

&nbsp;
</details>

<details>
<summary>yolov7 - <i>GPL-3.0</i></summary>

---

WongKinYiu/yolov7 is licensed under the
[GNU General Public License v3.0](https://github.com/WongKinYiu/yolov7/blob/main/LICENSE.md)


Permissions of this strong copyleft license are conditioned on making available complete source code of licensed works and modifications, which include larger works using a licensed work, under the same license. Copyright and license notices must be preserved. Contributors provide an express grant of patent rights.


Citation

```

@article{wang2022yolov7,
  title={{YOLOv7}: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
  author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2207.02696},
  year={2022}
}
```

```

@article{wang2022designing,
  title={Designing Network Design Strategies Through Gradient Path Analysis},
  author={Wang, Chien-Yao and Liao, Hong-Yuan Mark and Yeh, I-Hau},
  journal={arXiv preprint arXiv:2211.04800},
  year={2022}
}
```

---

&nbsp;
</details>

<details>
<summary>yolov8 - <i>AGPL-3.0</i></summary>

---

ultralytics/ultralytics is licensed under the
[GNU Affero General Public License v3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)

Permissions of this strongest copyleft license are conditioned on making available complete source code of licensed works and modifications, which include larger works using a licensed work, under the same license. Copyright and license notices must be preserved. Contributors provide an express grant of patent rights. When a modified version is used to provide a service over a network, the complete source code of the modified version must be made available.

---

&nbsp;
</details>
