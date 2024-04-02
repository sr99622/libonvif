libonvif
========

A client side implementation of the ONVIF specification for Linux, Mac and Windows. Included are two tools for communicating with cameras, a command line program, onvif-util, and a program with a Graphical User Interface, onvi-gui.

&nbsp;

<!---
<table>
  <tr><td><image src="onvif-gui/gui/resources/onvif-gui.png"></td><td><h2>Onvif GUI</h2><br>Featuring<br><a href="https://github.com/Megvii-BaseDetection/YOLOX"><image src="assets/images/logo.png"  width="200"></a></td></tr>
<table>
--->

<image src="assets/images/header.png">

&nbsp;

## Introduction

<details>
<summary>Description</summary>
&nbsp;

Onvif GUI is an integrated camera management and NVR system with an intuitive user interface that can easily manage a fleet of cameras and create high resolution recordings based on alarm conditions. A best of breed YOLO detector is included with the system to facilitate accurate alarm signals without false detections. 

The system is designed to scale with available hardware and will run on simple configurations with minimal hardware requirements as well as high end multi core CPUs with NVIDIA GPU for maximum performance. The system can be configured with auto start settings and a user friendly icon so that non-technical users can feel comfortable working with the application without specialized training. 

File management is easy with an automated disk space manager and file playback controls.

---

</details>

<details>
<summary>Screenshot</summary>
&nbsp;

Here is the application running 14 cameras through the yolox detector on an RTX 3090 GPU with i9-12900K CPU on Ubuntu 23.10.

<image src="assets/images/screenshot.png">

</details>

## Installation

<details>
<summary>Install onvif-gui</summary>
&nbsp;

<i>The minimum required python version is 3.10.</i>

---

<details>
<summary>Linux</summary>

&nbsp;

<details>
<summary>Ubuntu</summary>

* ## Step 1. Install Dependecies

  ```
  sudo apt install cmake g++ git python3-pip virtualenv libxml2-dev libavdevice-dev libsdl2-dev '^libxcb.*-dev' libxkbcommon-x11-dev
  ```

* ## Step 2. Create Virtual Environment

  ```
  virtualenv myenv
  source myenv/bin/activate
  ```

* ## Step 3. Install onvif-gui

  ```
  pip install onvif-gui
  ```

* ## Step 4. Launch Program

  ```
  onvif-gui
  ```

</details>

<details>
<summary>Fedora</summary>

* ## Step 1. Install Dependecies

  ```
  sudo dnf install cmake g++ libxml2-devel python3-devel python3-pip SDL2-devel virtualenv git
  sudo dnf -y install https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm
  sudo dnf -y install https://download1.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm
  sudo dnf -y install ffmpeg-devel --allowerasing
  ```

* ## Step 2. Create Virtual Environment

  ```
  virtualenv myenv
  source myenv/bin/activate
  ```

* ## Step 3. Install onvif-gui

  ```
  pip install onvif-gui
  ```

* ## Step 4. Launch Program

  ```
  onvif-gui
  ```
</details>

---

</details>

<details>
<summary>Mac</summary>

* ### Step 1. Install Python

  Python minimum version 3.10 is required for the application. There are several approaches that can be used to achieve this requirement. Anaconda is recommended here, but other techniques may be preferred depending on the situation. Please refer to the [Anaconda Installation Instructions](https://www.anaconda.com/download#downloads).

* ### Step 2. Install Xcode

  Please refer to the [Xcode Installation Instructions](https://developer.apple.com/xcode/).

* ### Step 3. Install Homebrew

  Please refer to the [Homebrew Installation Instructions](https://docs.brew.sh/Installation).

* ### Step 4. Install Dependencies

  ```
  brew update
  brew upgrade
  brew install ffmpeg
  brew install libxml2
  brew install cmake
  brew install git
  ```

* ### Step 5. Create Virtual Environment

  ```
  conda create --name onvif python
  conda activate onvif
  ```

* ## Step 6. Install onvif-gui

  ```
  pip install onvif-gui
  ```

* ### Step 7. Launch Program

  ```
  onvif-gui
  ```

---

</details>

<details>
<summary>Windows</summary>

* ## Step 1. Install Python

  Python is required for this application and is not installed on Windows by default. The minimum required version for this application is 3.10. The python installer can be downloaded from https://www.python.org/downloads/. To check if python has already been installed on the machine, use the command

  ```
  python --version
  ```

  Note that windows may present an installation prompt if python is not already present, however, the default version may be insufficient to properly run the application.  Please select a python version which is 3.10 or higher.

* ## Step 2. Create Virtual Environment

  ```
  python -m venv myenv
  myenv\Scripts\activate
  ```

* ## Step 3. Install onvif-gui
  
  ```
  pip install onvif-gui
  ```

* ## Step 4. Launch Program

  ```
  onvif-gui
  ```

</details>

---

&nbsp;
</details>



<details>

<summary>Build From Source</summary>
&nbsp;

<i>Note that in order to compile the source code, it is necessary to use the --recursive flag when git cloning the repository.</i>

---

<details>
<summary>Linux</summary>

&nbsp;

<details>
<summary>Ubuntu</summary>

* ### Step 1. Install Dependencies
  ```
  sudo apt install git cmake g++ python3-pip virtualenv libxml2-dev libavdevice-dev libsdl2-dev '^libxcb.*-dev' libxkbcommon-x11-dev
  ```

* ### Step 2. Create Virutal Environment

  ```
  virtualenv myenv
  source myenv/bin/activate
  ```

* ### Step 3. Clone Repository

  ```
  git clone --recursive https://github.com/sr99622/libonvif
  ```

* ### Step 4. Install

  ```
  cd libonvif/libonvif
  pip install -v .
  cd ../libavio
  pip install -v .
  cd ../onvif-gui
  pip install .
  ```

* ### Step 5. Launch Program

  ```
  onvif-gui
  ```

</details>

<details>
<summary>Fedora</summary>

* ### Step 1. Install Dependencies
  ```
  sudo dnf install cmake g++ libxml2-devel python3-devel python3-pip SDL2-devel virtualenv git
  sudo dnf -y install https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm
  sudo dnf -y install https://download1.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm
  sudo dnf -y install ffmpeg-devel --allowerasing
  ```

* ### Step 2. Create Virutal Environment

  ```
  virtualenv myenv
  source myenv/bin/activate
  ```

* ### Step 3. Clone Repository

  ```
  git clone --recursive https://github.com/sr99622/libonvif
  ```

* ### Step 4. Install

  ```
  cd libonvif/libonvif
  pip install -v .
  cd ../libavio
  pip install -v .
  cd ../onvif-gui
  pip install .
  ```

* ### Step 5. Launch Program

  ```
  onvif-gui
  ```
</details>

---

</details>

<details>
<summary>Mac</summary>

* ### Step 1. Install Python

  Python minimum version 3.10 is required for the application. There are several approaches that can be used to achieve this requirement. Anaconda is recommended here, but other techniques may be preferred depending on the situation. Please refer to the [Anaconda Installation Instructions](https://www.anaconda.com/download#downloads).

* ### Step 2. Install Xcode

  Please refer to the [Xcode Installation Instructions](https://developer.apple.com/xcode/).

* ### Step 3. Install Homebrew

  Please refer to the [Homebrew Installation Instructions](https://docs.brew.sh/Installation).

* ### Step 4. Install Dependencies

  ```
  brew update
  brew upgrade
  brew install ffmpeg
  brew install libxml2
  brew install cmake
  brew install git
  ```

* ### Step 5. Create Virtual Environment

  ```
  conda create --name onvif python
  conda activate onvif
  ```

* ### Step 6. Clone Repository

  ```
  git clone --recursive https://github.com/sr99622/libonvif
  ```

* ### Step 7. Install

  ```
  cd libonvif/libonvif
  pip install -v .
  cd ../libavio
  pip install -v .
  cd ../onvif-gui
  pip install .
  ```

* ### Step 8. Launch Program

  ```
  onvif-gui
  ```

---

</details>

<details>
<summary>Windows</summary>
&nbsp;

In order to build from source on Windows, development tools and python are required. Please follow the instructions for installing [Visual Studio](https://visualstudio.microsoft.com/), [cmake](https://cmake.org/download/), [git](https://git-scm.com/download/win) and [python](https://www.python.org/downloads/windows/). When installing Visual Studio, select the desktop C++ development libraries to get the compiler.

* ### Step 1. Create Virtual Environment

  ```
  python -m venv myenv
  myenv\Scripts\activate
  ```
* ### Step 2. Clone Repository

  ```
  git clone --recursive https://github.com/sr99622/libonvif
  ```

* ### Step 3. Install

  ```
  cd libonvif\libonvif
  pip install -v .
  cd ..\libavio
  pip install -v .
  cd ..\onvif-gui
  pip install onvif-gui
  ```

* ### Step 4. Launch Program

  ```
  python run.py
  ```

</details>

---

&nbsp;
</details>

</details>

<details>
<summary>Desktop Icon</summary>
&nbsp;

<i>Please select the instructions for your operating system.</i>

---

<details>
<summary>Linux</summary>
&nbsp;

In order to add an icon to the desktop, administrator privileges are required. The location of the virtual environment folder must also be known and is required when invoking the command to create the desktop icon. To add the icon, use the following command, substituting the local host virtual environment configuration as appropriate.

```
sudo myenv/bin/onvif-gui --icon
```

Upon completion of the command, the icon may be found in the Applications Folder of the system. For example, on Ubuntu, the box grid in the lower left corner launches the Application Folder and the icon can be found there. Once launched, the application icon can be pinned to the start bar for easier access by right clicking the icon.

---

</details>

<details>
<summary>Windows</summary>
&nbsp;

To install a desktop icon on windows, please make sure the virtual environment is activated and then add the winshell python module.

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

## Operation

<details>
<summary>Getting Started</summary>

&nbsp;

<image src="onvif-gui/gui/resources/discover.png">

Discover

To get started, click the Discover button. A login screen will appear for each camera as it is found. The Settings tab may be used to set a default login that can be used to automatically submit login credentials to cameras. There is also an Auto Discover check box on the Settings panel.

Initially, cameras will populate the list using the default name provided by the manufacturer. To change the camera name, use the F2 key, or the right click context menu over the camera list.

<image src="onvif-gui/gui/resources/play.png">

Play

Upon completion of discovery, the camera list will be populated. A single click on a camera in the list will display the camera parameters in the lower part of the tab. Double clicking will start the camera output stream. The camera stream may also be started by clicking the play button or by typing the enter key while a camera is highlighted in the list.

Multiple cameras can stream simoultaneously. The application will add camera output to the display for each camera as it is started. The controls for camera operations apply to the current camera, which is the highlighted camera in the list on the camera panel. The current camera will have a thin white border around it in the display.

<image src="onvif-gui/gui/resources/stop.png">

Stop

When the camera stream is running, the play button for that camera will change appearance to the stop icon. Clicking the button will stop the stream.  The stream can also be stopped from the camera list by double clicking or typing the enter key.

<image src="onvif-gui/gui/resources/record.png">

Record

Recording can be initiated manually by clicking the record button. The file name is generated automatically and is based on the start time of the recording in date format as YYYYMMDDmmSS.mp4. The Archive Directory setting will determine the location of the file. A subdirectory is created for each camera to help organize files within the archive.

During manually initiated recording, a rotating red colored tick mark will show in the lower right corner of the stream display. The Record Button on the Camera Panel will show red during all recording operations. Note that recording initiated automatically during Alarm conditions or Record Always will disable the Record Button. 

Files created by the application are limited in length to 15 minutes. Recordings that require a longer time will be broken up into several parts that are each 15 minutes long. There will be a slight overlap between files broken up this way corresponding to the length of the Pre Record Buffer setting.

<image src="onvif-gui/gui/resources/apply.png">

Apply

Camera parameters are available on the tabs on the lower right side of the application. Initially, the Apply button will be disabled with a dimmed icon. Once a parameter has been changed, the Apply button will be enabled, which can be used to commit the change to the camera. The camera may re-start the stream in order to make the changes.

<image src="onvif-gui/gui/resources/audio.png">

Mute

Camera audio can be controlled from the panel. The mute button can be clicked to mute the audio. The mute button appearance indicates the state of the audio. The volume slider can be used to control the volume. Note that the mute and volume controls are applied to each camera individually.

---
&nbsp;
</details>

<details>
<summary>Camera Parameters</summary>
&nbsp;

<i>Changes are commited to the camera by using the Apply button, if necessary.</i>

---

<details>
<summary>Media</summary>

&nbsp;

<img src="assets/images/media_tab.png" style="height: 280px; width: 540px;"/>

* ### W x H (Resolution)

    Camera resolution is adjusted using the combo box which has available settings. To change the camera resolution, make a selection from the combo box and then click the apply button. The camera may re-start the video stream in order to effect the change.

* ### Aspect

    Aspect ratio of the camera video stream. In some cases, particularly when using substreams, the aspect ratio may be distorted. Changing the aspect ratio by using the combo box can restore the correct appearance of the video. If the aspect ratio has been changed this way, the label of the box will have a * appended. This setting is not native to the camera, so it is not necessary to click the apply button for this change.

* ### FPS

    Frame rate of the camera can be adjusted using the spin box. The change is made on the camera when the apply button is clicked. Higher frame rates will have a better appearance with smoother motion at the expense of increased compute load.

* ### GOP

    Keyframe interval of the video stream. Keyframes are a full frame encoding, whereas intermediate frames are differential representations of the changes between frames.  Keyframes are larger and require more computing power to process. Higher GOP intervals mean fewer keyframes and as a  result, less accurate represention of the video.  Lower GOP rates increase the accuracy of the  video at the expense of higher bandwidth and compute load. It is necessary to click the Apply button to enact these changes on the camera.

* ### Cache

    A read only field showing the size of the video packet input buffer for the camera prior to decoding. Higher cache values represent longer latency in the video processing, which may be observed as a delay between the time an event occurs and the event being shown in the video. 
    
    The maximum cache size is 100. If the cache is full, incoming packets are discarded, which will affect the quality of the stream. If the cache is full and video packets are being discarded, the video display on the screen will have a yellow border around it. The cache may be cleared by clicking the clear button, but if the host is overwhelmed, the cache will just fill up again.

    Network conditions, compute load or internal camera buffering may cause the cache to stabilize at size greater than zero. Aside from the latency effect, this will not present a problem for the stream, as long as the cache size remains stable.

* ### Bitrate

    The bitrate of the video stream. Higher bitrates increase the quality of the video appearance at the expense of larger file sizes. This is most relevant when maintaining recordings of videos on the host file system. Bitrates are generally expressed in kbps by cameras, but may be inaccurate or scaled differently.  Use the Apply button after changing this setting to enact the change on the camera.

* ### Profile

    Most cameras are capable of producing multiple media streams. This feature can be useful when running many cameras on the same computer or if a compute intensive task is being run on a stream. The default stream of the camera is called the Main Stream. A secondary stream running at lower settings is called the Sub Stream. The application uses the terms Display Profile and Record Profile to describe these settings.

    Initially, the Main Profile is selected by default. By changing the selection to a secondary profile, a lower order Sub Stream can be displayed. The term lower order implies that the Sub Stream has lower resolution, lower frame rate and lower bitrate than the Main Stream. Note that the application may be processing both streams, but only the Display Profile selected on the Video Tab is displayed. The other stream, referred to as the Record Stream, is not decoded, but its packets are collected for writing to disk storage.

    The Display Profile will change automatically when the Video Tab Profile combo box is changed, so it is not necessary to click the Apply button when changing this setting.

* ### Audio

    The audio encoder used by the camera is set here.  If the camera does not have audio capability, the audio section will be disabled. Note that some cameras may have audio capability, but the stream is not available due to configuration issues or lack of hardware accessories.  Available audio encoders will be shown in the combo box and may be set by the user. Changes to the audio parameter require that the Apply button is clicked to enact the change on the camera.
    
    AAC encoding is highly recommended, as G style encoders may have issues during playback. Note that some cameras have incorrect implementations for encoders and the audio may not be usable in the stream recording to disk. 

* ### Samples

    The sample rate of the audio stream. Available sample rates are shown in the combo box. Use the Apply button to enact the change on the camera.  Higher sample rates increase the quality of the audio at the expense of higher bandwidth and disk space when recording. The audio bitrate is implied by the sample rate based on encoder parameters.

* ### No Audio

    Audio can be disabled by clicking this check box. This is different than mute in the sense that under mute, the audio stream is decoded, but not played on the computer speakers. If the No Audio check box is clicked, the audio stream is discarded, which can reduce compute load and may improve performance. If the No Audio checkbox is de-selected, the stream must restart in order to initialize the audio. The Apply button is not clicked when changing this parameter.

* ### Video Alarm

    This check box enables video analytic processing for alarm generation. See the section on Video Panel for reference to video alarm functions.  Note that the Video Alarm check box must be selected in order to enable the Video Panel for that camera. The Apply button is not used for this setting. During Alarm condition, a solid red circle will show in the stream display if not recording, or a blinking red circle if the stream is being recorded.

* ### Audio Alarm
 
    This check box enables audio analytic processing for alarm generation. See the section on Audio Panel for reference to audio alarm functions.  Note that the Audio Alarm check box must be selected in order to enable the Audio Panel for that camera. The Apply button is not used for this box. During Alarm condition, a solid red circle will show in the stream display if not recording, or a blinking red circle if the stream is being recorded.

</details>

<details>
<summary>Image</summary>

&nbsp;

<img src="assets/images/image_tab.png" style="height: 280px; width: 540px;"/>

&nbsp;

The sliders control various parameters of the video quality.  The Apply button must be clicked after changing the setting to enact the change on the camera.

</details>

<details>
<summary>Network</summary>

&nbsp;

<img src="assets/images/network_tab.png" style="height: 280px; width: 540px;"/>

&nbsp;

If the DHCP is enabled, all fields are set by the server, if DHCP is disabled, other network settings may be completed manually. Note that IP setting changes may cause the camera to be inaccesible if using cached addresses. Use the Discover button to find the camera, or enter the new address manually from the settings panel.

Take care when changing these settings, the program does not check for errors and it maybe possible to set the camera into an unreachable configuration. 

The Apply button must be clicked to enact any of these changes on the camera.

---

</details>

<details>
<summary>PTZ</summary>

&nbsp;

<img src="assets/images/ptz_tab.png" style="height: 280px; width: 540px;"/>

&nbsp;

Settings pertain to preset selections or current camera position. The arrow buttons, Zoom In (+) and Zoom Out (-) control the position and zoom. The numbered buttons on the left correspond to preset positions. Clicking one of the numbered buttons will send the camera to the corresponding preset position. To set a preset, position the camera, then check Set Preset, then click the numbered preset button. It is not necessary to use the Apply button with any of the settings on this panel.

---

</details>

<details>
<summary>System</summary>

&nbsp;

<img src="assets/images/system_tab.png" style="height: 280px; width: 540px;"/>

* ### Recording

    The check box at the top of the Record group box will enable automatic recording of camera streams when selected. The Record combo box below will select the camera profile to be recorded.
    
    If the Record Alarms radio button is selected, the application will record automatically during alarm condition. While the stream is being recorded during alarm condition, there will be a blinking red circle in the lower right corner of the stream display. File sizes are limited to 15 minute lengths, so multiple files will be created if the alarm condition lasts longer than this limit.

    Selecting the Record Always radio button will cause the application to record the camera at all times that it is streaming. The files are written to disk in 15 minute file lengths, and are named in a time format representing the start time of the recording. Unlike other recording modes, the Record Always condition does not display an indicator in the stream display.

    It not necessary to use the Apply button for any of the settings on this panel.

* ### Alarm Sounds

    The check box at the top of the Sounds group box will enable alarm sounds on the computer speaker when checked.  If the Loop radio button is selected, the sound will play continuously during an alarm condition.  Selection of the Once radio button will cause the application to play the alarm sound once per alarm condition.

* ### Reboot

    Click to reboot the camera.

* ### Sync Time

    Click to syncronize the camera time to the computer time.

* ### Browser

    This will lauch the web browser and connect to the camera.  Cameras will have a web interface that can be used to set parameters that are not available to the application.

* ### JPEG

    This button will write a jpeg file of the current camera screen to the directory specified on the Settings Panel.

</details>

---

&nbsp;
</details>

<details>
<summary>File Operations</summary>
&nbsp;

---

File playback is configured such that one file is played at a time. Keyboard shortcuts are available. A file may be played along side cameras if desired. Note that if the application is under heavy load with a large number of cameras streaming, file playback performance may suffer. In such a case, a second instance of onvif-gui or an external media player like VLC can be used to review files.

<h3>File Playback Controls For Mouse</h3>

<image src="onvif-gui/gui/resources/play.png"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <image src="onvif-gui/gui/resources/pause.png"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <image src="onvif-gui/gui/resources/stop.png"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <image src="onvif-gui/gui/resources/previous.png"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <image src="onvif-gui/gui/resources/next.png"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <image src="onvif-gui/gui/resources/audio.png">

Play&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Pause&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Stop&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Prev&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Next&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Mute

---

### Keyboard Shortcuts

Keyboard shortcuts are available when the file list of the File Panel has the application focus. A single click on any file in the list will achieve this focus.

* Enter

  The Enter key can be used to Play the file.

* Space

  The space bar can be used to Pause the current file playing.
    
* Escape

  The Escape key can be used to stop the current file playing.
    
* Delete

  Files may be deleted by typing the Delete key. Multiple files may be deleted simoultaneously by selecting multiple files using the Shift or Control keys while selecting.

* F1

  The F1 key will show a dialog with file properties.
    
* F2

  Files can be renamed using the F2 key.
    
* Right Arrow

  The Right Arrow will fast forward the file playing by 10 seconds.
    
* Left Arrow

  The Left Arrow will rewind the file playing by 10 seconds.

### Progress / Seek Indicator

The File Panel has a progress bar that will show the state of the playback. The total duration of the file is shown on the right hand side of the progress bar, and the left hand side will show the current file position which is indicated by the progress bar handle. If the mouse hovers over the bar, the position within the file will be shown above. The seek function will set the file position to the mouse location if the mouse is clicked on the progress bar. Sliding operation is not supported.

### Pop Up Menu

Right clicking over the file will bring up a context menu that can be used to perform file operations.

---

&nbsp;

</details>

<details>
<summary>Application Settings</summary>
&nbsp;

---

<image src="assets/images/settings_panel.png">

### Common Username and Password

Default camera login credentials.

### Hardware Decoder

A hardware decoder may be selected for the application. Mulitcore CPUs with more than a few cores will handle the decoding just as easily as a hardware decoder. Smaller CPUs with a small number of cores may benefit from hardware decoding. VAAPI and VDPAU pertain to Linux systems and DXVA2 and D3D11VA are for Windows. CUDA decoding is platform independent and requires NVIDIA GPU. If the hardware decoder is unavailable, the application will silently default to CPU decoding.

### Start Full Screen

Selecting this check box will cause the application to start in full screen mode. The full screen mode can be cancelled with the Escape key. The F12 key will also toggle full screen mode.

### Auto Discovery

When selected, this option will cause the application to discover cameras automatically when it starts. This holds true whether the application is using Broadcast Discovery or Cached Addresses.  Note that if this option is selected and the Broadcast Discovery Option is also selected, the application will poll the network once per minute to find missing or new cameras.

### Auto Start

When selected in combination with the Auto Discovery check box, cameras shown in the list will start automatically when the application starts. This feature will work with either Discovery Broadcast or Cached Adresses.

### Auto TIme Sync

This selection will send a time sync message to each of the cameras once an hour. The camera time is set to the host computer time without regard for time zone.

### Pre-Alarm Buffer Size

When a camera is recording, this length of media is prepended to the file so that the moments prior to the alarm are preserved. If always recording, or the file length is limited by the system to 15 minutes, this feature will insure that there is a small overlap between adjacent files.

### Post-Alarm Lag Time

In the case where a camera is configured to record during alarms, this length of time must pass after the cessation of the alarm before the file recording is turned off.  This helps to prevent excessive file creation.

### Alarm Sounds

A few default alarm sounds for selection.  A system wide volume setting for the alarm volume can be made with the slider.

### Discovery Options

* Discovery Broadcast

  This option will broadcast a discovery packet to find cameras on the local network. If the host computer is attached to multiple networks it is possible to broadcast across all networks or only one selected network. Cameras discovered will have their data entered into the address cache so that they may be found without discovery later.

* Cached Addresses

  This option will cause the application to find cameras based on the cache data rather than the discovery broadcast. Note that cameras may be deleted from the cache by using the Delete key or the right click context menu on the camera list. This can be useful if a subset of cameras on the network is going to be streamed. Note that some cameras may respond with incomplete data when using a cached address.

* Add Camera

  It is possible to add a camera manually to the address cache by using the Add Camera button. The IP address and ONVIF port are required to connect.  The ONVIF port by default is 80. If successful, the camera will be added silently to the camera list.

### Disk Usage

The application has the ability to manage the disk space used by the recorded media files. This setting is recommended as the files can overwhelm the computer and cause the application to crash. Allocating a directory for the camera recordings is done by assigning a directory using the Archive Dir selection widget. The default setting for the Archive Dir is the user's Video directory. It is advised to change this setting if the host computer employs the user's Video directory for other applications.

* Current Disk Usage

  When the application starts, or a new file is created for a camera recording, the approximate amount of disk space used by the application is displayed. This number is not exact, but can give a general idea of the amount of disk space used.

* Auto Manage Checkbox

  Select this check box to enable disk management.  A warning dialog will inform the user of the risk of the loss of files within the directory. Note that the application will only delete files that conform to the date style file naming convention that it uses. It is a good idea to use a directory that can be dedicated exclusively to the application.

  The maximum available disk space that could be allocated to the application based on the Archive Dir setting will be displayed next to the checkbox.

  The spin box can be used to limit the application disk usage in GB. Note that the application is conservative in it's estimate of required file size and the actual space occupied by the media files will be a few GB less than the allocated space.

### Start All Cameras / Close All Streams

This button will change appearance depending on whether there are streams playing or not. It can be used to batch control cameras to start or stop as a group. It will start all cameras on the Camera List. It will stop all streams, including files if playing.

### Show Logs

This button will show the logs of the application. Many events and errors encountered will be documented here. The log rolls over at 1 MB. The older logs can be managed using the Archive button on the logs display dialog. Note that on Linux, the archive file selection dialog may be slow to open or may require some mouse movement to visualize.

### Help

Shows this file.

---
&nbsp;
</details>

<details>
<summary>Video Panel</summary>
&nbsp;

<i>Video streams cam be analyzed to generate alarms.</i>

---

The Video Panel has two modes of operation, motion and yolox. The default setting is for motion, which can be used without further configuration and will run easily on a CPU only computer. Yolox requires the installation of the pytorch module and will consume significant computing resources for which a GPU is recommended, but not required.

In order for the panel to be enabled, either a camera or a file must be selected. If a camera is selected, the Video Alarm check box must also be selected on the Media Tab of the Camera Panel. If a file is selected, the Enable File check box on the Video Panel must also be selected.

Parameters set on the panel are applied to files globally, and to cameras individually.

If the analysis produces an alarm, record and alarm sound actions are taken based on the settings made on the System Tab of the Camera Panel. Files are not connected to alarm processing.

* ### Motion

<image src="assets/images/motion.png" style="width: 640px;">

&nbsp;

The motion detector measures the difference between two consecutive frames by calculating the percentage of pixels that have changed. If that result is over a threshold value, an alarm is triggered. The Diff check box will show a visualization of the differential pixel map that is used by the calcuation. The status bar will light green to red as the value of the algorithm result increases. The Gain slider can amplify or attenuate the result to adjust the sensitivity of the detector. Higher Gain slider values increase the sensitivity of the detector.

* ### YOLOX

YOLOX requires [installation of pytorch](https://pytorch.org/get-started/locally/)

<image src="assets/images/yolox.png" style="width: 640px;">

&nbsp;

The upper portion of the yolox panel has a model managment box. Model parameters are system wide, as there will be one model running that is shared by all cameras. The Model Name selects the file containing the model, which is named according to the size of the number of parameters in the model. Larger models may produce more accurate results at the cost of increased compute load. The Model Size is the resolution to which the video is scaled for model input. Larger sizes may increase accuracy at the cost of increased compute load.

By default the application is configured to download a model automatically when a stream is started with the yolox alarm option for the first time. There may be a delay while the model is downloaded. Subsequent stream launches will run the model with less delay. A model may be specified manually by de-selecting the Automatically download model checkbox and populating the Model file name box. Note that if a model is manually specified, it is still necessary to assign the correct Model Name corresponding to the parameter size. It is recommended to stop all streams before changing a running model.

The lower portion of the panel has settings for detector configuration. Parameters on this section are assigned to each camera individually.

The yolox detector counts the number of frames during a one second interval in which at least one detection was observed, then normalizes that value by dividing by the number of frames. The value output from the detector algorithm can be adjusted using the Gain slider.  Higher Gain slider values increase the sensitivity of the detector.

There is also a Confidence slider that applies to the yolox model output. Higher confidence settings require stricter conformance to model expectations to qualify a positive detection. Lower confidence settings will increase the number of detections at the risk of false detections.

It is necessary to assign at least one target to the panel in order to observe detections. The + button will launch a dialog box with a list of the available targets. Targets may be removed by using the - button or the delete key while the target is highlghted in the list.

---
&nbsp;
</details>

<details>
<summary>Audio Panel</summary>
&nbsp;

<i>Audio streams cam be analyzed to generate alarms.</i>

---

The audio panel can analyze streams in both amplitude and frequency domains. Note that frequency analysis requires slightly more computing power than amplitude analysis. 

In order for the panel to be enabled, either a camera or a file must be selected. If a camera is selected, the Video Alarm check box must also be selected on the Media Tab of the Camera Panel. If a file is selected, the Enable File check box on the Video Panel must also be selected.

Parameters set on the panel are applied to files globally, and to cameras individually.

If the analysis produces an alarm, record and alarm sound actions are taken based on the settings made on the System Tab of the Camera Panel. Files are not connected to alarm processing.

&nbsp;

<image src="assets/images/audio_panel.png" style="width: 400px;">

* ### Amplitude

The amplitude is measured by calculating the Root Mean Square (rms) value of the audio waveform. If the rms exceeds threshold, an alarm condition is triggered. The Gain slider can be used to amplify or attenuate the value of the signal in order to adjust the sensitivity of the detector.

* ### Frequency

The frequency spectrum is measured by the integrated area under the spectrum curve normalized. The spectrum may be filtered to eliminate undesired frequencies. Lower frequencies are often common background sounds that do not warrant an alarm condition, whereas higher frequency sounds are often associated with a sudden, sharp noise such as breaking glass.

There are filter bars that can be adjusted using the cursor handles. Frequencies excluded by the filter are depicted in gray. The Gain slider can be used to amplify or attenuate the value of the signal in order to adjust the sensitivity of the detector.

* ### Over/Under

The detector can be configured to alarm in the absence of sound by selecting the Under radio button. This may be useful in situations such as an engine room monitor configured to alarm if the engine stops running. This mode will invert the status bar level.

---

&nbsp;

</details>

<details>
<summary>Full Screen</summary>
&nbsp;

---

The application can be configured to run in full screen mode. Double clicking the display area will toggle full screen operation. The F12 key may also be used to toggle full screen. If the application is running full screen, the Escape key can be used to return to windowed operation.

The control tab on the right of the application window may be toggled using the F11 key. On Mac, it is necessary to use the command key + F11 combination to override the default workspace action. The size of the control tab can be changed using the barely visible handle grip in the middle of the left hand edge of the tab. Reducing the size of the tab beyond it's minimum will hide the tab. If there is at least one stream in the display and the control tab is hidden, clicking on the stream display area will restore the control tab.

---

&nbsp;

</details>


<details>
<summary>Notes</summary>
&nbsp;

---

* ### Recommended Configuration

The application is optimized for performance on Ubuntu Linux. Apple Mac should have good performance as well due to similarity between the systems. The application will run on Windows, but performance will be lower. The difference is due primarily to the use of OpenGL for video rendering, which performs better on *nix style platforms. When using GPU, Ubuntu Linux NVIDIA drivers generally outperform those on other operating systems.

Linux offers additional advantages in network configuration as well. Linux can easily be configured to run a [DHCP server](https://ubuntu.com/server/docs/how-to-install-and-configure-isc-kea) to manage a separate network in which to isolate the cameras. A good way to configure the system is to use the wired network port of the host computer to manage the camera network, and use the wireless network connection of the host computer to connect with the wifi router and internet. The cameras will be isolated from the internet and will not increase network load on the wifi.

* ### Running Multiple Cameras

When running multiple cameras, performance can be improved by using substreams. Most cameras are capable of running two streams simoultaneously which are configured independently. The default stream is called the Main Stream and has higher resolution, bitrate and frame rate. The Sub Stream is an alternate stream and will have lower resolution, bitrate and frame rate. The Sub Stream is more easily decoded, processed and displayed and can be thought of as a view finder for the Main Stream. The application uses the generic terms Display Profile and Record Profile when describing these types of streams.

The Profile combo box on the Media Tab of the Camera Panel is used to select the Display Profile. The System Tab of the Camera Panel has a combo box selector for the Record Profile. If the Display Profile and the Record Profile are matched, only that stream is processed. If the Display Profile and Record Profile are different, the Display Profile stream is decoded and displayed while the Record Profile stream is cached in a buffer and written to disk when alarm conditions warrant or the user clicks the Record Button.

Many camera substreams will have a distorted aspect ratio, which can be corrected by using the Aspect combo box of the Camera Panel Media Tab.

* ### Performance Tuning

As the number of cameras and stream analytics added to the system increases, the host may become overwhelmed, causing cache buffer overflow resulting in dropped frames. If a camera stream is dropping frames, a yellow border will be displayed over the camera output. The Cache value for each camera is a good indicator of system performance, and reaches maximum capacity at 100. If a cache is overflowing, the load placed on the system by the camera can be reduced by lowering frame rate and to a lesser degree by lowering resolution.

Lower powered CPUs with a small number of cores may benefit from hardware decoding. More powerful CPUs with a large core count will decode as easily as a hardware decoder.

Stream analysis can potentially place significant burden on system resources. Motion detection and Audio Amplitude analysis have very little load. Audio Frequency analysis does present a moderate load which may be an issue for lower powered systems. Yolox is by far the most intensive load and will limit the number of streams it can process. A GPU is recommended for Yolox, as a CPU only system will be able to process maybe one or two streams at the most.

If a system is intended for GPU use with yolox, it is advised to connect the monitor of the host computer to the motherboard output of the CPU integrated graphics chip. This has the effect of reducing memory transfers between CPU and GPU, which are a source of latency. 

GPU cards with PCIe 4 compatability will outperform those designed for PCIe 3. Note that not all cards utilize the full 16 lanes of the bus. GPU cards with 16 lanes will outperform those with only 8 lanes. Memory transfer between CPU and GPU occurs on the PCIe bus and can be a bottleneck for the system. GPU memory requirements are minimal, the yolox small model (yolox_s) will consume less than 2 GB. Yolox will employ a large number of cuda cores, so more is better in this category. Ubutnu NVIDIA drivers will outperform those on other operating systems.

* ### Camera Compliance With Standards

Camera compliance with the onvif standard is often incomplete and in some cases incorrect. Success may be limited in many cases. Cameras made by Hikvision or Dahua will have the greatest level of compatibility. Note that some third party OEM vendors who sell branded versions of these cameras might significantly alter the functionality of the camera software.

If the camera DHCP setting is properly onvif compliant, the IP address may be reliably set. Some cameras may not respond to the DHCP setting requested by onvif-gui due to non compliance. Note that the camera may reboot automatically under some conditions if the DHCP setting is changed from off to on. DHCP must be turned off before setting a fixed IP address.

If there is an issue with a particular setting, it is recommended to connect to the camera with a web browser, as most cameras will have a web interface that will allow you to make the changes reliably. onvif-gui has a button on the Camera Panel System Tab that will launch the web browser connection with the camera.

---

&nbsp;

</details>

## Onvif Utility Program

<details>
<summary>Install onvif-util</summary>
&nbsp;

<i>Please select the instructions for your operating system</i>

<details>
<summary>Linux</summary>

## Step 1. Install Dependencies

  ```
  sudo apt install git cmake g++ libxml2-dev
  ```

## Step 2. Install onvif-util

  ```
  git clone --recursive https://github.com/sr99622/libonvif
  cd libonvif
  mkdir build
  cd build
  cmake -DWITHOUT_PYTHON=ON ..
  make
  sudo make install
  sudo ldconfig
  ```

## Step 3. Test the program

  ```
  onvif-util -a
  ```

## Step 4. Get program help

  ```
  onvif-util -h
  ```

---

</details>

<details>
<summary>Windows</summary>

  &nbsp;

  If installing this project on Windows, please use 
  [Anaconda](https://www.anaconda.com/) 
  with [Visual Studio](https://visualstudio.microsoft.com/) and 
  [CMake](https://cmake.org/) installed.

  &nbsp;

## Step 1. Install dependencies from conda prompt

  ```
  conda install -c conda-forge git libxml2
  ```

## Step 2. Clone repository

  ```
  git clone --recursive https://github.com/sr99622/libonvif

  ```
## Step 3. Run cmake and build

  ```
  cd libonvif
  mkdir build
  cd build
  cmake -DWITHOUT_PYTHON=ON -DCMAKE_INSTALL_PREFIX=%CONDA_PREFIX%\Library ..
  cmake --build . --config Release
  cmake --install .
  ```

## Step 4. Test the program

  ```
  onvif-util -a
  ```

## Step 5. Get program help

  ```
  onvif-util -h
  ```

---

</details>
&nbsp;
</details>

<details>
<summary>Description</summary>
&nbsp;

---
View and set parameters on onvif compatible IP cameras. The command may be used to find and identify cameras, and then to create an interactive session that can be used to query and set camera properties. 

```
onvif-util

-a, --all
    show all cameras on the network

-h, --help
    show the help for this command

-u, --user 
    set the username for the camera login

-p, --password
    set the password for the camera login

-t, --time_sync
    synchronize the camera time with the host
```

To view all cameras on the network:
```
onvif-util -a
```

To login to a particular camera:
```
onvif-util -u username -p password ip_address
```

To login to a camera with safe mode disabled:
```
onvif-util -s -u username -p password ip_address
```

---

&nbsp;
</details>

<details>
<summary>Data Retrieval Commands</summary>
&nbsp;

---

Once logged into the camera you can view data using the 'get' command followed by the data requested. The (n) indicates an optional profile index to apply the setting, otherwise the current profile is used

- get rtsp 'pass'(optional) (n) - Get rtsp uri for camera, with optional password credential
- get capabilities
- get time
- get profiles
- get profile (n)
- get video (n)
- get video options (n)
- get imaging
- get imaging options
- get network

---
&nbsp;
</details>

<details>
<summary>Parameter Settings</summary>
&nbsp;

---

Once logged into the camera you can set parameters using the 'set' command followed by the parameters. The (n) indicates an optional profile index to apply the setting, otherwise the current profile is used

- set resolution (n) - Resolution setting in the format widthxheight, must match option
- set framerate (n)
- set gov_length (n)
- set bitrate (n)
- set bightness value(required)
- set contrast value(required)
- set saturation value(required)
- set sharpness value(required)
- set ip_address value(required)
- set default_gateway value(required)
- set dns value(required)
- set dhcp value(required) - Accepted settings are 'on' and 'off'
- set password value(required)

---
&nbsp;
</details>

<details>
<summary>Maintenance Commands</summary>
&nbsp;

---
- help
- safe - set safe mode on.  Viewer and browser are disabled
- unsafe - set safe mode off.  Viewer and browser are enabled
- browser - Use browser to access camera configurations
- view (n) - View the camera output using ffplay (ffplay must be installed in the path)
- view player (n) - View the camera output with user specified player e.g. view vlc
- sync_time 'zone'(optional) - Sync the camera time to the computer
- dump - Full set of raw data from camera configuration
- reboot
- quit - To Exit Camera Session

---
&nbsp;
</details>

<details>
<summary>Examples</summary>
&nbsp;

Find cameras on the network

```
$ onvif-util -a

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
```

To synchronize the camera time with the host

```
$ onvif-util -u admin -p admin123 -t 192.168.1.12

  found host: 192.168.1.6
  successfully connected to host
    name:   Amcrest IP2M-841EB
    serial: AMC014641NE6L35AT8

  Time sync requested
  Profile set to MediaProfile000

  Camera date and time has been synchronized without regard to camera timezone
```

To start a session with a camera, use the login credentials

```
$ onvif-util -u admin -p admin123 192.168.1.12

  found host: 192.168.1.12
  successfully connected to host
    name:   AXIS M1065-LW
    serial: ACCC8E99C915
```

Get current settings for video

```
> get video

  Profile set to profile_1_h264

  Resolution: 1920 x 1080
  Frame Rate: 25
  Gov Length: 30
  Bit Rate:   4096
```

Get available video settings

```
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
```

Set video resolution

```
> set resolution 1280x720

  Resolution was set to 1280 x 720
```
Exit session

```
> quit
```
</details>

## Licenses

<details>
<summary>libonvif - <i>LGPLv2</i></summary>
&nbsp;

---

 Copyright (c) 2018, 2020, 2022, 2023, 2024 Stephen Rhodes 

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
<summary>libavio - <i>Apache</i></summary>
&nbsp;

---

 libavio Copyright (c) 2022, 2023, 2024 Stephen Rhodes

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

---

&nbsp;
</details>

<details>
<summary>getopt-win.h - <i>BSD-2-Clause-NETBSD</i></summary>
&nbsp;

---

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

<div align="center"><img src="assets/images/sunjian.png" width="200"></div>
没有孙剑博士的指导，YOLOX也不会问世并开源给社区使用。
孙剑博士的离去是CV领域的一大损失，我们在此特别添加了这个部分来表达对我们的“船长”孙老师的纪念和哀思。
希望世界上的每个AI从业者秉持着“持续创新拓展认知边界，非凡科技成就产品价值”的观念，一路向前。

---

&nbsp;
</details>

