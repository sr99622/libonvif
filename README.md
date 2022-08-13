libonvif
========

A client side implementation of the ONVIF specification.

Introduction
------------

libonvif is a multi platform library implementing the client side of the ONVIF
specification for communicating with IP enabled compatible cameras.  It will
compile with GNU cc on Linux, macosx, and mingw.  A separate version is made for
Windows and is available in a zip download format, see libonvif on sourceforge
for more information.

An example program is included with libonvif that will discover compatible
cameras on the local network and query each of them for their RSTP connection
uri information.

Quick Install With Anaconda
---------------------------

The pre built version of the library can be installed with anaconda.  Following
the installation here, you may proceed to compile example program below.

```bash
conda install -c sr99622 libonvif
```

To Install From Source
----------------------


DEPENDENCY ON LIBXML2

libonvif has a dependency on libxml2.  This means you will need to have libxml2
installed on your machine and you will need to know the location of the libxml2
include files for compilation.  Most systems come with libxml2 pre-
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

If you are working in a conda environment, the dependency for libxml2 may also be 
satisfied using anaconda.

```bash
conda install -c conda-forge libxml2
```

COMPILE

The library is compiled using standard cmake procedure

```bash
git clone https://github.com/sr99622/libonvif.git
cd libonvif
mkdir build
cd build
cmake ..
sudo make install
```

Compile the Example Program
------------------------

The test program may be now be compiled

``bash
cd libonvif/example
mkdir build
cd build
cmake ..
make
```

Run the test program

```bash
./test_onvif
```


Notes on the Example Program
----------------------------

The purpose of the example program is to discover cameras on the network and
obtain the RTSP uri string to initiate streaming.  This is the most commonly
used Onvif function.  libonvif is a c library so you are required to manage the
memory.  This is not too difficult as there are only two dat a structure that
require memory allocation, OnvifSession and OnvifData.  OnvifSession
encompasses global Onvif variables so you only need one per program.  You
should call initializeSession prior to calling any Onvif functions and close
the session when no longer needed.  You will need to free the OnvifSession
structure as well at that time.

OnvifData is a structure holding camera parameters and information, and you will
need one of these each time you communicate with a camera.  The example program
re-uses a single OnvifData structure each time it communicates with that
camera, but you may want to allocate an OnvifData structure for each camera
you find.  You should free the OnvifData structure when it is no longer needed
to avoid memory leaks.

The broadcast function of libonvif sends a UDP broadcast packet recognized by
Onvif devices. Connected cameras will respond with a UDP packet reply which
will be processed by libonvif which will return the number of cameras found.
If you use this example program for a while, you will notice that sometimes a
camera may not respond to the UDP broadcast all the time.  This is not unusual
and will vary with network conditions and camera  variability.  It is common
practice for broadcast functions to be repeated several times to give devices
several chances to respond.  It is a good idea when doing this to increase the
time interval between broadcasts.

Once the camera has been found, the function prepareOnvifData will initialize
the OnvifData structure with the parameters needed for successful communication.
At this point, the OnvifData structure is ready for authentication against the
camera and the username and password are collected from the terminal prompt.
The function fillRTSP will get the RTSP uri string from the camera.  fillRTSP
will return a non-zero integer in the case of communication error, and the
error message can be found in the last_error field of OnvifData. 
