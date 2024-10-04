/*******************************************************************************
* onvif.cpp
*
* copyright 2023, 2024 Stephen Rhodes
*
* This library is free software; you can redistribute it and/or
* modify it under the terms of the GNU Lesser General Public
* License as published by the Free Software Foundation; either
* version 2.1 of the License, or (at your option) any later version.
*
* This library is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
* Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public
* License along with this library; if not, write to the Free Software
* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
*
*******************************************************************************/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/functional.h>
#include "onvif_data.h"
#include "session.h"

namespace py = pybind11;

namespace libonvif
{

PYBIND11_MODULE(libonvif, m)
{
    m.doc() = "pybind11 onvif plugin";
    py::class_<Session>(m, "Session")
        .def(py::init<>())
        .def("active_interface", &Session::active_interface)
        .def("getActiveInterfaces", &Session::getActiveInterfaces)
        .def("startDiscover", &Session::startDiscover)
        .def_readwrite("discovered", &Session::discovered)
        .def_readwrite("getCredential", &Session::getCredential)
        .def_readwrite("getData", &Session::getData)
        .def_readwrite("interface", &Session::interface)
        .def_readwrite("abort", &Session::abort);
    
    py::class_<Data>(m, "Data")
        .def(py::init<>())
        .def(py::init<const Data&>())
        .def("xaddrs", &Data::xaddrs)
        .def("setXAddrs", &Data::setXAddrs)
        .def("device_service", &Data::device_service)
        .def("setDeviceService", &Data::setDeviceService)
        .def("event_service", &Data::event_service)
        .def("manual_fill", &Data::manual_fill)
        .def("stream_uri", &Data::stream_uri)
        .def("uri", &Data::uri)
        .def("serial_number", &Data::serial_number)
        .def("camera_name", &Data::camera_name)
        .def("setCameraName", &Data::setCameraName)
        .def("host", &Data::host)
        .def("last_error", &Data::last_error)
        .def("profile", &Data::profile)
        .def("setProfile", &Data::setProfile)
        .def("resolutions_buf", &Data::resolutions_buf)
        .def("width", &Data::width)
        .def("setWidth", &Data::setWidth)
        .def("height", &Data::height)
        .def("setHeight", &Data::setHeight)
        .def("frame_rate_max", &Data::frame_rate_max)
        .def("frame_rate_min", &Data::frame_rate_min)
        .def("frame_rate", &Data::frame_rate)
        .def("setFrameRate", &Data::setFrameRate)
        .def("gov_length_max", &Data::gov_length_max)
        .def("gov_length_min", &Data::gov_length_min)
        .def("gov_length", &Data::gov_length)
        .def("setGovLength", &Data::setGovLength)
        .def("bitrate_max", &Data::bitrate_max)
        .def("bitrate_min", &Data::bitrate_min)
        .def("bitrate", &Data::bitrate)
        .def("setBitrate", &Data::setBitrate)
        .def("brightness_max", &Data::brightness_max)
        .def("brightness_min", &Data::brightness_min)
        .def("brightness", &Data::brightness)
        .def("setBrightness", &Data::setBrightness)
        .def("saturation_max", &Data::saturation_max)
        .def("saturation_min", &Data::saturation_min)
        .def("saturation", &Data::saturation)
        .def("setSaturation", &Data::setSaturation)
        .def("contrast_max", &Data::contrast_max)
        .def("contrast_min", &Data::contrast_min)
        .def("contrast", &Data::contrast)
        .def("setContrast", &Data::setContrast)
        .def("sharpness_max", &Data::sharpness_max)
        .def("sharpness_min", &Data::sharpness_min)
        .def("sharpness", &Data::sharpness)
        .def("setSharpness", &Data::setSharpness)
        .def("encoding", &Data::encoding)
        .def("dhcp_enabled", &Data::dhcp_enabled)
        .def("setDHCPEnabled", &Data::setDHCPEnabled)
        .def("ip_address_buf", &Data::ip_address_buf)
        .def("setIPAddressBuf", &Data::setIPAddressBuf)
        .def("default_gateway_buf", &Data::default_gateway_buf)
        .def("setDefaultGatewayBuf", &Data::setDefaultGatewayBuf)
        .def("dns_buf", &Data::dns_buf)
        .def("setDNSBuf", &Data::setDNSBuf)
        .def("prefix_length", &Data::prefix_length)
        .def("setPrefixLength", &Data::setPrefixLength)
        .def("mask_buf", &Data::mask_buf)
        .def("setMaskBuf", &Data::setMaskBuf)
        .def("audio_encoding", &Data::audio_encoding)
        .def("setAudioEncoding", &Data::setAudioEncoding)
        .def("audio_encoders", &Data::audio_encoders)
        .def("audio_name", &Data::audio_name)
        .def("audio_bitrate", &Data::audio_bitrate)
        .def("setAudioBitrate", &Data::setAudioBitrate)
        .def("audio_bitrates", &Data::audio_bitrates)
        .def("audio_sample_rate", &Data::audio_sample_rate)
        .def("setAudioSampleRate", &Data::setAudioSampleRate)
        .def("audio_sample_rates", &Data::audio_sample_rates)
        .def("audio_session_timeout", &Data::audio_session_timeout)
        .def("audio_multicast_type", &Data::audio_multicast_type)
        .def("audio_multicast_address", &Data::audio_multicast_address)
        .def("audio_use_count", &Data::audio_use_count)
        .def("audio_multicast_port", &Data::audio_multicast_port)
        .def("audio_multicast_TTL", &Data::audio_multicast_TTL)
        .def("audio_multicast_auto_start", &Data::audio_multicast_auto_start)
        .def("username", &Data::username)
        .def("setUsername", &Data::setUsername)
        .def("password", &Data::password)
        .def("setPassword", &Data::setPassword)
        .def("time_offset", &Data::time_offset)
        .def("setTimeOffset", &Data::setTimeOffset)
        .def("timezone", &Data::timezone)
        .def("dst", &Data::dst)
        .def("clear", &Data::clear)
        .def("updateTime" , &Data::updateTime)
        .def("startFill", &Data::startFill)
        .def("startManualFill", &Data::startManualFill)
        .def("startUpdateVideo", &Data::startUpdateVideo)
        .def("startUpdateAudio", &Data::startUpdateAudio)
        .def("startUpdateImage", &Data::startUpdateImage)
        .def("startUpdateNetwork", &Data::startUpdateNetwork)
        .def("startUpdateTime" ,&Data::startUpdateTime)
        .def("startReboot", &Data::startReboot)
        .def("startReset", &Data::startReset)
        .def("startSetUser", &Data::startSetUser)
        .def("startSet", &Data::startSet)
        .def("startSetGotoPreset", &Data::startSetGotoPreset)
        .def("startMove", &Data::startMove)
        .def("startStop", &Data::startStop)
        .def("getDisableVideo", &Data::getDisableVideo)
        .def("setDisableVideo", &Data::setDisableVideo)
        .def("getAnalyzeVideo", &Data::getAnalyzeVideo)
        .def("setAnalyzeVideo", &Data::setAnalyzeVideo)
        .def("getDisableAudio", &Data::getDisableAudio)
        .def("setDisableAudio", &Data::setDisableAudio)
        .def("getAnalyzeAudio", &Data::getAnalyzeAudio)
        .def("setAnalyzeAudio", &Data::setAnalyzeAudio)
        .def("getDesiredAspect", &Data::getDesiredAspect)
        .def("setDesiredAspect", &Data::setDesiredAspect)
        .def("getSyncAudio", &Data::getSyncAudio)
        .def("setSyncAudio", &Data::setSyncAudio)
        .def("getHidden", &Data::getHidden)
        .def("setHidden", &Data::setHidden)
        .def("getCacheMax", &Data::getCacheMax)
        .def("setCacheMax", &Data::setCacheMax)
        .def("toString", &Data::toString)
        .def(py::self == py::self)
        .def_readwrite("profiles", &Data::profiles)
        .def_readwrite("displayProfile", &Data::displayProfile)
        .def_readwrite("filled", &Data::filled)
        .def_readwrite("getData", &Data::getData)
        .def_readwrite("getCredential", &Data::getCredential)
        .def_readwrite("discovered", &Data::discovered)
        .def_readwrite("getSetting", &Data::getSetting)
        .def_readwrite("setSetting", &Data::setSetting)
        .def_readwrite("getProxyURI", &Data::getProxyURI)
        .def_readwrite("errorCallback", &Data::errorCallback)
        .def_readwrite("preset", &Data::preset)
        .def_readwrite("x", &Data::x)
        .def_readwrite("y", &Data::y)
        .def_readwrite("z", &Data::z)
        .def_readwrite("stop_type", &Data::stop_type)
        .def_readwrite("alias", &Data::alias)
        .def_readwrite("cancelled", &Data::cancelled);

    m.attr("__version__") = "3.2.1";
}


}