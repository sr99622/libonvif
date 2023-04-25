/*******************************************************************************
* onvif.cpp
*
* copyright 2023 Stephen Rhodes
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
#include "onvifboss.h"

namespace py = pybind11;

namespace libonvif
{

PYBIND11_MODULE(libonvif, m)
{
    m.doc() = "pybind11 onvif plugin";
    py::class_<Session>(m, "Session")
        .def(py::init<>())
        .def("active_interface", &Session::active_interface)
        .def("getActiveInterfaces", &Session::getActiveInterfaces);
    py::class_<Manager>(m, "Manager")
        .def(py::init<>())
        .def("startDiscover", &Manager::startDiscover)
        .def("startFill", &Manager::startFill)
        .def("startUpdateVideo", &Manager::startUpdateVideo)
        .def("startUpdateImage", &Manager::startUpdateImage)
        .def("startUpdateNetwork", &Manager::startUpdateNetwork)
        .def("startUpdateTime" ,&Manager::startUpdateTime)
        .def("startReboot", &Manager::startReboot)
        .def("startReset", &Manager::startReset)
        .def("startSetUser", &Manager::startSetUser)
        .def("startSet", &Manager::startSet)
        .def("startSetGotoPreset", &Manager::startSetGotoPreset)
        .def("startMove", &Manager::startMove)
        .def("startStop", &Manager::startStop)
        .def_readwrite("discovered", &Manager::discovered)
        .def_readwrite("getCredential", &Manager::getCredential)
        .def_readwrite("getData", &Manager::getData)
        .def_readwrite("filled", &Manager::filled)
        .def_readwrite("preset", &Manager::preset)
        .def_readwrite("x", &Manager::x)
        .def_readwrite("y", &Manager::y)
        .def_readwrite("z", &Manager::z)
        .def_readwrite("stop_type", &Manager::stop_type)
        .def_readwrite("new_password", &Manager::new_password)
        .def_readwrite("interface", &Manager::interface)
        .def_readwrite("onvif_data", &Manager::onvif_data);
    py::class_<Data>(m, "Data")
        .def(py::init<>())
        .def(py::init<const Data&>())
        .def("xaddrs", &Data::xaddrs)
        .def("stream_uri", &Data::stream_uri)
        .def("serial_number", &Data::serial_number)
        .def("camera_name", &Data::camera_name)
        .def("host", &Data::host)
        .def("last_error", &Data::last_error)
        .def("clearLastError", &Data::clearLastError)
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
        .def("username", &Data::username)
        .def("setUsername", &Data::setUsername)
        .def("password", &Data::password)
        .def("setPassword", &Data::setPassword)
        .def("time_offset", &Data::time_offset)
        .def("setTimeOffset", &Data::setTimeOffset)
        .def("clear", &Data::clear)
        .def(py::self == py::self)
        .def_readwrite("ipAddressChanged", &Data::ipAddressChanged)
        .def_readwrite("alias", &Data::alias)
        .def_readwrite("filled", &Data::filled)
        .def_readwrite("cancelled", &Data::cancelled);

    m.attr("__version__") = "2.0.8";
}


}