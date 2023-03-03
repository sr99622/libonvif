#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "onvifboss.h"

namespace py = pybind11;

namespace onvif
{

PYBIND11_MODULE(onvif, m)
{
    m.doc() = "pybind11 onvif plugin";
    py::class_<Manager>(m, "Manager")
        .def(py::init<>())
        .def("startPyDiscover", &Manager::startPyDiscover)
        .def("startPyFill", &Manager::startPyFill)
        .def("startPySet", &Manager::startPySet)
        .def("startPySetPreset", &Manager::startPySetPreset)
        .def("startPyMove", &Manager::startPyMove)
        .def("startPyStop", &Manager::startPyStop)
        .def_readwrite("discovered", &Manager::discovered)
        .def_readwrite("getCredential", &Manager::getCredential)
        .def_readwrite("getData", &Manager::getData)
        .def_readwrite("filled", &Manager::filled)
        .def_readwrite("preset", &Manager::preset)
        .def_readwrite("x", &Manager::x)
        .def_readwrite("y", &Manager::y)
        .def_readwrite("z", &Manager::z)
        .def_readwrite("stop_type", &Manager::stop_type)
        .def_readwrite("onvif_data", &Manager::onvif_data);
    py::class_<Data>(m, "Data")
        .def(py::init<>())
        .def(py::init<const Data&>())
        .def("xaddrs", &Data::xaddrs)
        .def("stream_uri", &Data::stream_uri)
        .def("serial_number", &Data::serial_number)
        .def("camera_name", &Data::camera_name)
        .def("resolutions_buf", &Data::resolutions_buf)
        .def("width", &Data::width)
        .def("height", &Data::height)
        .def("frame_rate_max", &Data::frame_rate_max)
        .def("frame_rate_min", &Data::frame_rate_min)
        .def("frame_rate", &Data::frame_rate)
        .def("gov_length_max", &Data::gov_length_max)
        .def("gov_length_min", &Data::gov_length_min)
        .def("gov_length", &Data::gov_length)
        .def("bitrate_max", &Data::bitrate_max)
        .def("bitrate_min", &Data::bitrate_min)
        .def("bitrate", &Data::bitrate)
        .def("brightness_max", &Data::brightness_max)
        .def("brightness_min", &Data::brightness_min)
        .def("brightness", &Data::brightness)
        .def("saturation_max", &Data::saturation_max)
        .def("saturation_min", &Data::saturation_min)
        .def("saturation", &Data::saturation)
        .def("contrast_max", &Data::contrast_max)
        .def("contrast_min", &Data::contrast_min)
        .def("contrast", &Data::contrast)
        .def("sharpness_max", &Data::sharpness_max)
        .def("sharpness_min", &Data::sharpness_min)
        .def("sharpness", &Data::sharpness)
        .def("dhcp_enabled", &Data::dhcp_enabled)
        .def("ip_address_buf", &Data::ip_address_buf)
        .def("default_gateway_buf", &Data::default_gateway_buf)
        .def("dns_buf", &Data::dns_buf)
        .def("prefix_length", &Data::prefix_length)
        .def("username", &Data::username)
        .def("setUsername", &Data::setUsername)
        .def("password", &Data::password)
        .def("setPassword", &Data::setPassword);

    m.attr("__version__") = "dev-00";
}


}