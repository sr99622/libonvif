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
        .def("startDiscover", &Manager::startDiscover)
        .def("startTest", &Manager::startTest)
        .def_readwrite("finished", &Manager::finished)
        .def_readwrite("quack", &Manager::quack);
    py::class_<Data>(m, "Data")
        .def(py::init<>())
        .def(py::init<const Data&>())
        .def("stream_uri", &Data::stream_uri)
        .def("setStreamUri", &Data::setStreamUri)
        .def("username", &Data::username)
        .def("setUsername", &Data::setUsername)
        .def("password", &Data::password)
        .def("setPassword", &Data::setPassword);

    m.attr("__version__") = "dev-00";
}


}