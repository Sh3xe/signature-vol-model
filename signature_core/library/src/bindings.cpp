#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include "signatures.hpp"

namespace py = pybind11;
using cdouble = std::complex<double>;

PYBIND11_MODULE(signature_core, m) {
    // py::class_<Signature>(m, "Signature")
    //     .def(py::init<size_t, cdouble>(), py::arg("order"), py::arg("fill_value") = cdouble(0, 0))
    //     .def("order", &Signature::order)
    //     .def("get_element", &Signature::get_element)
    //     .def("copy", &Signature::copy)
    //     .def("__iadd__", &Signature::operator+=, py::is_operator())
    //     .def("__imul__", &Signature::operator*=, py::is_operator())
    //     .def("matmul", &Signature::matmul)
    //     .def("shuffle", &Signature::shuffle)
    //     .def("projection_on", &Signature::projection_on);

    // m.def("is_signature", &is_signature);
    // m.def("shuffle", &shuffle);
}