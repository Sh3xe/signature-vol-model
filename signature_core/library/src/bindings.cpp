#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#include <memory>
#include <string>
#include <optional>
#include <stdexcept>
#include <cmath>

#include "signatures.hpp"

PYBIND11_MODULE(signature_core_cpp, m)
{
    py::class_<Sig2D>(m, "Sig2D")
        .def(py::init<size_t, cdouble>(),
            "Construct a signature object of order [order] with all values set to [fill_value]",
            py::arg("order") = 1, py::arg("fill_value") = 0.0)
        
        .def("order", &Sig2D::order,
            "Returns the order of the signature (defined as the largest order of non-null tensor of the signature)")

        .def("copy", &Sig2D::copy)

        .def("__str__", &to_string)
        
        .def("get_element", static_cast<cdouble (Sig2D::*)(uint32_t, uint32_t) const>(&Sig2D::get_element),
            "Return the constant associated with the basis defined by [coordinates/order]. For example, in dimension 2, coordinates 1211 will be defined as coordinates=0b0100 and coord_order=4",
            py::arg("coordinates"), py::arg("coord_order"))
        
        .def("get_element", static_cast<cdouble (Sig2D::*)(const coords &) const>(&Sig2D::get_element),
            "Return the constant associated with the basis defined by a list of dimension indices.",
            py::arg("coordinates"))
        
        .def("set_element", static_cast<void (Sig2D::*)(uint32_t, uint32_t, cdouble)>(&Sig2D::set_element),
            "Set the constant associated with the basis defined by bitwise coordinates and order.",
            py::arg("coordinates"), py::arg("coord_order"), py::arg("el"))
        
        .def("set_element", static_cast<void (Sig2D::*)(const coords &, cdouble)>(&Sig2D::set_element),
            "Set the constant associated with the basis defined by a list of dimension indices.",
            py::arg("coordinates"), py::arg("el"))
        
        .def(py::self += py::self)
        .def(py::self -= py::self)
        .def(py::self *= cdouble())
        .def(py::self /= cdouble())
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * cdouble())
        .def(py::self / cdouble())
        .def("__rmul__", [](const Sig2D &self, cdouble left) {
            return left * self;
        });

    // Cache class
    py::class_<ShuffleCache, std::shared_ptr<ShuffleCache>>(m, "ShuffleCache");

    // Operations and functions
    m.def("compute_shuffle_cache", &compute_shuffle_cache,
          "Precomputes and returns a shared pointer cache for shuffle products up to a given truncation order.",
          py::arg("truncation"));

    m.def("shuffle", static_cast<Sig2D (*)(const Sig2D&, const Sig2D&, size_t, const std::shared_ptr<ShuffleCache>&)>(shuffle),
          "Computes the shuffle product of two signatures up to a given truncation order using a precomputed cache.",
          py::arg("left"), py::arg("right"), py::arg("truncation"), py::arg("cache"));

    m.def("shuffle", static_cast<Sig2D (*)(const Sig2D&, const Sig2D&, size_t)>(shuffle),
          "Computes the shuffle product of two signatures up to a given truncation order.",
          py::arg("left"), py::arg("right"), py::arg("truncation"));

    m.def("matmul", &matmul,
          "Computes the tensor multiplication (matrix multiplication equivalent) of two signatures up to a given truncation order.",
          py::arg("left"), py::arg("right"), py::arg("truncation"));

    m.def("bracket", &bracket,
          "Computes the inner product (bracket pairing) between two signatures.",
          py::arg("left"), py::arg("right"));

    m.def("projection_on", &projection_on,
          "Extracts the projection of a signature onto a specific basis element defined by its bitwise coordinates and order.",
          py::arg("sig"), py::arg("coordinates"), py::arg("coord_order"));

    m.def("to_string", &to_string,
        "Converts the signature to a print-able string",
        py::arg("sig"));
}