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
#include "pricing.hpp"

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

    m.def("shuffle", [](const Sig2D& left, const Sig2D& right, size_t truncation, const std::shared_ptr<ShuffleCache>& cache) {
        if (!cache) {
            return static_cast<Sig2D (*)(const Sig2D&, const Sig2D&, size_t)>(shuffle)(left, right, truncation);
        }
        return static_cast<Sig2D (*)(const Sig2D&, const Sig2D&, size_t, const std::shared_ptr<ShuffleCache>&)>(shuffle)(left, right, truncation, cache);
    },
        "Computes the shuffle product of two signatures with an optional precomputed cache.",
        py::arg("left"), 
        py::arg("right"), 
        py::arg("truncation"), 
        py::arg("cache") = py::none()
    );


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

    m.def("european_call_integrand_vr", &european_call_integrand_vr,
        "Returns f(u) = e^{i(u-i/2)k_0 + psi_0} using variance reduction for numerical stability.\n\n"
        "Args:\n"
        "    u (float): Integrand parameter\n"
        "    k_0 (float): log(S_0 / K)\n"
        "    maturity (float): Time to maturity\n"
        "    model_sig (Sig2D): Model signature parameter\n"
        "    model_sig_squared (Sig2D): Precomputed shuffle(model_sig, model_sig)\n"
        "    rho (float): Correlation parameter\n"
        "    r_bs (float): Black-Scholes risk-free rate\n"
        "    vol_bs (float): Black-Scholes volatility\n"
        "    trunc (int): Signature truncation order\n"
        "    rk_subdivs (int): Runge-Kutta subdivisions\n"
        "    upper_bound (float): Numerical stability threshold\n"
        "    cache (ShuffleCache): Precomputed cache instance\n\n"
        "Returns:\n"
        "    float: Real part of the integrand value",
        py::arg("u"),
        py::arg("k_0"),
        py::arg("maturity"),
        py::arg("model_sig"),
        py::arg("model_sig_squared"),
        py::arg("rho"),
        py::arg("r_bs"),
        py::arg("vol_bs"),
        py::arg("trunc"),
        py::arg("rk_subdivs"),
        py::arg("upper_bound"),
        py::arg("cache")
    );
}