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
        .def(py::init<size_t, cdouble>(), py::arg("order") = 1, py::arg("fill_value") = 0.0 )

        .def(py::init([](size_t order, py::array_t<cdouble, py::array::c_style | py::array::forcecast> array) {
            py::buffer_info buf = array.request();
            if (buf.ndim != 1) {
                throw std::runtime_error("Sig2D initializer expects a 1D NumPy array.");
            }
            
            // Construct the std::vector directly using iterator bounds over the NumPy memory buffer
            std::vector<cdouble> values(static_cast<cdouble*>(buf.ptr), 
                                        static_cast<cdouble*>(buf.ptr) + buf.size);
            
            return std::make_unique<Sig2D>(order, values);
        }), py::arg("order"), py::arg("array"))
        
        .def("order", &Sig2D::order)
        .def("copy", &Sig2D::copy)
        .def("__str__", &to_string)
        
        .def("get_element", static_cast<cdouble (Sig2D::*)(uint32_t, uint32_t) const>(&Sig2D::get_element), py::arg("coordinates"), py::arg("coord_order"))
        
        .def("get_element", static_cast<cdouble (Sig2D::*)(const coords &) const>(&Sig2D::get_element), py::arg("coordinates"))
        
        .def("set_element", static_cast<void (Sig2D::*)(uint32_t, uint32_t, cdouble)>(&Sig2D::set_element), py::arg("coordinates"), py::arg("coord_order"), py::arg("el"))
        
        .def("set_element", static_cast<void (Sig2D::*)(const coords &, cdouble)>(&Sig2D::set_element), py::arg("coordinates"), py::arg("el"))
        
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
    m.def("compute_shuffle_cache", &compute_shuffle_cache, py::arg("truncation"));

    m.def("shuffle", [](const Sig2D& left, const Sig2D& right, size_t truncation, const std::shared_ptr<ShuffleCache>& cache) {
        if (!cache) {
            return static_cast<Sig2D (*)(const Sig2D&, const Sig2D&, size_t)>(shuffle)(left, right, truncation);
        }
        return static_cast<Sig2D (*)(const Sig2D&, const Sig2D&, size_t, const std::shared_ptr<ShuffleCache>&)>(shuffle)(left, right, truncation, cache);
    }, py::arg("left"), py::arg("right"), py::arg("truncation"), py::arg("cache") = py::none() );

    m.def("matmul", &matmul,
          py::arg("left"), py::arg("right"), py::arg("truncation"));

    m.def("bracket", &bracket,
          py::arg("left"), py::arg("right"));

    m.def("projection_on", &projection_on,
          py::arg("sig"), py::arg("coordinates"), py::arg("coord_order"));

    m.def("to_string", &to_string,
        py::arg("sig"));

    m.def("european_call_integrand_vr", &european_call_integrand_vr,
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

    m.def("european_call_integrand", &european_call_integrand,
        py::arg("u"),
        py::arg("k_0"),
        py::arg("maturity"),
        py::arg("model_sig"),
        py::arg("model_sig_squared"),
        py::arg("rho"),
        py::arg("trunc"),
        py::arg("rk_subdivs"),
        py::arg("upper_bound"),
        py::arg("cache")
    );

    m.def("european_call_sig", &european_call_sig,
        py::arg("initial_price"),
        py::arg("maturity"),
        py::arg("strike"),
        py::arg("model_signature"),
        py::arg("rho"),
        py::arg("trunc"),
        py::arg("rk_subdivs"),
        py::arg("integral_subdivs"),
        py::arg("cache")
    );

    m.def("european_call_sig_vr", &european_call_sig_vr,
        py::arg("initial_price"),
        py::arg("maturity"),
        py::arg("strike"),
        py::arg("model_signature"),
        py::arg("rho"),
        py::arg("trunc"),
        py::arg("rk_subdivs"),
        py::arg("integral_subdivs"),
        py::arg("r_bs"),
        py::arg("vol_bs"),
        py::arg("cache")
    );
}