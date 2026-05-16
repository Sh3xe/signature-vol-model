#include "signatures.hpp"

Signature::Signature()
{
    m_data.push_back(0.0);
}

void Signature::set_data(double a)
{
    m_data[0] = a;
}

double Signature::get_data()
{
    return m_data[0];
}

PYBIND11_MODULE(signature_core_cpp, m, py::mod_gil_not_used())
{
    py::class_<Signature>(m, "Signature")
        .def("set_data", &Signature::set_data)
        .def("get_data", &Signature::get_data)
        .def(py::init<>());
}