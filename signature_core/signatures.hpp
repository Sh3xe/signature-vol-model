#pragma once

#include <cstdint>
#include <vector>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// Assumed dimension: 2
class Signature
{
public:
    Signature();

    void set_data(double a);
    double get_data();

private:
    uint32_t m_order;
    std::vector<double> m_data;
};