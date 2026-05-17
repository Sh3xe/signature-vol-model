#include "signatures.hpp"
#include <iostream>

using namespace std::complex_literals;

std::ostream &operator<<(std::ostream &os, const Signature &sig)
{
    os << "Signature (Max Order: " << sig.order() << "):\n";
    
    for(size_t i = 0; i <= sig.order(); ++i)
    {
        os << "  Order " << i << ": ";
        
        size_t offset = (1ULL << i) - 1;
        size_t num_elements = (1ULL << i);
        
        // Print all opening brackets for multi-dimensional formatting
        if (i == 0) {
            os << "[ ";
        } else {
            for (size_t b = 0; b < i; ++b) os << "[";
            os << " ";
        }
        
        for(size_t j = 0; j < num_elements; ++j)
        {
            size_t actual_index = offset + j;
            if (actual_index >= sig.m_data.size()) break;
            
            os << sig.m_data[actual_index];
            
            // Format trailing separators and closing/re-opening brackets
            if (j + 1 < num_elements) {
                // Find how many dimensions close at this specific position
                size_t trailing_zeros = 0;
                size_t temp = j + 1;
                while ((temp & 1) == 0) {
                    trailing_zeros++;
                    temp >>= 1;
                }
                
                if (trailing_zeros == 0) {
                    os << ", "; // Same deep row block
                } else {
                    os << " ";
                    for (size_t b = 0; b < trailing_zeros; ++b) os << "]";
                    os << ", ";
                    for (size_t b = 0; b < trailing_zeros; ++b) os << "[";
                    os << " ";
                }
            }
        }
        
        // Print closing brackets for the level
        if (i == 0) {
            os << " ]\n";
        } else {
            os << " ";
            for (size_t b = 0; b < i; ++b) os << "]";
            os << "\n";
        }
    }

    return os;
}

bool check_accessors()
{
    // Tensor product test
    auto sig1 = Signature(3, 0.0);
    sig1.set_element({0, 0}, 1.2);
    sig1.set_element({0, 1}, -6.0);
    sig1.set_element({1, 0}, 0.08);
    sig1.set_element({1, 1}, 16.54);

    auto sig2 = Signature(3, 0.0);
    sig2.set_element({0}, 0.98);
    sig2.set_element({1}, -0.5);

    if( sig2.get_element({0}) != 0.98 || sig2.get_element({1}) != -0.5) {
        std::cout << "element accessors are broken" << std::endl;
        std::cout << sig2.get_element({0}) << " "  << sig2.get_element({1}) << std::endl;

        return false;
    }

    if( 
        sig1.get_element({0, 0}) != 1.2 || sig1.get_element({0, 1}) != -6.0 ||
        sig1.get_element({1, 0}) != 0.08 || sig1.get_element({1, 1}) != 16.54
    ) {
        std::cout << "element accessors are broken" << std::endl;
        std::cout << sig1.get_element({0, 0}) << " " << sig1.get_element({0, 1}) << std::endl;
        std::cout << sig1.get_element({1, 0}) << " " << sig1.get_element({1, 1}) << std::endl;

        return false;
    }

    return true;
}

bool check_matmul()
{
    // Tensor product test
    auto sig1 = Signature(2, 0.0);
    sig1.set_element({0}, 0.0);
    sig1.set_element({1}, 1.0);

    auto sig2 = Signature(2, 0.0);
    sig2.set_element({0}, 1.0);
    sig2.set_element({1}, 0.0);

    auto res = matmul(sig1, sig2, 3);

    if( 
        res.get_element({0, 0}) != 0.0 || res.get_element({0, 1}) != 0.0 ||
        res.get_element({1, 0}) != 1.0 || res.get_element({1, 1}) != 0.0
    ) {
        std::cout << "matmul is broken" << std::endl;
        std::cout << res << std::endl;

        return false;
    }

    sig1.set_element({0, 0}, 1.2);
    sig1.set_element({0, 1}, -6.0);
    sig1.set_element({1, 0}, 0.08);
    sig1.set_element({1, 1}, 16.54);

    sig2.set_element({0}, 0.98);
    sig2.set_element({1}, -5.0);

    res = matmul(sig1, sig2, 3);

    double target_hash = -47.5164;
    double hash = 0.0;
    for(size_t i1 = 0; i1 < 2; ++i1)
    for(size_t i2 = 0; i2 < 2; ++i2)
    for(size_t i3 = 0; i3 < 2; ++i3) {
        hash += res.get_element({i1,i2,i3}).real();
    }

    if( std::abs(hash-target_hash) > 0.01)
    {
        std::cout << "matmul is broken" << std::endl;
        std::cout << res << std::endl;
        return false;
    }

    return true;
}

int main()
{
    if( check_accessors() )
    {
        std::cout << "Element accessors OK" << std::endl;
    }

    if( check_matmul() )
    {
        std::cout << "Matmul OK" << std::endl;
    }


    return 1;
}