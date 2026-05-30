#include "utilities.hpp"
#include "signatures.hpp"

#include <sstream>

std::ostream &operator<<(std::ostream &os, const Sig2D &sig)
{
    os << "Sig2D (Max Order: " << sig.order() << "):\n";
    
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

std::string to_string(const Sig2D &sig)
{
    std::stringstream ss;

    ss << sig;

    return ss.str();
}