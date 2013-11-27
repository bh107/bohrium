/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium team:
http://bohrium.bitbucket.org

Bohrium is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as 
published by the Free Software Foundation, either version 3 
of the License, or (at your option) any later version.

Bohrium is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the 
GNU Lesser General Public License along with Bohrium. 

If not, see <http://www.gnu.org/licenses/>.
*/

//
//  WARN:   This file is generated; changes to it will be overwritten.
//          If you wish to change its functionality then change the code-generator for this file.
//          Take a look at: codegen/README
//

#ifndef __BOHRIUM_BRIDGE_CPP_OPERATORS
#define __BOHRIUM_BRIDGE_CPP_OPERATORS
#include "bh.h"

namespace bh {

#include <sstream>

//
//  Internally defined operator overloads
//


template <typename T>
inline multi_array<T>& multi_array<T>::operator+= (const T& rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_ADD, *this, *this, rhs);
    return *this;
}

template <typename T>
inline multi_array<T>& multi_array<T>::operator+= (multi_array<T>& rhs)
{
    multi_array<T>* input = &rhs;
                                            
    if (this->getRank() < input->getRank()) {           // This would be illogical...
        std::stringstream s;
        s << "BINARY-BUNKERS " << "+=: " << this->getRank() << ", " << input->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!same_shape(*this, *input)) {                   // We need to broadcast
        input = &Runtime::instance().temp_view(rhs);   // Create view pointing to rhs as base
        
        if (!broadcast(rhs, *this, *input)) {
            throw std::runtime_error("Right-handside is not broadcastable.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_ADD, *this, *this, *input);
    return *this;
}

template <typename T>
inline multi_array<T>& multi_array<T>::operator-= (const T& rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, *this, *this, rhs);
    return *this;
}

template <typename T>
inline multi_array<T>& multi_array<T>::operator-= (multi_array<T>& rhs)
{
    multi_array<T>* input = &rhs;
                                            
    if (this->getRank() < input->getRank()) {           // This would be illogical...
        std::stringstream s;
        s << "BINARY-BUNKERS " << "-=: " << this->getRank() << ", " << input->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!same_shape(*this, *input)) {                   // We need to broadcast
        input = &Runtime::instance().temp_view(rhs);   // Create view pointing to rhs as base
        
        if (!broadcast(rhs, *this, *input)) {
            throw std::runtime_error("Right-handside is not broadcastable.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, *this, *this, *input);
    return *this;
}

template <typename T>
inline multi_array<T>& multi_array<T>::operator*= (const T& rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, *this, *this, rhs);
    return *this;
}

template <typename T>
inline multi_array<T>& multi_array<T>::operator*= (multi_array<T>& rhs)
{
    multi_array<T>* input = &rhs;
                                            
    if (this->getRank() < input->getRank()) {           // This would be illogical...
        std::stringstream s;
        s << "BINARY-BUNKERS " << "*=: " << this->getRank() << ", " << input->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!same_shape(*this, *input)) {                   // We need to broadcast
        input = &Runtime::instance().temp_view(rhs);   // Create view pointing to rhs as base
        
        if (!broadcast(rhs, *this, *input)) {
            throw std::runtime_error("Right-handside is not broadcastable.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, *this, *this, *input);
    return *this;
}

template <typename T>
inline multi_array<T>& multi_array<T>::operator/= (const T& rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE, *this, *this, rhs);
    return *this;
}

template <typename T>
inline multi_array<T>& multi_array<T>::operator/= (multi_array<T>& rhs)
{
    multi_array<T>* input = &rhs;
                                            
    if (this->getRank() < input->getRank()) {           // This would be illogical...
        std::stringstream s;
        s << "BINARY-BUNKERS " << "/=: " << this->getRank() << ", " << input->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!same_shape(*this, *input)) {                   // We need to broadcast
        input = &Runtime::instance().temp_view(rhs);   // Create view pointing to rhs as base
        
        if (!broadcast(rhs, *this, *input)) {
            throw std::runtime_error("Right-handside is not broadcastable.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE, *this, *this, *input);
    return *this;
}

template <typename T>
inline multi_array<T>& multi_array<T>::operator%= (const T& rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_MOD, *this, *this, rhs);
    return *this;
}

template <typename T>
inline multi_array<T>& multi_array<T>::operator%= (multi_array<T>& rhs)
{
    multi_array<T>* input = &rhs;
                                            
    if (this->getRank() < input->getRank()) {           // This would be illogical...
        std::stringstream s;
        s << "BINARY-BUNKERS " << "%=: " << this->getRank() << ", " << input->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!same_shape(*this, *input)) {                   // We need to broadcast
        input = &Runtime::instance().temp_view(rhs);   // Create view pointing to rhs as base
        
        if (!broadcast(rhs, *this, *input)) {
            throw std::runtime_error("Right-handside is not broadcastable.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_MOD, *this, *this, *input);
    return *this;
}

template <typename T>
inline multi_array<T>& multi_array<T>::operator&= (const T& rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_AND, *this, *this, rhs);
    return *this;
}

template <typename T>
inline multi_array<T>& multi_array<T>::operator&= (multi_array<T>& rhs)
{
    multi_array<T>* input = &rhs;
                                            
    if (this->getRank() < input->getRank()) {           // This would be illogical...
        std::stringstream s;
        s << "BINARY-BUNKERS " << "&=: " << this->getRank() << ", " << input->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!same_shape(*this, *input)) {                   // We need to broadcast
        input = &Runtime::instance().temp_view(rhs);   // Create view pointing to rhs as base
        
        if (!broadcast(rhs, *this, *input)) {
            throw std::runtime_error("Right-handside is not broadcastable.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_AND, *this, *this, *input);
    return *this;
}

template <typename T>
inline multi_array<T>& multi_array<T>::operator|= (const T& rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_OR, *this, *this, rhs);
    return *this;
}

template <typename T>
inline multi_array<T>& multi_array<T>::operator|= (multi_array<T>& rhs)
{
    multi_array<T>* input = &rhs;
                                            
    if (this->getRank() < input->getRank()) {           // This would be illogical...
        std::stringstream s;
        s << "BINARY-BUNKERS " << "|=: " << this->getRank() << ", " << input->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!same_shape(*this, *input)) {                   // We need to broadcast
        input = &Runtime::instance().temp_view(rhs);   // Create view pointing to rhs as base
        
        if (!broadcast(rhs, *this, *input)) {
            throw std::runtime_error("Right-handside is not broadcastable.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_OR, *this, *this, *input);
    return *this;
}

template <typename T>
inline multi_array<T>& multi_array<T>::operator^= (const T& rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_XOR, *this, *this, rhs);
    return *this;
}

template <typename T>
inline multi_array<T>& multi_array<T>::operator^= (multi_array<T>& rhs)
{
    multi_array<T>* input = &rhs;
                                            
    if (this->getRank() < input->getRank()) {           // This would be illogical...
        std::stringstream s;
        s << "BINARY-BUNKERS " << "^=: " << this->getRank() << ", " << input->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!same_shape(*this, *input)) {                   // We need to broadcast
        input = &Runtime::instance().temp_view(rhs);   // Create view pointing to rhs as base
        
        if (!broadcast(rhs, *this, *input)) {
            throw std::runtime_error("Right-handside is not broadcastable.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_XOR, *this, *this, *input);
    return *this;
}

//
//  Binary operators such as:
//  Mapping "a + b" to BH_ADD(t, a, b)
//  Mapping "a + 1.0" to BH_ADD(t, a, 1.0)
//  Mapping "1.0 + a" to BH_ADD(t, 1.0, a)
//

inline multi_array<int8_t>& operator+ (multi_array<int8_t>& lhs, multi_array<int8_t>& rhs)
{
    multi_array<int8_t>* result;  
    multi_array<int8_t>* left    = &lhs;
    multi_array<int8_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<int8_t, int8_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int8_t, int8_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int8_t, int8_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_ADD, *result, *left, *right);

    return *result;
}

inline multi_array<int8_t> & operator+ (multi_array<int8_t>& lhs, const int8_t& rhs)
{
    multi_array<int8_t>* result = &Runtime::instance().temp<int8_t, int8_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_ADD, *result, lhs, rhs);

    return *result;
}

inline multi_array<int8_t> & operator+ (const int8_t& lhs, multi_array<int8_t>& rhs)
{
    multi_array<int8_t>* result = &Runtime::instance().temp<int8_t, int8_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_ADD, *result, lhs, rhs);

    return *result;
}

inline multi_array<double>& operator+ (multi_array<double>& lhs, multi_array<double>& rhs)
{
    multi_array<double>* result;  
    multi_array<double>* left    = &lhs;
    multi_array<double>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<double, double>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<double, double>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<double, double>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_ADD, *result, *left, *right);

    return *result;
}

inline multi_array<double> & operator+ (multi_array<double>& lhs, const double& rhs)
{
    multi_array<double>* result = &Runtime::instance().temp<double, double>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_ADD, *result, lhs, rhs);

    return *result;
}

inline multi_array<double> & operator+ (const double& lhs, multi_array<double>& rhs)
{
    multi_array<double>* result = &Runtime::instance().temp<double, double>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_ADD, *result, lhs, rhs);

    return *result;
}

inline multi_array<uint16_t>& operator+ (multi_array<uint16_t>& lhs, multi_array<uint16_t>& rhs)
{
    multi_array<uint16_t>* result;  
    multi_array<uint16_t>* left    = &lhs;
    multi_array<uint16_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<uint16_t, uint16_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<uint16_t, uint16_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<uint16_t, uint16_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_ADD, *result, *left, *right);

    return *result;
}

inline multi_array<uint16_t> & operator+ (multi_array<uint16_t>& lhs, const uint16_t& rhs)
{
    multi_array<uint16_t>* result = &Runtime::instance().temp<uint16_t, uint16_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_ADD, *result, lhs, rhs);

    return *result;
}

inline multi_array<uint16_t> & operator+ (const uint16_t& lhs, multi_array<uint16_t>& rhs)
{
    multi_array<uint16_t>* result = &Runtime::instance().temp<uint16_t, uint16_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_ADD, *result, lhs, rhs);

    return *result;
}

inline multi_array<uint64_t>& operator+ (multi_array<uint64_t>& lhs, multi_array<uint64_t>& rhs)
{
    multi_array<uint64_t>* result;  
    multi_array<uint64_t>* left    = &lhs;
    multi_array<uint64_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<uint64_t, uint64_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<uint64_t, uint64_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<uint64_t, uint64_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_ADD, *result, *left, *right);

    return *result;
}

inline multi_array<uint64_t> & operator+ (multi_array<uint64_t>& lhs, const uint64_t& rhs)
{
    multi_array<uint64_t>* result = &Runtime::instance().temp<uint64_t, uint64_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_ADD, *result, lhs, rhs);

    return *result;
}

inline multi_array<uint64_t> & operator+ (const uint64_t& lhs, multi_array<uint64_t>& rhs)
{
    multi_array<uint64_t>* result = &Runtime::instance().temp<uint64_t, uint64_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_ADD, *result, lhs, rhs);

    return *result;
}

inline multi_array<int16_t>& operator+ (multi_array<int16_t>& lhs, multi_array<int16_t>& rhs)
{
    multi_array<int16_t>* result;  
    multi_array<int16_t>* left    = &lhs;
    multi_array<int16_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<int16_t, int16_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int16_t, int16_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int16_t, int16_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_ADD, *result, *left, *right);

    return *result;
}

inline multi_array<int16_t> & operator+ (multi_array<int16_t>& lhs, const int16_t& rhs)
{
    multi_array<int16_t>* result = &Runtime::instance().temp<int16_t, int16_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_ADD, *result, lhs, rhs);

    return *result;
}

inline multi_array<int16_t> & operator+ (const int16_t& lhs, multi_array<int16_t>& rhs)
{
    multi_array<int16_t>* result = &Runtime::instance().temp<int16_t, int16_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_ADD, *result, lhs, rhs);

    return *result;
}

inline multi_array<float>& operator+ (multi_array<float>& lhs, multi_array<float>& rhs)
{
    multi_array<float>* result;  
    multi_array<float>* left    = &lhs;
    multi_array<float>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<float, float>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<float, float>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<float, float>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_ADD, *result, *left, *right);

    return *result;
}

inline multi_array<float> & operator+ (multi_array<float>& lhs, const float& rhs)
{
    multi_array<float>* result = &Runtime::instance().temp<float, float>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_ADD, *result, lhs, rhs);

    return *result;
}

inline multi_array<float> & operator+ (const float& lhs, multi_array<float>& rhs)
{
    multi_array<float>* result = &Runtime::instance().temp<float, float>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_ADD, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator+ (multi_array<unsigned char>& lhs, multi_array<unsigned char>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<unsigned char>* left    = &lhs;
    multi_array<unsigned char>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, unsigned char>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, unsigned char>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, unsigned char>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_ADD, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator+ (multi_array<unsigned char>& lhs, const unsigned char& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, unsigned char>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_ADD, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator+ (const unsigned char& lhs, multi_array<unsigned char>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, unsigned char>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_ADD, *result, lhs, rhs);

    return *result;
}

inline multi_array<int32_t>& operator+ (multi_array<int32_t>& lhs, multi_array<int32_t>& rhs)
{
    multi_array<int32_t>* result;  
    multi_array<int32_t>* left    = &lhs;
    multi_array<int32_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<int32_t, int32_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int32_t, int32_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int32_t, int32_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_ADD, *result, *left, *right);

    return *result;
}

inline multi_array<int32_t> & operator+ (multi_array<int32_t>& lhs, const int32_t& rhs)
{
    multi_array<int32_t>* result = &Runtime::instance().temp<int32_t, int32_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_ADD, *result, lhs, rhs);

    return *result;
}

inline multi_array<int32_t> & operator+ (const int32_t& lhs, multi_array<int32_t>& rhs)
{
    multi_array<int32_t>* result = &Runtime::instance().temp<int32_t, int32_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_ADD, *result, lhs, rhs);

    return *result;
}

inline multi_array<int64_t>& operator+ (multi_array<int64_t>& lhs, multi_array<int64_t>& rhs)
{
    multi_array<int64_t>* result;  
    multi_array<int64_t>* left    = &lhs;
    multi_array<int64_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<int64_t, int64_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int64_t, int64_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int64_t, int64_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_ADD, *result, *left, *right);

    return *result;
}

inline multi_array<int64_t> & operator+ (multi_array<int64_t>& lhs, const int64_t& rhs)
{
    multi_array<int64_t>* result = &Runtime::instance().temp<int64_t, int64_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_ADD, *result, lhs, rhs);

    return *result;
}

inline multi_array<int64_t> & operator+ (const int64_t& lhs, multi_array<int64_t>& rhs)
{
    multi_array<int64_t>* result = &Runtime::instance().temp<int64_t, int64_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_ADD, *result, lhs, rhs);

    return *result;
}

inline multi_array<uint32_t>& operator+ (multi_array<uint32_t>& lhs, multi_array<uint32_t>& rhs)
{
    multi_array<uint32_t>* result;  
    multi_array<uint32_t>* left    = &lhs;
    multi_array<uint32_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<uint32_t, uint32_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<uint32_t, uint32_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<uint32_t, uint32_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_ADD, *result, *left, *right);

    return *result;
}

inline multi_array<uint32_t> & operator+ (multi_array<uint32_t>& lhs, const uint32_t& rhs)
{
    multi_array<uint32_t>* result = &Runtime::instance().temp<uint32_t, uint32_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_ADD, *result, lhs, rhs);

    return *result;
}

inline multi_array<uint32_t> & operator+ (const uint32_t& lhs, multi_array<uint32_t>& rhs)
{
    multi_array<uint32_t>* result = &Runtime::instance().temp<uint32_t, uint32_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_ADD, *result, lhs, rhs);

    return *result;
}

inline multi_array<std::complex<float> >& operator+ (multi_array<std::complex<float> >& lhs, multi_array<std::complex<float> >& rhs)
{
    multi_array<std::complex<float> >* result;  
    multi_array<std::complex<float> >* left    = &lhs;
    multi_array<std::complex<float> >* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<std::complex<float> , std::complex<float> >(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<std::complex<float> , std::complex<float> >(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<std::complex<float> , std::complex<float> >(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_ADD, *result, *left, *right);

    return *result;
}

inline multi_array<std::complex<float> > & operator+ (multi_array<std::complex<float> >& lhs, const std::complex<float> & rhs)
{
    multi_array<std::complex<float> >* result = &Runtime::instance().temp<std::complex<float> , std::complex<float> >(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_ADD, *result, lhs, rhs);

    return *result;
}

inline multi_array<std::complex<float> > & operator+ (const std::complex<float> & lhs, multi_array<std::complex<float> >& rhs)
{
    multi_array<std::complex<float> >* result = &Runtime::instance().temp<std::complex<float> , std::complex<float> >(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_ADD, *result, lhs, rhs);

    return *result;
}

inline multi_array<std::complex<double> >& operator+ (multi_array<std::complex<double> >& lhs, multi_array<std::complex<double> >& rhs)
{
    multi_array<std::complex<double> >* result;  
    multi_array<std::complex<double> >* left    = &lhs;
    multi_array<std::complex<double> >* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<std::complex<double> , std::complex<double> >(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<std::complex<double> , std::complex<double> >(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<std::complex<double> , std::complex<double> >(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_ADD, *result, *left, *right);

    return *result;
}

inline multi_array<std::complex<double> > & operator+ (multi_array<std::complex<double> >& lhs, const std::complex<double> & rhs)
{
    multi_array<std::complex<double> >* result = &Runtime::instance().temp<std::complex<double> , std::complex<double> >(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_ADD, *result, lhs, rhs);

    return *result;
}

inline multi_array<std::complex<double> > & operator+ (const std::complex<double> & lhs, multi_array<std::complex<double> >& rhs)
{
    multi_array<std::complex<double> >* result = &Runtime::instance().temp<std::complex<double> , std::complex<double> >(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_ADD, *result, lhs, rhs);

    return *result;
}

inline multi_array<int8_t>& operator- (multi_array<int8_t>& lhs, multi_array<int8_t>& rhs)
{
    multi_array<int8_t>* result;  
    multi_array<int8_t>* left    = &lhs;
    multi_array<int8_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<int8_t, int8_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int8_t, int8_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int8_t, int8_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, *result, *left, *right);

    return *result;
}

inline multi_array<int8_t> & operator- (multi_array<int8_t>& lhs, const int8_t& rhs)
{
    multi_array<int8_t>* result = &Runtime::instance().temp<int8_t, int8_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, *result, lhs, rhs);

    return *result;
}

inline multi_array<int8_t> & operator- (const int8_t& lhs, multi_array<int8_t>& rhs)
{
    multi_array<int8_t>* result = &Runtime::instance().temp<int8_t, int8_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, *result, lhs, rhs);

    return *result;
}

inline multi_array<double>& operator- (multi_array<double>& lhs, multi_array<double>& rhs)
{
    multi_array<double>* result;  
    multi_array<double>* left    = &lhs;
    multi_array<double>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<double, double>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<double, double>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<double, double>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, *result, *left, *right);

    return *result;
}

inline multi_array<double> & operator- (multi_array<double>& lhs, const double& rhs)
{
    multi_array<double>* result = &Runtime::instance().temp<double, double>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, *result, lhs, rhs);

    return *result;
}

inline multi_array<double> & operator- (const double& lhs, multi_array<double>& rhs)
{
    multi_array<double>* result = &Runtime::instance().temp<double, double>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, *result, lhs, rhs);

    return *result;
}

inline multi_array<uint16_t>& operator- (multi_array<uint16_t>& lhs, multi_array<uint16_t>& rhs)
{
    multi_array<uint16_t>* result;  
    multi_array<uint16_t>* left    = &lhs;
    multi_array<uint16_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<uint16_t, uint16_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<uint16_t, uint16_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<uint16_t, uint16_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, *result, *left, *right);

    return *result;
}

inline multi_array<uint16_t> & operator- (multi_array<uint16_t>& lhs, const uint16_t& rhs)
{
    multi_array<uint16_t>* result = &Runtime::instance().temp<uint16_t, uint16_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, *result, lhs, rhs);

    return *result;
}

inline multi_array<uint16_t> & operator- (const uint16_t& lhs, multi_array<uint16_t>& rhs)
{
    multi_array<uint16_t>* result = &Runtime::instance().temp<uint16_t, uint16_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, *result, lhs, rhs);

    return *result;
}

inline multi_array<uint64_t>& operator- (multi_array<uint64_t>& lhs, multi_array<uint64_t>& rhs)
{
    multi_array<uint64_t>* result;  
    multi_array<uint64_t>* left    = &lhs;
    multi_array<uint64_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<uint64_t, uint64_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<uint64_t, uint64_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<uint64_t, uint64_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, *result, *left, *right);

    return *result;
}

inline multi_array<uint64_t> & operator- (multi_array<uint64_t>& lhs, const uint64_t& rhs)
{
    multi_array<uint64_t>* result = &Runtime::instance().temp<uint64_t, uint64_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, *result, lhs, rhs);

    return *result;
}

inline multi_array<uint64_t> & operator- (const uint64_t& lhs, multi_array<uint64_t>& rhs)
{
    multi_array<uint64_t>* result = &Runtime::instance().temp<uint64_t, uint64_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, *result, lhs, rhs);

    return *result;
}

inline multi_array<int16_t>& operator- (multi_array<int16_t>& lhs, multi_array<int16_t>& rhs)
{
    multi_array<int16_t>* result;  
    multi_array<int16_t>* left    = &lhs;
    multi_array<int16_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<int16_t, int16_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int16_t, int16_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int16_t, int16_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, *result, *left, *right);

    return *result;
}

inline multi_array<int16_t> & operator- (multi_array<int16_t>& lhs, const int16_t& rhs)
{
    multi_array<int16_t>* result = &Runtime::instance().temp<int16_t, int16_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, *result, lhs, rhs);

    return *result;
}

inline multi_array<int16_t> & operator- (const int16_t& lhs, multi_array<int16_t>& rhs)
{
    multi_array<int16_t>* result = &Runtime::instance().temp<int16_t, int16_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, *result, lhs, rhs);

    return *result;
}

inline multi_array<float>& operator- (multi_array<float>& lhs, multi_array<float>& rhs)
{
    multi_array<float>* result;  
    multi_array<float>* left    = &lhs;
    multi_array<float>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<float, float>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<float, float>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<float, float>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, *result, *left, *right);

    return *result;
}

inline multi_array<float> & operator- (multi_array<float>& lhs, const float& rhs)
{
    multi_array<float>* result = &Runtime::instance().temp<float, float>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, *result, lhs, rhs);

    return *result;
}

inline multi_array<float> & operator- (const float& lhs, multi_array<float>& rhs)
{
    multi_array<float>* result = &Runtime::instance().temp<float, float>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator- (multi_array<unsigned char>& lhs, multi_array<unsigned char>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<unsigned char>* left    = &lhs;
    multi_array<unsigned char>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, unsigned char>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, unsigned char>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, unsigned char>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator- (multi_array<unsigned char>& lhs, const unsigned char& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, unsigned char>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator- (const unsigned char& lhs, multi_array<unsigned char>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, unsigned char>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, *result, lhs, rhs);

    return *result;
}

inline multi_array<int32_t>& operator- (multi_array<int32_t>& lhs, multi_array<int32_t>& rhs)
{
    multi_array<int32_t>* result;  
    multi_array<int32_t>* left    = &lhs;
    multi_array<int32_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<int32_t, int32_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int32_t, int32_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int32_t, int32_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, *result, *left, *right);

    return *result;
}

inline multi_array<int32_t> & operator- (multi_array<int32_t>& lhs, const int32_t& rhs)
{
    multi_array<int32_t>* result = &Runtime::instance().temp<int32_t, int32_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, *result, lhs, rhs);

    return *result;
}

inline multi_array<int32_t> & operator- (const int32_t& lhs, multi_array<int32_t>& rhs)
{
    multi_array<int32_t>* result = &Runtime::instance().temp<int32_t, int32_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, *result, lhs, rhs);

    return *result;
}

inline multi_array<int64_t>& operator- (multi_array<int64_t>& lhs, multi_array<int64_t>& rhs)
{
    multi_array<int64_t>* result;  
    multi_array<int64_t>* left    = &lhs;
    multi_array<int64_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<int64_t, int64_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int64_t, int64_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int64_t, int64_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, *result, *left, *right);

    return *result;
}

inline multi_array<int64_t> & operator- (multi_array<int64_t>& lhs, const int64_t& rhs)
{
    multi_array<int64_t>* result = &Runtime::instance().temp<int64_t, int64_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, *result, lhs, rhs);

    return *result;
}

inline multi_array<int64_t> & operator- (const int64_t& lhs, multi_array<int64_t>& rhs)
{
    multi_array<int64_t>* result = &Runtime::instance().temp<int64_t, int64_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, *result, lhs, rhs);

    return *result;
}

inline multi_array<uint32_t>& operator- (multi_array<uint32_t>& lhs, multi_array<uint32_t>& rhs)
{
    multi_array<uint32_t>* result;  
    multi_array<uint32_t>* left    = &lhs;
    multi_array<uint32_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<uint32_t, uint32_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<uint32_t, uint32_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<uint32_t, uint32_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, *result, *left, *right);

    return *result;
}

inline multi_array<uint32_t> & operator- (multi_array<uint32_t>& lhs, const uint32_t& rhs)
{
    multi_array<uint32_t>* result = &Runtime::instance().temp<uint32_t, uint32_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, *result, lhs, rhs);

    return *result;
}

inline multi_array<uint32_t> & operator- (const uint32_t& lhs, multi_array<uint32_t>& rhs)
{
    multi_array<uint32_t>* result = &Runtime::instance().temp<uint32_t, uint32_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, *result, lhs, rhs);

    return *result;
}

inline multi_array<std::complex<float> >& operator- (multi_array<std::complex<float> >& lhs, multi_array<std::complex<float> >& rhs)
{
    multi_array<std::complex<float> >* result;  
    multi_array<std::complex<float> >* left    = &lhs;
    multi_array<std::complex<float> >* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<std::complex<float> , std::complex<float> >(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<std::complex<float> , std::complex<float> >(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<std::complex<float> , std::complex<float> >(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, *result, *left, *right);

    return *result;
}

inline multi_array<std::complex<float> > & operator- (multi_array<std::complex<float> >& lhs, const std::complex<float> & rhs)
{
    multi_array<std::complex<float> >* result = &Runtime::instance().temp<std::complex<float> , std::complex<float> >(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, *result, lhs, rhs);

    return *result;
}

inline multi_array<std::complex<float> > & operator- (const std::complex<float> & lhs, multi_array<std::complex<float> >& rhs)
{
    multi_array<std::complex<float> >* result = &Runtime::instance().temp<std::complex<float> , std::complex<float> >(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, *result, lhs, rhs);

    return *result;
}

inline multi_array<std::complex<double> >& operator- (multi_array<std::complex<double> >& lhs, multi_array<std::complex<double> >& rhs)
{
    multi_array<std::complex<double> >* result;  
    multi_array<std::complex<double> >* left    = &lhs;
    multi_array<std::complex<double> >* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<std::complex<double> , std::complex<double> >(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<std::complex<double> , std::complex<double> >(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<std::complex<double> , std::complex<double> >(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, *result, *left, *right);

    return *result;
}

inline multi_array<std::complex<double> > & operator- (multi_array<std::complex<double> >& lhs, const std::complex<double> & rhs)
{
    multi_array<std::complex<double> >* result = &Runtime::instance().temp<std::complex<double> , std::complex<double> >(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, *result, lhs, rhs);

    return *result;
}

inline multi_array<std::complex<double> > & operator- (const std::complex<double> & lhs, multi_array<std::complex<double> >& rhs)
{
    multi_array<std::complex<double> >* result = &Runtime::instance().temp<std::complex<double> , std::complex<double> >(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, *result, lhs, rhs);

    return *result;
}

inline multi_array<int8_t>& operator* (multi_array<int8_t>& lhs, multi_array<int8_t>& rhs)
{
    multi_array<int8_t>* result;  
    multi_array<int8_t>* left    = &lhs;
    multi_array<int8_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<int8_t, int8_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int8_t, int8_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int8_t, int8_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, *result, *left, *right);

    return *result;
}

inline multi_array<int8_t> & operator* (multi_array<int8_t>& lhs, const int8_t& rhs)
{
    multi_array<int8_t>* result = &Runtime::instance().temp<int8_t, int8_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, *result, lhs, rhs);

    return *result;
}

inline multi_array<int8_t> & operator* (const int8_t& lhs, multi_array<int8_t>& rhs)
{
    multi_array<int8_t>* result = &Runtime::instance().temp<int8_t, int8_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, *result, lhs, rhs);

    return *result;
}

inline multi_array<double>& operator* (multi_array<double>& lhs, multi_array<double>& rhs)
{
    multi_array<double>* result;  
    multi_array<double>* left    = &lhs;
    multi_array<double>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<double, double>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<double, double>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<double, double>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, *result, *left, *right);

    return *result;
}

inline multi_array<double> & operator* (multi_array<double>& lhs, const double& rhs)
{
    multi_array<double>* result = &Runtime::instance().temp<double, double>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, *result, lhs, rhs);

    return *result;
}

inline multi_array<double> & operator* (const double& lhs, multi_array<double>& rhs)
{
    multi_array<double>* result = &Runtime::instance().temp<double, double>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, *result, lhs, rhs);

    return *result;
}

inline multi_array<uint16_t>& operator* (multi_array<uint16_t>& lhs, multi_array<uint16_t>& rhs)
{
    multi_array<uint16_t>* result;  
    multi_array<uint16_t>* left    = &lhs;
    multi_array<uint16_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<uint16_t, uint16_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<uint16_t, uint16_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<uint16_t, uint16_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, *result, *left, *right);

    return *result;
}

inline multi_array<uint16_t> & operator* (multi_array<uint16_t>& lhs, const uint16_t& rhs)
{
    multi_array<uint16_t>* result = &Runtime::instance().temp<uint16_t, uint16_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, *result, lhs, rhs);

    return *result;
}

inline multi_array<uint16_t> & operator* (const uint16_t& lhs, multi_array<uint16_t>& rhs)
{
    multi_array<uint16_t>* result = &Runtime::instance().temp<uint16_t, uint16_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, *result, lhs, rhs);

    return *result;
}

inline multi_array<uint64_t>& operator* (multi_array<uint64_t>& lhs, multi_array<uint64_t>& rhs)
{
    multi_array<uint64_t>* result;  
    multi_array<uint64_t>* left    = &lhs;
    multi_array<uint64_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<uint64_t, uint64_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<uint64_t, uint64_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<uint64_t, uint64_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, *result, *left, *right);

    return *result;
}

inline multi_array<uint64_t> & operator* (multi_array<uint64_t>& lhs, const uint64_t& rhs)
{
    multi_array<uint64_t>* result = &Runtime::instance().temp<uint64_t, uint64_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, *result, lhs, rhs);

    return *result;
}

inline multi_array<uint64_t> & operator* (const uint64_t& lhs, multi_array<uint64_t>& rhs)
{
    multi_array<uint64_t>* result = &Runtime::instance().temp<uint64_t, uint64_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, *result, lhs, rhs);

    return *result;
}

inline multi_array<int16_t>& operator* (multi_array<int16_t>& lhs, multi_array<int16_t>& rhs)
{
    multi_array<int16_t>* result;  
    multi_array<int16_t>* left    = &lhs;
    multi_array<int16_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<int16_t, int16_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int16_t, int16_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int16_t, int16_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, *result, *left, *right);

    return *result;
}

inline multi_array<int16_t> & operator* (multi_array<int16_t>& lhs, const int16_t& rhs)
{
    multi_array<int16_t>* result = &Runtime::instance().temp<int16_t, int16_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, *result, lhs, rhs);

    return *result;
}

inline multi_array<int16_t> & operator* (const int16_t& lhs, multi_array<int16_t>& rhs)
{
    multi_array<int16_t>* result = &Runtime::instance().temp<int16_t, int16_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, *result, lhs, rhs);

    return *result;
}

inline multi_array<float>& operator* (multi_array<float>& lhs, multi_array<float>& rhs)
{
    multi_array<float>* result;  
    multi_array<float>* left    = &lhs;
    multi_array<float>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<float, float>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<float, float>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<float, float>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, *result, *left, *right);

    return *result;
}

inline multi_array<float> & operator* (multi_array<float>& lhs, const float& rhs)
{
    multi_array<float>* result = &Runtime::instance().temp<float, float>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, *result, lhs, rhs);

    return *result;
}

inline multi_array<float> & operator* (const float& lhs, multi_array<float>& rhs)
{
    multi_array<float>* result = &Runtime::instance().temp<float, float>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator* (multi_array<unsigned char>& lhs, multi_array<unsigned char>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<unsigned char>* left    = &lhs;
    multi_array<unsigned char>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, unsigned char>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, unsigned char>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, unsigned char>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator* (multi_array<unsigned char>& lhs, const unsigned char& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, unsigned char>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator* (const unsigned char& lhs, multi_array<unsigned char>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, unsigned char>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, *result, lhs, rhs);

    return *result;
}

inline multi_array<int32_t>& operator* (multi_array<int32_t>& lhs, multi_array<int32_t>& rhs)
{
    multi_array<int32_t>* result;  
    multi_array<int32_t>* left    = &lhs;
    multi_array<int32_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<int32_t, int32_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int32_t, int32_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int32_t, int32_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, *result, *left, *right);

    return *result;
}

inline multi_array<int32_t> & operator* (multi_array<int32_t>& lhs, const int32_t& rhs)
{
    multi_array<int32_t>* result = &Runtime::instance().temp<int32_t, int32_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, *result, lhs, rhs);

    return *result;
}

inline multi_array<int32_t> & operator* (const int32_t& lhs, multi_array<int32_t>& rhs)
{
    multi_array<int32_t>* result = &Runtime::instance().temp<int32_t, int32_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, *result, lhs, rhs);

    return *result;
}

inline multi_array<int64_t>& operator* (multi_array<int64_t>& lhs, multi_array<int64_t>& rhs)
{
    multi_array<int64_t>* result;  
    multi_array<int64_t>* left    = &lhs;
    multi_array<int64_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<int64_t, int64_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int64_t, int64_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int64_t, int64_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, *result, *left, *right);

    return *result;
}

inline multi_array<int64_t> & operator* (multi_array<int64_t>& lhs, const int64_t& rhs)
{
    multi_array<int64_t>* result = &Runtime::instance().temp<int64_t, int64_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, *result, lhs, rhs);

    return *result;
}

inline multi_array<int64_t> & operator* (const int64_t& lhs, multi_array<int64_t>& rhs)
{
    multi_array<int64_t>* result = &Runtime::instance().temp<int64_t, int64_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, *result, lhs, rhs);

    return *result;
}

inline multi_array<uint32_t>& operator* (multi_array<uint32_t>& lhs, multi_array<uint32_t>& rhs)
{
    multi_array<uint32_t>* result;  
    multi_array<uint32_t>* left    = &lhs;
    multi_array<uint32_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<uint32_t, uint32_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<uint32_t, uint32_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<uint32_t, uint32_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, *result, *left, *right);

    return *result;
}

inline multi_array<uint32_t> & operator* (multi_array<uint32_t>& lhs, const uint32_t& rhs)
{
    multi_array<uint32_t>* result = &Runtime::instance().temp<uint32_t, uint32_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, *result, lhs, rhs);

    return *result;
}

inline multi_array<uint32_t> & operator* (const uint32_t& lhs, multi_array<uint32_t>& rhs)
{
    multi_array<uint32_t>* result = &Runtime::instance().temp<uint32_t, uint32_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, *result, lhs, rhs);

    return *result;
}

inline multi_array<std::complex<float> >& operator* (multi_array<std::complex<float> >& lhs, multi_array<std::complex<float> >& rhs)
{
    multi_array<std::complex<float> >* result;  
    multi_array<std::complex<float> >* left    = &lhs;
    multi_array<std::complex<float> >* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<std::complex<float> , std::complex<float> >(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<std::complex<float> , std::complex<float> >(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<std::complex<float> , std::complex<float> >(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, *result, *left, *right);

    return *result;
}

inline multi_array<std::complex<float> > & operator* (multi_array<std::complex<float> >& lhs, const std::complex<float> & rhs)
{
    multi_array<std::complex<float> >* result = &Runtime::instance().temp<std::complex<float> , std::complex<float> >(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, *result, lhs, rhs);

    return *result;
}

inline multi_array<std::complex<float> > & operator* (const std::complex<float> & lhs, multi_array<std::complex<float> >& rhs)
{
    multi_array<std::complex<float> >* result = &Runtime::instance().temp<std::complex<float> , std::complex<float> >(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, *result, lhs, rhs);

    return *result;
}

inline multi_array<std::complex<double> >& operator* (multi_array<std::complex<double> >& lhs, multi_array<std::complex<double> >& rhs)
{
    multi_array<std::complex<double> >* result;  
    multi_array<std::complex<double> >* left    = &lhs;
    multi_array<std::complex<double> >* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<std::complex<double> , std::complex<double> >(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<std::complex<double> , std::complex<double> >(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<std::complex<double> , std::complex<double> >(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, *result, *left, *right);

    return *result;
}

inline multi_array<std::complex<double> > & operator* (multi_array<std::complex<double> >& lhs, const std::complex<double> & rhs)
{
    multi_array<std::complex<double> >* result = &Runtime::instance().temp<std::complex<double> , std::complex<double> >(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, *result, lhs, rhs);

    return *result;
}

inline multi_array<std::complex<double> > & operator* (const std::complex<double> & lhs, multi_array<std::complex<double> >& rhs)
{
    multi_array<std::complex<double> >* result = &Runtime::instance().temp<std::complex<double> , std::complex<double> >(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, *result, lhs, rhs);

    return *result;
}

inline multi_array<int8_t>& operator/ (multi_array<int8_t>& lhs, multi_array<int8_t>& rhs)
{
    multi_array<int8_t>* result;  
    multi_array<int8_t>* left    = &lhs;
    multi_array<int8_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<int8_t, int8_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int8_t, int8_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int8_t, int8_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE, *result, *left, *right);

    return *result;
}

inline multi_array<int8_t> & operator/ (multi_array<int8_t>& lhs, const int8_t& rhs)
{
    multi_array<int8_t>* result = &Runtime::instance().temp<int8_t, int8_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE, *result, lhs, rhs);

    return *result;
}

inline multi_array<int8_t> & operator/ (const int8_t& lhs, multi_array<int8_t>& rhs)
{
    multi_array<int8_t>* result = &Runtime::instance().temp<int8_t, int8_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE, *result, lhs, rhs);

    return *result;
}

inline multi_array<double>& operator/ (multi_array<double>& lhs, multi_array<double>& rhs)
{
    multi_array<double>* result;  
    multi_array<double>* left    = &lhs;
    multi_array<double>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<double, double>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<double, double>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<double, double>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE, *result, *left, *right);

    return *result;
}

inline multi_array<double> & operator/ (multi_array<double>& lhs, const double& rhs)
{
    multi_array<double>* result = &Runtime::instance().temp<double, double>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE, *result, lhs, rhs);

    return *result;
}

inline multi_array<double> & operator/ (const double& lhs, multi_array<double>& rhs)
{
    multi_array<double>* result = &Runtime::instance().temp<double, double>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE, *result, lhs, rhs);

    return *result;
}

inline multi_array<uint16_t>& operator/ (multi_array<uint16_t>& lhs, multi_array<uint16_t>& rhs)
{
    multi_array<uint16_t>* result;  
    multi_array<uint16_t>* left    = &lhs;
    multi_array<uint16_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<uint16_t, uint16_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<uint16_t, uint16_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<uint16_t, uint16_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE, *result, *left, *right);

    return *result;
}

inline multi_array<uint16_t> & operator/ (multi_array<uint16_t>& lhs, const uint16_t& rhs)
{
    multi_array<uint16_t>* result = &Runtime::instance().temp<uint16_t, uint16_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE, *result, lhs, rhs);

    return *result;
}

inline multi_array<uint16_t> & operator/ (const uint16_t& lhs, multi_array<uint16_t>& rhs)
{
    multi_array<uint16_t>* result = &Runtime::instance().temp<uint16_t, uint16_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE, *result, lhs, rhs);

    return *result;
}

inline multi_array<uint64_t>& operator/ (multi_array<uint64_t>& lhs, multi_array<uint64_t>& rhs)
{
    multi_array<uint64_t>* result;  
    multi_array<uint64_t>* left    = &lhs;
    multi_array<uint64_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<uint64_t, uint64_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<uint64_t, uint64_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<uint64_t, uint64_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE, *result, *left, *right);

    return *result;
}

inline multi_array<uint64_t> & operator/ (multi_array<uint64_t>& lhs, const uint64_t& rhs)
{
    multi_array<uint64_t>* result = &Runtime::instance().temp<uint64_t, uint64_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE, *result, lhs, rhs);

    return *result;
}

inline multi_array<uint64_t> & operator/ (const uint64_t& lhs, multi_array<uint64_t>& rhs)
{
    multi_array<uint64_t>* result = &Runtime::instance().temp<uint64_t, uint64_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE, *result, lhs, rhs);

    return *result;
}

inline multi_array<int16_t>& operator/ (multi_array<int16_t>& lhs, multi_array<int16_t>& rhs)
{
    multi_array<int16_t>* result;  
    multi_array<int16_t>* left    = &lhs;
    multi_array<int16_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<int16_t, int16_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int16_t, int16_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int16_t, int16_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE, *result, *left, *right);

    return *result;
}

inline multi_array<int16_t> & operator/ (multi_array<int16_t>& lhs, const int16_t& rhs)
{
    multi_array<int16_t>* result = &Runtime::instance().temp<int16_t, int16_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE, *result, lhs, rhs);

    return *result;
}

inline multi_array<int16_t> & operator/ (const int16_t& lhs, multi_array<int16_t>& rhs)
{
    multi_array<int16_t>* result = &Runtime::instance().temp<int16_t, int16_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE, *result, lhs, rhs);

    return *result;
}

inline multi_array<float>& operator/ (multi_array<float>& lhs, multi_array<float>& rhs)
{
    multi_array<float>* result;  
    multi_array<float>* left    = &lhs;
    multi_array<float>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<float, float>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<float, float>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<float, float>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE, *result, *left, *right);

    return *result;
}

inline multi_array<float> & operator/ (multi_array<float>& lhs, const float& rhs)
{
    multi_array<float>* result = &Runtime::instance().temp<float, float>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE, *result, lhs, rhs);

    return *result;
}

inline multi_array<float> & operator/ (const float& lhs, multi_array<float>& rhs)
{
    multi_array<float>* result = &Runtime::instance().temp<float, float>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE, *result, lhs, rhs);

    return *result;
}

inline multi_array<int32_t>& operator/ (multi_array<int32_t>& lhs, multi_array<int32_t>& rhs)
{
    multi_array<int32_t>* result;  
    multi_array<int32_t>* left    = &lhs;
    multi_array<int32_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<int32_t, int32_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int32_t, int32_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int32_t, int32_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE, *result, *left, *right);

    return *result;
}

inline multi_array<int32_t> & operator/ (multi_array<int32_t>& lhs, const int32_t& rhs)
{
    multi_array<int32_t>* result = &Runtime::instance().temp<int32_t, int32_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE, *result, lhs, rhs);

    return *result;
}

inline multi_array<int32_t> & operator/ (const int32_t& lhs, multi_array<int32_t>& rhs)
{
    multi_array<int32_t>* result = &Runtime::instance().temp<int32_t, int32_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE, *result, lhs, rhs);

    return *result;
}

inline multi_array<int64_t>& operator/ (multi_array<int64_t>& lhs, multi_array<int64_t>& rhs)
{
    multi_array<int64_t>* result;  
    multi_array<int64_t>* left    = &lhs;
    multi_array<int64_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<int64_t, int64_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int64_t, int64_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int64_t, int64_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE, *result, *left, *right);

    return *result;
}

inline multi_array<int64_t> & operator/ (multi_array<int64_t>& lhs, const int64_t& rhs)
{
    multi_array<int64_t>* result = &Runtime::instance().temp<int64_t, int64_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE, *result, lhs, rhs);

    return *result;
}

inline multi_array<int64_t> & operator/ (const int64_t& lhs, multi_array<int64_t>& rhs)
{
    multi_array<int64_t>* result = &Runtime::instance().temp<int64_t, int64_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE, *result, lhs, rhs);

    return *result;
}

inline multi_array<uint32_t>& operator/ (multi_array<uint32_t>& lhs, multi_array<uint32_t>& rhs)
{
    multi_array<uint32_t>* result;  
    multi_array<uint32_t>* left    = &lhs;
    multi_array<uint32_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<uint32_t, uint32_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<uint32_t, uint32_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<uint32_t, uint32_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE, *result, *left, *right);

    return *result;
}

inline multi_array<uint32_t> & operator/ (multi_array<uint32_t>& lhs, const uint32_t& rhs)
{
    multi_array<uint32_t>* result = &Runtime::instance().temp<uint32_t, uint32_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE, *result, lhs, rhs);

    return *result;
}

inline multi_array<uint32_t> & operator/ (const uint32_t& lhs, multi_array<uint32_t>& rhs)
{
    multi_array<uint32_t>* result = &Runtime::instance().temp<uint32_t, uint32_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE, *result, lhs, rhs);

    return *result;
}

inline multi_array<std::complex<float> >& operator/ (multi_array<std::complex<float> >& lhs, multi_array<std::complex<float> >& rhs)
{
    multi_array<std::complex<float> >* result;  
    multi_array<std::complex<float> >* left    = &lhs;
    multi_array<std::complex<float> >* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<std::complex<float> , std::complex<float> >(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<std::complex<float> , std::complex<float> >(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<std::complex<float> , std::complex<float> >(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE, *result, *left, *right);

    return *result;
}

inline multi_array<std::complex<float> > & operator/ (multi_array<std::complex<float> >& lhs, const std::complex<float> & rhs)
{
    multi_array<std::complex<float> >* result = &Runtime::instance().temp<std::complex<float> , std::complex<float> >(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE, *result, lhs, rhs);

    return *result;
}

inline multi_array<std::complex<float> > & operator/ (const std::complex<float> & lhs, multi_array<std::complex<float> >& rhs)
{
    multi_array<std::complex<float> >* result = &Runtime::instance().temp<std::complex<float> , std::complex<float> >(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE, *result, lhs, rhs);

    return *result;
}

inline multi_array<std::complex<double> >& operator/ (multi_array<std::complex<double> >& lhs, multi_array<std::complex<double> >& rhs)
{
    multi_array<std::complex<double> >* result;  
    multi_array<std::complex<double> >* left    = &lhs;
    multi_array<std::complex<double> >* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<std::complex<double> , std::complex<double> >(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<std::complex<double> , std::complex<double> >(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<std::complex<double> , std::complex<double> >(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE, *result, *left, *right);

    return *result;
}

inline multi_array<std::complex<double> > & operator/ (multi_array<std::complex<double> >& lhs, const std::complex<double> & rhs)
{
    multi_array<std::complex<double> >* result = &Runtime::instance().temp<std::complex<double> , std::complex<double> >(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE, *result, lhs, rhs);

    return *result;
}

inline multi_array<std::complex<double> > & operator/ (const std::complex<double> & lhs, multi_array<std::complex<double> >& rhs)
{
    multi_array<std::complex<double> >* result = &Runtime::instance().temp<std::complex<double> , std::complex<double> >(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE, *result, lhs, rhs);

    return *result;
}

inline multi_array<int8_t>& operator% (multi_array<int8_t>& lhs, multi_array<int8_t>& rhs)
{
    multi_array<int8_t>* result;  
    multi_array<int8_t>* left    = &lhs;
    multi_array<int8_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<int8_t, int8_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int8_t, int8_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int8_t, int8_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MOD, *result, *left, *right);

    return *result;
}

inline multi_array<int8_t> & operator% (multi_array<int8_t>& lhs, const int8_t& rhs)
{
    multi_array<int8_t>* result = &Runtime::instance().temp<int8_t, int8_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MOD, *result, lhs, rhs);

    return *result;
}

inline multi_array<int8_t> & operator% (const int8_t& lhs, multi_array<int8_t>& rhs)
{
    multi_array<int8_t>* result = &Runtime::instance().temp<int8_t, int8_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MOD, *result, lhs, rhs);

    return *result;
}

inline multi_array<double>& operator% (multi_array<double>& lhs, multi_array<double>& rhs)
{
    multi_array<double>* result;  
    multi_array<double>* left    = &lhs;
    multi_array<double>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<double, double>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<double, double>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<double, double>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MOD, *result, *left, *right);

    return *result;
}

inline multi_array<double> & operator% (multi_array<double>& lhs, const double& rhs)
{
    multi_array<double>* result = &Runtime::instance().temp<double, double>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MOD, *result, lhs, rhs);

    return *result;
}

inline multi_array<double> & operator% (const double& lhs, multi_array<double>& rhs)
{
    multi_array<double>* result = &Runtime::instance().temp<double, double>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MOD, *result, lhs, rhs);

    return *result;
}

inline multi_array<uint16_t>& operator% (multi_array<uint16_t>& lhs, multi_array<uint16_t>& rhs)
{
    multi_array<uint16_t>* result;  
    multi_array<uint16_t>* left    = &lhs;
    multi_array<uint16_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<uint16_t, uint16_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<uint16_t, uint16_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<uint16_t, uint16_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MOD, *result, *left, *right);

    return *result;
}

inline multi_array<uint16_t> & operator% (multi_array<uint16_t>& lhs, const uint16_t& rhs)
{
    multi_array<uint16_t>* result = &Runtime::instance().temp<uint16_t, uint16_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MOD, *result, lhs, rhs);

    return *result;
}

inline multi_array<uint16_t> & operator% (const uint16_t& lhs, multi_array<uint16_t>& rhs)
{
    multi_array<uint16_t>* result = &Runtime::instance().temp<uint16_t, uint16_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MOD, *result, lhs, rhs);

    return *result;
}

inline multi_array<uint64_t>& operator% (multi_array<uint64_t>& lhs, multi_array<uint64_t>& rhs)
{
    multi_array<uint64_t>* result;  
    multi_array<uint64_t>* left    = &lhs;
    multi_array<uint64_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<uint64_t, uint64_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<uint64_t, uint64_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<uint64_t, uint64_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MOD, *result, *left, *right);

    return *result;
}

inline multi_array<uint64_t> & operator% (multi_array<uint64_t>& lhs, const uint64_t& rhs)
{
    multi_array<uint64_t>* result = &Runtime::instance().temp<uint64_t, uint64_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MOD, *result, lhs, rhs);

    return *result;
}

inline multi_array<uint64_t> & operator% (const uint64_t& lhs, multi_array<uint64_t>& rhs)
{
    multi_array<uint64_t>* result = &Runtime::instance().temp<uint64_t, uint64_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MOD, *result, lhs, rhs);

    return *result;
}

inline multi_array<int16_t>& operator% (multi_array<int16_t>& lhs, multi_array<int16_t>& rhs)
{
    multi_array<int16_t>* result;  
    multi_array<int16_t>* left    = &lhs;
    multi_array<int16_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<int16_t, int16_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int16_t, int16_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int16_t, int16_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MOD, *result, *left, *right);

    return *result;
}

inline multi_array<int16_t> & operator% (multi_array<int16_t>& lhs, const int16_t& rhs)
{
    multi_array<int16_t>* result = &Runtime::instance().temp<int16_t, int16_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MOD, *result, lhs, rhs);

    return *result;
}

inline multi_array<int16_t> & operator% (const int16_t& lhs, multi_array<int16_t>& rhs)
{
    multi_array<int16_t>* result = &Runtime::instance().temp<int16_t, int16_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MOD, *result, lhs, rhs);

    return *result;
}

inline multi_array<float>& operator% (multi_array<float>& lhs, multi_array<float>& rhs)
{
    multi_array<float>* result;  
    multi_array<float>* left    = &lhs;
    multi_array<float>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<float, float>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<float, float>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<float, float>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MOD, *result, *left, *right);

    return *result;
}

inline multi_array<float> & operator% (multi_array<float>& lhs, const float& rhs)
{
    multi_array<float>* result = &Runtime::instance().temp<float, float>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MOD, *result, lhs, rhs);

    return *result;
}

inline multi_array<float> & operator% (const float& lhs, multi_array<float>& rhs)
{
    multi_array<float>* result = &Runtime::instance().temp<float, float>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MOD, *result, lhs, rhs);

    return *result;
}

inline multi_array<int32_t>& operator% (multi_array<int32_t>& lhs, multi_array<int32_t>& rhs)
{
    multi_array<int32_t>* result;  
    multi_array<int32_t>* left    = &lhs;
    multi_array<int32_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<int32_t, int32_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int32_t, int32_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int32_t, int32_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MOD, *result, *left, *right);

    return *result;
}

inline multi_array<int32_t> & operator% (multi_array<int32_t>& lhs, const int32_t& rhs)
{
    multi_array<int32_t>* result = &Runtime::instance().temp<int32_t, int32_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MOD, *result, lhs, rhs);

    return *result;
}

inline multi_array<int32_t> & operator% (const int32_t& lhs, multi_array<int32_t>& rhs)
{
    multi_array<int32_t>* result = &Runtime::instance().temp<int32_t, int32_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MOD, *result, lhs, rhs);

    return *result;
}

inline multi_array<int64_t>& operator% (multi_array<int64_t>& lhs, multi_array<int64_t>& rhs)
{
    multi_array<int64_t>* result;  
    multi_array<int64_t>* left    = &lhs;
    multi_array<int64_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<int64_t, int64_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int64_t, int64_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int64_t, int64_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MOD, *result, *left, *right);

    return *result;
}

inline multi_array<int64_t> & operator% (multi_array<int64_t>& lhs, const int64_t& rhs)
{
    multi_array<int64_t>* result = &Runtime::instance().temp<int64_t, int64_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MOD, *result, lhs, rhs);

    return *result;
}

inline multi_array<int64_t> & operator% (const int64_t& lhs, multi_array<int64_t>& rhs)
{
    multi_array<int64_t>* result = &Runtime::instance().temp<int64_t, int64_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MOD, *result, lhs, rhs);

    return *result;
}

inline multi_array<uint32_t>& operator% (multi_array<uint32_t>& lhs, multi_array<uint32_t>& rhs)
{
    multi_array<uint32_t>* result;  
    multi_array<uint32_t>* left    = &lhs;
    multi_array<uint32_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<uint32_t, uint32_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<uint32_t, uint32_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<uint32_t, uint32_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MOD, *result, *left, *right);

    return *result;
}

inline multi_array<uint32_t> & operator% (multi_array<uint32_t>& lhs, const uint32_t& rhs)
{
    multi_array<uint32_t>* result = &Runtime::instance().temp<uint32_t, uint32_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MOD, *result, lhs, rhs);

    return *result;
}

inline multi_array<uint32_t> & operator% (const uint32_t& lhs, multi_array<uint32_t>& rhs)
{
    multi_array<uint32_t>* result = &Runtime::instance().temp<uint32_t, uint32_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_MOD, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator== (multi_array<int8_t>& lhs, multi_array<int8_t>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<int8_t>* left    = &lhs;
    multi_array<int8_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, int8_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, int8_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, int8_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator== (multi_array<int8_t>& lhs, const int8_t& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, int8_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator== (const int8_t& lhs, multi_array<int8_t>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, int8_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator== (multi_array<double>& lhs, multi_array<double>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<double>* left    = &lhs;
    multi_array<double>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, double>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, double>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, double>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator== (multi_array<double>& lhs, const double& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, double>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator== (const double& lhs, multi_array<double>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, double>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator== (multi_array<uint16_t>& lhs, multi_array<uint16_t>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<uint16_t>* left    = &lhs;
    multi_array<uint16_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, uint16_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, uint16_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, uint16_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator== (multi_array<uint16_t>& lhs, const uint16_t& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, uint16_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator== (const uint16_t& lhs, multi_array<uint16_t>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, uint16_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator== (multi_array<uint64_t>& lhs, multi_array<uint64_t>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<uint64_t>* left    = &lhs;
    multi_array<uint64_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, uint64_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, uint64_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, uint64_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator== (multi_array<uint64_t>& lhs, const uint64_t& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, uint64_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator== (const uint64_t& lhs, multi_array<uint64_t>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, uint64_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator== (multi_array<int16_t>& lhs, multi_array<int16_t>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<int16_t>* left    = &lhs;
    multi_array<int16_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, int16_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, int16_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, int16_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator== (multi_array<int16_t>& lhs, const int16_t& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, int16_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator== (const int16_t& lhs, multi_array<int16_t>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, int16_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator== (multi_array<float>& lhs, multi_array<float>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<float>* left    = &lhs;
    multi_array<float>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, float>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, float>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, float>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator== (multi_array<float>& lhs, const float& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, float>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator== (const float& lhs, multi_array<float>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, float>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator== (multi_array<unsigned char>& lhs, multi_array<unsigned char>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<unsigned char>* left    = &lhs;
    multi_array<unsigned char>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, unsigned char>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, unsigned char>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, unsigned char>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator== (multi_array<unsigned char>& lhs, const unsigned char& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, unsigned char>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator== (const unsigned char& lhs, multi_array<unsigned char>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, unsigned char>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator== (multi_array<int32_t>& lhs, multi_array<int32_t>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<int32_t>* left    = &lhs;
    multi_array<int32_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, int32_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, int32_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, int32_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator== (multi_array<int32_t>& lhs, const int32_t& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, int32_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator== (const int32_t& lhs, multi_array<int32_t>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, int32_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator== (multi_array<int64_t>& lhs, multi_array<int64_t>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<int64_t>* left    = &lhs;
    multi_array<int64_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, int64_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, int64_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, int64_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator== (multi_array<int64_t>& lhs, const int64_t& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, int64_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator== (const int64_t& lhs, multi_array<int64_t>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, int64_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator== (multi_array<uint32_t>& lhs, multi_array<uint32_t>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<uint32_t>* left    = &lhs;
    multi_array<uint32_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, uint32_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, uint32_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, uint32_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator== (multi_array<uint32_t>& lhs, const uint32_t& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, uint32_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator== (const uint32_t& lhs, multi_array<uint32_t>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, uint32_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator== (multi_array<std::complex<float> >& lhs, multi_array<std::complex<float> >& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<std::complex<float> >* left    = &lhs;
    multi_array<std::complex<float> >* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, std::complex<float> >(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, std::complex<float> >(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, std::complex<float> >(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator== (multi_array<std::complex<float> >& lhs, const std::complex<float> & rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, std::complex<float> >(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator== (const std::complex<float> & lhs, multi_array<std::complex<float> >& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, std::complex<float> >(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator== (multi_array<std::complex<double> >& lhs, multi_array<std::complex<double> >& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<std::complex<double> >* left    = &lhs;
    multi_array<std::complex<double> >* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, std::complex<double> >(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, std::complex<double> >(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, std::complex<double> >(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator== (multi_array<std::complex<double> >& lhs, const std::complex<double> & rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, std::complex<double> >(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator== (const std::complex<double> & lhs, multi_array<std::complex<double> >& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, std::complex<double> >(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator!= (multi_array<int8_t>& lhs, multi_array<int8_t>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<int8_t>* left    = &lhs;
    multi_array<int8_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, int8_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, int8_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, int8_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator!= (multi_array<int8_t>& lhs, const int8_t& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, int8_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator!= (const int8_t& lhs, multi_array<int8_t>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, int8_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator!= (multi_array<double>& lhs, multi_array<double>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<double>* left    = &lhs;
    multi_array<double>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, double>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, double>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, double>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator!= (multi_array<double>& lhs, const double& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, double>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator!= (const double& lhs, multi_array<double>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, double>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator!= (multi_array<uint16_t>& lhs, multi_array<uint16_t>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<uint16_t>* left    = &lhs;
    multi_array<uint16_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, uint16_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, uint16_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, uint16_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator!= (multi_array<uint16_t>& lhs, const uint16_t& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, uint16_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator!= (const uint16_t& lhs, multi_array<uint16_t>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, uint16_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator!= (multi_array<uint64_t>& lhs, multi_array<uint64_t>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<uint64_t>* left    = &lhs;
    multi_array<uint64_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, uint64_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, uint64_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, uint64_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator!= (multi_array<uint64_t>& lhs, const uint64_t& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, uint64_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator!= (const uint64_t& lhs, multi_array<uint64_t>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, uint64_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator!= (multi_array<int16_t>& lhs, multi_array<int16_t>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<int16_t>* left    = &lhs;
    multi_array<int16_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, int16_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, int16_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, int16_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator!= (multi_array<int16_t>& lhs, const int16_t& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, int16_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator!= (const int16_t& lhs, multi_array<int16_t>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, int16_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator!= (multi_array<float>& lhs, multi_array<float>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<float>* left    = &lhs;
    multi_array<float>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, float>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, float>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, float>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator!= (multi_array<float>& lhs, const float& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, float>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator!= (const float& lhs, multi_array<float>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, float>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator!= (multi_array<unsigned char>& lhs, multi_array<unsigned char>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<unsigned char>* left    = &lhs;
    multi_array<unsigned char>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, unsigned char>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, unsigned char>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, unsigned char>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator!= (multi_array<unsigned char>& lhs, const unsigned char& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, unsigned char>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator!= (const unsigned char& lhs, multi_array<unsigned char>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, unsigned char>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator!= (multi_array<int32_t>& lhs, multi_array<int32_t>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<int32_t>* left    = &lhs;
    multi_array<int32_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, int32_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, int32_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, int32_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator!= (multi_array<int32_t>& lhs, const int32_t& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, int32_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator!= (const int32_t& lhs, multi_array<int32_t>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, int32_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator!= (multi_array<int64_t>& lhs, multi_array<int64_t>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<int64_t>* left    = &lhs;
    multi_array<int64_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, int64_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, int64_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, int64_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator!= (multi_array<int64_t>& lhs, const int64_t& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, int64_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator!= (const int64_t& lhs, multi_array<int64_t>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, int64_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator!= (multi_array<uint32_t>& lhs, multi_array<uint32_t>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<uint32_t>* left    = &lhs;
    multi_array<uint32_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, uint32_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, uint32_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, uint32_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator!= (multi_array<uint32_t>& lhs, const uint32_t& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, uint32_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator!= (const uint32_t& lhs, multi_array<uint32_t>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, uint32_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator!= (multi_array<std::complex<float> >& lhs, multi_array<std::complex<float> >& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<std::complex<float> >* left    = &lhs;
    multi_array<std::complex<float> >* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, std::complex<float> >(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, std::complex<float> >(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, std::complex<float> >(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator!= (multi_array<std::complex<float> >& lhs, const std::complex<float> & rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, std::complex<float> >(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator!= (const std::complex<float> & lhs, multi_array<std::complex<float> >& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, std::complex<float> >(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator!= (multi_array<std::complex<double> >& lhs, multi_array<std::complex<double> >& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<std::complex<double> >* left    = &lhs;
    multi_array<std::complex<double> >* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, std::complex<double> >(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, std::complex<double> >(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, std::complex<double> >(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator!= (multi_array<std::complex<double> >& lhs, const std::complex<double> & rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, std::complex<double> >(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator!= (const std::complex<double> & lhs, multi_array<std::complex<double> >& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, std::complex<double> >(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator> (multi_array<int8_t>& lhs, multi_array<int8_t>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<int8_t>* left    = &lhs;
    multi_array<int8_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, int8_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, int8_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, int8_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator> (multi_array<int8_t>& lhs, const int8_t& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, int8_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator> (const int8_t& lhs, multi_array<int8_t>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, int8_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator> (multi_array<double>& lhs, multi_array<double>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<double>* left    = &lhs;
    multi_array<double>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, double>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, double>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, double>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator> (multi_array<double>& lhs, const double& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, double>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator> (const double& lhs, multi_array<double>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, double>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator> (multi_array<uint16_t>& lhs, multi_array<uint16_t>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<uint16_t>* left    = &lhs;
    multi_array<uint16_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, uint16_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, uint16_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, uint16_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator> (multi_array<uint16_t>& lhs, const uint16_t& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, uint16_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator> (const uint16_t& lhs, multi_array<uint16_t>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, uint16_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator> (multi_array<uint64_t>& lhs, multi_array<uint64_t>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<uint64_t>* left    = &lhs;
    multi_array<uint64_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, uint64_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, uint64_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, uint64_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator> (multi_array<uint64_t>& lhs, const uint64_t& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, uint64_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator> (const uint64_t& lhs, multi_array<uint64_t>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, uint64_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator> (multi_array<int16_t>& lhs, multi_array<int16_t>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<int16_t>* left    = &lhs;
    multi_array<int16_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, int16_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, int16_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, int16_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator> (multi_array<int16_t>& lhs, const int16_t& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, int16_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator> (const int16_t& lhs, multi_array<int16_t>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, int16_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator> (multi_array<float>& lhs, multi_array<float>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<float>* left    = &lhs;
    multi_array<float>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, float>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, float>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, float>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator> (multi_array<float>& lhs, const float& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, float>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator> (const float& lhs, multi_array<float>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, float>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator> (multi_array<unsigned char>& lhs, multi_array<unsigned char>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<unsigned char>* left    = &lhs;
    multi_array<unsigned char>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, unsigned char>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, unsigned char>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, unsigned char>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator> (multi_array<unsigned char>& lhs, const unsigned char& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, unsigned char>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator> (const unsigned char& lhs, multi_array<unsigned char>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, unsigned char>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator> (multi_array<int32_t>& lhs, multi_array<int32_t>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<int32_t>* left    = &lhs;
    multi_array<int32_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, int32_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, int32_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, int32_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator> (multi_array<int32_t>& lhs, const int32_t& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, int32_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator> (const int32_t& lhs, multi_array<int32_t>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, int32_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator> (multi_array<int64_t>& lhs, multi_array<int64_t>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<int64_t>* left    = &lhs;
    multi_array<int64_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, int64_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, int64_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, int64_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator> (multi_array<int64_t>& lhs, const int64_t& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, int64_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator> (const int64_t& lhs, multi_array<int64_t>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, int64_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator> (multi_array<uint32_t>& lhs, multi_array<uint32_t>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<uint32_t>* left    = &lhs;
    multi_array<uint32_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, uint32_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, uint32_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, uint32_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator> (multi_array<uint32_t>& lhs, const uint32_t& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, uint32_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator> (const uint32_t& lhs, multi_array<uint32_t>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, uint32_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator>= (multi_array<int8_t>& lhs, multi_array<int8_t>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<int8_t>* left    = &lhs;
    multi_array<int8_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, int8_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, int8_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, int8_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER_EQUAL, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator>= (multi_array<int8_t>& lhs, const int8_t& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, int8_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator>= (const int8_t& lhs, multi_array<int8_t>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, int8_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator>= (multi_array<double>& lhs, multi_array<double>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<double>* left    = &lhs;
    multi_array<double>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, double>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, double>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, double>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER_EQUAL, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator>= (multi_array<double>& lhs, const double& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, double>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator>= (const double& lhs, multi_array<double>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, double>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator>= (multi_array<uint16_t>& lhs, multi_array<uint16_t>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<uint16_t>* left    = &lhs;
    multi_array<uint16_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, uint16_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, uint16_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, uint16_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER_EQUAL, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator>= (multi_array<uint16_t>& lhs, const uint16_t& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, uint16_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator>= (const uint16_t& lhs, multi_array<uint16_t>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, uint16_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator>= (multi_array<uint64_t>& lhs, multi_array<uint64_t>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<uint64_t>* left    = &lhs;
    multi_array<uint64_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, uint64_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, uint64_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, uint64_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER_EQUAL, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator>= (multi_array<uint64_t>& lhs, const uint64_t& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, uint64_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator>= (const uint64_t& lhs, multi_array<uint64_t>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, uint64_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator>= (multi_array<int16_t>& lhs, multi_array<int16_t>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<int16_t>* left    = &lhs;
    multi_array<int16_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, int16_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, int16_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, int16_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER_EQUAL, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator>= (multi_array<int16_t>& lhs, const int16_t& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, int16_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator>= (const int16_t& lhs, multi_array<int16_t>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, int16_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator>= (multi_array<float>& lhs, multi_array<float>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<float>* left    = &lhs;
    multi_array<float>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, float>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, float>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, float>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER_EQUAL, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator>= (multi_array<float>& lhs, const float& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, float>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator>= (const float& lhs, multi_array<float>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, float>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator>= (multi_array<unsigned char>& lhs, multi_array<unsigned char>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<unsigned char>* left    = &lhs;
    multi_array<unsigned char>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, unsigned char>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, unsigned char>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, unsigned char>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER_EQUAL, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator>= (multi_array<unsigned char>& lhs, const unsigned char& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, unsigned char>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator>= (const unsigned char& lhs, multi_array<unsigned char>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, unsigned char>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator>= (multi_array<int32_t>& lhs, multi_array<int32_t>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<int32_t>* left    = &lhs;
    multi_array<int32_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, int32_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, int32_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, int32_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER_EQUAL, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator>= (multi_array<int32_t>& lhs, const int32_t& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, int32_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator>= (const int32_t& lhs, multi_array<int32_t>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, int32_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator>= (multi_array<int64_t>& lhs, multi_array<int64_t>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<int64_t>* left    = &lhs;
    multi_array<int64_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, int64_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, int64_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, int64_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER_EQUAL, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator>= (multi_array<int64_t>& lhs, const int64_t& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, int64_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator>= (const int64_t& lhs, multi_array<int64_t>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, int64_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator>= (multi_array<uint32_t>& lhs, multi_array<uint32_t>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<uint32_t>* left    = &lhs;
    multi_array<uint32_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, uint32_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, uint32_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, uint32_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER_EQUAL, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator>= (multi_array<uint32_t>& lhs, const uint32_t& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, uint32_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator>= (const uint32_t& lhs, multi_array<uint32_t>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, uint32_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_GREATER_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator< (multi_array<int8_t>& lhs, multi_array<int8_t>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<int8_t>* left    = &lhs;
    multi_array<int8_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, int8_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, int8_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, int8_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator< (multi_array<int8_t>& lhs, const int8_t& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, int8_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator< (const int8_t& lhs, multi_array<int8_t>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, int8_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator< (multi_array<double>& lhs, multi_array<double>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<double>* left    = &lhs;
    multi_array<double>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, double>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, double>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, double>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator< (multi_array<double>& lhs, const double& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, double>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator< (const double& lhs, multi_array<double>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, double>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator< (multi_array<uint16_t>& lhs, multi_array<uint16_t>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<uint16_t>* left    = &lhs;
    multi_array<uint16_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, uint16_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, uint16_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, uint16_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator< (multi_array<uint16_t>& lhs, const uint16_t& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, uint16_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator< (const uint16_t& lhs, multi_array<uint16_t>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, uint16_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator< (multi_array<uint64_t>& lhs, multi_array<uint64_t>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<uint64_t>* left    = &lhs;
    multi_array<uint64_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, uint64_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, uint64_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, uint64_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator< (multi_array<uint64_t>& lhs, const uint64_t& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, uint64_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator< (const uint64_t& lhs, multi_array<uint64_t>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, uint64_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator< (multi_array<int16_t>& lhs, multi_array<int16_t>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<int16_t>* left    = &lhs;
    multi_array<int16_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, int16_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, int16_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, int16_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator< (multi_array<int16_t>& lhs, const int16_t& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, int16_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator< (const int16_t& lhs, multi_array<int16_t>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, int16_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator< (multi_array<float>& lhs, multi_array<float>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<float>* left    = &lhs;
    multi_array<float>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, float>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, float>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, float>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator< (multi_array<float>& lhs, const float& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, float>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator< (const float& lhs, multi_array<float>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, float>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator< (multi_array<unsigned char>& lhs, multi_array<unsigned char>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<unsigned char>* left    = &lhs;
    multi_array<unsigned char>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, unsigned char>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, unsigned char>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, unsigned char>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator< (multi_array<unsigned char>& lhs, const unsigned char& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, unsigned char>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator< (const unsigned char& lhs, multi_array<unsigned char>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, unsigned char>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator< (multi_array<int32_t>& lhs, multi_array<int32_t>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<int32_t>* left    = &lhs;
    multi_array<int32_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, int32_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, int32_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, int32_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator< (multi_array<int32_t>& lhs, const int32_t& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, int32_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator< (const int32_t& lhs, multi_array<int32_t>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, int32_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator< (multi_array<int64_t>& lhs, multi_array<int64_t>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<int64_t>* left    = &lhs;
    multi_array<int64_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, int64_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, int64_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, int64_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator< (multi_array<int64_t>& lhs, const int64_t& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, int64_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator< (const int64_t& lhs, multi_array<int64_t>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, int64_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator< (multi_array<uint32_t>& lhs, multi_array<uint32_t>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<uint32_t>* left    = &lhs;
    multi_array<uint32_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, uint32_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, uint32_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, uint32_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator< (multi_array<uint32_t>& lhs, const uint32_t& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, uint32_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator< (const uint32_t& lhs, multi_array<uint32_t>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, uint32_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator<= (multi_array<int8_t>& lhs, multi_array<int8_t>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<int8_t>* left    = &lhs;
    multi_array<int8_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, int8_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, int8_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, int8_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS_EQUAL, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator<= (multi_array<int8_t>& lhs, const int8_t& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, int8_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator<= (const int8_t& lhs, multi_array<int8_t>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, int8_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator<= (multi_array<double>& lhs, multi_array<double>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<double>* left    = &lhs;
    multi_array<double>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, double>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, double>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, double>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS_EQUAL, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator<= (multi_array<double>& lhs, const double& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, double>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator<= (const double& lhs, multi_array<double>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, double>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator<= (multi_array<uint16_t>& lhs, multi_array<uint16_t>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<uint16_t>* left    = &lhs;
    multi_array<uint16_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, uint16_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, uint16_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, uint16_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS_EQUAL, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator<= (multi_array<uint16_t>& lhs, const uint16_t& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, uint16_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator<= (const uint16_t& lhs, multi_array<uint16_t>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, uint16_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator<= (multi_array<uint64_t>& lhs, multi_array<uint64_t>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<uint64_t>* left    = &lhs;
    multi_array<uint64_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, uint64_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, uint64_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, uint64_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS_EQUAL, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator<= (multi_array<uint64_t>& lhs, const uint64_t& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, uint64_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator<= (const uint64_t& lhs, multi_array<uint64_t>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, uint64_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator<= (multi_array<int16_t>& lhs, multi_array<int16_t>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<int16_t>* left    = &lhs;
    multi_array<int16_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, int16_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, int16_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, int16_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS_EQUAL, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator<= (multi_array<int16_t>& lhs, const int16_t& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, int16_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator<= (const int16_t& lhs, multi_array<int16_t>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, int16_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator<= (multi_array<float>& lhs, multi_array<float>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<float>* left    = &lhs;
    multi_array<float>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, float>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, float>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, float>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS_EQUAL, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator<= (multi_array<float>& lhs, const float& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, float>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator<= (const float& lhs, multi_array<float>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, float>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator<= (multi_array<unsigned char>& lhs, multi_array<unsigned char>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<unsigned char>* left    = &lhs;
    multi_array<unsigned char>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, unsigned char>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, unsigned char>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, unsigned char>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS_EQUAL, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator<= (multi_array<unsigned char>& lhs, const unsigned char& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, unsigned char>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator<= (const unsigned char& lhs, multi_array<unsigned char>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, unsigned char>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator<= (multi_array<int32_t>& lhs, multi_array<int32_t>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<int32_t>* left    = &lhs;
    multi_array<int32_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, int32_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, int32_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, int32_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS_EQUAL, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator<= (multi_array<int32_t>& lhs, const int32_t& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, int32_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator<= (const int32_t& lhs, multi_array<int32_t>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, int32_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator<= (multi_array<int64_t>& lhs, multi_array<int64_t>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<int64_t>* left    = &lhs;
    multi_array<int64_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, int64_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, int64_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, int64_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS_EQUAL, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator<= (multi_array<int64_t>& lhs, const int64_t& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, int64_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator<= (const int64_t& lhs, multi_array<int64_t>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, int64_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator<= (multi_array<uint32_t>& lhs, multi_array<uint32_t>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<uint32_t>* left    = &lhs;
    multi_array<uint32_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, uint32_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, uint32_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, uint32_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS_EQUAL, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator<= (multi_array<uint32_t>& lhs, const uint32_t& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, uint32_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator<= (const uint32_t& lhs, multi_array<uint32_t>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, uint32_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LESS_EQUAL, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator&& (multi_array<unsigned char>& lhs, multi_array<unsigned char>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<unsigned char>* left    = &lhs;
    multi_array<unsigned char>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, unsigned char>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, unsigned char>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, unsigned char>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_AND, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator&& (multi_array<unsigned char>& lhs, const unsigned char& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, unsigned char>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_AND, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator&& (const unsigned char& lhs, multi_array<unsigned char>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, unsigned char>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_AND, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator|| (multi_array<unsigned char>& lhs, multi_array<unsigned char>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<unsigned char>* left    = &lhs;
    multi_array<unsigned char>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, unsigned char>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, unsigned char>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, unsigned char>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_OR, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator|| (multi_array<unsigned char>& lhs, const unsigned char& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, unsigned char>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_OR, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator|| (const unsigned char& lhs, multi_array<unsigned char>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, unsigned char>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_OR, *result, lhs, rhs);

    return *result;
}

inline multi_array<int8_t>& operator& (multi_array<int8_t>& lhs, multi_array<int8_t>& rhs)
{
    multi_array<int8_t>* result;  
    multi_array<int8_t>* left    = &lhs;
    multi_array<int8_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<int8_t, int8_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int8_t, int8_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int8_t, int8_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_AND, *result, *left, *right);

    return *result;
}

inline multi_array<int8_t> & operator& (multi_array<int8_t>& lhs, const int8_t& rhs)
{
    multi_array<int8_t>* result = &Runtime::instance().temp<int8_t, int8_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_AND, *result, lhs, rhs);

    return *result;
}

inline multi_array<int8_t> & operator& (const int8_t& lhs, multi_array<int8_t>& rhs)
{
    multi_array<int8_t>* result = &Runtime::instance().temp<int8_t, int8_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_AND, *result, lhs, rhs);

    return *result;
}

inline multi_array<uint16_t>& operator& (multi_array<uint16_t>& lhs, multi_array<uint16_t>& rhs)
{
    multi_array<uint16_t>* result;  
    multi_array<uint16_t>* left    = &lhs;
    multi_array<uint16_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<uint16_t, uint16_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<uint16_t, uint16_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<uint16_t, uint16_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_AND, *result, *left, *right);

    return *result;
}

inline multi_array<uint16_t> & operator& (multi_array<uint16_t>& lhs, const uint16_t& rhs)
{
    multi_array<uint16_t>* result = &Runtime::instance().temp<uint16_t, uint16_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_AND, *result, lhs, rhs);

    return *result;
}

inline multi_array<uint16_t> & operator& (const uint16_t& lhs, multi_array<uint16_t>& rhs)
{
    multi_array<uint16_t>* result = &Runtime::instance().temp<uint16_t, uint16_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_AND, *result, lhs, rhs);

    return *result;
}

inline multi_array<uint64_t>& operator& (multi_array<uint64_t>& lhs, multi_array<uint64_t>& rhs)
{
    multi_array<uint64_t>* result;  
    multi_array<uint64_t>* left    = &lhs;
    multi_array<uint64_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<uint64_t, uint64_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<uint64_t, uint64_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<uint64_t, uint64_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_AND, *result, *left, *right);

    return *result;
}

inline multi_array<uint64_t> & operator& (multi_array<uint64_t>& lhs, const uint64_t& rhs)
{
    multi_array<uint64_t>* result = &Runtime::instance().temp<uint64_t, uint64_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_AND, *result, lhs, rhs);

    return *result;
}

inline multi_array<uint64_t> & operator& (const uint64_t& lhs, multi_array<uint64_t>& rhs)
{
    multi_array<uint64_t>* result = &Runtime::instance().temp<uint64_t, uint64_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_AND, *result, lhs, rhs);

    return *result;
}

inline multi_array<int16_t>& operator& (multi_array<int16_t>& lhs, multi_array<int16_t>& rhs)
{
    multi_array<int16_t>* result;  
    multi_array<int16_t>* left    = &lhs;
    multi_array<int16_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<int16_t, int16_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int16_t, int16_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int16_t, int16_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_AND, *result, *left, *right);

    return *result;
}

inline multi_array<int16_t> & operator& (multi_array<int16_t>& lhs, const int16_t& rhs)
{
    multi_array<int16_t>* result = &Runtime::instance().temp<int16_t, int16_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_AND, *result, lhs, rhs);

    return *result;
}

inline multi_array<int16_t> & operator& (const int16_t& lhs, multi_array<int16_t>& rhs)
{
    multi_array<int16_t>* result = &Runtime::instance().temp<int16_t, int16_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_AND, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator& (multi_array<unsigned char>& lhs, multi_array<unsigned char>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<unsigned char>* left    = &lhs;
    multi_array<unsigned char>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, unsigned char>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, unsigned char>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, unsigned char>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_AND, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator& (multi_array<unsigned char>& lhs, const unsigned char& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, unsigned char>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_AND, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator& (const unsigned char& lhs, multi_array<unsigned char>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, unsigned char>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_AND, *result, lhs, rhs);

    return *result;
}

inline multi_array<int32_t>& operator& (multi_array<int32_t>& lhs, multi_array<int32_t>& rhs)
{
    multi_array<int32_t>* result;  
    multi_array<int32_t>* left    = &lhs;
    multi_array<int32_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<int32_t, int32_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int32_t, int32_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int32_t, int32_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_AND, *result, *left, *right);

    return *result;
}

inline multi_array<int32_t> & operator& (multi_array<int32_t>& lhs, const int32_t& rhs)
{
    multi_array<int32_t>* result = &Runtime::instance().temp<int32_t, int32_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_AND, *result, lhs, rhs);

    return *result;
}

inline multi_array<int32_t> & operator& (const int32_t& lhs, multi_array<int32_t>& rhs)
{
    multi_array<int32_t>* result = &Runtime::instance().temp<int32_t, int32_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_AND, *result, lhs, rhs);

    return *result;
}

inline multi_array<int64_t>& operator& (multi_array<int64_t>& lhs, multi_array<int64_t>& rhs)
{
    multi_array<int64_t>* result;  
    multi_array<int64_t>* left    = &lhs;
    multi_array<int64_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<int64_t, int64_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int64_t, int64_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int64_t, int64_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_AND, *result, *left, *right);

    return *result;
}

inline multi_array<int64_t> & operator& (multi_array<int64_t>& lhs, const int64_t& rhs)
{
    multi_array<int64_t>* result = &Runtime::instance().temp<int64_t, int64_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_AND, *result, lhs, rhs);

    return *result;
}

inline multi_array<int64_t> & operator& (const int64_t& lhs, multi_array<int64_t>& rhs)
{
    multi_array<int64_t>* result = &Runtime::instance().temp<int64_t, int64_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_AND, *result, lhs, rhs);

    return *result;
}

inline multi_array<uint32_t>& operator& (multi_array<uint32_t>& lhs, multi_array<uint32_t>& rhs)
{
    multi_array<uint32_t>* result;  
    multi_array<uint32_t>* left    = &lhs;
    multi_array<uint32_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<uint32_t, uint32_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<uint32_t, uint32_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<uint32_t, uint32_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_AND, *result, *left, *right);

    return *result;
}

inline multi_array<uint32_t> & operator& (multi_array<uint32_t>& lhs, const uint32_t& rhs)
{
    multi_array<uint32_t>* result = &Runtime::instance().temp<uint32_t, uint32_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_AND, *result, lhs, rhs);

    return *result;
}

inline multi_array<uint32_t> & operator& (const uint32_t& lhs, multi_array<uint32_t>& rhs)
{
    multi_array<uint32_t>* result = &Runtime::instance().temp<uint32_t, uint32_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_AND, *result, lhs, rhs);

    return *result;
}

inline multi_array<int8_t>& operator| (multi_array<int8_t>& lhs, multi_array<int8_t>& rhs)
{
    multi_array<int8_t>* result;  
    multi_array<int8_t>* left    = &lhs;
    multi_array<int8_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<int8_t, int8_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int8_t, int8_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int8_t, int8_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_OR, *result, *left, *right);

    return *result;
}

inline multi_array<int8_t> & operator| (multi_array<int8_t>& lhs, const int8_t& rhs)
{
    multi_array<int8_t>* result = &Runtime::instance().temp<int8_t, int8_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_OR, *result, lhs, rhs);

    return *result;
}

inline multi_array<int8_t> & operator| (const int8_t& lhs, multi_array<int8_t>& rhs)
{
    multi_array<int8_t>* result = &Runtime::instance().temp<int8_t, int8_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_OR, *result, lhs, rhs);

    return *result;
}

inline multi_array<uint16_t>& operator| (multi_array<uint16_t>& lhs, multi_array<uint16_t>& rhs)
{
    multi_array<uint16_t>* result;  
    multi_array<uint16_t>* left    = &lhs;
    multi_array<uint16_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<uint16_t, uint16_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<uint16_t, uint16_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<uint16_t, uint16_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_OR, *result, *left, *right);

    return *result;
}

inline multi_array<uint16_t> & operator| (multi_array<uint16_t>& lhs, const uint16_t& rhs)
{
    multi_array<uint16_t>* result = &Runtime::instance().temp<uint16_t, uint16_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_OR, *result, lhs, rhs);

    return *result;
}

inline multi_array<uint16_t> & operator| (const uint16_t& lhs, multi_array<uint16_t>& rhs)
{
    multi_array<uint16_t>* result = &Runtime::instance().temp<uint16_t, uint16_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_OR, *result, lhs, rhs);

    return *result;
}

inline multi_array<uint64_t>& operator| (multi_array<uint64_t>& lhs, multi_array<uint64_t>& rhs)
{
    multi_array<uint64_t>* result;  
    multi_array<uint64_t>* left    = &lhs;
    multi_array<uint64_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<uint64_t, uint64_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<uint64_t, uint64_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<uint64_t, uint64_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_OR, *result, *left, *right);

    return *result;
}

inline multi_array<uint64_t> & operator| (multi_array<uint64_t>& lhs, const uint64_t& rhs)
{
    multi_array<uint64_t>* result = &Runtime::instance().temp<uint64_t, uint64_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_OR, *result, lhs, rhs);

    return *result;
}

inline multi_array<uint64_t> & operator| (const uint64_t& lhs, multi_array<uint64_t>& rhs)
{
    multi_array<uint64_t>* result = &Runtime::instance().temp<uint64_t, uint64_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_OR, *result, lhs, rhs);

    return *result;
}

inline multi_array<int16_t>& operator| (multi_array<int16_t>& lhs, multi_array<int16_t>& rhs)
{
    multi_array<int16_t>* result;  
    multi_array<int16_t>* left    = &lhs;
    multi_array<int16_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<int16_t, int16_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int16_t, int16_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int16_t, int16_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_OR, *result, *left, *right);

    return *result;
}

inline multi_array<int16_t> & operator| (multi_array<int16_t>& lhs, const int16_t& rhs)
{
    multi_array<int16_t>* result = &Runtime::instance().temp<int16_t, int16_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_OR, *result, lhs, rhs);

    return *result;
}

inline multi_array<int16_t> & operator| (const int16_t& lhs, multi_array<int16_t>& rhs)
{
    multi_array<int16_t>* result = &Runtime::instance().temp<int16_t, int16_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_OR, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator| (multi_array<unsigned char>& lhs, multi_array<unsigned char>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<unsigned char>* left    = &lhs;
    multi_array<unsigned char>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, unsigned char>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, unsigned char>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, unsigned char>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_OR, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator| (multi_array<unsigned char>& lhs, const unsigned char& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, unsigned char>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_OR, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator| (const unsigned char& lhs, multi_array<unsigned char>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, unsigned char>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_OR, *result, lhs, rhs);

    return *result;
}

inline multi_array<int32_t>& operator| (multi_array<int32_t>& lhs, multi_array<int32_t>& rhs)
{
    multi_array<int32_t>* result;  
    multi_array<int32_t>* left    = &lhs;
    multi_array<int32_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<int32_t, int32_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int32_t, int32_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int32_t, int32_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_OR, *result, *left, *right);

    return *result;
}

inline multi_array<int32_t> & operator| (multi_array<int32_t>& lhs, const int32_t& rhs)
{
    multi_array<int32_t>* result = &Runtime::instance().temp<int32_t, int32_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_OR, *result, lhs, rhs);

    return *result;
}

inline multi_array<int32_t> & operator| (const int32_t& lhs, multi_array<int32_t>& rhs)
{
    multi_array<int32_t>* result = &Runtime::instance().temp<int32_t, int32_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_OR, *result, lhs, rhs);

    return *result;
}

inline multi_array<int64_t>& operator| (multi_array<int64_t>& lhs, multi_array<int64_t>& rhs)
{
    multi_array<int64_t>* result;  
    multi_array<int64_t>* left    = &lhs;
    multi_array<int64_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<int64_t, int64_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int64_t, int64_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int64_t, int64_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_OR, *result, *left, *right);

    return *result;
}

inline multi_array<int64_t> & operator| (multi_array<int64_t>& lhs, const int64_t& rhs)
{
    multi_array<int64_t>* result = &Runtime::instance().temp<int64_t, int64_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_OR, *result, lhs, rhs);

    return *result;
}

inline multi_array<int64_t> & operator| (const int64_t& lhs, multi_array<int64_t>& rhs)
{
    multi_array<int64_t>* result = &Runtime::instance().temp<int64_t, int64_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_OR, *result, lhs, rhs);

    return *result;
}

inline multi_array<uint32_t>& operator| (multi_array<uint32_t>& lhs, multi_array<uint32_t>& rhs)
{
    multi_array<uint32_t>* result;  
    multi_array<uint32_t>* left    = &lhs;
    multi_array<uint32_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<uint32_t, uint32_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<uint32_t, uint32_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<uint32_t, uint32_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_OR, *result, *left, *right);

    return *result;
}

inline multi_array<uint32_t> & operator| (multi_array<uint32_t>& lhs, const uint32_t& rhs)
{
    multi_array<uint32_t>* result = &Runtime::instance().temp<uint32_t, uint32_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_OR, *result, lhs, rhs);

    return *result;
}

inline multi_array<uint32_t> & operator| (const uint32_t& lhs, multi_array<uint32_t>& rhs)
{
    multi_array<uint32_t>* result = &Runtime::instance().temp<uint32_t, uint32_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_OR, *result, lhs, rhs);

    return *result;
}

inline multi_array<int8_t>& operator^ (multi_array<int8_t>& lhs, multi_array<int8_t>& rhs)
{
    multi_array<int8_t>* result;  
    multi_array<int8_t>* left    = &lhs;
    multi_array<int8_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<int8_t, int8_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int8_t, int8_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int8_t, int8_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_XOR, *result, *left, *right);

    return *result;
}

inline multi_array<int8_t> & operator^ (multi_array<int8_t>& lhs, const int8_t& rhs)
{
    multi_array<int8_t>* result = &Runtime::instance().temp<int8_t, int8_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_XOR, *result, lhs, rhs);

    return *result;
}

inline multi_array<int8_t> & operator^ (const int8_t& lhs, multi_array<int8_t>& rhs)
{
    multi_array<int8_t>* result = &Runtime::instance().temp<int8_t, int8_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_XOR, *result, lhs, rhs);

    return *result;
}

inline multi_array<uint16_t>& operator^ (multi_array<uint16_t>& lhs, multi_array<uint16_t>& rhs)
{
    multi_array<uint16_t>* result;  
    multi_array<uint16_t>* left    = &lhs;
    multi_array<uint16_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<uint16_t, uint16_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<uint16_t, uint16_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<uint16_t, uint16_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_XOR, *result, *left, *right);

    return *result;
}

inline multi_array<uint16_t> & operator^ (multi_array<uint16_t>& lhs, const uint16_t& rhs)
{
    multi_array<uint16_t>* result = &Runtime::instance().temp<uint16_t, uint16_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_XOR, *result, lhs, rhs);

    return *result;
}

inline multi_array<uint16_t> & operator^ (const uint16_t& lhs, multi_array<uint16_t>& rhs)
{
    multi_array<uint16_t>* result = &Runtime::instance().temp<uint16_t, uint16_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_XOR, *result, lhs, rhs);

    return *result;
}

inline multi_array<uint64_t>& operator^ (multi_array<uint64_t>& lhs, multi_array<uint64_t>& rhs)
{
    multi_array<uint64_t>* result;  
    multi_array<uint64_t>* left    = &lhs;
    multi_array<uint64_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<uint64_t, uint64_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<uint64_t, uint64_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<uint64_t, uint64_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_XOR, *result, *left, *right);

    return *result;
}

inline multi_array<uint64_t> & operator^ (multi_array<uint64_t>& lhs, const uint64_t& rhs)
{
    multi_array<uint64_t>* result = &Runtime::instance().temp<uint64_t, uint64_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_XOR, *result, lhs, rhs);

    return *result;
}

inline multi_array<uint64_t> & operator^ (const uint64_t& lhs, multi_array<uint64_t>& rhs)
{
    multi_array<uint64_t>* result = &Runtime::instance().temp<uint64_t, uint64_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_XOR, *result, lhs, rhs);

    return *result;
}

inline multi_array<int16_t>& operator^ (multi_array<int16_t>& lhs, multi_array<int16_t>& rhs)
{
    multi_array<int16_t>* result;  
    multi_array<int16_t>* left    = &lhs;
    multi_array<int16_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<int16_t, int16_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int16_t, int16_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int16_t, int16_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_XOR, *result, *left, *right);

    return *result;
}

inline multi_array<int16_t> & operator^ (multi_array<int16_t>& lhs, const int16_t& rhs)
{
    multi_array<int16_t>* result = &Runtime::instance().temp<int16_t, int16_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_XOR, *result, lhs, rhs);

    return *result;
}

inline multi_array<int16_t> & operator^ (const int16_t& lhs, multi_array<int16_t>& rhs)
{
    multi_array<int16_t>* result = &Runtime::instance().temp<int16_t, int16_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_XOR, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char>& operator^ (multi_array<unsigned char>& lhs, multi_array<unsigned char>& rhs)
{
    multi_array<unsigned char>* result;  
    multi_array<unsigned char>* left    = &lhs;
    multi_array<unsigned char>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<unsigned char, unsigned char>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, unsigned char>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<unsigned char, unsigned char>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_XOR, *result, *left, *right);

    return *result;
}

inline multi_array<unsigned char> & operator^ (multi_array<unsigned char>& lhs, const unsigned char& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, unsigned char>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_XOR, *result, lhs, rhs);

    return *result;
}

inline multi_array<unsigned char> & operator^ (const unsigned char& lhs, multi_array<unsigned char>& rhs)
{
    multi_array<unsigned char>* result = &Runtime::instance().temp<unsigned char, unsigned char>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_XOR, *result, lhs, rhs);

    return *result;
}

inline multi_array<int32_t>& operator^ (multi_array<int32_t>& lhs, multi_array<int32_t>& rhs)
{
    multi_array<int32_t>* result;  
    multi_array<int32_t>* left    = &lhs;
    multi_array<int32_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<int32_t, int32_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int32_t, int32_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int32_t, int32_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_XOR, *result, *left, *right);

    return *result;
}

inline multi_array<int32_t> & operator^ (multi_array<int32_t>& lhs, const int32_t& rhs)
{
    multi_array<int32_t>* result = &Runtime::instance().temp<int32_t, int32_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_XOR, *result, lhs, rhs);

    return *result;
}

inline multi_array<int32_t> & operator^ (const int32_t& lhs, multi_array<int32_t>& rhs)
{
    multi_array<int32_t>* result = &Runtime::instance().temp<int32_t, int32_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_XOR, *result, lhs, rhs);

    return *result;
}

inline multi_array<int64_t>& operator^ (multi_array<int64_t>& lhs, multi_array<int64_t>& rhs)
{
    multi_array<int64_t>* result;  
    multi_array<int64_t>* left    = &lhs;
    multi_array<int64_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<int64_t, int64_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int64_t, int64_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<int64_t, int64_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_XOR, *result, *left, *right);

    return *result;
}

inline multi_array<int64_t> & operator^ (multi_array<int64_t>& lhs, const int64_t& rhs)
{
    multi_array<int64_t>* result = &Runtime::instance().temp<int64_t, int64_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_XOR, *result, lhs, rhs);

    return *result;
}

inline multi_array<int64_t> & operator^ (const int64_t& lhs, multi_array<int64_t>& rhs)
{
    multi_array<int64_t>* result = &Runtime::instance().temp<int64_t, int64_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_XOR, *result, lhs, rhs);

    return *result;
}

inline multi_array<uint32_t>& operator^ (multi_array<uint32_t>& lhs, multi_array<uint32_t>& rhs)
{
    multi_array<uint32_t>* result;  
    multi_array<uint32_t>* left    = &lhs;
    multi_array<uint32_t>* right   = &rhs;

    if (same_shape(lhs, rhs)) {
        result = &Runtime::instance().temp<uint32_t, uint32_t>(lhs);
    } else {                                    // Broadcasting

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            left    = &Runtime::instance().temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<uint32_t, uint32_t>(*left);

        } else {                                // Right-handside has lowest rank
            left    = &lhs;
            right   = &Runtime::instance().temp_view(rhs);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            result = &Runtime::instance().temp<uint32_t, uint32_t>(*right);
        }
    }

    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_XOR, *result, *left, *right);

    return *result;
}

inline multi_array<uint32_t> & operator^ (multi_array<uint32_t>& lhs, const uint32_t& rhs)
{
    multi_array<uint32_t>* result = &Runtime::instance().temp<uint32_t, uint32_t>(lhs); 
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_XOR, *result, lhs, rhs);

    return *result;
}

inline multi_array<uint32_t> & operator^ (const uint32_t& lhs, multi_array<uint32_t>& rhs)
{
    multi_array<uint32_t>* result = &Runtime::instance().temp<uint32_t, uint32_t>(rhs);
    result->link();
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_XOR, *result, lhs, rhs);

    return *result;
}

//
//  Unary operators such as:
//  Mapping "!a" to BH_NEGATE(t, a)
//

template <typename T>
inline multi_array<T> & operator! (multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_NOT, *result, rhs);

    return *result;
}

template <typename T>
inline multi_array<T> & operator~ (multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T,T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_INVERT, *result, rhs);

    return *result;
}

}
#endif
