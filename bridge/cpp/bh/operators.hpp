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
multi_array<T>& multi_array<T>::operator+= (const T& rhs)
{
    Runtime::instance()->enqueue((bh_opcode)BH_ADD, *this, *this, rhs);
    return *this;
}

template <typename T>
multi_array<T>& multi_array<T>::operator+= (multi_array<T>& rhs)
{
    multi_array<T>* input = &rhs;
                                            
    if (this->getRank() < input->getRank()) {           // This would be illogical...
        std::stringstream s;
        s << "BINARY-BUNKERS " << "+=: " << this->getRank() << ", " << input->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!compatible_shape(*this, *input)) {             // We need to broadcast
        input = &Runtime::instance()->temp_view(rhs);   // Create view pointing to rhs as base
        
        if (!broadcast(rhs, *this, *input)) {
            throw std::runtime_error("Right-handside is not broadcastable.");
        }
    }

    Runtime::instance()->enqueue((bh_opcode)BH_ADD, *this, *this, *input);
    return *this;
}

template <typename T>
multi_array<T>& multi_array<T>::operator-= (const T& rhs)
{
    Runtime::instance()->enqueue((bh_opcode)BH_SUBTRACT, *this, *this, rhs);
    return *this;
}

template <typename T>
multi_array<T>& multi_array<T>::operator-= (multi_array<T>& rhs)
{
    multi_array<T>* input = &rhs;
                                            
    if (this->getRank() < input->getRank()) {           // This would be illogical...
        std::stringstream s;
        s << "BINARY-BUNKERS " << "-=: " << this->getRank() << ", " << input->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!compatible_shape(*this, *input)) {             // We need to broadcast
        input = &Runtime::instance()->temp_view(rhs);   // Create view pointing to rhs as base
        
        if (!broadcast(rhs, *this, *input)) {
            throw std::runtime_error("Right-handside is not broadcastable.");
        }
    }

    Runtime::instance()->enqueue((bh_opcode)BH_SUBTRACT, *this, *this, *input);
    return *this;
}

template <typename T>
multi_array<T>& multi_array<T>::operator*= (const T& rhs)
{
    Runtime::instance()->enqueue((bh_opcode)BH_MULTIPLY, *this, *this, rhs);
    return *this;
}

template <typename T>
multi_array<T>& multi_array<T>::operator*= (multi_array<T>& rhs)
{
    multi_array<T>* input = &rhs;
                                            
    if (this->getRank() < input->getRank()) {           // This would be illogical...
        std::stringstream s;
        s << "BINARY-BUNKERS " << "*=: " << this->getRank() << ", " << input->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!compatible_shape(*this, *input)) {             // We need to broadcast
        input = &Runtime::instance()->temp_view(rhs);   // Create view pointing to rhs as base
        
        if (!broadcast(rhs, *this, *input)) {
            throw std::runtime_error("Right-handside is not broadcastable.");
        }
    }

    Runtime::instance()->enqueue((bh_opcode)BH_MULTIPLY, *this, *this, *input);
    return *this;
}

template <typename T>
multi_array<T>& multi_array<T>::operator/= (const T& rhs)
{
    Runtime::instance()->enqueue((bh_opcode)BH_DIVIDE, *this, *this, rhs);
    return *this;
}

template <typename T>
multi_array<T>& multi_array<T>::operator/= (multi_array<T>& rhs)
{
    multi_array<T>* input = &rhs;
                                            
    if (this->getRank() < input->getRank()) {           // This would be illogical...
        std::stringstream s;
        s << "BINARY-BUNKERS " << "/=: " << this->getRank() << ", " << input->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!compatible_shape(*this, *input)) {             // We need to broadcast
        input = &Runtime::instance()->temp_view(rhs);   // Create view pointing to rhs as base
        
        if (!broadcast(rhs, *this, *input)) {
            throw std::runtime_error("Right-handside is not broadcastable.");
        }
    }

    Runtime::instance()->enqueue((bh_opcode)BH_DIVIDE, *this, *this, *input);
    return *this;
}

template <typename T>
multi_array<T>& multi_array<T>::operator%= (const T& rhs)
{
    Runtime::instance()->enqueue((bh_opcode)BH_MOD, *this, *this, rhs);
    return *this;
}

template <typename T>
multi_array<T>& multi_array<T>::operator%= (multi_array<T>& rhs)
{
    multi_array<T>* input = &rhs;
                                            
    if (this->getRank() < input->getRank()) {           // This would be illogical...
        std::stringstream s;
        s << "BINARY-BUNKERS " << "%=: " << this->getRank() << ", " << input->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!compatible_shape(*this, *input)) {             // We need to broadcast
        input = &Runtime::instance()->temp_view(rhs);   // Create view pointing to rhs as base
        
        if (!broadcast(rhs, *this, *input)) {
            throw std::runtime_error("Right-handside is not broadcastable.");
        }
    }

    Runtime::instance()->enqueue((bh_opcode)BH_MOD, *this, *this, *input);
    return *this;
}

template <typename T>
multi_array<T>& multi_array<T>::operator&= (const T& rhs)
{
    Runtime::instance()->enqueue((bh_opcode)BH_BITWISE_AND, *this, *this, rhs);
    return *this;
}

template <typename T>
multi_array<T>& multi_array<T>::operator&= (multi_array<T>& rhs)
{
    multi_array<T>* input = &rhs;
                                            
    if (this->getRank() < input->getRank()) {           // This would be illogical...
        std::stringstream s;
        s << "BINARY-BUNKERS " << "&=: " << this->getRank() << ", " << input->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!compatible_shape(*this, *input)) {             // We need to broadcast
        input = &Runtime::instance()->temp_view(rhs);   // Create view pointing to rhs as base
        
        if (!broadcast(rhs, *this, *input)) {
            throw std::runtime_error("Right-handside is not broadcastable.");
        }
    }

    Runtime::instance()->enqueue((bh_opcode)BH_BITWISE_AND, *this, *this, *input);
    return *this;
}

template <typename T>
multi_array<T>& multi_array<T>::operator|= (const T& rhs)
{
    Runtime::instance()->enqueue((bh_opcode)BH_BITWISE_OR, *this, *this, rhs);
    return *this;
}

template <typename T>
multi_array<T>& multi_array<T>::operator|= (multi_array<T>& rhs)
{
    multi_array<T>* input = &rhs;
                                            
    if (this->getRank() < input->getRank()) {           // This would be illogical...
        std::stringstream s;
        s << "BINARY-BUNKERS " << "|=: " << this->getRank() << ", " << input->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!compatible_shape(*this, *input)) {             // We need to broadcast
        input = &Runtime::instance()->temp_view(rhs);   // Create view pointing to rhs as base
        
        if (!broadcast(rhs, *this, *input)) {
            throw std::runtime_error("Right-handside is not broadcastable.");
        }
    }

    Runtime::instance()->enqueue((bh_opcode)BH_BITWISE_OR, *this, *this, *input);
    return *this;
}

template <typename T>
multi_array<T>& multi_array<T>::operator^= (const T& rhs)
{
    Runtime::instance()->enqueue((bh_opcode)BH_BITWISE_XOR, *this, *this, rhs);
    return *this;
}

template <typename T>
multi_array<T>& multi_array<T>::operator^= (multi_array<T>& rhs)
{
    multi_array<T>* input = &rhs;
                                            
    if (this->getRank() < input->getRank()) {           // This would be illogical...
        std::stringstream s;
        s << "BINARY-BUNKERS " << "^=: " << this->getRank() << ", " << input->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!compatible_shape(*this, *input)) {             // We need to broadcast
        input = &Runtime::instance()->temp_view(rhs);   // Create view pointing to rhs as base
        
        if (!broadcast(rhs, *this, *input)) {
            throw std::runtime_error("Right-handside is not broadcastable.");
        }
    }

    Runtime::instance()->enqueue((bh_opcode)BH_BITWISE_XOR, *this, *this, *input);
    return *this;
}

template <typename T>
multi_array<T>& multi_array<T>::operator= (const T& rhs)
{
    Runtime::instance()->enqueue((bh_opcode)BH_IDENTITY, *this, rhs);
    return *this;
}

template <typename T>
multi_array<T>& multi_array<T>::operator= (multi_array<T>& rhs)
{
    DEBUG_PRINT("{{ %ld = ", this->getRank());
    multi_array<T>* input = &rhs;
    DEBUG_PRINT("%ld\n", input->getRank());
                                            
    if (this->getRank() < input->getRank()) {           // This would be illogical...
        std::stringstream s;
        s << "Incompatible shape in 'operator='; lrank(" << this->getRank() << ") is less than rrank(" << input->getRank() << ")." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!compatible_shape(*this, *input)) {             // We need to broadcast
        input = &Runtime::instance()->temp_view(rhs);   // Create view pointing to rhs as base
        
        if (!broadcast(rhs, *this, *input)) {
            throw std::runtime_error("Right-handside is not broadcastable.");
        }
    }

    Runtime::instance()->enqueue((bh_opcode)BH_IDENTITY, *this, *input);

    DEBUG_PRINT("}}\n");
    return *this;
}

//
//  Binary operators such as:
//  Mapping "a + b" to BH_ADD(t, a, b)
//  Mapping "a + 1.0" to BH_ADD(t, a, 1.0)
//  Mapping "1.0 + a" to BH_ADD(t, 1.0, a)
//

template <typename T>
multi_array<T>& operator+ (multi_array<T>& lhs, multi_array<T>& rhs)
{
    DEBUG_PRINT("> %ld operator+ %ld\n", lhs.getRank(), rhs.getRank());
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    multi_array<T>* result; 

    if (compatible_shape(lhs, rhs)) {
        result = &Runtime::instance()->temp(lhs);
    } else {
        DEBUG_PRINT("> Incompatible shape, possibly broadcastable.\n");

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            DEBUG_PRINT("> Creating view of left\n");
            left    = &Runtime::instance()->temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            DEBUG_PRINT("> Creating temp\n");
            result  = &Runtime::instance()->temp(*left);
        } else {                                // Right-handside has lowest rank
            DEBUG_PRINT("> Creating view of right\n");
            left    = &lhs;
            right   = &Runtime::instance()->temp_view(rhs);
            right->setTemp(true);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            DEBUG_PRINT("> Creating temp\n");
            result  = &Runtime::instance()->temp(*right);
        }
        
    }
    #ifdef DEBUG
    bh_pprint_array(&storage[left->getKey()]);
    bh_pprint_array(&storage[right->getKey()]);
    bh_pprint_array(&storage[result->getKey()]);
    #endif
    Runtime::instance()->enqueue((bh_opcode)BH_ADD, *result, *left, *right);
    DEBUG_PRINT("< operator+\n");
    return *result;
}

template <typename T>
multi_array<T> & operator+ (multi_array<T>& lhs, const T& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(lhs);

    Runtime::instance()->enqueue((bh_opcode)BH_ADD, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T> & operator+ (const T& lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(rhs);

    Runtime::instance()->enqueue((bh_opcode)BH_ADD, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T>& operator- (multi_array<T>& lhs, multi_array<T>& rhs)
{
    DEBUG_PRINT("> %ld operator- %ld\n", lhs.getRank(), rhs.getRank());
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    multi_array<T>* result; 

    if (compatible_shape(lhs, rhs)) {
        result = &Runtime::instance()->temp(lhs);
    } else {
        DEBUG_PRINT("> Incompatible shape, possibly broadcastable.\n");

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            DEBUG_PRINT("> Creating view of left\n");
            left    = &Runtime::instance()->temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            DEBUG_PRINT("> Creating temp\n");
            result  = &Runtime::instance()->temp(*left);
        } else {                                // Right-handside has lowest rank
            DEBUG_PRINT("> Creating view of right\n");
            left    = &lhs;
            right   = &Runtime::instance()->temp_view(rhs);
            right->setTemp(true);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            DEBUG_PRINT("> Creating temp\n");
            result  = &Runtime::instance()->temp(*right);
        }
        
    }
    #ifdef DEBUG
    bh_pprint_array(&storage[left->getKey()]);
    bh_pprint_array(&storage[right->getKey()]);
    bh_pprint_array(&storage[result->getKey()]);
    #endif
    Runtime::instance()->enqueue((bh_opcode)BH_SUBTRACT, *result, *left, *right);
    DEBUG_PRINT("< operator-\n");
    return *result;
}

template <typename T>
multi_array<T> & operator- (multi_array<T>& lhs, const T& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(lhs);

    Runtime::instance()->enqueue((bh_opcode)BH_SUBTRACT, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T> & operator- (const T& lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(rhs);

    Runtime::instance()->enqueue((bh_opcode)BH_SUBTRACT, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T>& operator* (multi_array<T>& lhs, multi_array<T>& rhs)
{
    DEBUG_PRINT("> %ld operator* %ld\n", lhs.getRank(), rhs.getRank());
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    multi_array<T>* result; 

    if (compatible_shape(lhs, rhs)) {
        result = &Runtime::instance()->temp(lhs);
    } else {
        DEBUG_PRINT("> Incompatible shape, possibly broadcastable.\n");

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            DEBUG_PRINT("> Creating view of left\n");
            left    = &Runtime::instance()->temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            DEBUG_PRINT("> Creating temp\n");
            result  = &Runtime::instance()->temp(*left);
        } else {                                // Right-handside has lowest rank
            DEBUG_PRINT("> Creating view of right\n");
            left    = &lhs;
            right   = &Runtime::instance()->temp_view(rhs);
            right->setTemp(true);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            DEBUG_PRINT("> Creating temp\n");
            result  = &Runtime::instance()->temp(*right);
        }
        
    }
    #ifdef DEBUG
    bh_pprint_array(&storage[left->getKey()]);
    bh_pprint_array(&storage[right->getKey()]);
    bh_pprint_array(&storage[result->getKey()]);
    #endif
    Runtime::instance()->enqueue((bh_opcode)BH_MULTIPLY, *result, *left, *right);
    DEBUG_PRINT("< operator*\n");
    return *result;
}

template <typename T>
multi_array<T> & operator* (multi_array<T>& lhs, const T& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(lhs);

    Runtime::instance()->enqueue((bh_opcode)BH_MULTIPLY, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T> & operator* (const T& lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(rhs);

    Runtime::instance()->enqueue((bh_opcode)BH_MULTIPLY, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T>& operator/ (multi_array<T>& lhs, multi_array<T>& rhs)
{
    DEBUG_PRINT("> %ld operator/ %ld\n", lhs.getRank(), rhs.getRank());
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    multi_array<T>* result; 

    if (compatible_shape(lhs, rhs)) {
        result = &Runtime::instance()->temp(lhs);
    } else {
        DEBUG_PRINT("> Incompatible shape, possibly broadcastable.\n");

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            DEBUG_PRINT("> Creating view of left\n");
            left    = &Runtime::instance()->temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            DEBUG_PRINT("> Creating temp\n");
            result  = &Runtime::instance()->temp(*left);
        } else {                                // Right-handside has lowest rank
            DEBUG_PRINT("> Creating view of right\n");
            left    = &lhs;
            right   = &Runtime::instance()->temp_view(rhs);
            right->setTemp(true);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            DEBUG_PRINT("> Creating temp\n");
            result  = &Runtime::instance()->temp(*right);
        }
        
    }
    #ifdef DEBUG
    bh_pprint_array(&storage[left->getKey()]);
    bh_pprint_array(&storage[right->getKey()]);
    bh_pprint_array(&storage[result->getKey()]);
    #endif
    Runtime::instance()->enqueue((bh_opcode)BH_DIVIDE, *result, *left, *right);
    DEBUG_PRINT("< operator/\n");
    return *result;
}

template <typename T>
multi_array<T> & operator/ (multi_array<T>& lhs, const T& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(lhs);

    Runtime::instance()->enqueue((bh_opcode)BH_DIVIDE, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T> & operator/ (const T& lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(rhs);

    Runtime::instance()->enqueue((bh_opcode)BH_DIVIDE, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T>& operator% (multi_array<T>& lhs, multi_array<T>& rhs)
{
    DEBUG_PRINT("> %ld operator% %ld\n", lhs.getRank(), rhs.getRank());
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    multi_array<T>* result; 

    if (compatible_shape(lhs, rhs)) {
        result = &Runtime::instance()->temp(lhs);
    } else {
        DEBUG_PRINT("> Incompatible shape, possibly broadcastable.\n");

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            DEBUG_PRINT("> Creating view of left\n");
            left    = &Runtime::instance()->temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            DEBUG_PRINT("> Creating temp\n");
            result  = &Runtime::instance()->temp(*left);
        } else {                                // Right-handside has lowest rank
            DEBUG_PRINT("> Creating view of right\n");
            left    = &lhs;
            right   = &Runtime::instance()->temp_view(rhs);
            right->setTemp(true);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            DEBUG_PRINT("> Creating temp\n");
            result  = &Runtime::instance()->temp(*right);
        }
        
    }
    #ifdef DEBUG
    bh_pprint_array(&storage[left->getKey()]);
    bh_pprint_array(&storage[right->getKey()]);
    bh_pprint_array(&storage[result->getKey()]);
    #endif
    Runtime::instance()->enqueue((bh_opcode)BH_MOD, *result, *left, *right);
    DEBUG_PRINT("< operator%\n");
    return *result;
}

template <typename T>
multi_array<T> & operator% (multi_array<T>& lhs, const T& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(lhs);

    Runtime::instance()->enqueue((bh_opcode)BH_MOD, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T> & operator% (const T& lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(rhs);

    Runtime::instance()->enqueue((bh_opcode)BH_MOD, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T>& operator== (multi_array<T>& lhs, multi_array<T>& rhs)
{
    DEBUG_PRINT("> %ld operator== %ld\n", lhs.getRank(), rhs.getRank());
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    multi_array<T>* result; 

    if (compatible_shape(lhs, rhs)) {
        result = &Runtime::instance()->temp(lhs);
    } else {
        DEBUG_PRINT("> Incompatible shape, possibly broadcastable.\n");

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            DEBUG_PRINT("> Creating view of left\n");
            left    = &Runtime::instance()->temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            DEBUG_PRINT("> Creating temp\n");
            result  = &Runtime::instance()->temp(*left);
        } else {                                // Right-handside has lowest rank
            DEBUG_PRINT("> Creating view of right\n");
            left    = &lhs;
            right   = &Runtime::instance()->temp_view(rhs);
            right->setTemp(true);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            DEBUG_PRINT("> Creating temp\n");
            result  = &Runtime::instance()->temp(*right);
        }
        
    }
    #ifdef DEBUG
    bh_pprint_array(&storage[left->getKey()]);
    bh_pprint_array(&storage[right->getKey()]);
    bh_pprint_array(&storage[result->getKey()]);
    #endif
    Runtime::instance()->enqueue((bh_opcode)BH_EQUAL, *result, *left, *right);
    DEBUG_PRINT("< operator==\n");
    return *result;
}

template <typename T>
multi_array<T> & operator== (multi_array<T>& lhs, const T& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(lhs);

    Runtime::instance()->enqueue((bh_opcode)BH_EQUAL, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T> & operator== (const T& lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(rhs);

    Runtime::instance()->enqueue((bh_opcode)BH_EQUAL, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T>& operator!= (multi_array<T>& lhs, multi_array<T>& rhs)
{
    DEBUG_PRINT("> %ld operator!= %ld\n", lhs.getRank(), rhs.getRank());
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    multi_array<T>* result; 

    if (compatible_shape(lhs, rhs)) {
        result = &Runtime::instance()->temp(lhs);
    } else {
        DEBUG_PRINT("> Incompatible shape, possibly broadcastable.\n");

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            DEBUG_PRINT("> Creating view of left\n");
            left    = &Runtime::instance()->temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            DEBUG_PRINT("> Creating temp\n");
            result  = &Runtime::instance()->temp(*left);
        } else {                                // Right-handside has lowest rank
            DEBUG_PRINT("> Creating view of right\n");
            left    = &lhs;
            right   = &Runtime::instance()->temp_view(rhs);
            right->setTemp(true);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            DEBUG_PRINT("> Creating temp\n");
            result  = &Runtime::instance()->temp(*right);
        }
        
    }
    #ifdef DEBUG
    bh_pprint_array(&storage[left->getKey()]);
    bh_pprint_array(&storage[right->getKey()]);
    bh_pprint_array(&storage[result->getKey()]);
    #endif
    Runtime::instance()->enqueue((bh_opcode)BH_NOT_EQUAL, *result, *left, *right);
    DEBUG_PRINT("< operator!=\n");
    return *result;
}

template <typename T>
multi_array<T> & operator!= (multi_array<T>& lhs, const T& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(lhs);

    Runtime::instance()->enqueue((bh_opcode)BH_NOT_EQUAL, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T> & operator!= (const T& lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(rhs);

    Runtime::instance()->enqueue((bh_opcode)BH_NOT_EQUAL, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T>& operator> (multi_array<T>& lhs, multi_array<T>& rhs)
{
    DEBUG_PRINT("> %ld operator> %ld\n", lhs.getRank(), rhs.getRank());
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    multi_array<T>* result; 

    if (compatible_shape(lhs, rhs)) {
        result = &Runtime::instance()->temp(lhs);
    } else {
        DEBUG_PRINT("> Incompatible shape, possibly broadcastable.\n");

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            DEBUG_PRINT("> Creating view of left\n");
            left    = &Runtime::instance()->temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            DEBUG_PRINT("> Creating temp\n");
            result  = &Runtime::instance()->temp(*left);
        } else {                                // Right-handside has lowest rank
            DEBUG_PRINT("> Creating view of right\n");
            left    = &lhs;
            right   = &Runtime::instance()->temp_view(rhs);
            right->setTemp(true);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            DEBUG_PRINT("> Creating temp\n");
            result  = &Runtime::instance()->temp(*right);
        }
        
    }
    #ifdef DEBUG
    bh_pprint_array(&storage[left->getKey()]);
    bh_pprint_array(&storage[right->getKey()]);
    bh_pprint_array(&storage[result->getKey()]);
    #endif
    Runtime::instance()->enqueue((bh_opcode)BH_GREATER, *result, *left, *right);
    DEBUG_PRINT("< operator>\n");
    return *result;
}

template <typename T>
multi_array<T> & operator> (multi_array<T>& lhs, const T& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(lhs);

    Runtime::instance()->enqueue((bh_opcode)BH_GREATER, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T> & operator> (const T& lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(rhs);

    Runtime::instance()->enqueue((bh_opcode)BH_GREATER, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T>& operator>= (multi_array<T>& lhs, multi_array<T>& rhs)
{
    DEBUG_PRINT("> %ld operator>= %ld\n", lhs.getRank(), rhs.getRank());
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    multi_array<T>* result; 

    if (compatible_shape(lhs, rhs)) {
        result = &Runtime::instance()->temp(lhs);
    } else {
        DEBUG_PRINT("> Incompatible shape, possibly broadcastable.\n");

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            DEBUG_PRINT("> Creating view of left\n");
            left    = &Runtime::instance()->temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            DEBUG_PRINT("> Creating temp\n");
            result  = &Runtime::instance()->temp(*left);
        } else {                                // Right-handside has lowest rank
            DEBUG_PRINT("> Creating view of right\n");
            left    = &lhs;
            right   = &Runtime::instance()->temp_view(rhs);
            right->setTemp(true);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            DEBUG_PRINT("> Creating temp\n");
            result  = &Runtime::instance()->temp(*right);
        }
        
    }
    #ifdef DEBUG
    bh_pprint_array(&storage[left->getKey()]);
    bh_pprint_array(&storage[right->getKey()]);
    bh_pprint_array(&storage[result->getKey()]);
    #endif
    Runtime::instance()->enqueue((bh_opcode)BH_GREATER_EQUAL, *result, *left, *right);
    DEBUG_PRINT("< operator>=\n");
    return *result;
}

template <typename T>
multi_array<T> & operator>= (multi_array<T>& lhs, const T& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(lhs);

    Runtime::instance()->enqueue((bh_opcode)BH_GREATER_EQUAL, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T> & operator>= (const T& lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(rhs);

    Runtime::instance()->enqueue((bh_opcode)BH_GREATER_EQUAL, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T>& operator< (multi_array<T>& lhs, multi_array<T>& rhs)
{
    DEBUG_PRINT("> %ld operator< %ld\n", lhs.getRank(), rhs.getRank());
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    multi_array<T>* result; 

    if (compatible_shape(lhs, rhs)) {
        result = &Runtime::instance()->temp(lhs);
    } else {
        DEBUG_PRINT("> Incompatible shape, possibly broadcastable.\n");

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            DEBUG_PRINT("> Creating view of left\n");
            left    = &Runtime::instance()->temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            DEBUG_PRINT("> Creating temp\n");
            result  = &Runtime::instance()->temp(*left);
        } else {                                // Right-handside has lowest rank
            DEBUG_PRINT("> Creating view of right\n");
            left    = &lhs;
            right   = &Runtime::instance()->temp_view(rhs);
            right->setTemp(true);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            DEBUG_PRINT("> Creating temp\n");
            result  = &Runtime::instance()->temp(*right);
        }
        
    }
    #ifdef DEBUG
    bh_pprint_array(&storage[left->getKey()]);
    bh_pprint_array(&storage[right->getKey()]);
    bh_pprint_array(&storage[result->getKey()]);
    #endif
    Runtime::instance()->enqueue((bh_opcode)BH_LESS, *result, *left, *right);
    DEBUG_PRINT("< operator<\n");
    return *result;
}

template <typename T>
multi_array<T> & operator< (multi_array<T>& lhs, const T& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(lhs);

    Runtime::instance()->enqueue((bh_opcode)BH_LESS, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T> & operator< (const T& lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(rhs);

    Runtime::instance()->enqueue((bh_opcode)BH_LESS, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T>& operator<= (multi_array<T>& lhs, multi_array<T>& rhs)
{
    DEBUG_PRINT("> %ld operator<= %ld\n", lhs.getRank(), rhs.getRank());
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    multi_array<T>* result; 

    if (compatible_shape(lhs, rhs)) {
        result = &Runtime::instance()->temp(lhs);
    } else {
        DEBUG_PRINT("> Incompatible shape, possibly broadcastable.\n");

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            DEBUG_PRINT("> Creating view of left\n");
            left    = &Runtime::instance()->temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            DEBUG_PRINT("> Creating temp\n");
            result  = &Runtime::instance()->temp(*left);
        } else {                                // Right-handside has lowest rank
            DEBUG_PRINT("> Creating view of right\n");
            left    = &lhs;
            right   = &Runtime::instance()->temp_view(rhs);
            right->setTemp(true);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            DEBUG_PRINT("> Creating temp\n");
            result  = &Runtime::instance()->temp(*right);
        }
        
    }
    #ifdef DEBUG
    bh_pprint_array(&storage[left->getKey()]);
    bh_pprint_array(&storage[right->getKey()]);
    bh_pprint_array(&storage[result->getKey()]);
    #endif
    Runtime::instance()->enqueue((bh_opcode)BH_LESS_EQUAL, *result, *left, *right);
    DEBUG_PRINT("< operator<=\n");
    return *result;
}

template <typename T>
multi_array<T> & operator<= (multi_array<T>& lhs, const T& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(lhs);

    Runtime::instance()->enqueue((bh_opcode)BH_LESS_EQUAL, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T> & operator<= (const T& lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(rhs);

    Runtime::instance()->enqueue((bh_opcode)BH_LESS_EQUAL, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T>& operator&& (multi_array<T>& lhs, multi_array<T>& rhs)
{
    DEBUG_PRINT("> %ld operator&& %ld\n", lhs.getRank(), rhs.getRank());
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    multi_array<T>* result; 

    if (compatible_shape(lhs, rhs)) {
        result = &Runtime::instance()->temp(lhs);
    } else {
        DEBUG_PRINT("> Incompatible shape, possibly broadcastable.\n");

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            DEBUG_PRINT("> Creating view of left\n");
            left    = &Runtime::instance()->temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            DEBUG_PRINT("> Creating temp\n");
            result  = &Runtime::instance()->temp(*left);
        } else {                                // Right-handside has lowest rank
            DEBUG_PRINT("> Creating view of right\n");
            left    = &lhs;
            right   = &Runtime::instance()->temp_view(rhs);
            right->setTemp(true);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            DEBUG_PRINT("> Creating temp\n");
            result  = &Runtime::instance()->temp(*right);
        }
        
    }
    #ifdef DEBUG
    bh_pprint_array(&storage[left->getKey()]);
    bh_pprint_array(&storage[right->getKey()]);
    bh_pprint_array(&storage[result->getKey()]);
    #endif
    Runtime::instance()->enqueue((bh_opcode)BH_LOGICAL_AND, *result, *left, *right);
    DEBUG_PRINT("< operator&&\n");
    return *result;
}

template <typename T>
multi_array<T> & operator&& (multi_array<T>& lhs, const T& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(lhs);

    Runtime::instance()->enqueue((bh_opcode)BH_LOGICAL_AND, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T> & operator&& (const T& lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(rhs);

    Runtime::instance()->enqueue((bh_opcode)BH_LOGICAL_AND, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T>& operator|| (multi_array<T>& lhs, multi_array<T>& rhs)
{
    DEBUG_PRINT("> %ld operator|| %ld\n", lhs.getRank(), rhs.getRank());
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    multi_array<T>* result; 

    if (compatible_shape(lhs, rhs)) {
        result = &Runtime::instance()->temp(lhs);
    } else {
        DEBUG_PRINT("> Incompatible shape, possibly broadcastable.\n");

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            DEBUG_PRINT("> Creating view of left\n");
            left    = &Runtime::instance()->temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            DEBUG_PRINT("> Creating temp\n");
            result  = &Runtime::instance()->temp(*left);
        } else {                                // Right-handside has lowest rank
            DEBUG_PRINT("> Creating view of right\n");
            left    = &lhs;
            right   = &Runtime::instance()->temp_view(rhs);
            right->setTemp(true);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            DEBUG_PRINT("> Creating temp\n");
            result  = &Runtime::instance()->temp(*right);
        }
        
    }
    #ifdef DEBUG
    bh_pprint_array(&storage[left->getKey()]);
    bh_pprint_array(&storage[right->getKey()]);
    bh_pprint_array(&storage[result->getKey()]);
    #endif
    Runtime::instance()->enqueue((bh_opcode)BH_LOGICAL_OR, *result, *left, *right);
    DEBUG_PRINT("< operator||\n");
    return *result;
}

template <typename T>
multi_array<T> & operator|| (multi_array<T>& lhs, const T& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(lhs);

    Runtime::instance()->enqueue((bh_opcode)BH_LOGICAL_OR, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T> & operator|| (const T& lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(rhs);

    Runtime::instance()->enqueue((bh_opcode)BH_LOGICAL_OR, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T>& operator& (multi_array<T>& lhs, multi_array<T>& rhs)
{
    DEBUG_PRINT("> %ld operator& %ld\n", lhs.getRank(), rhs.getRank());
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    multi_array<T>* result; 

    if (compatible_shape(lhs, rhs)) {
        result = &Runtime::instance()->temp(lhs);
    } else {
        DEBUG_PRINT("> Incompatible shape, possibly broadcastable.\n");

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            DEBUG_PRINT("> Creating view of left\n");
            left    = &Runtime::instance()->temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            DEBUG_PRINT("> Creating temp\n");
            result  = &Runtime::instance()->temp(*left);
        } else {                                // Right-handside has lowest rank
            DEBUG_PRINT("> Creating view of right\n");
            left    = &lhs;
            right   = &Runtime::instance()->temp_view(rhs);
            right->setTemp(true);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            DEBUG_PRINT("> Creating temp\n");
            result  = &Runtime::instance()->temp(*right);
        }
        
    }
    #ifdef DEBUG
    bh_pprint_array(&storage[left->getKey()]);
    bh_pprint_array(&storage[right->getKey()]);
    bh_pprint_array(&storage[result->getKey()]);
    #endif
    Runtime::instance()->enqueue((bh_opcode)BH_BITWISE_AND, *result, *left, *right);
    DEBUG_PRINT("< operator&\n");
    return *result;
}

template <typename T>
multi_array<T> & operator& (multi_array<T>& lhs, const T& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(lhs);

    Runtime::instance()->enqueue((bh_opcode)BH_BITWISE_AND, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T> & operator& (const T& lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(rhs);

    Runtime::instance()->enqueue((bh_opcode)BH_BITWISE_AND, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T>& operator| (multi_array<T>& lhs, multi_array<T>& rhs)
{
    DEBUG_PRINT("> %ld operator| %ld\n", lhs.getRank(), rhs.getRank());
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    multi_array<T>* result; 

    if (compatible_shape(lhs, rhs)) {
        result = &Runtime::instance()->temp(lhs);
    } else {
        DEBUG_PRINT("> Incompatible shape, possibly broadcastable.\n");

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            DEBUG_PRINT("> Creating view of left\n");
            left    = &Runtime::instance()->temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            DEBUG_PRINT("> Creating temp\n");
            result  = &Runtime::instance()->temp(*left);
        } else {                                // Right-handside has lowest rank
            DEBUG_PRINT("> Creating view of right\n");
            left    = &lhs;
            right   = &Runtime::instance()->temp_view(rhs);
            right->setTemp(true);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            DEBUG_PRINT("> Creating temp\n");
            result  = &Runtime::instance()->temp(*right);
        }
        
    }
    #ifdef DEBUG
    bh_pprint_array(&storage[left->getKey()]);
    bh_pprint_array(&storage[right->getKey()]);
    bh_pprint_array(&storage[result->getKey()]);
    #endif
    Runtime::instance()->enqueue((bh_opcode)BH_BITWISE_OR, *result, *left, *right);
    DEBUG_PRINT("< operator|\n");
    return *result;
}

template <typename T>
multi_array<T> & operator| (multi_array<T>& lhs, const T& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(lhs);

    Runtime::instance()->enqueue((bh_opcode)BH_BITWISE_OR, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T> & operator| (const T& lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(rhs);

    Runtime::instance()->enqueue((bh_opcode)BH_BITWISE_OR, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T>& operator^ (multi_array<T>& lhs, multi_array<T>& rhs)
{
    DEBUG_PRINT("> %ld operator^ %ld\n", lhs.getRank(), rhs.getRank());
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    multi_array<T>* result; 

    if (compatible_shape(lhs, rhs)) {
        result = &Runtime::instance()->temp(lhs);
    } else {
        DEBUG_PRINT("> Incompatible shape, possibly broadcastable.\n");

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            DEBUG_PRINT("> Creating view of left\n");
            left    = &Runtime::instance()->temp_view(lhs);
            right   = &rhs;
            if (!broadcast(lhs, rhs, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            DEBUG_PRINT("> Creating temp\n");
            result  = &Runtime::instance()->temp(*left);
        } else {                                // Right-handside has lowest rank
            DEBUG_PRINT("> Creating view of right\n");
            left    = &lhs;
            right   = &Runtime::instance()->temp_view(rhs);
            right->setTemp(true);
            if (!broadcast(rhs, lhs, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
            DEBUG_PRINT("> Creating temp\n");
            result  = &Runtime::instance()->temp(*right);
        }
        
    }
    #ifdef DEBUG
    bh_pprint_array(&storage[left->getKey()]);
    bh_pprint_array(&storage[right->getKey()]);
    bh_pprint_array(&storage[result->getKey()]);
    #endif
    Runtime::instance()->enqueue((bh_opcode)BH_BITWISE_XOR, *result, *left, *right);
    DEBUG_PRINT("< operator^\n");
    return *result;
}

template <typename T>
multi_array<T> & operator^ (multi_array<T>& lhs, const T& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(lhs);

    Runtime::instance()->enqueue((bh_opcode)BH_BITWISE_XOR, *result, lhs, rhs);

    return *result;
}

template <typename T>
multi_array<T> & operator^ (const T& lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(rhs);

    Runtime::instance()->enqueue((bh_opcode)BH_BITWISE_XOR, *result, lhs, rhs);

    return *result;
}

//
//  Unary operators such as:
//  Mapping "!a" to BH_NEGATE(t, a)
//

template <typename T>
multi_array<T> & operator! (multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(rhs);

    Runtime::instance()->enqueue((bh_opcode)BH_LOGICAL_NOT, *result, rhs);

    return *result;
}

template <typename T>
multi_array<T> & operator~ (multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance()->temp(rhs);

    Runtime::instance()->enqueue((bh_opcode)BH_INVERT, *result, rhs);

    return *result;
}

}
#endif
