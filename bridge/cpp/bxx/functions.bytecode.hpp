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

#ifndef __BOHRIUM_BRIDGE_CPP_BYTECODE_FUNCTIONS
#define __BOHRIUM_BRIDGE_CPP_BYTECODE_FUNCTIONS
#include "bh.h"

namespace bxx {

template <typename T>
multi_array<T>& bh_identity (multi_array<T> res, multi_array<T> &rhs)
{
    multi_array<T>* right = &rhs;

    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_IDENTITY, res, *right);
    return res;
}
template <typename T>
multi_array<T>& bh_add (multi_array<T>& res, multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(lhs, rhs)) {
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }

    // Check that broadcast is usefull
    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }

    Runtime::instance().enqueue((bh_opcode)BH_ADD, res, *left, *right);

    return res;
}

template <typename T>
multi_array<T>& bh_add (multi_array<T>& res, multi_array<T> &lhs, const T &rhs)
{
    multi_array<T>* left = &lhs;

    if (!same_shape(res, *left)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_ADD, res, *left, rhs);
    return res;
}

template <typename T>
multi_array<T>& bh_add (multi_array<T>& res, const T &lhs, multi_array<T> &rhs)
{
    multi_array<T>* right = &rhs;

    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_ADD, res, lhs, *right);
    return res;
}

template <typename T>
multi_array<T>& bh_subtract (multi_array<T>& res, multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(lhs, rhs)) {
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }

    // Check that broadcast is usefull
    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }

    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, res, *left, *right);

    return res;
}

template <typename T>
multi_array<T>& bh_subtract (multi_array<T>& res, multi_array<T> &lhs, const T &rhs)
{
    multi_array<T>* left = &lhs;

    if (!same_shape(res, *left)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, res, *left, rhs);
    return res;
}

template <typename T>
multi_array<T>& bh_subtract (multi_array<T>& res, const T &lhs, multi_array<T> &rhs)
{
    multi_array<T>* right = &rhs;

    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, res, lhs, *right);
    return res;
}

template <typename T>
multi_array<T>& bh_multiply (multi_array<T>& res, multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(lhs, rhs)) {
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }

    // Check that broadcast is usefull
    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }

    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, res, *left, *right);

    return res;
}

template <typename T>
multi_array<T>& bh_multiply (multi_array<T>& res, multi_array<T> &lhs, const T &rhs)
{
    multi_array<T>* left = &lhs;

    if (!same_shape(res, *left)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, res, *left, rhs);
    return res;
}

template <typename T>
multi_array<T>& bh_multiply (multi_array<T>& res, const T &lhs, multi_array<T> &rhs)
{
    multi_array<T>* right = &rhs;

    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, res, lhs, *right);
    return res;
}

template <typename T>
multi_array<T>& bh_divide (multi_array<T>& res, multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(lhs, rhs)) {
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }

    // Check that broadcast is usefull
    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }

    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE, res, *left, *right);

    return res;
}

template <typename T>
multi_array<T>& bh_divide (multi_array<T>& res, multi_array<T> &lhs, const T &rhs)
{
    multi_array<T>* left = &lhs;

    if (!same_shape(res, *left)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE, res, *left, rhs);
    return res;
}

template <typename T>
multi_array<T>& bh_divide (multi_array<T>& res, const T &lhs, multi_array<T> &rhs)
{
    multi_array<T>* right = &rhs;

    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE, res, lhs, *right);
    return res;
}

template <typename T>
multi_array<T>& bh_mod (multi_array<T>& res, multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(lhs, rhs)) {
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }

    // Check that broadcast is usefull
    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }

    Runtime::instance().enqueue((bh_opcode)BH_MOD, res, *left, *right);

    return res;
}

template <typename T>
multi_array<T>& bh_mod (multi_array<T>& res, multi_array<T> &lhs, const T &rhs)
{
    multi_array<T>* left = &lhs;

    if (!same_shape(res, *left)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_MOD, res, *left, rhs);
    return res;
}

template <typename T>
multi_array<T>& bh_mod (multi_array<T>& res, const T &lhs, multi_array<T> &rhs)
{
    multi_array<T>* right = &rhs;

    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_MOD, res, lhs, *right);
    return res;
}

template <typename T>
multi_array<T>& bh_bitwise_and (multi_array<T>& res, multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(lhs, rhs)) {
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }

    // Check that broadcast is usefull
    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_AND, res, *left, *right);

    return res;
}

template <typename T>
multi_array<T>& bh_bitwise_and (multi_array<T>& res, multi_array<T> &lhs, const T &rhs)
{
    multi_array<T>* left = &lhs;

    if (!same_shape(res, *left)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_AND, res, *left, rhs);
    return res;
}

template <typename T>
multi_array<T>& bh_bitwise_and (multi_array<T>& res, const T &lhs, multi_array<T> &rhs)
{
    multi_array<T>* right = &rhs;

    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_AND, res, lhs, *right);
    return res;
}

template <typename T>
multi_array<T>& bh_bitwise_or (multi_array<T>& res, multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(lhs, rhs)) {
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }

    // Check that broadcast is usefull
    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_OR, res, *left, *right);

    return res;
}

template <typename T>
multi_array<T>& bh_bitwise_or (multi_array<T>& res, multi_array<T> &lhs, const T &rhs)
{
    multi_array<T>* left = &lhs;

    if (!same_shape(res, *left)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_OR, res, *left, rhs);
    return res;
}

template <typename T>
multi_array<T>& bh_bitwise_or (multi_array<T>& res, const T &lhs, multi_array<T> &rhs)
{
    multi_array<T>* right = &rhs;

    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_OR, res, lhs, *right);
    return res;
}

template <typename T>
multi_array<T>& bh_bitwise_xor (multi_array<T>& res, multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(lhs, rhs)) {
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }

    // Check that broadcast is usefull
    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_XOR, res, *left, *right);

    return res;
}

template <typename T>
multi_array<T>& bh_bitwise_xor (multi_array<T>& res, multi_array<T> &lhs, const T &rhs)
{
    multi_array<T>* left = &lhs;

    if (!same_shape(res, *left)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_XOR, res, *left, rhs);
    return res;
}

template <typename T>
multi_array<T>& bh_bitwise_xor (multi_array<T>& res, const T &lhs, multi_array<T> &rhs)
{
    multi_array<T>* right = &rhs;

    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_XOR, res, lhs, *right);
    return res;
}

template <typename T>
multi_array<T>& bh_left_shift (multi_array<T>& res, multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(lhs, rhs)) {
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }

    // Check that broadcast is usefull
    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }

    Runtime::instance().enqueue((bh_opcode)BH_LEFT_SHIFT, res, *left, *right);

    return res;
}

template <typename T>
multi_array<T>& bh_left_shift (multi_array<T>& res, multi_array<T> &lhs, const T &rhs)
{
    multi_array<T>* left = &lhs;

    if (!same_shape(res, *left)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_LEFT_SHIFT, res, *left, rhs);
    return res;
}

template <typename T>
multi_array<T>& bh_left_shift (multi_array<T>& res, const T &lhs, multi_array<T> &rhs)
{
    multi_array<T>* right = &rhs;

    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_LEFT_SHIFT, res, lhs, *right);
    return res;
}

template <typename T>
multi_array<T>& bh_right_shift (multi_array<T>& res, multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(lhs, rhs)) {
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }

    // Check that broadcast is usefull
    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }

    Runtime::instance().enqueue((bh_opcode)BH_RIGHT_SHIFT, res, *left, *right);

    return res;
}

template <typename T>
multi_array<T>& bh_right_shift (multi_array<T>& res, multi_array<T> &lhs, const T &rhs)
{
    multi_array<T>* left = &lhs;

    if (!same_shape(res, *left)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_RIGHT_SHIFT, res, *left, rhs);
    return res;
}

template <typename T>
multi_array<T>& bh_right_shift (multi_array<T>& res, const T &lhs, multi_array<T> &rhs)
{
    multi_array<T>* right = &rhs;

    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_RIGHT_SHIFT, res, lhs, *right);
    return res;
}

template <typename T>
multi_array<T>& bh_equal (multi_array<T>& res, multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(lhs, rhs)) {
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }

    // Check that broadcast is usefull
    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }

    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, res, *left, *right);

    return res;
}

template <typename T>
multi_array<T>& bh_equal (multi_array<T>& res, multi_array<T> &lhs, const T &rhs)
{
    multi_array<T>* left = &lhs;

    if (!same_shape(res, *left)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, res, *left, rhs);
    return res;
}

template <typename T>
multi_array<T>& bh_equal (multi_array<T>& res, const T &lhs, multi_array<T> &rhs)
{
    multi_array<T>* right = &rhs;

    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, res, lhs, *right);
    return res;
}

template <typename T>
multi_array<T>& bh_not_equal (multi_array<T>& res, multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(lhs, rhs)) {
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }

    // Check that broadcast is usefull
    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }

    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, res, *left, *right);

    return res;
}

template <typename T>
multi_array<T>& bh_not_equal (multi_array<T>& res, multi_array<T> &lhs, const T &rhs)
{
    multi_array<T>* left = &lhs;

    if (!same_shape(res, *left)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, res, *left, rhs);
    return res;
}

template <typename T>
multi_array<T>& bh_not_equal (multi_array<T>& res, const T &lhs, multi_array<T> &rhs)
{
    multi_array<T>* right = &rhs;

    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, res, lhs, *right);
    return res;
}

template <typename T>
multi_array<T>& bh_greater (multi_array<T>& res, multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(lhs, rhs)) {
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }

    // Check that broadcast is usefull
    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }

    Runtime::instance().enqueue((bh_opcode)BH_GREATER, res, *left, *right);

    return res;
}

template <typename T>
multi_array<T>& bh_greater (multi_array<T>& res, multi_array<T> &lhs, const T &rhs)
{
    multi_array<T>* left = &lhs;

    if (!same_shape(res, *left)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_GREATER, res, *left, rhs);
    return res;
}

template <typename T>
multi_array<T>& bh_greater (multi_array<T>& res, const T &lhs, multi_array<T> &rhs)
{
    multi_array<T>* right = &rhs;

    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_GREATER, res, lhs, *right);
    return res;
}

template <typename T>
multi_array<T>& greater_equal (multi_array<T>& res, multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(lhs, rhs)) {
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }

    // Check that broadcast is usefull
    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }

    Runtime::instance().enqueue((bh_opcode)BH_GREATER_EQUAL, res, *left, *right);

    return res;
}

template <typename T>
multi_array<T>& greater_equal (multi_array<T>& res, multi_array<T> &lhs, const T &rhs)
{
    multi_array<T>* left = &lhs;

    if (!same_shape(res, *left)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_GREATER_EQUAL, res, *left, rhs);
    return res;
}

template <typename T>
multi_array<T>& greater_equal (multi_array<T>& res, const T &lhs, multi_array<T> &rhs)
{
    multi_array<T>* right = &rhs;

    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_GREATER_EQUAL, res, lhs, *right);
    return res;
}

template <typename T>
multi_array<T>& bh_less (multi_array<T>& res, multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(lhs, rhs)) {
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }

    // Check that broadcast is usefull
    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }

    Runtime::instance().enqueue((bh_opcode)BH_LESS, res, *left, *right);

    return res;
}

template <typename T>
multi_array<T>& bh_less (multi_array<T>& res, multi_array<T> &lhs, const T &rhs)
{
    multi_array<T>* left = &lhs;

    if (!same_shape(res, *left)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_LESS, res, *left, rhs);
    return res;
}

template <typename T>
multi_array<T>& bh_less (multi_array<T>& res, const T &lhs, multi_array<T> &rhs)
{
    multi_array<T>* right = &rhs;

    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_LESS, res, lhs, *right);
    return res;
}

template <typename T>
multi_array<T>& less_equal (multi_array<T>& res, multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(lhs, rhs)) {
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }

    // Check that broadcast is usefull
    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }

    Runtime::instance().enqueue((bh_opcode)BH_LESS_EQUAL, res, *left, *right);

    return res;
}

template <typename T>
multi_array<T>& less_equal (multi_array<T>& res, multi_array<T> &lhs, const T &rhs)
{
    multi_array<T>* left = &lhs;

    if (!same_shape(res, *left)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_LESS_EQUAL, res, *left, rhs);
    return res;
}

template <typename T>
multi_array<T>& less_equal (multi_array<T>& res, const T &lhs, multi_array<T> &rhs)
{
    multi_array<T>* right = &rhs;

    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_LESS_EQUAL, res, lhs, *right);
    return res;
}

template <typename T>
multi_array<T>& bh_logical_and (multi_array<T>& res, multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(lhs, rhs)) {
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }

    // Check that broadcast is usefull
    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_AND, res, *left, *right);

    return res;
}

template <typename T>
multi_array<T>& bh_logical_and (multi_array<T>& res, multi_array<T> &lhs, const T &rhs)
{
    multi_array<T>* left = &lhs;

    if (!same_shape(res, *left)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_AND, res, *left, rhs);
    return res;
}

template <typename T>
multi_array<T>& bh_logical_and (multi_array<T>& res, const T &lhs, multi_array<T> &rhs)
{
    multi_array<T>* right = &rhs;

    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_AND, res, lhs, *right);
    return res;
}

template <typename T>
multi_array<T>& bh_logical_or (multi_array<T>& res, multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(lhs, rhs)) {
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }

    // Check that broadcast is usefull
    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_OR, res, *left, *right);

    return res;
}

template <typename T>
multi_array<T>& bh_logical_or (multi_array<T>& res, multi_array<T> &lhs, const T &rhs)
{
    multi_array<T>* left = &lhs;

    if (!same_shape(res, *left)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_OR, res, *left, rhs);
    return res;
}

template <typename T>
multi_array<T>& bh_logical_or (multi_array<T>& res, const T &lhs, multi_array<T> &rhs)
{
    multi_array<T>* right = &rhs;

    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_OR, res, lhs, *right);
    return res;
}

template <typename T>
multi_array<T>& bh_logical_xor (multi_array<T>& res, multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(lhs, rhs)) {
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }

    // Check that broadcast is usefull
    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_XOR, res, *left, *right);

    return res;
}

template <typename T>
multi_array<T>& bh_logical_xor (multi_array<T>& res, multi_array<T> &lhs, const T &rhs)
{
    multi_array<T>* left = &lhs;

    if (!same_shape(res, *left)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_XOR, res, *left, rhs);
    return res;
}

template <typename T>
multi_array<T>& bh_logical_xor (multi_array<T>& res, const T &lhs, multi_array<T> &rhs)
{
    multi_array<T>* right = &rhs;

    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_XOR, res, lhs, *right);
    return res;
}

template <typename T>
multi_array<T>& bh_logical_not (multi_array<T> res, multi_array<T> &rhs)
{
    multi_array<T>* right = &rhs;

    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_NOT, res, *right);
    return res;
}
template <typename T>
multi_array<T>& bh_invert (multi_array<T> res, multi_array<T> &rhs)
{
    multi_array<T>* right = &rhs;

    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_INVERT, res, *right);
    return res;
}
template <typename T>
multi_array<T>& bh_power (multi_array<T>& res, multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(lhs, rhs)) {
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }

    // Check that broadcast is usefull
    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }

    Runtime::instance().enqueue((bh_opcode)BH_POWER, res, *left, *right);

    return res;
}

template <typename T>
multi_array<T>& bh_power (multi_array<T>& res, multi_array<T> &lhs, const T &rhs)
{
    multi_array<T>* left = &lhs;

    if (!same_shape(res, *left)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_POWER, res, *left, rhs);
    return res;
}

template <typename T>
multi_array<T>& bh_power (multi_array<T>& res, const T &lhs, multi_array<T> &rhs)
{
    multi_array<T>* right = &rhs;

    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_POWER, res, lhs, *right);
    return res;
}

template <typename T>
multi_array<T>& bh_absolute (multi_array<T> res, multi_array<T> &rhs)
{
    multi_array<T>* right = &rhs;

    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_ABSOLUTE, res, *right);
    return res;
}
template <typename T>
multi_array<T>& bh_maximum (multi_array<T>& res, multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(lhs, rhs)) {
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }

    // Check that broadcast is usefull
    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }

    Runtime::instance().enqueue((bh_opcode)BH_MAXIMUM, res, *left, *right);

    return res;
}

template <typename T>
multi_array<T>& bh_maximum (multi_array<T>& res, multi_array<T> &lhs, const T &rhs)
{
    multi_array<T>* left = &lhs;

    if (!same_shape(res, *left)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_MAXIMUM, res, *left, rhs);
    return res;
}

template <typename T>
multi_array<T>& bh_maximum (multi_array<T>& res, const T &lhs, multi_array<T> &rhs)
{
    multi_array<T>* right = &rhs;

    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_MAXIMUM, res, lhs, *right);
    return res;
}

template <typename T>
multi_array<T>& bh_minimum (multi_array<T>& res, multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(lhs, rhs)) {
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }

    // Check that broadcast is usefull
    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }

    Runtime::instance().enqueue((bh_opcode)BH_MINIMUM, res, *left, *right);

    return res;
}

template <typename T>
multi_array<T>& bh_minimum (multi_array<T>& res, multi_array<T> &lhs, const T &rhs)
{
    multi_array<T>* left = &lhs;

    if (!same_shape(res, *left)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_MINIMUM, res, *left, rhs);
    return res;
}

template <typename T>
multi_array<T>& bh_minimum (multi_array<T>& res, const T &lhs, multi_array<T> &rhs)
{
    multi_array<T>* right = &rhs;

    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_MINIMUM, res, lhs, *right);
    return res;
}

template <typename T>
multi_array<T>& bh_sin (multi_array<T> res, multi_array<T> &rhs)
{
    multi_array<T>* right = &rhs;

    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_SIN, res, *right);
    return res;
}
template <typename T>
multi_array<T>& bh_cos (multi_array<T> res, multi_array<T> &rhs)
{
    multi_array<T>* right = &rhs;

    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_COS, res, *right);
    return res;
}
template <typename T>
multi_array<T>& bh_tan (multi_array<T> res, multi_array<T> &rhs)
{
    multi_array<T>* right = &rhs;

    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_TAN, res, *right);
    return res;
}
template <typename T>
multi_array<T>& bh_sinh (multi_array<T> res, multi_array<T> &rhs)
{
    multi_array<T>* right = &rhs;

    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_SINH, res, *right);
    return res;
}
template <typename T>
multi_array<T>& bh_cosh (multi_array<T> res, multi_array<T> &rhs)
{
    multi_array<T>* right = &rhs;

    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_COSH, res, *right);
    return res;
}
template <typename T>
multi_array<T>& bh_tanh (multi_array<T> res, multi_array<T> &rhs)
{
    multi_array<T>* right = &rhs;

    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_TANH, res, *right);
    return res;
}
template <typename T>
multi_array<T>& bh_asin (multi_array<T> res, multi_array<T> &rhs)
{
    multi_array<T>* right = &rhs;

    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_ARCSIN, res, *right);
    return res;
}
template <typename T>
multi_array<T>& bh_acos (multi_array<T> res, multi_array<T> &rhs)
{
    multi_array<T>* right = &rhs;

    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_ARCCOS, res, *right);
    return res;
}
template <typename T>
multi_array<T>& bh_atan (multi_array<T> res, multi_array<T> &rhs)
{
    multi_array<T>* right = &rhs;

    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_ARCTAN, res, *right);
    return res;
}
template <typename T>
multi_array<T>& bh_atan2 (multi_array<T>& res, multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(lhs, rhs)) {
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(*left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(*right, *left)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        }
    }

    // Check that broadcast is usefull
    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }

    Runtime::instance().enqueue((bh_opcode)BH_ARCTAN2, res, *left, *right);

    return res;
}

template <typename T>
multi_array<T>& bh_atan2 (multi_array<T>& res, multi_array<T> &lhs, const T &rhs)
{
    multi_array<T>* left = &lhs;

    if (!same_shape(res, *left)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_ARCTAN2, res, *left, rhs);
    return res;
}

template <typename T>
multi_array<T>& bh_atan2 (multi_array<T>& res, const T &lhs, multi_array<T> &rhs)
{
    multi_array<T>* right = &rhs;

    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_ARCTAN2, res, lhs, *right);
    return res;
}

template <typename T>
multi_array<T>& bh_asinh (multi_array<T> res, multi_array<T> &rhs)
{
    multi_array<T>* right = &rhs;

    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_ARCSINH, res, *right);
    return res;
}
template <typename T>
multi_array<T>& bh_acosh (multi_array<T> res, multi_array<T> &rhs)
{
    multi_array<T>* right = &rhs;

    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_ARCCOSH, res, *right);
    return res;
}
template <typename T>
multi_array<T>& bh_atanh (multi_array<T> res, multi_array<T> &rhs)
{
    multi_array<T>* right = &rhs;

    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_ARCTANH, res, *right);
    return res;
}
template <typename T>
multi_array<T>& bh_exp (multi_array<T> res, multi_array<T> &rhs)
{
    multi_array<T>* right = &rhs;

    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_EXP, res, *right);
    return res;
}
template <typename T>
multi_array<T>& bh_exp2 (multi_array<T> res, multi_array<T> &rhs)
{
    multi_array<T>* right = &rhs;

    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_EXP2, res, *right);
    return res;
}
template <typename T>
multi_array<T>& bh_expm1 (multi_array<T> res, multi_array<T> &rhs)
{
    multi_array<T>* right = &rhs;

    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_EXPM1, res, *right);
    return res;
}
template <typename T>
multi_array<T>& bh_isnan (multi_array<T> res, multi_array<T> &rhs)
{
    multi_array<T>* right = &rhs;

    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_ISNAN, res, *right);
    return res;
}
template <typename T>
multi_array<T>& bh_isinf (multi_array<T> res, multi_array<T> &rhs)
{
    multi_array<T>* right = &rhs;

    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_ISINF, res, *right);
    return res;
}
template <typename T>
multi_array<T>& bh_log (multi_array<T> res, multi_array<T> &rhs)
{
    multi_array<T>* right = &rhs;

    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_LOG, res, *right);
    return res;
}
template <typename T>
multi_array<T>& bh_log2 (multi_array<T> res, multi_array<T> &rhs)
{
    multi_array<T>* right = &rhs;

    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_LOG2, res, *right);
    return res;
}
template <typename T>
multi_array<T>& bh_log10 (multi_array<T> res, multi_array<T> &rhs)
{
    multi_array<T>* right = &rhs;

    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_LOG10, res, *right);
    return res;
}
template <typename T>
multi_array<T>& bh_log1p (multi_array<T> res, multi_array<T> &rhs)
{
    multi_array<T>* right = &rhs;

    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_LOG1P, res, *right);
    return res;
}
template <typename T>
multi_array<T>& bh_sqrt (multi_array<T> res, multi_array<T> &rhs)
{
    multi_array<T>* right = &rhs;

    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_SQRT, res, *right);
    return res;
}
template <typename T>
multi_array<T>& bh_ceil (multi_array<T> res, multi_array<T> &rhs)
{
    multi_array<T>* right = &rhs;

    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_CEIL, res, *right);
    return res;
}
template <typename T>
multi_array<T>& bh_trunc (multi_array<T> res, multi_array<T> &rhs)
{
    multi_array<T>* right = &rhs;

    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_TRUNC, res, *right);
    return res;
}
template <typename T>
multi_array<T>& bh_floor (multi_array<T> res, multi_array<T> &rhs)
{
    multi_array<T>* right = &rhs;

    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_FLOOR, res, *right);
    return res;
}
template <typename T>
multi_array<T>& bh_rint (multi_array<T> res, multi_array<T> &rhs)
{
    multi_array<T>* right = &rhs;

    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_RINT, res, *right);
    return res;
}

// TODO: Fix this hack-slash support for BH_REAL/IMAG
template <typename InT, typename OutT>
multi_array<OutT>& bh_real (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    multi_array<InT>* right = &rhs;

    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_REAL, res, *right);
    return res;
}
template <typename InT, typename OutT>
multi_array<OutT>& bh_imag (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    multi_array<InT>* right = &rhs;

    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes.");
    }
    Runtime::instance().enqueue((bh_opcode)BH_IMAG, res, *right);
    return res;
}

}
#endif

