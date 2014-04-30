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

#ifndef __BOHRIUM_BRIDGE_CPP_RUNTIME_OPERATIONS
#define __BOHRIUM_BRIDGE_CPP_RUNTIME_OPERATIONS
#include "bh.h"

namespace bxx {


// Explicit result array
template <typename OutT, typename InT>
multi_array<OutT>& bh_add (multi_array<OutT>& res, multi_array<InT>& lhs, multi_array<InT>& rhs)
{
    multi_array<InT>* left    = &lhs;
    multi_array<InT>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(*left, *right)) {
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

    // Check that operands are compatible with the output
    // TODO: Broadcasting should also be done in relation to output
    //       for now we simply fail...
    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes of output and input.");
    }

    Runtime::instance().enqueue((bh_opcode)BH_ADD, res, *left, *right);

    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_add (multi_array<OutT>& res, multi_array<InT>& lhs, const InT rhs)
{
    multi_array<InT>* left = &lhs;

    // Check for unbroadcastable situation
    if (res.getRank() < left->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_add: " << res.getRank() << ", " << left->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    // Broadcast
    if (!same_shape(res, *left)) {
        left = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *left)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *left)) {
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_ADD, res, *left, rhs);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_add (multi_array<OutT>& res, const InT lhs, multi_array<InT>& rhs)
{
    multi_array<InT>* right = &rhs;

    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_add: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    // Broadcast
    if (!same_shape(res, *right)) {
        right = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *right)) {
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_ADD, res, lhs, *right);
    return res;
}

//
// Implicit temporary result array
//
template <typename T>
multi_array<T>& bh_add (multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(*left, *right)) {
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

    // Construct output / result array
    multi_array<T>* result = &Runtime::instance().temp<T, T>(*left);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_ADD, *result, *left, *right);
    return *result;
}

template <typename T>
multi_array<T>& bh_add (multi_array<T>& lhs, const T rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T, T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_ADD, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& bh_add (const T lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T, T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_ADD, *result, lhs, rhs);
    return *result;
}

// Explicit result array
template <typename OutT, typename InT>
multi_array<OutT>& bh_subtract (multi_array<OutT>& res, multi_array<InT>& lhs, multi_array<InT>& rhs)
{
    multi_array<InT>* left    = &lhs;
    multi_array<InT>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(*left, *right)) {
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

    // Check that operands are compatible with the output
    // TODO: Broadcasting should also be done in relation to output
    //       for now we simply fail...
    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes of output and input.");
    }

    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, res, *left, *right);

    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_subtract (multi_array<OutT>& res, multi_array<InT>& lhs, const InT rhs)
{
    multi_array<InT>* left = &lhs;

    // Check for unbroadcastable situation
    if (res.getRank() < left->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_subtract: " << res.getRank() << ", " << left->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    // Broadcast
    if (!same_shape(res, *left)) {
        left = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *left)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *left)) {
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, res, *left, rhs);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_subtract (multi_array<OutT>& res, const InT lhs, multi_array<InT>& rhs)
{
    multi_array<InT>* right = &rhs;

    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_subtract: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    // Broadcast
    if (!same_shape(res, *right)) {
        right = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *right)) {
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, res, lhs, *right);
    return res;
}

//
// Implicit temporary result array
//
template <typename T>
multi_array<T>& bh_subtract (multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(*left, *right)) {
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

    // Construct output / result array
    multi_array<T>* result = &Runtime::instance().temp<T, T>(*left);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, *result, *left, *right);
    return *result;
}

template <typename T>
multi_array<T>& bh_subtract (multi_array<T>& lhs, const T rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T, T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& bh_subtract (const T lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T, T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, *result, lhs, rhs);
    return *result;
}

// Explicit result array
template <typename OutT, typename InT>
multi_array<OutT>& bh_multiply (multi_array<OutT>& res, multi_array<InT>& lhs, multi_array<InT>& rhs)
{
    multi_array<InT>* left    = &lhs;
    multi_array<InT>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(*left, *right)) {
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

    // Check that operands are compatible with the output
    // TODO: Broadcasting should also be done in relation to output
    //       for now we simply fail...
    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes of output and input.");
    }

    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, res, *left, *right);

    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_multiply (multi_array<OutT>& res, multi_array<InT>& lhs, const InT rhs)
{
    multi_array<InT>* left = &lhs;

    // Check for unbroadcastable situation
    if (res.getRank() < left->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_multiply: " << res.getRank() << ", " << left->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    // Broadcast
    if (!same_shape(res, *left)) {
        left = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *left)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *left)) {
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, res, *left, rhs);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_multiply (multi_array<OutT>& res, const InT lhs, multi_array<InT>& rhs)
{
    multi_array<InT>* right = &rhs;

    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_multiply: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    // Broadcast
    if (!same_shape(res, *right)) {
        right = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *right)) {
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, res, lhs, *right);
    return res;
}

//
// Implicit temporary result array
//
template <typename T>
multi_array<T>& bh_multiply (multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(*left, *right)) {
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

    // Construct output / result array
    multi_array<T>* result = &Runtime::instance().temp<T, T>(*left);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, *result, *left, *right);
    return *result;
}

template <typename T>
multi_array<T>& bh_multiply (multi_array<T>& lhs, const T rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T, T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& bh_multiply (const T lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T, T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, *result, lhs, rhs);
    return *result;
}

// Explicit result array
template <typename OutT, typename InT>
multi_array<OutT>& bh_divide (multi_array<OutT>& res, multi_array<InT>& lhs, multi_array<InT>& rhs)
{
    multi_array<InT>* left    = &lhs;
    multi_array<InT>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(*left, *right)) {
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

    // Check that operands are compatible with the output
    // TODO: Broadcasting should also be done in relation to output
    //       for now we simply fail...
    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes of output and input.");
    }

    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE, res, *left, *right);

    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_divide (multi_array<OutT>& res, multi_array<InT>& lhs, const InT rhs)
{
    multi_array<InT>* left = &lhs;

    // Check for unbroadcastable situation
    if (res.getRank() < left->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_divide: " << res.getRank() << ", " << left->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    // Broadcast
    if (!same_shape(res, *left)) {
        left = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *left)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *left)) {
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE, res, *left, rhs);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_divide (multi_array<OutT>& res, const InT lhs, multi_array<InT>& rhs)
{
    multi_array<InT>* right = &rhs;

    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_divide: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    // Broadcast
    if (!same_shape(res, *right)) {
        right = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *right)) {
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE, res, lhs, *right);
    return res;
}

//
// Implicit temporary result array
//
template <typename T>
multi_array<T>& bh_divide (multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(*left, *right)) {
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

    // Construct output / result array
    multi_array<T>* result = &Runtime::instance().temp<T, T>(*left);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE, *result, *left, *right);
    return *result;
}

template <typename T>
multi_array<T>& bh_divide (multi_array<T>& lhs, const T rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T, T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& bh_divide (const T lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T, T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE, *result, lhs, rhs);
    return *result;
}

// Explicit result array
template <typename OutT, typename InT>
multi_array<OutT>& bh_mod (multi_array<OutT>& res, multi_array<InT>& lhs, multi_array<InT>& rhs)
{
    multi_array<InT>* left    = &lhs;
    multi_array<InT>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(*left, *right)) {
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

    // Check that operands are compatible with the output
    // TODO: Broadcasting should also be done in relation to output
    //       for now we simply fail...
    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes of output and input.");
    }

    Runtime::instance().enqueue((bh_opcode)BH_MOD, res, *left, *right);

    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_mod (multi_array<OutT>& res, multi_array<InT>& lhs, const InT rhs)
{
    multi_array<InT>* left = &lhs;

    // Check for unbroadcastable situation
    if (res.getRank() < left->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_mod: " << res.getRank() << ", " << left->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    // Broadcast
    if (!same_shape(res, *left)) {
        left = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *left)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *left)) {
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_MOD, res, *left, rhs);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_mod (multi_array<OutT>& res, const InT lhs, multi_array<InT>& rhs)
{
    multi_array<InT>* right = &rhs;

    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_mod: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    // Broadcast
    if (!same_shape(res, *right)) {
        right = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *right)) {
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_MOD, res, lhs, *right);
    return res;
}

//
// Implicit temporary result array
//
template <typename T>
multi_array<T>& bh_mod (multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(*left, *right)) {
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

    // Construct output / result array
    multi_array<T>* result = &Runtime::instance().temp<T, T>(*left);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_MOD, *result, *left, *right);
    return *result;
}

template <typename T>
multi_array<T>& bh_mod (multi_array<T>& lhs, const T rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T, T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_MOD, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& bh_mod (const T lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T, T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_MOD, *result, lhs, rhs);
    return *result;
}

// Explicit result array
template <typename OutT, typename InT>
multi_array<OutT>& bh_bitwise_and (multi_array<OutT>& res, multi_array<InT>& lhs, multi_array<InT>& rhs)
{
    multi_array<InT>* left    = &lhs;
    multi_array<InT>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(*left, *right)) {
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

    // Check that operands are compatible with the output
    // TODO: Broadcasting should also be done in relation to output
    //       for now we simply fail...
    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes of output and input.");
    }

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_AND, res, *left, *right);

    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_bitwise_and (multi_array<OutT>& res, multi_array<InT>& lhs, const InT rhs)
{
    multi_array<InT>* left = &lhs;

    // Check for unbroadcastable situation
    if (res.getRank() < left->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_bitwise_and: " << res.getRank() << ", " << left->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    // Broadcast
    if (!same_shape(res, *left)) {
        left = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *left)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *left)) {
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_AND, res, *left, rhs);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_bitwise_and (multi_array<OutT>& res, const InT lhs, multi_array<InT>& rhs)
{
    multi_array<InT>* right = &rhs;

    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_bitwise_and: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    // Broadcast
    if (!same_shape(res, *right)) {
        right = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *right)) {
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_AND, res, lhs, *right);
    return res;
}

//
// Implicit temporary result array
//
template <typename T>
multi_array<T>& bh_bitwise_and (multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(*left, *right)) {
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

    // Construct output / result array
    multi_array<T>* result = &Runtime::instance().temp<T, T>(*left);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_AND, *result, *left, *right);
    return *result;
}

template <typename T>
multi_array<T>& bh_bitwise_and (multi_array<T>& lhs, const T rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T, T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_AND, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& bh_bitwise_and (const T lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T, T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_AND, *result, lhs, rhs);
    return *result;
}

// Explicit result array
template <typename OutT, typename InT>
multi_array<OutT>& bh_bitwise_or (multi_array<OutT>& res, multi_array<InT>& lhs, multi_array<InT>& rhs)
{
    multi_array<InT>* left    = &lhs;
    multi_array<InT>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(*left, *right)) {
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

    // Check that operands are compatible with the output
    // TODO: Broadcasting should also be done in relation to output
    //       for now we simply fail...
    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes of output and input.");
    }

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_OR, res, *left, *right);

    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_bitwise_or (multi_array<OutT>& res, multi_array<InT>& lhs, const InT rhs)
{
    multi_array<InT>* left = &lhs;

    // Check for unbroadcastable situation
    if (res.getRank() < left->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_bitwise_or: " << res.getRank() << ", " << left->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    // Broadcast
    if (!same_shape(res, *left)) {
        left = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *left)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *left)) {
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_OR, res, *left, rhs);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_bitwise_or (multi_array<OutT>& res, const InT lhs, multi_array<InT>& rhs)
{
    multi_array<InT>* right = &rhs;

    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_bitwise_or: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    // Broadcast
    if (!same_shape(res, *right)) {
        right = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *right)) {
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_OR, res, lhs, *right);
    return res;
}

//
// Implicit temporary result array
//
template <typename T>
multi_array<T>& bh_bitwise_or (multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(*left, *right)) {
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

    // Construct output / result array
    multi_array<T>* result = &Runtime::instance().temp<T, T>(*left);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_OR, *result, *left, *right);
    return *result;
}

template <typename T>
multi_array<T>& bh_bitwise_or (multi_array<T>& lhs, const T rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T, T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_OR, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& bh_bitwise_or (const T lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T, T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_OR, *result, lhs, rhs);
    return *result;
}

// Explicit result array
template <typename OutT, typename InT>
multi_array<OutT>& bh_bitwise_xor (multi_array<OutT>& res, multi_array<InT>& lhs, multi_array<InT>& rhs)
{
    multi_array<InT>* left    = &lhs;
    multi_array<InT>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(*left, *right)) {
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

    // Check that operands are compatible with the output
    // TODO: Broadcasting should also be done in relation to output
    //       for now we simply fail...
    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes of output and input.");
    }

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_XOR, res, *left, *right);

    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_bitwise_xor (multi_array<OutT>& res, multi_array<InT>& lhs, const InT rhs)
{
    multi_array<InT>* left = &lhs;

    // Check for unbroadcastable situation
    if (res.getRank() < left->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_bitwise_xor: " << res.getRank() << ", " << left->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    // Broadcast
    if (!same_shape(res, *left)) {
        left = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *left)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *left)) {
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_XOR, res, *left, rhs);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_bitwise_xor (multi_array<OutT>& res, const InT lhs, multi_array<InT>& rhs)
{
    multi_array<InT>* right = &rhs;

    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_bitwise_xor: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    // Broadcast
    if (!same_shape(res, *right)) {
        right = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *right)) {
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_XOR, res, lhs, *right);
    return res;
}

//
// Implicit temporary result array
//
template <typename T>
multi_array<T>& bh_bitwise_xor (multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(*left, *right)) {
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

    // Construct output / result array
    multi_array<T>* result = &Runtime::instance().temp<T, T>(*left);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_XOR, *result, *left, *right);
    return *result;
}

template <typename T>
multi_array<T>& bh_bitwise_xor (multi_array<T>& lhs, const T rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T, T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_XOR, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& bh_bitwise_xor (const T lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T, T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_XOR, *result, lhs, rhs);
    return *result;
}

// Explicit result array
template <typename OutT, typename InT>
multi_array<OutT>& bh_left_shift (multi_array<OutT>& res, multi_array<InT>& lhs, multi_array<InT>& rhs)
{
    multi_array<InT>* left    = &lhs;
    multi_array<InT>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(*left, *right)) {
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

    // Check that operands are compatible with the output
    // TODO: Broadcasting should also be done in relation to output
    //       for now we simply fail...
    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes of output and input.");
    }

    Runtime::instance().enqueue((bh_opcode)BH_LEFT_SHIFT, res, *left, *right);

    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_left_shift (multi_array<OutT>& res, multi_array<InT>& lhs, const InT rhs)
{
    multi_array<InT>* left = &lhs;

    // Check for unbroadcastable situation
    if (res.getRank() < left->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_left_shift: " << res.getRank() << ", " << left->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    // Broadcast
    if (!same_shape(res, *left)) {
        left = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *left)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *left)) {
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_LEFT_SHIFT, res, *left, rhs);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_left_shift (multi_array<OutT>& res, const InT lhs, multi_array<InT>& rhs)
{
    multi_array<InT>* right = &rhs;

    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_left_shift: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    // Broadcast
    if (!same_shape(res, *right)) {
        right = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *right)) {
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_LEFT_SHIFT, res, lhs, *right);
    return res;
}

//
// Implicit temporary result array
//
template <typename T>
multi_array<T>& bh_left_shift (multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(*left, *right)) {
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

    // Construct output / result array
    multi_array<T>* result = &Runtime::instance().temp<T, T>(*left);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LEFT_SHIFT, *result, *left, *right);
    return *result;
}

template <typename T>
multi_array<T>& bh_left_shift (multi_array<T>& lhs, const T rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T, T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LEFT_SHIFT, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& bh_left_shift (const T lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T, T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LEFT_SHIFT, *result, lhs, rhs);
    return *result;
}

// Explicit result array
template <typename OutT, typename InT>
multi_array<OutT>& bh_right_shift (multi_array<OutT>& res, multi_array<InT>& lhs, multi_array<InT>& rhs)
{
    multi_array<InT>* left    = &lhs;
    multi_array<InT>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(*left, *right)) {
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

    // Check that operands are compatible with the output
    // TODO: Broadcasting should also be done in relation to output
    //       for now we simply fail...
    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes of output and input.");
    }

    Runtime::instance().enqueue((bh_opcode)BH_RIGHT_SHIFT, res, *left, *right);

    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_right_shift (multi_array<OutT>& res, multi_array<InT>& lhs, const InT rhs)
{
    multi_array<InT>* left = &lhs;

    // Check for unbroadcastable situation
    if (res.getRank() < left->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_right_shift: " << res.getRank() << ", " << left->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    // Broadcast
    if (!same_shape(res, *left)) {
        left = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *left)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *left)) {
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_RIGHT_SHIFT, res, *left, rhs);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_right_shift (multi_array<OutT>& res, const InT lhs, multi_array<InT>& rhs)
{
    multi_array<InT>* right = &rhs;

    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_right_shift: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    // Broadcast
    if (!same_shape(res, *right)) {
        right = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *right)) {
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_RIGHT_SHIFT, res, lhs, *right);
    return res;
}

//
// Implicit temporary result array
//
template <typename T>
multi_array<T>& bh_right_shift (multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(*left, *right)) {
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

    // Construct output / result array
    multi_array<T>* result = &Runtime::instance().temp<T, T>(*left);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_RIGHT_SHIFT, *result, *left, *right);
    return *result;
}

template <typename T>
multi_array<T>& bh_right_shift (multi_array<T>& lhs, const T rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T, T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_RIGHT_SHIFT, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& bh_right_shift (const T lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T, T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_RIGHT_SHIFT, *result, lhs, rhs);
    return *result;
}

// Explicit result array
template <typename OutT, typename InT>
multi_array<OutT>& bh_equal (multi_array<OutT>& res, multi_array<InT>& lhs, multi_array<InT>& rhs)
{
    multi_array<InT>* left    = &lhs;
    multi_array<InT>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(*left, *right)) {
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

    // Check that operands are compatible with the output
    // TODO: Broadcasting should also be done in relation to output
    //       for now we simply fail...
    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes of output and input.");
    }

    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, res, *left, *right);

    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_equal (multi_array<OutT>& res, multi_array<InT>& lhs, const InT rhs)
{
    multi_array<InT>* left = &lhs;

    // Check for unbroadcastable situation
    if (res.getRank() < left->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_equal: " << res.getRank() << ", " << left->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    // Broadcast
    if (!same_shape(res, *left)) {
        left = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *left)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *left)) {
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, res, *left, rhs);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_equal (multi_array<OutT>& res, const InT lhs, multi_array<InT>& rhs)
{
    multi_array<InT>* right = &rhs;

    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_equal: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    // Broadcast
    if (!same_shape(res, *right)) {
        right = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *right)) {
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, res, lhs, *right);
    return res;
}

//
// Implicit temporary result array
//
template <typename T>
multi_array<T>& bh_equal (multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(*left, *right)) {
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

    // Construct output / result array
    multi_array<T>* result = &Runtime::instance().temp<T, T>(*left);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, *result, *left, *right);
    return *result;
}

template <typename T>
multi_array<T>& bh_equal (multi_array<T>& lhs, const T rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T, T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& bh_equal (const T lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T, T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, *result, lhs, rhs);
    return *result;
}

// Explicit result array
template <typename OutT, typename InT>
multi_array<OutT>& bh_not_equal (multi_array<OutT>& res, multi_array<InT>& lhs, multi_array<InT>& rhs)
{
    multi_array<InT>* left    = &lhs;
    multi_array<InT>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(*left, *right)) {
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

    // Check that operands are compatible with the output
    // TODO: Broadcasting should also be done in relation to output
    //       for now we simply fail...
    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes of output and input.");
    }

    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, res, *left, *right);

    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_not_equal (multi_array<OutT>& res, multi_array<InT>& lhs, const InT rhs)
{
    multi_array<InT>* left = &lhs;

    // Check for unbroadcastable situation
    if (res.getRank() < left->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_not_equal: " << res.getRank() << ", " << left->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    // Broadcast
    if (!same_shape(res, *left)) {
        left = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *left)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *left)) {
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, res, *left, rhs);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_not_equal (multi_array<OutT>& res, const InT lhs, multi_array<InT>& rhs)
{
    multi_array<InT>* right = &rhs;

    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_not_equal: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    // Broadcast
    if (!same_shape(res, *right)) {
        right = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *right)) {
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, res, lhs, *right);
    return res;
}

//
// Implicit temporary result array
//
template <typename T>
multi_array<T>& bh_not_equal (multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(*left, *right)) {
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

    // Construct output / result array
    multi_array<T>* result = &Runtime::instance().temp<T, T>(*left);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, *result, *left, *right);
    return *result;
}

template <typename T>
multi_array<T>& bh_not_equal (multi_array<T>& lhs, const T rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T, T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& bh_not_equal (const T lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T, T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, *result, lhs, rhs);
    return *result;
}

// Explicit result array
template <typename OutT, typename InT>
multi_array<OutT>& bh_greater (multi_array<OutT>& res, multi_array<InT>& lhs, multi_array<InT>& rhs)
{
    multi_array<InT>* left    = &lhs;
    multi_array<InT>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(*left, *right)) {
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

    // Check that operands are compatible with the output
    // TODO: Broadcasting should also be done in relation to output
    //       for now we simply fail...
    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes of output and input.");
    }

    Runtime::instance().enqueue((bh_opcode)BH_GREATER, res, *left, *right);

    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_greater (multi_array<OutT>& res, multi_array<InT>& lhs, const InT rhs)
{
    multi_array<InT>* left = &lhs;

    // Check for unbroadcastable situation
    if (res.getRank() < left->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_greater: " << res.getRank() << ", " << left->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    // Broadcast
    if (!same_shape(res, *left)) {
        left = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *left)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *left)) {
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_GREATER, res, *left, rhs);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_greater (multi_array<OutT>& res, const InT lhs, multi_array<InT>& rhs)
{
    multi_array<InT>* right = &rhs;

    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_greater: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    // Broadcast
    if (!same_shape(res, *right)) {
        right = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *right)) {
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_GREATER, res, lhs, *right);
    return res;
}

//
// Implicit temporary result array
//
template <typename T>
multi_array<T>& bh_greater (multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(*left, *right)) {
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

    // Construct output / result array
    multi_array<T>* result = &Runtime::instance().temp<T, T>(*left);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_GREATER, *result, *left, *right);
    return *result;
}

template <typename T>
multi_array<T>& bh_greater (multi_array<T>& lhs, const T rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T, T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_GREATER, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& bh_greater (const T lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T, T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_GREATER, *result, lhs, rhs);
    return *result;
}

// Explicit result array
template <typename OutT, typename InT>
multi_array<OutT>& greater_equal (multi_array<OutT>& res, multi_array<InT>& lhs, multi_array<InT>& rhs)
{
    multi_array<InT>* left    = &lhs;
    multi_array<InT>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(*left, *right)) {
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

    // Check that operands are compatible with the output
    // TODO: Broadcasting should also be done in relation to output
    //       for now we simply fail...
    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes of output and input.");
    }

    Runtime::instance().enqueue((bh_opcode)BH_GREATER_EQUAL, res, *left, *right);

    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& greater_equal (multi_array<OutT>& res, multi_array<InT>& lhs, const InT rhs)
{
    multi_array<InT>* left = &lhs;

    // Check for unbroadcastable situation
    if (res.getRank() < left->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "greater_equal: " << res.getRank() << ", " << left->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    // Broadcast
    if (!same_shape(res, *left)) {
        left = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *left)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *left)) {
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_GREATER_EQUAL, res, *left, rhs);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& greater_equal (multi_array<OutT>& res, const InT lhs, multi_array<InT>& rhs)
{
    multi_array<InT>* right = &rhs;

    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "greater_equal: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    // Broadcast
    if (!same_shape(res, *right)) {
        right = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *right)) {
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_GREATER_EQUAL, res, lhs, *right);
    return res;
}

//
// Implicit temporary result array
//
template <typename T>
multi_array<T>& greater_equal (multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(*left, *right)) {
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

    // Construct output / result array
    multi_array<T>* result = &Runtime::instance().temp<T, T>(*left);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_GREATER_EQUAL, *result, *left, *right);
    return *result;
}

template <typename T>
multi_array<T>& greater_equal (multi_array<T>& lhs, const T rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T, T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_GREATER_EQUAL, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& greater_equal (const T lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T, T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_GREATER_EQUAL, *result, lhs, rhs);
    return *result;
}

// Explicit result array
template <typename OutT, typename InT>
multi_array<OutT>& bh_less (multi_array<OutT>& res, multi_array<InT>& lhs, multi_array<InT>& rhs)
{
    multi_array<InT>* left    = &lhs;
    multi_array<InT>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(*left, *right)) {
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

    // Check that operands are compatible with the output
    // TODO: Broadcasting should also be done in relation to output
    //       for now we simply fail...
    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes of output and input.");
    }

    Runtime::instance().enqueue((bh_opcode)BH_LESS, res, *left, *right);

    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_less (multi_array<OutT>& res, multi_array<InT>& lhs, const InT rhs)
{
    multi_array<InT>* left = &lhs;

    // Check for unbroadcastable situation
    if (res.getRank() < left->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_less: " << res.getRank() << ", " << left->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    // Broadcast
    if (!same_shape(res, *left)) {
        left = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *left)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *left)) {
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_LESS, res, *left, rhs);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_less (multi_array<OutT>& res, const InT lhs, multi_array<InT>& rhs)
{
    multi_array<InT>* right = &rhs;

    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_less: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    // Broadcast
    if (!same_shape(res, *right)) {
        right = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *right)) {
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_LESS, res, lhs, *right);
    return res;
}

//
// Implicit temporary result array
//
template <typename T>
multi_array<T>& bh_less (multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(*left, *right)) {
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

    // Construct output / result array
    multi_array<T>* result = &Runtime::instance().temp<T, T>(*left);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LESS, *result, *left, *right);
    return *result;
}

template <typename T>
multi_array<T>& bh_less (multi_array<T>& lhs, const T rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T, T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LESS, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& bh_less (const T lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T, T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LESS, *result, lhs, rhs);
    return *result;
}

// Explicit result array
template <typename OutT, typename InT>
multi_array<OutT>& less_equal (multi_array<OutT>& res, multi_array<InT>& lhs, multi_array<InT>& rhs)
{
    multi_array<InT>* left    = &lhs;
    multi_array<InT>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(*left, *right)) {
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

    // Check that operands are compatible with the output
    // TODO: Broadcasting should also be done in relation to output
    //       for now we simply fail...
    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes of output and input.");
    }

    Runtime::instance().enqueue((bh_opcode)BH_LESS_EQUAL, res, *left, *right);

    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& less_equal (multi_array<OutT>& res, multi_array<InT>& lhs, const InT rhs)
{
    multi_array<InT>* left = &lhs;

    // Check for unbroadcastable situation
    if (res.getRank() < left->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "less_equal: " << res.getRank() << ", " << left->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    // Broadcast
    if (!same_shape(res, *left)) {
        left = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *left)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *left)) {
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_LESS_EQUAL, res, *left, rhs);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& less_equal (multi_array<OutT>& res, const InT lhs, multi_array<InT>& rhs)
{
    multi_array<InT>* right = &rhs;

    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "less_equal: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    // Broadcast
    if (!same_shape(res, *right)) {
        right = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *right)) {
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_LESS_EQUAL, res, lhs, *right);
    return res;
}

//
// Implicit temporary result array
//
template <typename T>
multi_array<T>& less_equal (multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(*left, *right)) {
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

    // Construct output / result array
    multi_array<T>* result = &Runtime::instance().temp<T, T>(*left);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LESS_EQUAL, *result, *left, *right);
    return *result;
}

template <typename T>
multi_array<T>& less_equal (multi_array<T>& lhs, const T rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T, T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LESS_EQUAL, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& less_equal (const T lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T, T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LESS_EQUAL, *result, lhs, rhs);
    return *result;
}

// Explicit result array
template <typename OutT, typename InT>
multi_array<OutT>& bh_logical_and (multi_array<OutT>& res, multi_array<InT>& lhs, multi_array<InT>& rhs)
{
    multi_array<InT>* left    = &lhs;
    multi_array<InT>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(*left, *right)) {
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

    // Check that operands are compatible with the output
    // TODO: Broadcasting should also be done in relation to output
    //       for now we simply fail...
    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes of output and input.");
    }

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_AND, res, *left, *right);

    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_logical_and (multi_array<OutT>& res, multi_array<InT>& lhs, const InT rhs)
{
    multi_array<InT>* left = &lhs;

    // Check for unbroadcastable situation
    if (res.getRank() < left->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_logical_and: " << res.getRank() << ", " << left->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    // Broadcast
    if (!same_shape(res, *left)) {
        left = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *left)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *left)) {
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_AND, res, *left, rhs);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_logical_and (multi_array<OutT>& res, const InT lhs, multi_array<InT>& rhs)
{
    multi_array<InT>* right = &rhs;

    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_logical_and: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    // Broadcast
    if (!same_shape(res, *right)) {
        right = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *right)) {
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_AND, res, lhs, *right);
    return res;
}

//
// Implicit temporary result array
//
template <typename T>
multi_array<T>& bh_logical_and (multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(*left, *right)) {
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

    // Construct output / result array
    multi_array<T>* result = &Runtime::instance().temp<T, T>(*left);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_AND, *result, *left, *right);
    return *result;
}

template <typename T>
multi_array<T>& bh_logical_and (multi_array<T>& lhs, const T rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T, T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_AND, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& bh_logical_and (const T lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T, T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_AND, *result, lhs, rhs);
    return *result;
}

// Explicit result array
template <typename OutT, typename InT>
multi_array<OutT>& bh_logical_or (multi_array<OutT>& res, multi_array<InT>& lhs, multi_array<InT>& rhs)
{
    multi_array<InT>* left    = &lhs;
    multi_array<InT>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(*left, *right)) {
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

    // Check that operands are compatible with the output
    // TODO: Broadcasting should also be done in relation to output
    //       for now we simply fail...
    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes of output and input.");
    }

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_OR, res, *left, *right);

    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_logical_or (multi_array<OutT>& res, multi_array<InT>& lhs, const InT rhs)
{
    multi_array<InT>* left = &lhs;

    // Check for unbroadcastable situation
    if (res.getRank() < left->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_logical_or: " << res.getRank() << ", " << left->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    // Broadcast
    if (!same_shape(res, *left)) {
        left = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *left)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *left)) {
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_OR, res, *left, rhs);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_logical_or (multi_array<OutT>& res, const InT lhs, multi_array<InT>& rhs)
{
    multi_array<InT>* right = &rhs;

    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_logical_or: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    // Broadcast
    if (!same_shape(res, *right)) {
        right = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *right)) {
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_OR, res, lhs, *right);
    return res;
}

//
// Implicit temporary result array
//
template <typename T>
multi_array<T>& bh_logical_or (multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(*left, *right)) {
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

    // Construct output / result array
    multi_array<T>* result = &Runtime::instance().temp<T, T>(*left);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_OR, *result, *left, *right);
    return *result;
}

template <typename T>
multi_array<T>& bh_logical_or (multi_array<T>& lhs, const T rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T, T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_OR, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& bh_logical_or (const T lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T, T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_OR, *result, lhs, rhs);
    return *result;
}

// Explicit result array
template <typename OutT, typename InT>
multi_array<OutT>& bh_logical_xor (multi_array<OutT>& res, multi_array<InT>& lhs, multi_array<InT>& rhs)
{
    multi_array<InT>* left    = &lhs;
    multi_array<InT>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(*left, *right)) {
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

    // Check that operands are compatible with the output
    // TODO: Broadcasting should also be done in relation to output
    //       for now we simply fail...
    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes of output and input.");
    }

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_XOR, res, *left, *right);

    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_logical_xor (multi_array<OutT>& res, multi_array<InT>& lhs, const InT rhs)
{
    multi_array<InT>* left = &lhs;

    // Check for unbroadcastable situation
    if (res.getRank() < left->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_logical_xor: " << res.getRank() << ", " << left->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    // Broadcast
    if (!same_shape(res, *left)) {
        left = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *left)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *left)) {
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_XOR, res, *left, rhs);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_logical_xor (multi_array<OutT>& res, const InT lhs, multi_array<InT>& rhs)
{
    multi_array<InT>* right = &rhs;

    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_logical_xor: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    // Broadcast
    if (!same_shape(res, *right)) {
        right = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *right)) {
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_XOR, res, lhs, *right);
    return res;
}

//
// Implicit temporary result array
//
template <typename T>
multi_array<T>& bh_logical_xor (multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(*left, *right)) {
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

    // Construct output / result array
    multi_array<T>* result = &Runtime::instance().temp<T, T>(*left);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_XOR, *result, *left, *right);
    return *result;
}

template <typename T>
multi_array<T>& bh_logical_xor (multi_array<T>& lhs, const T rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T, T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_XOR, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& bh_logical_xor (const T lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T, T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_XOR, *result, lhs, rhs);
    return *result;
}

// Explicit result array
template <typename OutT, typename InT>
multi_array<OutT>& bh_power (multi_array<OutT>& res, multi_array<InT>& lhs, multi_array<InT>& rhs)
{
    multi_array<InT>* left    = &lhs;
    multi_array<InT>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(*left, *right)) {
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

    // Check that operands are compatible with the output
    // TODO: Broadcasting should also be done in relation to output
    //       for now we simply fail...
    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes of output and input.");
    }

    Runtime::instance().enqueue((bh_opcode)BH_POWER, res, *left, *right);

    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_power (multi_array<OutT>& res, multi_array<InT>& lhs, const InT rhs)
{
    multi_array<InT>* left = &lhs;

    // Check for unbroadcastable situation
    if (res.getRank() < left->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_power: " << res.getRank() << ", " << left->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    // Broadcast
    if (!same_shape(res, *left)) {
        left = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *left)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *left)) {
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_POWER, res, *left, rhs);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_power (multi_array<OutT>& res, const InT lhs, multi_array<InT>& rhs)
{
    multi_array<InT>* right = &rhs;

    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_power: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    // Broadcast
    if (!same_shape(res, *right)) {
        right = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *right)) {
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_POWER, res, lhs, *right);
    return res;
}

//
// Implicit temporary result array
//
template <typename T>
multi_array<T>& bh_power (multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(*left, *right)) {
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

    // Construct output / result array
    multi_array<T>* result = &Runtime::instance().temp<T, T>(*left);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_POWER, *result, *left, *right);
    return *result;
}

template <typename T>
multi_array<T>& bh_power (multi_array<T>& lhs, const T rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T, T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_POWER, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& bh_power (const T lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T, T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_POWER, *result, lhs, rhs);
    return *result;
}

// Explicit result array
template <typename OutT, typename InT>
multi_array<OutT>& bh_maximum (multi_array<OutT>& res, multi_array<InT>& lhs, multi_array<InT>& rhs)
{
    multi_array<InT>* left    = &lhs;
    multi_array<InT>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(*left, *right)) {
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

    // Check that operands are compatible with the output
    // TODO: Broadcasting should also be done in relation to output
    //       for now we simply fail...
    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes of output and input.");
    }

    Runtime::instance().enqueue((bh_opcode)BH_MAXIMUM, res, *left, *right);

    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_maximum (multi_array<OutT>& res, multi_array<InT>& lhs, const InT rhs)
{
    multi_array<InT>* left = &lhs;

    // Check for unbroadcastable situation
    if (res.getRank() < left->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_maximum: " << res.getRank() << ", " << left->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    // Broadcast
    if (!same_shape(res, *left)) {
        left = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *left)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *left)) {
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_MAXIMUM, res, *left, rhs);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_maximum (multi_array<OutT>& res, const InT lhs, multi_array<InT>& rhs)
{
    multi_array<InT>* right = &rhs;

    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_maximum: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    // Broadcast
    if (!same_shape(res, *right)) {
        right = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *right)) {
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_MAXIMUM, res, lhs, *right);
    return res;
}

//
// Implicit temporary result array
//
template <typename T>
multi_array<T>& bh_maximum (multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(*left, *right)) {
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

    // Construct output / result array
    multi_array<T>* result = &Runtime::instance().temp<T, T>(*left);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_MAXIMUM, *result, *left, *right);
    return *result;
}

template <typename T>
multi_array<T>& bh_maximum (multi_array<T>& lhs, const T rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T, T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_MAXIMUM, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& bh_maximum (const T lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T, T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_MAXIMUM, *result, lhs, rhs);
    return *result;
}

// Explicit result array
template <typename OutT, typename InT>
multi_array<OutT>& bh_minimum (multi_array<OutT>& res, multi_array<InT>& lhs, multi_array<InT>& rhs)
{
    multi_array<InT>* left    = &lhs;
    multi_array<InT>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(*left, *right)) {
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

    // Check that operands are compatible with the output
    // TODO: Broadcasting should also be done in relation to output
    //       for now we simply fail...
    if (!same_shape(res, *right)) {
        throw std::runtime_error("Incompatible shapes of output and input.");
    }

    Runtime::instance().enqueue((bh_opcode)BH_MINIMUM, res, *left, *right);

    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_minimum (multi_array<OutT>& res, multi_array<InT>& lhs, const InT rhs)
{
    multi_array<InT>* left = &lhs;

    // Check for unbroadcastable situation
    if (res.getRank() < left->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_minimum: " << res.getRank() << ", " << left->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    // Broadcast
    if (!same_shape(res, *left)) {
        left = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *left)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *left)) {
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_MINIMUM, res, *left, rhs);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_minimum (multi_array<OutT>& res, const InT lhs, multi_array<InT>& rhs)
{
    multi_array<InT>* right = &rhs;

    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_minimum: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    // Broadcast
    if (!same_shape(res, *right)) {
        right = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *right)) {
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_MINIMUM, res, lhs, *right);
    return res;
}

//
// Implicit temporary result array
//
template <typename T>
multi_array<T>& bh_minimum (multi_array<T> &lhs, multi_array<T> &rhs)
{
    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    // Broadcast
    if (!same_shape(*left, *right)) {
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

    // Construct output / result array
    multi_array<T>* result = &Runtime::instance().temp<T, T>(*left);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_MINIMUM, *result, *left, *right);
    return *result;
}

template <typename T>
multi_array<T>& bh_minimum (multi_array<T>& lhs, const T rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T, T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_MINIMUM, *result, lhs, rhs);
    return *result;
}

template <typename T>
multi_array<T>& bh_minimum (const T lhs, multi_array<T>& rhs)
{
    multi_array<T>* result = &Runtime::instance().temp<T, T>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_MINIMUM, *result, lhs, rhs);
    return *result;
}




//
//  Explicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_identity (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    multi_array<OutT>* right = &rhs;
    
    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_identity: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    //
    // Broadcast
    if (!same_shape(res, *right)) {
        right = &Runtime::instance().temp_view(rhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("Right-handside is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *right)) {
            throw std::runtime_error("Incompatable shapes, even after broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_IDENTITY, res, *right);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_identity (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_IDENTITY, res, rhs);
    return res;
}

//
//  Implicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_identity (multi_array<InT>& rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_IDENTITY, *result, rhs);
    return result;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_identity (const InT rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(1);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_IDENTITY, *result, rhs);
    return result;
}


//
//  Explicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_logical_not (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    multi_array<OutT>* right = &rhs;
    
    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_logical_not: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    //
    // Broadcast
    if (!same_shape(res, *right)) {
        right = &Runtime::instance().temp_view(rhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("Right-handside is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *right)) {
            throw std::runtime_error("Incompatable shapes, even after broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_NOT, res, *right);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_logical_not (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_NOT, res, rhs);
    return res;
}

//
//  Implicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_logical_not (multi_array<InT>& rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_NOT, *result, rhs);
    return result;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_logical_not (const InT rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(1);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_NOT, *result, rhs);
    return result;
}


//
//  Explicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_invert (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    multi_array<OutT>* right = &rhs;
    
    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_invert: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    //
    // Broadcast
    if (!same_shape(res, *right)) {
        right = &Runtime::instance().temp_view(rhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("Right-handside is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *right)) {
            throw std::runtime_error("Incompatable shapes, even after broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_INVERT, res, *right);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_invert (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_INVERT, res, rhs);
    return res;
}

//
//  Implicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_invert (multi_array<InT>& rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_INVERT, *result, rhs);
    return result;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_invert (const InT rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(1);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_INVERT, *result, rhs);
    return result;
}


//
//  Explicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_imag (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    multi_array<OutT>* right = &rhs;
    
    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_imag: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    //
    // Broadcast
    if (!same_shape(res, *right)) {
        right = &Runtime::instance().temp_view(rhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("Right-handside is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *right)) {
            throw std::runtime_error("Incompatable shapes, even after broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_IMAG, res, *right);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_imag (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_IMAG, res, rhs);
    return res;
}

//
//  Implicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_imag (multi_array<InT>& rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_IMAG, *result, rhs);
    return result;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_imag (const InT rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(1);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_IMAG, *result, rhs);
    return result;
}


//
//  Explicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_real (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    multi_array<OutT>* right = &rhs;
    
    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_real: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    //
    // Broadcast
    if (!same_shape(res, *right)) {
        right = &Runtime::instance().temp_view(rhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("Right-handside is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *right)) {
            throw std::runtime_error("Incompatable shapes, even after broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_REAL, res, *right);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_real (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_REAL, res, rhs);
    return res;
}

//
//  Implicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_real (multi_array<InT>& rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_REAL, *result, rhs);
    return result;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_real (const InT rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(1);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_REAL, *result, rhs);
    return result;
}


//
//  Explicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_absolute (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    multi_array<OutT>* right = &rhs;
    
    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_absolute: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    //
    // Broadcast
    if (!same_shape(res, *right)) {
        right = &Runtime::instance().temp_view(rhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("Right-handside is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *right)) {
            throw std::runtime_error("Incompatable shapes, even after broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_ABSOLUTE, res, *right);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_absolute (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_ABSOLUTE, res, rhs);
    return res;
}

//
//  Implicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_absolute (multi_array<InT>& rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_ABSOLUTE, *result, rhs);
    return result;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_absolute (const InT rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(1);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_ABSOLUTE, *result, rhs);
    return result;
}


//
//  Explicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_sin (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    multi_array<OutT>* right = &rhs;
    
    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_sin: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    //
    // Broadcast
    if (!same_shape(res, *right)) {
        right = &Runtime::instance().temp_view(rhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("Right-handside is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *right)) {
            throw std::runtime_error("Incompatable shapes, even after broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_SIN, res, *right);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_sin (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_SIN, res, rhs);
    return res;
}

//
//  Implicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_sin (multi_array<InT>& rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_SIN, *result, rhs);
    return result;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_sin (const InT rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(1);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_SIN, *result, rhs);
    return result;
}


//
//  Explicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_cos (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    multi_array<OutT>* right = &rhs;
    
    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_cos: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    //
    // Broadcast
    if (!same_shape(res, *right)) {
        right = &Runtime::instance().temp_view(rhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("Right-handside is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *right)) {
            throw std::runtime_error("Incompatable shapes, even after broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_COS, res, *right);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_cos (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_COS, res, rhs);
    return res;
}

//
//  Implicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_cos (multi_array<InT>& rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_COS, *result, rhs);
    return result;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_cos (const InT rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(1);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_COS, *result, rhs);
    return result;
}


//
//  Explicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_tan (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    multi_array<OutT>* right = &rhs;
    
    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_tan: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    //
    // Broadcast
    if (!same_shape(res, *right)) {
        right = &Runtime::instance().temp_view(rhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("Right-handside is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *right)) {
            throw std::runtime_error("Incompatable shapes, even after broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_TAN, res, *right);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_tan (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_TAN, res, rhs);
    return res;
}

//
//  Implicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_tan (multi_array<InT>& rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_TAN, *result, rhs);
    return result;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_tan (const InT rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(1);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_TAN, *result, rhs);
    return result;
}


//
//  Explicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_sinh (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    multi_array<OutT>* right = &rhs;
    
    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_sinh: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    //
    // Broadcast
    if (!same_shape(res, *right)) {
        right = &Runtime::instance().temp_view(rhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("Right-handside is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *right)) {
            throw std::runtime_error("Incompatable shapes, even after broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_SINH, res, *right);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_sinh (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_SINH, res, rhs);
    return res;
}

//
//  Implicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_sinh (multi_array<InT>& rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_SINH, *result, rhs);
    return result;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_sinh (const InT rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(1);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_SINH, *result, rhs);
    return result;
}


//
//  Explicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_cosh (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    multi_array<OutT>* right = &rhs;
    
    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_cosh: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    //
    // Broadcast
    if (!same_shape(res, *right)) {
        right = &Runtime::instance().temp_view(rhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("Right-handside is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *right)) {
            throw std::runtime_error("Incompatable shapes, even after broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_COSH, res, *right);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_cosh (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_COSH, res, rhs);
    return res;
}

//
//  Implicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_cosh (multi_array<InT>& rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_COSH, *result, rhs);
    return result;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_cosh (const InT rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(1);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_COSH, *result, rhs);
    return result;
}


//
//  Explicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_tanh (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    multi_array<OutT>* right = &rhs;
    
    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_tanh: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    //
    // Broadcast
    if (!same_shape(res, *right)) {
        right = &Runtime::instance().temp_view(rhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("Right-handside is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *right)) {
            throw std::runtime_error("Incompatable shapes, even after broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_TANH, res, *right);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_tanh (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_TANH, res, rhs);
    return res;
}

//
//  Implicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_tanh (multi_array<InT>& rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_TANH, *result, rhs);
    return result;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_tanh (const InT rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(1);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_TANH, *result, rhs);
    return result;
}


//
//  Explicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_asin (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    multi_array<OutT>* right = &rhs;
    
    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_asin: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    //
    // Broadcast
    if (!same_shape(res, *right)) {
        right = &Runtime::instance().temp_view(rhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("Right-handside is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *right)) {
            throw std::runtime_error("Incompatable shapes, even after broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_ARCSIN, res, *right);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_asin (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_ARCSIN, res, rhs);
    return res;
}

//
//  Implicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_asin (multi_array<InT>& rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_ARCSIN, *result, rhs);
    return result;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_asin (const InT rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(1);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_ARCSIN, *result, rhs);
    return result;
}


//
//  Explicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_acos (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    multi_array<OutT>* right = &rhs;
    
    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_acos: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    //
    // Broadcast
    if (!same_shape(res, *right)) {
        right = &Runtime::instance().temp_view(rhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("Right-handside is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *right)) {
            throw std::runtime_error("Incompatable shapes, even after broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_ARCCOS, res, *right);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_acos (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_ARCCOS, res, rhs);
    return res;
}

//
//  Implicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_acos (multi_array<InT>& rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_ARCCOS, *result, rhs);
    return result;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_acos (const InT rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(1);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_ARCCOS, *result, rhs);
    return result;
}


//
//  Explicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_atan (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    multi_array<OutT>* right = &rhs;
    
    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_atan: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    //
    // Broadcast
    if (!same_shape(res, *right)) {
        right = &Runtime::instance().temp_view(rhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("Right-handside is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *right)) {
            throw std::runtime_error("Incompatable shapes, even after broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_ARCTAN, res, *right);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_atan (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_ARCTAN, res, rhs);
    return res;
}

//
//  Implicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_atan (multi_array<InT>& rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_ARCTAN, *result, rhs);
    return result;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_atan (const InT rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(1);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_ARCTAN, *result, rhs);
    return result;
}


//
//  Explicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_atan2 (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    multi_array<OutT>* right = &rhs;
    
    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_atan2: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    //
    // Broadcast
    if (!same_shape(res, *right)) {
        right = &Runtime::instance().temp_view(rhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("Right-handside is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *right)) {
            throw std::runtime_error("Incompatable shapes, even after broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_ARCTAN2, res, *right);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_atan2 (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_ARCTAN2, res, rhs);
    return res;
}

//
//  Implicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_atan2 (multi_array<InT>& rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_ARCTAN2, *result, rhs);
    return result;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_atan2 (const InT rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(1);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_ARCTAN2, *result, rhs);
    return result;
}


//
//  Explicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_asinh (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    multi_array<OutT>* right = &rhs;
    
    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_asinh: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    //
    // Broadcast
    if (!same_shape(res, *right)) {
        right = &Runtime::instance().temp_view(rhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("Right-handside is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *right)) {
            throw std::runtime_error("Incompatable shapes, even after broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_ARCSINH, res, *right);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_asinh (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_ARCSINH, res, rhs);
    return res;
}

//
//  Implicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_asinh (multi_array<InT>& rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_ARCSINH, *result, rhs);
    return result;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_asinh (const InT rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(1);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_ARCSINH, *result, rhs);
    return result;
}


//
//  Explicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_acosh (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    multi_array<OutT>* right = &rhs;
    
    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_acosh: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    //
    // Broadcast
    if (!same_shape(res, *right)) {
        right = &Runtime::instance().temp_view(rhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("Right-handside is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *right)) {
            throw std::runtime_error("Incompatable shapes, even after broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_ARCCOSH, res, *right);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_acosh (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_ARCCOSH, res, rhs);
    return res;
}

//
//  Implicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_acosh (multi_array<InT>& rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_ARCCOSH, *result, rhs);
    return result;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_acosh (const InT rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(1);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_ARCCOSH, *result, rhs);
    return result;
}


//
//  Explicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_atanh (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    multi_array<OutT>* right = &rhs;
    
    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_atanh: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    //
    // Broadcast
    if (!same_shape(res, *right)) {
        right = &Runtime::instance().temp_view(rhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("Right-handside is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *right)) {
            throw std::runtime_error("Incompatable shapes, even after broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_ARCTANH, res, *right);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_atanh (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_ARCTANH, res, rhs);
    return res;
}

//
//  Implicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_atanh (multi_array<InT>& rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_ARCTANH, *result, rhs);
    return result;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_atanh (const InT rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(1);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_ARCTANH, *result, rhs);
    return result;
}


//
//  Explicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_exp (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    multi_array<OutT>* right = &rhs;
    
    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_exp: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    //
    // Broadcast
    if (!same_shape(res, *right)) {
        right = &Runtime::instance().temp_view(rhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("Right-handside is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *right)) {
            throw std::runtime_error("Incompatable shapes, even after broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_EXP, res, *right);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_exp (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_EXP, res, rhs);
    return res;
}

//
//  Implicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_exp (multi_array<InT>& rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_EXP, *result, rhs);
    return result;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_exp (const InT rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(1);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_EXP, *result, rhs);
    return result;
}


//
//  Explicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_exp2 (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    multi_array<OutT>* right = &rhs;
    
    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_exp2: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    //
    // Broadcast
    if (!same_shape(res, *right)) {
        right = &Runtime::instance().temp_view(rhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("Right-handside is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *right)) {
            throw std::runtime_error("Incompatable shapes, even after broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_EXP2, res, *right);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_exp2 (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_EXP2, res, rhs);
    return res;
}

//
//  Implicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_exp2 (multi_array<InT>& rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_EXP2, *result, rhs);
    return result;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_exp2 (const InT rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(1);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_EXP2, *result, rhs);
    return result;
}


//
//  Explicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_expm1 (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    multi_array<OutT>* right = &rhs;
    
    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_expm1: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    //
    // Broadcast
    if (!same_shape(res, *right)) {
        right = &Runtime::instance().temp_view(rhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("Right-handside is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *right)) {
            throw std::runtime_error("Incompatable shapes, even after broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_EXPM1, res, *right);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_expm1 (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_EXPM1, res, rhs);
    return res;
}

//
//  Implicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_expm1 (multi_array<InT>& rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_EXPM1, *result, rhs);
    return result;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_expm1 (const InT rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(1);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_EXPM1, *result, rhs);
    return result;
}


//
//  Explicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_isnan (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    multi_array<OutT>* right = &rhs;
    
    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_isnan: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    //
    // Broadcast
    if (!same_shape(res, *right)) {
        right = &Runtime::instance().temp_view(rhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("Right-handside is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *right)) {
            throw std::runtime_error("Incompatable shapes, even after broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_ISNAN, res, *right);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_isnan (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_ISNAN, res, rhs);
    return res;
}

//
//  Implicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_isnan (multi_array<InT>& rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_ISNAN, *result, rhs);
    return result;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_isnan (const InT rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(1);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_ISNAN, *result, rhs);
    return result;
}


//
//  Explicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_isinf (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    multi_array<OutT>* right = &rhs;
    
    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_isinf: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    //
    // Broadcast
    if (!same_shape(res, *right)) {
        right = &Runtime::instance().temp_view(rhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("Right-handside is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *right)) {
            throw std::runtime_error("Incompatable shapes, even after broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_ISINF, res, *right);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_isinf (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_ISINF, res, rhs);
    return res;
}

//
//  Implicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_isinf (multi_array<InT>& rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_ISINF, *result, rhs);
    return result;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_isinf (const InT rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(1);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_ISINF, *result, rhs);
    return result;
}


//
//  Explicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_log (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    multi_array<OutT>* right = &rhs;
    
    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_log: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    //
    // Broadcast
    if (!same_shape(res, *right)) {
        right = &Runtime::instance().temp_view(rhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("Right-handside is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *right)) {
            throw std::runtime_error("Incompatable shapes, even after broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_LOG, res, *right);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_log (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_LOG, res, rhs);
    return res;
}

//
//  Implicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_log (multi_array<InT>& rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LOG, *result, rhs);
    return result;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_log (const InT rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(1);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LOG, *result, rhs);
    return result;
}


//
//  Explicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_log2 (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    multi_array<OutT>* right = &rhs;
    
    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_log2: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    //
    // Broadcast
    if (!same_shape(res, *right)) {
        right = &Runtime::instance().temp_view(rhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("Right-handside is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *right)) {
            throw std::runtime_error("Incompatable shapes, even after broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_LOG2, res, *right);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_log2 (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_LOG2, res, rhs);
    return res;
}

//
//  Implicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_log2 (multi_array<InT>& rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LOG2, *result, rhs);
    return result;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_log2 (const InT rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(1);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LOG2, *result, rhs);
    return result;
}


//
//  Explicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_log10 (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    multi_array<OutT>* right = &rhs;
    
    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_log10: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    //
    // Broadcast
    if (!same_shape(res, *right)) {
        right = &Runtime::instance().temp_view(rhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("Right-handside is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *right)) {
            throw std::runtime_error("Incompatable shapes, even after broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_LOG10, res, *right);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_log10 (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_LOG10, res, rhs);
    return res;
}

//
//  Implicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_log10 (multi_array<InT>& rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LOG10, *result, rhs);
    return result;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_log10 (const InT rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(1);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LOG10, *result, rhs);
    return result;
}


//
//  Explicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_log1p (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    multi_array<OutT>* right = &rhs;
    
    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_log1p: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    //
    // Broadcast
    if (!same_shape(res, *right)) {
        right = &Runtime::instance().temp_view(rhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("Right-handside is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *right)) {
            throw std::runtime_error("Incompatable shapes, even after broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_LOG1P, res, *right);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_log1p (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_LOG1P, res, rhs);
    return res;
}

//
//  Implicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_log1p (multi_array<InT>& rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LOG1P, *result, rhs);
    return result;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_log1p (const InT rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(1);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_LOG1P, *result, rhs);
    return result;
}


//
//  Explicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_sqrt (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    multi_array<OutT>* right = &rhs;
    
    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_sqrt: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    //
    // Broadcast
    if (!same_shape(res, *right)) {
        right = &Runtime::instance().temp_view(rhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("Right-handside is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *right)) {
            throw std::runtime_error("Incompatable shapes, even after broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_SQRT, res, *right);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_sqrt (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_SQRT, res, rhs);
    return res;
}

//
//  Implicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_sqrt (multi_array<InT>& rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_SQRT, *result, rhs);
    return result;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_sqrt (const InT rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(1);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_SQRT, *result, rhs);
    return result;
}


//
//  Explicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_ceil (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    multi_array<OutT>* right = &rhs;
    
    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_ceil: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    //
    // Broadcast
    if (!same_shape(res, *right)) {
        right = &Runtime::instance().temp_view(rhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("Right-handside is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *right)) {
            throw std::runtime_error("Incompatable shapes, even after broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_CEIL, res, *right);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_ceil (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_CEIL, res, rhs);
    return res;
}

//
//  Implicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_ceil (multi_array<InT>& rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_CEIL, *result, rhs);
    return result;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_ceil (const InT rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(1);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_CEIL, *result, rhs);
    return result;
}


//
//  Explicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_trunc (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    multi_array<OutT>* right = &rhs;
    
    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_trunc: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    //
    // Broadcast
    if (!same_shape(res, *right)) {
        right = &Runtime::instance().temp_view(rhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("Right-handside is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *right)) {
            throw std::runtime_error("Incompatable shapes, even after broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_TRUNC, res, *right);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_trunc (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_TRUNC, res, rhs);
    return res;
}

//
//  Implicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_trunc (multi_array<InT>& rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_TRUNC, *result, rhs);
    return result;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_trunc (const InT rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(1);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_TRUNC, *result, rhs);
    return result;
}


//
//  Explicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_floor (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    multi_array<OutT>* right = &rhs;
    
    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_floor: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    //
    // Broadcast
    if (!same_shape(res, *right)) {
        right = &Runtime::instance().temp_view(rhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("Right-handside is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *right)) {
            throw std::runtime_error("Incompatable shapes, even after broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_FLOOR, res, *right);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_floor (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_FLOOR, res, rhs);
    return res;
}

//
//  Implicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_floor (multi_array<InT>& rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_FLOOR, *result, rhs);
    return result;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_floor (const InT rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(1);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_FLOOR, *result, rhs);
    return result;
}


//
//  Explicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_rint (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    multi_array<OutT>* right = &rhs;
    
    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_rint: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    //
    // Broadcast
    if (!same_shape(res, *right)) {
        right = &Runtime::instance().temp_view(rhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("Right-handside is not broadcastable.");
        }
        
        //
        // Re-check compatibility
        if (!same_shape(res, *right)) {
            throw std::runtime_error("Incompatable shapes, even after broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_RINT, res, *right);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_rint (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().enqueue((bh_opcode)BH_RINT, res, rhs);
    return res;
}

//
//  Implicit result array
//
template <typename OutT, typename InT>
multi_array<OutT>& bh_rint (multi_array<InT>& rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(rhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_RINT, *result, rhs);
    return result;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_rint (const InT rhs)
{
    multi_array<OutT>* result = &Runtime::instance().temp<OutT, InT>(1);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_RINT, *result, rhs);
    return result;
}




template <typename T>
multi_array<T>& bh_range (multi_array<T>& res)
{
    Runtime::instance().enqueue((bh_opcode)BH_RANGE, res);
    return res;
}




template <typename T>
multi_array<T>& bh_random (multi_array<T>& res, uint64_t in1, uint64_t in2)
{
    Runtime::instance().enqueue((bh_opcode)BH_RANDOM, res, in1, in2);

    return res;
}



template <typename T>
multi_array<T>& bh_add_accumulate (multi_array<T>& res, multi_array<T> &lhs, int64_t rhs)
{
    // Check axis
    if (rhs<0) {
        rhs = lhs.getRank()+rhs;
    }
    if (rhs >= (int64_t)lhs.getRank()) {
        throw std::runtime_error("Error: Axis out of bounds in reduction.\n");
    }
    // TODO:
    //  * shape-check
    //  * type-check
    Runtime::instance().enqueue((bh_opcode)BH_ADD_ACCUMULATE, res, lhs, rhs);

    return res;
}

// Todo: typecheck
template <typename T>
multi_array<T>& bh_add_accumulate (multi_array<T> &lhs, int64_t rhs)
{
    // Check axis
    if (rhs<0) {
        rhs = lhs.getRank()+rhs;
    }
    if (rhs >= (int64_t)lhs.getRank()) {
        throw std::runtime_error("Error: Axis out of bounds in reduction.\n");
    }

    // Construct result array
    multi_array<T>* result = &Runtime::instance().temp<T, T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_ADD_ACCUMULATE, *result, lhs, rhs);

    return *result;
}


template <typename T>
multi_array<T>& bh_multiply_accumulate (multi_array<T>& res, multi_array<T> &lhs, int64_t rhs)
{
    // Check axis
    if (rhs<0) {
        rhs = lhs.getRank()+rhs;
    }
    if (rhs >= (int64_t)lhs.getRank()) {
        throw std::runtime_error("Error: Axis out of bounds in reduction.\n");
    }
    // TODO:
    //  * shape-check
    //  * type-check
    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY_ACCUMULATE, res, lhs, rhs);

    return res;
}

// Todo: typecheck
template <typename T>
multi_array<T>& bh_multiply_accumulate (multi_array<T> &lhs, int64_t rhs)
{
    // Check axis
    if (rhs<0) {
        rhs = lhs.getRank()+rhs;
    }
    if (rhs >= (int64_t)lhs.getRank()) {
        throw std::runtime_error("Error: Axis out of bounds in reduction.\n");
    }

    // Construct result array
    multi_array<T>* result = &Runtime::instance().temp<T, T>(lhs);
    result->link();

    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY_ACCUMULATE, *result, lhs, rhs);

    return *result;
}




template <typename T>
multi_array<T>& bh_add_reduce (multi_array<T>& res, multi_array<T> &lhs, int64_t rhs)
{
    if (rhs<0) {
        rhs = lhs.getRank()+rhs;
    }
    if (rhs >= (int64_t)lhs.getRank()) {
        throw std::runtime_error("Error: Axis out of bounds in reduction.\n");
    }

    // TODO
    //  * Type-check
    //  * Shape-check
    Runtime::instance().enqueue((bh_opcode)BH_ADD_REDUCE, res, lhs, rhs);

    return res;
}

template <typename T>
multi_array<T>& bh_add_reduce (multi_array<T> &lhs, int64_t rhs)
{
    if (rhs<0) {
        rhs = lhs.getRank()+rhs;
    }
    if (rhs >= (int64_t)lhs.getRank()) {
        throw std::runtime_error("Error: Axis out of bounds in reduction.\n");
    }

    // Construct result array
    multi_array<T>* result = &Runtime::instance().temp<T>();

    result->meta.start = 0;                 // Update meta-data
    if (lhs.meta.ndim == 1) {                // Pseudo-scalar; one element
        result->meta.ndim      = 1;
        result->meta.shape[0]  = 1;
        result->meta.stride[0] = lhs.meta.stride[0];
    } else {                                // Remove axis
        result->meta.ndim  = lhs.meta.ndim -1;
        int64_t stride = 1; 
        for(int64_t i=lhs.meta.ndim-1, j=result->meta.ndim-1; i>=0; --i) {
            if (i!=(int64_t)rhs) {
                result->meta.shape[j]  = lhs.meta.shape[i];
                result->meta.stride[j] = stride;
                stride *= result->meta.shape[j];
                --j;
            }
        }
    }
    result->link();                         // Bind the base

    // TODO
    //  * Type-check
    //  * Shape-check
    Runtime::instance().enqueue((bh_opcode)BH_ADD_REDUCE, *result, lhs, rhs);

    return *result;
}


template <typename T>
multi_array<T>& bh_multiply_reduce (multi_array<T>& res, multi_array<T> &lhs, int64_t rhs)
{
    if (rhs<0) {
        rhs = lhs.getRank()+rhs;
    }
    if (rhs >= (int64_t)lhs.getRank()) {
        throw std::runtime_error("Error: Axis out of bounds in reduction.\n");
    }

    // TODO
    //  * Type-check
    //  * Shape-check
    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY_REDUCE, res, lhs, rhs);

    return res;
}

template <typename T>
multi_array<T>& bh_multiply_reduce (multi_array<T> &lhs, int64_t rhs)
{
    if (rhs<0) {
        rhs = lhs.getRank()+rhs;
    }
    if (rhs >= (int64_t)lhs.getRank()) {
        throw std::runtime_error("Error: Axis out of bounds in reduction.\n");
    }

    // Construct result array
    multi_array<T>* result = &Runtime::instance().temp<T>();

    result->meta.start = 0;                 // Update meta-data
    if (lhs.meta.ndim == 1) {                // Pseudo-scalar; one element
        result->meta.ndim      = 1;
        result->meta.shape[0]  = 1;
        result->meta.stride[0] = lhs.meta.stride[0];
    } else {                                // Remove axis
        result->meta.ndim  = lhs.meta.ndim -1;
        int64_t stride = 1; 
        for(int64_t i=lhs.meta.ndim-1, j=result->meta.ndim-1; i>=0; --i) {
            if (i!=(int64_t)rhs) {
                result->meta.shape[j]  = lhs.meta.shape[i];
                result->meta.stride[j] = stride;
                stride *= result->meta.shape[j];
                --j;
            }
        }
    }
    result->link();                         // Bind the base

    // TODO
    //  * Type-check
    //  * Shape-check
    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY_REDUCE, *result, lhs, rhs);

    return *result;
}


template <typename T>
multi_array<T>& bh_minimum_reduce (multi_array<T>& res, multi_array<T> &lhs, int64_t rhs)
{
    if (rhs<0) {
        rhs = lhs.getRank()+rhs;
    }
    if (rhs >= (int64_t)lhs.getRank()) {
        throw std::runtime_error("Error: Axis out of bounds in reduction.\n");
    }

    // TODO
    //  * Type-check
    //  * Shape-check
    Runtime::instance().enqueue((bh_opcode)BH_MINIMUM_REDUCE, res, lhs, rhs);

    return res;
}

template <typename T>
multi_array<T>& bh_minimum_reduce (multi_array<T> &lhs, int64_t rhs)
{
    if (rhs<0) {
        rhs = lhs.getRank()+rhs;
    }
    if (rhs >= (int64_t)lhs.getRank()) {
        throw std::runtime_error("Error: Axis out of bounds in reduction.\n");
    }

    // Construct result array
    multi_array<T>* result = &Runtime::instance().temp<T>();

    result->meta.start = 0;                 // Update meta-data
    if (lhs.meta.ndim == 1) {                // Pseudo-scalar; one element
        result->meta.ndim      = 1;
        result->meta.shape[0]  = 1;
        result->meta.stride[0] = lhs.meta.stride[0];
    } else {                                // Remove axis
        result->meta.ndim  = lhs.meta.ndim -1;
        int64_t stride = 1; 
        for(int64_t i=lhs.meta.ndim-1, j=result->meta.ndim-1; i>=0; --i) {
            if (i!=(int64_t)rhs) {
                result->meta.shape[j]  = lhs.meta.shape[i];
                result->meta.stride[j] = stride;
                stride *= result->meta.shape[j];
                --j;
            }
        }
    }
    result->link();                         // Bind the base

    // TODO
    //  * Type-check
    //  * Shape-check
    Runtime::instance().enqueue((bh_opcode)BH_MINIMUM_REDUCE, *result, lhs, rhs);

    return *result;
}


template <typename T>
multi_array<T>& bh_maximum_reduce (multi_array<T>& res, multi_array<T> &lhs, int64_t rhs)
{
    if (rhs<0) {
        rhs = lhs.getRank()+rhs;
    }
    if (rhs >= (int64_t)lhs.getRank()) {
        throw std::runtime_error("Error: Axis out of bounds in reduction.\n");
    }

    // TODO
    //  * Type-check
    //  * Shape-check
    Runtime::instance().enqueue((bh_opcode)BH_MAXIMUM_REDUCE, res, lhs, rhs);

    return res;
}

template <typename T>
multi_array<T>& bh_maximum_reduce (multi_array<T> &lhs, int64_t rhs)
{
    if (rhs<0) {
        rhs = lhs.getRank()+rhs;
    }
    if (rhs >= (int64_t)lhs.getRank()) {
        throw std::runtime_error("Error: Axis out of bounds in reduction.\n");
    }

    // Construct result array
    multi_array<T>* result = &Runtime::instance().temp<T>();

    result->meta.start = 0;                 // Update meta-data
    if (lhs.meta.ndim == 1) {                // Pseudo-scalar; one element
        result->meta.ndim      = 1;
        result->meta.shape[0]  = 1;
        result->meta.stride[0] = lhs.meta.stride[0];
    } else {                                // Remove axis
        result->meta.ndim  = lhs.meta.ndim -1;
        int64_t stride = 1; 
        for(int64_t i=lhs.meta.ndim-1, j=result->meta.ndim-1; i>=0; --i) {
            if (i!=(int64_t)rhs) {
                result->meta.shape[j]  = lhs.meta.shape[i];
                result->meta.stride[j] = stride;
                stride *= result->meta.shape[j];
                --j;
            }
        }
    }
    result->link();                         // Bind the base

    // TODO
    //  * Type-check
    //  * Shape-check
    Runtime::instance().enqueue((bh_opcode)BH_MAXIMUM_REDUCE, *result, lhs, rhs);

    return *result;
}


template <typename T>
multi_array<T>& bh_logical_and_reduce (multi_array<T>& res, multi_array<T> &lhs, int64_t rhs)
{
    if (rhs<0) {
        rhs = lhs.getRank()+rhs;
    }
    if (rhs >= (int64_t)lhs.getRank()) {
        throw std::runtime_error("Error: Axis out of bounds in reduction.\n");
    }

    // TODO
    //  * Type-check
    //  * Shape-check
    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_AND_REDUCE, res, lhs, rhs);

    return res;
}

template <typename T>
multi_array<T>& bh_logical_and_reduce (multi_array<T> &lhs, int64_t rhs)
{
    if (rhs<0) {
        rhs = lhs.getRank()+rhs;
    }
    if (rhs >= (int64_t)lhs.getRank()) {
        throw std::runtime_error("Error: Axis out of bounds in reduction.\n");
    }

    // Construct result array
    multi_array<T>* result = &Runtime::instance().temp<T>();

    result->meta.start = 0;                 // Update meta-data
    if (lhs.meta.ndim == 1) {                // Pseudo-scalar; one element
        result->meta.ndim      = 1;
        result->meta.shape[0]  = 1;
        result->meta.stride[0] = lhs.meta.stride[0];
    } else {                                // Remove axis
        result->meta.ndim  = lhs.meta.ndim -1;
        int64_t stride = 1; 
        for(int64_t i=lhs.meta.ndim-1, j=result->meta.ndim-1; i>=0; --i) {
            if (i!=(int64_t)rhs) {
                result->meta.shape[j]  = lhs.meta.shape[i];
                result->meta.stride[j] = stride;
                stride *= result->meta.shape[j];
                --j;
            }
        }
    }
    result->link();                         // Bind the base

    // TODO
    //  * Type-check
    //  * Shape-check
    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_AND_REDUCE, *result, lhs, rhs);

    return *result;
}


template <typename T>
multi_array<T>& bh_logical_or_reduce (multi_array<T>& res, multi_array<T> &lhs, int64_t rhs)
{
    if (rhs<0) {
        rhs = lhs.getRank()+rhs;
    }
    if (rhs >= (int64_t)lhs.getRank()) {
        throw std::runtime_error("Error: Axis out of bounds in reduction.\n");
    }

    // TODO
    //  * Type-check
    //  * Shape-check
    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_OR_REDUCE, res, lhs, rhs);

    return res;
}

template <typename T>
multi_array<T>& bh_logical_or_reduce (multi_array<T> &lhs, int64_t rhs)
{
    if (rhs<0) {
        rhs = lhs.getRank()+rhs;
    }
    if (rhs >= (int64_t)lhs.getRank()) {
        throw std::runtime_error("Error: Axis out of bounds in reduction.\n");
    }

    // Construct result array
    multi_array<T>* result = &Runtime::instance().temp<T>();

    result->meta.start = 0;                 // Update meta-data
    if (lhs.meta.ndim == 1) {                // Pseudo-scalar; one element
        result->meta.ndim      = 1;
        result->meta.shape[0]  = 1;
        result->meta.stride[0] = lhs.meta.stride[0];
    } else {                                // Remove axis
        result->meta.ndim  = lhs.meta.ndim -1;
        int64_t stride = 1; 
        for(int64_t i=lhs.meta.ndim-1, j=result->meta.ndim-1; i>=0; --i) {
            if (i!=(int64_t)rhs) {
                result->meta.shape[j]  = lhs.meta.shape[i];
                result->meta.stride[j] = stride;
                stride *= result->meta.shape[j];
                --j;
            }
        }
    }
    result->link();                         // Bind the base

    // TODO
    //  * Type-check
    //  * Shape-check
    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_OR_REDUCE, *result, lhs, rhs);

    return *result;
}


template <typename T>
multi_array<T>& bh_logical_xor_reduce (multi_array<T>& res, multi_array<T> &lhs, int64_t rhs)
{
    if (rhs<0) {
        rhs = lhs.getRank()+rhs;
    }
    if (rhs >= (int64_t)lhs.getRank()) {
        throw std::runtime_error("Error: Axis out of bounds in reduction.\n");
    }

    // TODO
    //  * Type-check
    //  * Shape-check
    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_XOR_REDUCE, res, lhs, rhs);

    return res;
}

template <typename T>
multi_array<T>& bh_logical_xor_reduce (multi_array<T> &lhs, int64_t rhs)
{
    if (rhs<0) {
        rhs = lhs.getRank()+rhs;
    }
    if (rhs >= (int64_t)lhs.getRank()) {
        throw std::runtime_error("Error: Axis out of bounds in reduction.\n");
    }

    // Construct result array
    multi_array<T>* result = &Runtime::instance().temp<T>();

    result->meta.start = 0;                 // Update meta-data
    if (lhs.meta.ndim == 1) {                // Pseudo-scalar; one element
        result->meta.ndim      = 1;
        result->meta.shape[0]  = 1;
        result->meta.stride[0] = lhs.meta.stride[0];
    } else {                                // Remove axis
        result->meta.ndim  = lhs.meta.ndim -1;
        int64_t stride = 1; 
        for(int64_t i=lhs.meta.ndim-1, j=result->meta.ndim-1; i>=0; --i) {
            if (i!=(int64_t)rhs) {
                result->meta.shape[j]  = lhs.meta.shape[i];
                result->meta.stride[j] = stride;
                stride *= result->meta.shape[j];
                --j;
            }
        }
    }
    result->link();                         // Bind the base

    // TODO
    //  * Type-check
    //  * Shape-check
    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_XOR_REDUCE, *result, lhs, rhs);

    return *result;
}


template <typename T>
multi_array<T>& bh_bitwise_and_reduce (multi_array<T>& res, multi_array<T> &lhs, int64_t rhs)
{
    if (rhs<0) {
        rhs = lhs.getRank()+rhs;
    }
    if (rhs >= (int64_t)lhs.getRank()) {
        throw std::runtime_error("Error: Axis out of bounds in reduction.\n");
    }

    // TODO
    //  * Type-check
    //  * Shape-check
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_AND_REDUCE, res, lhs, rhs);

    return res;
}

template <typename T>
multi_array<T>& bh_bitwise_and_reduce (multi_array<T> &lhs, int64_t rhs)
{
    if (rhs<0) {
        rhs = lhs.getRank()+rhs;
    }
    if (rhs >= (int64_t)lhs.getRank()) {
        throw std::runtime_error("Error: Axis out of bounds in reduction.\n");
    }

    // Construct result array
    multi_array<T>* result = &Runtime::instance().temp<T>();

    result->meta.start = 0;                 // Update meta-data
    if (lhs.meta.ndim == 1) {                // Pseudo-scalar; one element
        result->meta.ndim      = 1;
        result->meta.shape[0]  = 1;
        result->meta.stride[0] = lhs.meta.stride[0];
    } else {                                // Remove axis
        result->meta.ndim  = lhs.meta.ndim -1;
        int64_t stride = 1; 
        for(int64_t i=lhs.meta.ndim-1, j=result->meta.ndim-1; i>=0; --i) {
            if (i!=(int64_t)rhs) {
                result->meta.shape[j]  = lhs.meta.shape[i];
                result->meta.stride[j] = stride;
                stride *= result->meta.shape[j];
                --j;
            }
        }
    }
    result->link();                         // Bind the base

    // TODO
    //  * Type-check
    //  * Shape-check
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_AND_REDUCE, *result, lhs, rhs);

    return *result;
}


template <typename T>
multi_array<T>& bh_bitwise_or_reduce (multi_array<T>& res, multi_array<T> &lhs, int64_t rhs)
{
    if (rhs<0) {
        rhs = lhs.getRank()+rhs;
    }
    if (rhs >= (int64_t)lhs.getRank()) {
        throw std::runtime_error("Error: Axis out of bounds in reduction.\n");
    }

    // TODO
    //  * Type-check
    //  * Shape-check
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_OR_REDUCE, res, lhs, rhs);

    return res;
}

template <typename T>
multi_array<T>& bh_bitwise_or_reduce (multi_array<T> &lhs, int64_t rhs)
{
    if (rhs<0) {
        rhs = lhs.getRank()+rhs;
    }
    if (rhs >= (int64_t)lhs.getRank()) {
        throw std::runtime_error("Error: Axis out of bounds in reduction.\n");
    }

    // Construct result array
    multi_array<T>* result = &Runtime::instance().temp<T>();

    result->meta.start = 0;                 // Update meta-data
    if (lhs.meta.ndim == 1) {                // Pseudo-scalar; one element
        result->meta.ndim      = 1;
        result->meta.shape[0]  = 1;
        result->meta.stride[0] = lhs.meta.stride[0];
    } else {                                // Remove axis
        result->meta.ndim  = lhs.meta.ndim -1;
        int64_t stride = 1; 
        for(int64_t i=lhs.meta.ndim-1, j=result->meta.ndim-1; i>=0; --i) {
            if (i!=(int64_t)rhs) {
                result->meta.shape[j]  = lhs.meta.shape[i];
                result->meta.stride[j] = stride;
                stride *= result->meta.shape[j];
                --j;
            }
        }
    }
    result->link();                         // Bind the base

    // TODO
    //  * Type-check
    //  * Shape-check
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_OR_REDUCE, *result, lhs, rhs);

    return *result;
}


template <typename T>
multi_array<T>& bh_bitwise_xor_reduce (multi_array<T>& res, multi_array<T> &lhs, int64_t rhs)
{
    if (rhs<0) {
        rhs = lhs.getRank()+rhs;
    }
    if (rhs >= (int64_t)lhs.getRank()) {
        throw std::runtime_error("Error: Axis out of bounds in reduction.\n");
    }

    // TODO
    //  * Type-check
    //  * Shape-check
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_XOR_REDUCE, res, lhs, rhs);

    return res;
}

template <typename T>
multi_array<T>& bh_bitwise_xor_reduce (multi_array<T> &lhs, int64_t rhs)
{
    if (rhs<0) {
        rhs = lhs.getRank()+rhs;
    }
    if (rhs >= (int64_t)lhs.getRank()) {
        throw std::runtime_error("Error: Axis out of bounds in reduction.\n");
    }

    // Construct result array
    multi_array<T>* result = &Runtime::instance().temp<T>();

    result->meta.start = 0;                 // Update meta-data
    if (lhs.meta.ndim == 1) {                // Pseudo-scalar; one element
        result->meta.ndim      = 1;
        result->meta.shape[0]  = 1;
        result->meta.stride[0] = lhs.meta.stride[0];
    } else {                                // Remove axis
        result->meta.ndim  = lhs.meta.ndim -1;
        int64_t stride = 1; 
        for(int64_t i=lhs.meta.ndim-1, j=result->meta.ndim-1; i>=0; --i) {
            if (i!=(int64_t)rhs) {
                result->meta.shape[j]  = lhs.meta.shape[i];
                result->meta.stride[j] = stride;
                stride *= result->meta.shape[j];
                --j;
            }
        }
    }
    result->link();                         // Bind the base

    // TODO
    //  * Type-check
    //  * Shape-check
    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_XOR_REDUCE, *result, lhs, rhs);

    return *result;
}


}
#endif
