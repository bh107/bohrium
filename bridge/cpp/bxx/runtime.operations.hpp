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
#include <bh.h>

namespace bxx {



//
// bh_add - BH_ADD - runtime.binary - 3
//

//
// Explicit result array
template <typename TO, typename TL, typename TR>
multi_array<TO>& bh_add (multi_array<TO>& res, multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_ADD, TO, TL, TR>();

    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(res, *left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(res, *right, *left)) {
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

template <typename TO, typename TL, typename TR>
multi_array<TO>& bh_add (multi_array<TO>& res, multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_ADD, TO, TL, TR>();

    multi_array<TL>* left = &lhs;

    if (res.getRank() < left->getRank()) {  // Check for unbroadcastable situation
        std::stringstream s;
        s << "Incompatible shapes " << "bh_add: " << res.getRank() << ", " << left->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!same_shape(res, *left)) {          // Broadcast
        left = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *left)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        if (!same_shape(res, *left)) {      // Re-check compatibility
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_ADD, res, *left, rhs);
    return res;
}

template <typename TO, typename TL, typename TR>
multi_array<TO>& bh_add (multi_array<TO>& res, const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_ADD, TO, TL, TR>();

    multi_array<TR>* right = &rhs;

    if (res.getRank() < right->getRank()) { // Check for unbroadcastable situation
        std::stringstream s;
        s << "Incompatible shapes " << "bh_add: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!same_shape(res, *right)) {         // Broadcast
        right = &Runtime::instance().temp_view(rhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        if (!same_shape(res, *right)) {     // Re-check compatibility
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_ADD, res, lhs, *right);
    return res;
}

//
// Implicit temporary result array
template <typename TL, typename TR>
multi_array<TL>& bh_add (multi_array<TL> &lhs, multi_array<TR> &rhs)
{
    Runtime::instance().typecheck<BH_ADD, TL, TL, TR>();

    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
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
    multi_array<TL>* result = &Runtime::instance().temp<TL, TR>(*left);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_ADD, *result, *left, *right);
    result->setTemp(true);
    return *result;
}

template <typename TL, typename TR>
multi_array<TL>& bh_add (multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_ADD, TL, TL, TR>();

    multi_array<TL>* result = &Runtime::instance().temp<TL, TR>(lhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_ADD, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}

template <typename TL, typename TR>
multi_array<TL>& bh_add (const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_ADD, TL, TL, TR>();

    multi_array<TL>* result = &Runtime::instance().temp<TL, TR>(rhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_ADD, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}


//
// bh_subtract - BH_SUBTRACT - runtime.binary - 3
//

//
// Explicit result array
template <typename TO, typename TL, typename TR>
multi_array<TO>& bh_subtract (multi_array<TO>& res, multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_SUBTRACT, TO, TL, TR>();

    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(res, *left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(res, *right, *left)) {
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

template <typename TO, typename TL, typename TR>
multi_array<TO>& bh_subtract (multi_array<TO>& res, multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_SUBTRACT, TO, TL, TR>();

    multi_array<TL>* left = &lhs;

    if (res.getRank() < left->getRank()) {  // Check for unbroadcastable situation
        std::stringstream s;
        s << "Incompatible shapes " << "bh_subtract: " << res.getRank() << ", " << left->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!same_shape(res, *left)) {          // Broadcast
        left = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *left)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        if (!same_shape(res, *left)) {      // Re-check compatibility
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, res, *left, rhs);
    return res;
}

template <typename TO, typename TL, typename TR>
multi_array<TO>& bh_subtract (multi_array<TO>& res, const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_SUBTRACT, TO, TL, TR>();

    multi_array<TR>* right = &rhs;

    if (res.getRank() < right->getRank()) { // Check for unbroadcastable situation
        std::stringstream s;
        s << "Incompatible shapes " << "bh_subtract: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!same_shape(res, *right)) {         // Broadcast
        right = &Runtime::instance().temp_view(rhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        if (!same_shape(res, *right)) {     // Re-check compatibility
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, res, lhs, *right);
    return res;
}

//
// Implicit temporary result array
template <typename TL, typename TR>
multi_array<TL>& bh_subtract (multi_array<TL> &lhs, multi_array<TR> &rhs)
{
    Runtime::instance().typecheck<BH_SUBTRACT, TL, TL, TR>();

    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
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
    multi_array<TL>* result = &Runtime::instance().temp<TL, TR>(*left);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, *result, *left, *right);
    result->setTemp(true);
    return *result;
}

template <typename TL, typename TR>
multi_array<TL>& bh_subtract (multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_SUBTRACT, TL, TL, TR>();

    multi_array<TL>* result = &Runtime::instance().temp<TL, TR>(lhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}

template <typename TL, typename TR>
multi_array<TL>& bh_subtract (const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_SUBTRACT, TL, TL, TR>();

    multi_array<TL>* result = &Runtime::instance().temp<TL, TR>(rhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_SUBTRACT, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}


//
// bh_multiply - BH_MULTIPLY - runtime.binary - 3
//

//
// Explicit result array
template <typename TO, typename TL, typename TR>
multi_array<TO>& bh_multiply (multi_array<TO>& res, multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_MULTIPLY, TO, TL, TR>();

    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(res, *left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(res, *right, *left)) {
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

template <typename TO, typename TL, typename TR>
multi_array<TO>& bh_multiply (multi_array<TO>& res, multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_MULTIPLY, TO, TL, TR>();

    multi_array<TL>* left = &lhs;

    if (res.getRank() < left->getRank()) {  // Check for unbroadcastable situation
        std::stringstream s;
        s << "Incompatible shapes " << "bh_multiply: " << res.getRank() << ", " << left->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!same_shape(res, *left)) {          // Broadcast
        left = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *left)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        if (!same_shape(res, *left)) {      // Re-check compatibility
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, res, *left, rhs);
    return res;
}

template <typename TO, typename TL, typename TR>
multi_array<TO>& bh_multiply (multi_array<TO>& res, const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_MULTIPLY, TO, TL, TR>();

    multi_array<TR>* right = &rhs;

    if (res.getRank() < right->getRank()) { // Check for unbroadcastable situation
        std::stringstream s;
        s << "Incompatible shapes " << "bh_multiply: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!same_shape(res, *right)) {         // Broadcast
        right = &Runtime::instance().temp_view(rhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        if (!same_shape(res, *right)) {     // Re-check compatibility
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, res, lhs, *right);
    return res;
}

//
// Implicit temporary result array
template <typename TL, typename TR>
multi_array<TL>& bh_multiply (multi_array<TL> &lhs, multi_array<TR> &rhs)
{
    Runtime::instance().typecheck<BH_MULTIPLY, TL, TL, TR>();

    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
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
    multi_array<TL>* result = &Runtime::instance().temp<TL, TR>(*left);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, *result, *left, *right);
    result->setTemp(true);
    return *result;
}

template <typename TL, typename TR>
multi_array<TL>& bh_multiply (multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_MULTIPLY, TL, TL, TR>();

    multi_array<TL>* result = &Runtime::instance().temp<TL, TR>(lhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}

template <typename TL, typename TR>
multi_array<TL>& bh_multiply (const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_MULTIPLY, TL, TL, TR>();

    multi_array<TL>* result = &Runtime::instance().temp<TL, TR>(rhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}


//
// bh_divide - BH_DIVIDE - runtime.binary - 3
//

//
// Explicit result array
template <typename TO, typename TL, typename TR>
multi_array<TO>& bh_divide (multi_array<TO>& res, multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_DIVIDE, TO, TL, TR>();

    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(res, *left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(res, *right, *left)) {
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

template <typename TO, typename TL, typename TR>
multi_array<TO>& bh_divide (multi_array<TO>& res, multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_DIVIDE, TO, TL, TR>();

    multi_array<TL>* left = &lhs;

    if (res.getRank() < left->getRank()) {  // Check for unbroadcastable situation
        std::stringstream s;
        s << "Incompatible shapes " << "bh_divide: " << res.getRank() << ", " << left->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!same_shape(res, *left)) {          // Broadcast
        left = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *left)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        if (!same_shape(res, *left)) {      // Re-check compatibility
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE, res, *left, rhs);
    return res;
}

template <typename TO, typename TL, typename TR>
multi_array<TO>& bh_divide (multi_array<TO>& res, const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_DIVIDE, TO, TL, TR>();

    multi_array<TR>* right = &rhs;

    if (res.getRank() < right->getRank()) { // Check for unbroadcastable situation
        std::stringstream s;
        s << "Incompatible shapes " << "bh_divide: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!same_shape(res, *right)) {         // Broadcast
        right = &Runtime::instance().temp_view(rhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        if (!same_shape(res, *right)) {     // Re-check compatibility
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE, res, lhs, *right);
    return res;
}

//
// Implicit temporary result array
template <typename TL, typename TR>
multi_array<TL>& bh_divide (multi_array<TL> &lhs, multi_array<TR> &rhs)
{
    Runtime::instance().typecheck<BH_DIVIDE, TL, TL, TR>();

    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
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
    multi_array<TL>* result = &Runtime::instance().temp<TL, TR>(*left);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE, *result, *left, *right);
    result->setTemp(true);
    return *result;
}

template <typename TL, typename TR>
multi_array<TL>& bh_divide (multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_DIVIDE, TL, TL, TR>();

    multi_array<TL>* result = &Runtime::instance().temp<TL, TR>(lhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}

template <typename TL, typename TR>
multi_array<TL>& bh_divide (const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_DIVIDE, TL, TL, TR>();

    multi_array<TL>* result = &Runtime::instance().temp<TL, TR>(rhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_DIVIDE, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}


//
// bh_mod - BH_MOD - runtime.binary - 3
//

//
// Explicit result array
template <typename TO, typename TL, typename TR>
multi_array<TO>& bh_mod (multi_array<TO>& res, multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_MOD, TO, TL, TR>();

    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(res, *left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(res, *right, *left)) {
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

template <typename TO, typename TL, typename TR>
multi_array<TO>& bh_mod (multi_array<TO>& res, multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_MOD, TO, TL, TR>();

    multi_array<TL>* left = &lhs;

    if (res.getRank() < left->getRank()) {  // Check for unbroadcastable situation
        std::stringstream s;
        s << "Incompatible shapes " << "bh_mod: " << res.getRank() << ", " << left->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!same_shape(res, *left)) {          // Broadcast
        left = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *left)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        if (!same_shape(res, *left)) {      // Re-check compatibility
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_MOD, res, *left, rhs);
    return res;
}

template <typename TO, typename TL, typename TR>
multi_array<TO>& bh_mod (multi_array<TO>& res, const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_MOD, TO, TL, TR>();

    multi_array<TR>* right = &rhs;

    if (res.getRank() < right->getRank()) { // Check for unbroadcastable situation
        std::stringstream s;
        s << "Incompatible shapes " << "bh_mod: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!same_shape(res, *right)) {         // Broadcast
        right = &Runtime::instance().temp_view(rhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        if (!same_shape(res, *right)) {     // Re-check compatibility
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_MOD, res, lhs, *right);
    return res;
}

//
// Implicit temporary result array
template <typename TL, typename TR>
multi_array<TL>& bh_mod (multi_array<TL> &lhs, multi_array<TR> &rhs)
{
    Runtime::instance().typecheck<BH_MOD, TL, TL, TR>();

    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
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
    multi_array<TL>* result = &Runtime::instance().temp<TL, TR>(*left);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_MOD, *result, *left, *right);
    result->setTemp(true);
    return *result;
}

template <typename TL, typename TR>
multi_array<TL>& bh_mod (multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_MOD, TL, TL, TR>();

    multi_array<TL>* result = &Runtime::instance().temp<TL, TR>(lhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_MOD, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}

template <typename TL, typename TR>
multi_array<TL>& bh_mod (const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_MOD, TL, TL, TR>();

    multi_array<TL>* result = &Runtime::instance().temp<TL, TR>(rhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_MOD, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}


//
// bh_bitwise_and - BH_BITWISE_AND - runtime.binary - 3
//

//
// Explicit result array
template <typename TO, typename TL, typename TR>
multi_array<TO>& bh_bitwise_and (multi_array<TO>& res, multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_BITWISE_AND, TO, TL, TR>();

    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(res, *left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(res, *right, *left)) {
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

template <typename TO, typename TL, typename TR>
multi_array<TO>& bh_bitwise_and (multi_array<TO>& res, multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_BITWISE_AND, TO, TL, TR>();

    multi_array<TL>* left = &lhs;

    if (res.getRank() < left->getRank()) {  // Check for unbroadcastable situation
        std::stringstream s;
        s << "Incompatible shapes " << "bh_bitwise_and: " << res.getRank() << ", " << left->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!same_shape(res, *left)) {          // Broadcast
        left = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *left)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        if (!same_shape(res, *left)) {      // Re-check compatibility
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_AND, res, *left, rhs);
    return res;
}

template <typename TO, typename TL, typename TR>
multi_array<TO>& bh_bitwise_and (multi_array<TO>& res, const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_BITWISE_AND, TO, TL, TR>();

    multi_array<TR>* right = &rhs;

    if (res.getRank() < right->getRank()) { // Check for unbroadcastable situation
        std::stringstream s;
        s << "Incompatible shapes " << "bh_bitwise_and: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!same_shape(res, *right)) {         // Broadcast
        right = &Runtime::instance().temp_view(rhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        if (!same_shape(res, *right)) {     // Re-check compatibility
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_AND, res, lhs, *right);
    return res;
}

//
// Implicit temporary result array
template <typename TL, typename TR>
multi_array<TL>& bh_bitwise_and (multi_array<TL> &lhs, multi_array<TR> &rhs)
{
    Runtime::instance().typecheck<BH_BITWISE_AND, TL, TL, TR>();

    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
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
    multi_array<TL>* result = &Runtime::instance().temp<TL, TR>(*left);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_AND, *result, *left, *right);
    result->setTemp(true);
    return *result;
}

template <typename TL, typename TR>
multi_array<TL>& bh_bitwise_and (multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_BITWISE_AND, TL, TL, TR>();

    multi_array<TL>* result = &Runtime::instance().temp<TL, TR>(lhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_AND, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}

template <typename TL, typename TR>
multi_array<TL>& bh_bitwise_and (const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_BITWISE_AND, TL, TL, TR>();

    multi_array<TL>* result = &Runtime::instance().temp<TL, TR>(rhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_AND, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}


//
// bh_bitwise_or - BH_BITWISE_OR - runtime.binary - 3
//

//
// Explicit result array
template <typename TO, typename TL, typename TR>
multi_array<TO>& bh_bitwise_or (multi_array<TO>& res, multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_BITWISE_OR, TO, TL, TR>();

    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(res, *left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(res, *right, *left)) {
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

template <typename TO, typename TL, typename TR>
multi_array<TO>& bh_bitwise_or (multi_array<TO>& res, multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_BITWISE_OR, TO, TL, TR>();

    multi_array<TL>* left = &lhs;

    if (res.getRank() < left->getRank()) {  // Check for unbroadcastable situation
        std::stringstream s;
        s << "Incompatible shapes " << "bh_bitwise_or: " << res.getRank() << ", " << left->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!same_shape(res, *left)) {          // Broadcast
        left = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *left)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        if (!same_shape(res, *left)) {      // Re-check compatibility
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_OR, res, *left, rhs);
    return res;
}

template <typename TO, typename TL, typename TR>
multi_array<TO>& bh_bitwise_or (multi_array<TO>& res, const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_BITWISE_OR, TO, TL, TR>();

    multi_array<TR>* right = &rhs;

    if (res.getRank() < right->getRank()) { // Check for unbroadcastable situation
        std::stringstream s;
        s << "Incompatible shapes " << "bh_bitwise_or: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!same_shape(res, *right)) {         // Broadcast
        right = &Runtime::instance().temp_view(rhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        if (!same_shape(res, *right)) {     // Re-check compatibility
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_OR, res, lhs, *right);
    return res;
}

//
// Implicit temporary result array
template <typename TL, typename TR>
multi_array<TL>& bh_bitwise_or (multi_array<TL> &lhs, multi_array<TR> &rhs)
{
    Runtime::instance().typecheck<BH_BITWISE_OR, TL, TL, TR>();

    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
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
    multi_array<TL>* result = &Runtime::instance().temp<TL, TR>(*left);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_OR, *result, *left, *right);
    result->setTemp(true);
    return *result;
}

template <typename TL, typename TR>
multi_array<TL>& bh_bitwise_or (multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_BITWISE_OR, TL, TL, TR>();

    multi_array<TL>* result = &Runtime::instance().temp<TL, TR>(lhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_OR, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}

template <typename TL, typename TR>
multi_array<TL>& bh_bitwise_or (const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_BITWISE_OR, TL, TL, TR>();

    multi_array<TL>* result = &Runtime::instance().temp<TL, TR>(rhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_OR, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}


//
// bh_bitwise_xor - BH_BITWISE_XOR - runtime.binary - 3
//

//
// Explicit result array
template <typename TO, typename TL, typename TR>
multi_array<TO>& bh_bitwise_xor (multi_array<TO>& res, multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_BITWISE_XOR, TO, TL, TR>();

    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(res, *left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(res, *right, *left)) {
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

template <typename TO, typename TL, typename TR>
multi_array<TO>& bh_bitwise_xor (multi_array<TO>& res, multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_BITWISE_XOR, TO, TL, TR>();

    multi_array<TL>* left = &lhs;

    if (res.getRank() < left->getRank()) {  // Check for unbroadcastable situation
        std::stringstream s;
        s << "Incompatible shapes " << "bh_bitwise_xor: " << res.getRank() << ", " << left->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!same_shape(res, *left)) {          // Broadcast
        left = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *left)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        if (!same_shape(res, *left)) {      // Re-check compatibility
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_XOR, res, *left, rhs);
    return res;
}

template <typename TO, typename TL, typename TR>
multi_array<TO>& bh_bitwise_xor (multi_array<TO>& res, const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_BITWISE_XOR, TO, TL, TR>();

    multi_array<TR>* right = &rhs;

    if (res.getRank() < right->getRank()) { // Check for unbroadcastable situation
        std::stringstream s;
        s << "Incompatible shapes " << "bh_bitwise_xor: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!same_shape(res, *right)) {         // Broadcast
        right = &Runtime::instance().temp_view(rhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        if (!same_shape(res, *right)) {     // Re-check compatibility
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_XOR, res, lhs, *right);
    return res;
}

//
// Implicit temporary result array
template <typename TL, typename TR>
multi_array<TL>& bh_bitwise_xor (multi_array<TL> &lhs, multi_array<TR> &rhs)
{
    Runtime::instance().typecheck<BH_BITWISE_XOR, TL, TL, TR>();

    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
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
    multi_array<TL>* result = &Runtime::instance().temp<TL, TR>(*left);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_XOR, *result, *left, *right);
    result->setTemp(true);
    return *result;
}

template <typename TL, typename TR>
multi_array<TL>& bh_bitwise_xor (multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_BITWISE_XOR, TL, TL, TR>();

    multi_array<TL>* result = &Runtime::instance().temp<TL, TR>(lhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_XOR, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}

template <typename TL, typename TR>
multi_array<TL>& bh_bitwise_xor (const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_BITWISE_XOR, TL, TL, TR>();

    multi_array<TL>* result = &Runtime::instance().temp<TL, TR>(rhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_XOR, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}


//
// bh_left_shift - BH_LEFT_SHIFT - runtime.binary - 3
//

//
// Explicit result array
template <typename TO, typename TL, typename TR>
multi_array<TO>& bh_left_shift (multi_array<TO>& res, multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_LEFT_SHIFT, TO, TL, TR>();

    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(res, *left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(res, *right, *left)) {
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

template <typename TO, typename TL, typename TR>
multi_array<TO>& bh_left_shift (multi_array<TO>& res, multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_LEFT_SHIFT, TO, TL, TR>();

    multi_array<TL>* left = &lhs;

    if (res.getRank() < left->getRank()) {  // Check for unbroadcastable situation
        std::stringstream s;
        s << "Incompatible shapes " << "bh_left_shift: " << res.getRank() << ", " << left->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!same_shape(res, *left)) {          // Broadcast
        left = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *left)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        if (!same_shape(res, *left)) {      // Re-check compatibility
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_LEFT_SHIFT, res, *left, rhs);
    return res;
}

template <typename TO, typename TL, typename TR>
multi_array<TO>& bh_left_shift (multi_array<TO>& res, const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_LEFT_SHIFT, TO, TL, TR>();

    multi_array<TR>* right = &rhs;

    if (res.getRank() < right->getRank()) { // Check for unbroadcastable situation
        std::stringstream s;
        s << "Incompatible shapes " << "bh_left_shift: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!same_shape(res, *right)) {         // Broadcast
        right = &Runtime::instance().temp_view(rhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        if (!same_shape(res, *right)) {     // Re-check compatibility
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_LEFT_SHIFT, res, lhs, *right);
    return res;
}

//
// Implicit temporary result array
template <typename TL, typename TR>
multi_array<TL>& bh_left_shift (multi_array<TL> &lhs, multi_array<TR> &rhs)
{
    Runtime::instance().typecheck<BH_LEFT_SHIFT, TL, TL, TR>();

    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
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
    multi_array<TL>* result = &Runtime::instance().temp<TL, TR>(*left);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_LEFT_SHIFT, *result, *left, *right);
    result->setTemp(true);
    return *result;
}

template <typename TL, typename TR>
multi_array<TL>& bh_left_shift (multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_LEFT_SHIFT, TL, TL, TR>();

    multi_array<TL>* result = &Runtime::instance().temp<TL, TR>(lhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_LEFT_SHIFT, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}

template <typename TL, typename TR>
multi_array<TL>& bh_left_shift (const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_LEFT_SHIFT, TL, TL, TR>();

    multi_array<TL>* result = &Runtime::instance().temp<TL, TR>(rhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_LEFT_SHIFT, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}


//
// bh_right_shift - BH_RIGHT_SHIFT - runtime.binary - 3
//

//
// Explicit result array
template <typename TO, typename TL, typename TR>
multi_array<TO>& bh_right_shift (multi_array<TO>& res, multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_RIGHT_SHIFT, TO, TL, TR>();

    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(res, *left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(res, *right, *left)) {
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

template <typename TO, typename TL, typename TR>
multi_array<TO>& bh_right_shift (multi_array<TO>& res, multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_RIGHT_SHIFT, TO, TL, TR>();

    multi_array<TL>* left = &lhs;

    if (res.getRank() < left->getRank()) {  // Check for unbroadcastable situation
        std::stringstream s;
        s << "Incompatible shapes " << "bh_right_shift: " << res.getRank() << ", " << left->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!same_shape(res, *left)) {          // Broadcast
        left = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *left)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        if (!same_shape(res, *left)) {      // Re-check compatibility
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_RIGHT_SHIFT, res, *left, rhs);
    return res;
}

template <typename TO, typename TL, typename TR>
multi_array<TO>& bh_right_shift (multi_array<TO>& res, const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_RIGHT_SHIFT, TO, TL, TR>();

    multi_array<TR>* right = &rhs;

    if (res.getRank() < right->getRank()) { // Check for unbroadcastable situation
        std::stringstream s;
        s << "Incompatible shapes " << "bh_right_shift: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!same_shape(res, *right)) {         // Broadcast
        right = &Runtime::instance().temp_view(rhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        if (!same_shape(res, *right)) {     // Re-check compatibility
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_RIGHT_SHIFT, res, lhs, *right);
    return res;
}

//
// Implicit temporary result array
template <typename TL, typename TR>
multi_array<TL>& bh_right_shift (multi_array<TL> &lhs, multi_array<TR> &rhs)
{
    Runtime::instance().typecheck<BH_RIGHT_SHIFT, TL, TL, TR>();

    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
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
    multi_array<TL>* result = &Runtime::instance().temp<TL, TR>(*left);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_RIGHT_SHIFT, *result, *left, *right);
    result->setTemp(true);
    return *result;
}

template <typename TL, typename TR>
multi_array<TL>& bh_right_shift (multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_RIGHT_SHIFT, TL, TL, TR>();

    multi_array<TL>* result = &Runtime::instance().temp<TL, TR>(lhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_RIGHT_SHIFT, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}

template <typename TL, typename TR>
multi_array<TL>& bh_right_shift (const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_RIGHT_SHIFT, TL, TL, TR>();

    multi_array<TL>* result = &Runtime::instance().temp<TL, TR>(rhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_RIGHT_SHIFT, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}


//
// bh_logical_and - BH_LOGICAL_AND - runtime.binary - 3
//

//
// Explicit result array
template <typename TO, typename TL, typename TR>
multi_array<TO>& bh_logical_and (multi_array<TO>& res, multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_LOGICAL_AND, TO, TL, TR>();

    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(res, *left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(res, *right, *left)) {
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

template <typename TO, typename TL, typename TR>
multi_array<TO>& bh_logical_and (multi_array<TO>& res, multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_LOGICAL_AND, TO, TL, TR>();

    multi_array<TL>* left = &lhs;

    if (res.getRank() < left->getRank()) {  // Check for unbroadcastable situation
        std::stringstream s;
        s << "Incompatible shapes " << "bh_logical_and: " << res.getRank() << ", " << left->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!same_shape(res, *left)) {          // Broadcast
        left = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *left)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        if (!same_shape(res, *left)) {      // Re-check compatibility
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_AND, res, *left, rhs);
    return res;
}

template <typename TO, typename TL, typename TR>
multi_array<TO>& bh_logical_and (multi_array<TO>& res, const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_LOGICAL_AND, TO, TL, TR>();

    multi_array<TR>* right = &rhs;

    if (res.getRank() < right->getRank()) { // Check for unbroadcastable situation
        std::stringstream s;
        s << "Incompatible shapes " << "bh_logical_and: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!same_shape(res, *right)) {         // Broadcast
        right = &Runtime::instance().temp_view(rhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        if (!same_shape(res, *right)) {     // Re-check compatibility
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_AND, res, lhs, *right);
    return res;
}

//
// Implicit temporary result array
template <typename TL, typename TR>
multi_array<TL>& bh_logical_and (multi_array<TL> &lhs, multi_array<TR> &rhs)
{
    Runtime::instance().typecheck<BH_LOGICAL_AND, TL, TL, TR>();

    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
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
    multi_array<TL>* result = &Runtime::instance().temp<TL, TR>(*left);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_AND, *result, *left, *right);
    result->setTemp(true);
    return *result;
}

template <typename TL, typename TR>
multi_array<TL>& bh_logical_and (multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_LOGICAL_AND, TL, TL, TR>();

    multi_array<TL>* result = &Runtime::instance().temp<TL, TR>(lhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_AND, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}

template <typename TL, typename TR>
multi_array<TL>& bh_logical_and (const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_LOGICAL_AND, TL, TL, TR>();

    multi_array<TL>* result = &Runtime::instance().temp<TL, TR>(rhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_AND, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}


//
// bh_logical_or - BH_LOGICAL_OR - runtime.binary - 3
//

//
// Explicit result array
template <typename TO, typename TL, typename TR>
multi_array<TO>& bh_logical_or (multi_array<TO>& res, multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_LOGICAL_OR, TO, TL, TR>();

    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(res, *left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(res, *right, *left)) {
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

template <typename TO, typename TL, typename TR>
multi_array<TO>& bh_logical_or (multi_array<TO>& res, multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_LOGICAL_OR, TO, TL, TR>();

    multi_array<TL>* left = &lhs;

    if (res.getRank() < left->getRank()) {  // Check for unbroadcastable situation
        std::stringstream s;
        s << "Incompatible shapes " << "bh_logical_or: " << res.getRank() << ", " << left->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!same_shape(res, *left)) {          // Broadcast
        left = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *left)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        if (!same_shape(res, *left)) {      // Re-check compatibility
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_OR, res, *left, rhs);
    return res;
}

template <typename TO, typename TL, typename TR>
multi_array<TO>& bh_logical_or (multi_array<TO>& res, const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_LOGICAL_OR, TO, TL, TR>();

    multi_array<TR>* right = &rhs;

    if (res.getRank() < right->getRank()) { // Check for unbroadcastable situation
        std::stringstream s;
        s << "Incompatible shapes " << "bh_logical_or: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!same_shape(res, *right)) {         // Broadcast
        right = &Runtime::instance().temp_view(rhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        if (!same_shape(res, *right)) {     // Re-check compatibility
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_OR, res, lhs, *right);
    return res;
}

//
// Implicit temporary result array
template <typename TL, typename TR>
multi_array<TL>& bh_logical_or (multi_array<TL> &lhs, multi_array<TR> &rhs)
{
    Runtime::instance().typecheck<BH_LOGICAL_OR, TL, TL, TR>();

    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
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
    multi_array<TL>* result = &Runtime::instance().temp<TL, TR>(*left);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_OR, *result, *left, *right);
    result->setTemp(true);
    return *result;
}

template <typename TL, typename TR>
multi_array<TL>& bh_logical_or (multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_LOGICAL_OR, TL, TL, TR>();

    multi_array<TL>* result = &Runtime::instance().temp<TL, TR>(lhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_OR, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}

template <typename TL, typename TR>
multi_array<TL>& bh_logical_or (const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_LOGICAL_OR, TL, TL, TR>();

    multi_array<TL>* result = &Runtime::instance().temp<TL, TR>(rhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_OR, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}


//
// bh_logical_xor - BH_LOGICAL_XOR - runtime.binary - 3
//

//
// Explicit result array
template <typename TO, typename TL, typename TR>
multi_array<TO>& bh_logical_xor (multi_array<TO>& res, multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_LOGICAL_XOR, TO, TL, TR>();

    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(res, *left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(res, *right, *left)) {
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

template <typename TO, typename TL, typename TR>
multi_array<TO>& bh_logical_xor (multi_array<TO>& res, multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_LOGICAL_XOR, TO, TL, TR>();

    multi_array<TL>* left = &lhs;

    if (res.getRank() < left->getRank()) {  // Check for unbroadcastable situation
        std::stringstream s;
        s << "Incompatible shapes " << "bh_logical_xor: " << res.getRank() << ", " << left->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!same_shape(res, *left)) {          // Broadcast
        left = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *left)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        if (!same_shape(res, *left)) {      // Re-check compatibility
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_XOR, res, *left, rhs);
    return res;
}

template <typename TO, typename TL, typename TR>
multi_array<TO>& bh_logical_xor (multi_array<TO>& res, const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_LOGICAL_XOR, TO, TL, TR>();

    multi_array<TR>* right = &rhs;

    if (res.getRank() < right->getRank()) { // Check for unbroadcastable situation
        std::stringstream s;
        s << "Incompatible shapes " << "bh_logical_xor: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!same_shape(res, *right)) {         // Broadcast
        right = &Runtime::instance().temp_view(rhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        if (!same_shape(res, *right)) {     // Re-check compatibility
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_XOR, res, lhs, *right);
    return res;
}

//
// Implicit temporary result array
template <typename TL, typename TR>
multi_array<TL>& bh_logical_xor (multi_array<TL> &lhs, multi_array<TR> &rhs)
{
    Runtime::instance().typecheck<BH_LOGICAL_XOR, TL, TL, TR>();

    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
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
    multi_array<TL>* result = &Runtime::instance().temp<TL, TR>(*left);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_XOR, *result, *left, *right);
    result->setTemp(true);
    return *result;
}

template <typename TL, typename TR>
multi_array<TL>& bh_logical_xor (multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_LOGICAL_XOR, TL, TL, TR>();

    multi_array<TL>* result = &Runtime::instance().temp<TL, TR>(lhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_XOR, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}

template <typename TL, typename TR>
multi_array<TL>& bh_logical_xor (const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_LOGICAL_XOR, TL, TL, TR>();

    multi_array<TL>* result = &Runtime::instance().temp<TL, TR>(rhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_XOR, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}


//
// bh_power - BH_POWER - runtime.binary - 3
//

//
// Explicit result array
template <typename TO, typename TL, typename TR>
multi_array<TO>& bh_power (multi_array<TO>& res, multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_POWER, TO, TL, TR>();

    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(res, *left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(res, *right, *left)) {
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

template <typename TO, typename TL, typename TR>
multi_array<TO>& bh_power (multi_array<TO>& res, multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_POWER, TO, TL, TR>();

    multi_array<TL>* left = &lhs;

    if (res.getRank() < left->getRank()) {  // Check for unbroadcastable situation
        std::stringstream s;
        s << "Incompatible shapes " << "bh_power: " << res.getRank() << ", " << left->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!same_shape(res, *left)) {          // Broadcast
        left = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *left)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        if (!same_shape(res, *left)) {      // Re-check compatibility
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_POWER, res, *left, rhs);
    return res;
}

template <typename TO, typename TL, typename TR>
multi_array<TO>& bh_power (multi_array<TO>& res, const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_POWER, TO, TL, TR>();

    multi_array<TR>* right = &rhs;

    if (res.getRank() < right->getRank()) { // Check for unbroadcastable situation
        std::stringstream s;
        s << "Incompatible shapes " << "bh_power: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!same_shape(res, *right)) {         // Broadcast
        right = &Runtime::instance().temp_view(rhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        if (!same_shape(res, *right)) {     // Re-check compatibility
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_POWER, res, lhs, *right);
    return res;
}

//
// Implicit temporary result array
template <typename TL, typename TR>
multi_array<TL>& bh_power (multi_array<TL> &lhs, multi_array<TR> &rhs)
{
    Runtime::instance().typecheck<BH_POWER, TL, TL, TR>();

    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
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
    multi_array<TL>* result = &Runtime::instance().temp<TL, TR>(*left);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_POWER, *result, *left, *right);
    result->setTemp(true);
    return *result;
}

template <typename TL, typename TR>
multi_array<TL>& bh_power (multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_POWER, TL, TL, TR>();

    multi_array<TL>* result = &Runtime::instance().temp<TL, TR>(lhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_POWER, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}

template <typename TL, typename TR>
multi_array<TL>& bh_power (const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_POWER, TL, TL, TR>();

    multi_array<TL>* result = &Runtime::instance().temp<TL, TR>(rhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_POWER, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}


//
// bh_maximum - BH_MAXIMUM - runtime.binary - 3
//

//
// Explicit result array
template <typename TO, typename TL, typename TR>
multi_array<TO>& bh_maximum (multi_array<TO>& res, multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_MAXIMUM, TO, TL, TR>();

    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(res, *left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(res, *right, *left)) {
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

template <typename TO, typename TL, typename TR>
multi_array<TO>& bh_maximum (multi_array<TO>& res, multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_MAXIMUM, TO, TL, TR>();

    multi_array<TL>* left = &lhs;

    if (res.getRank() < left->getRank()) {  // Check for unbroadcastable situation
        std::stringstream s;
        s << "Incompatible shapes " << "bh_maximum: " << res.getRank() << ", " << left->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!same_shape(res, *left)) {          // Broadcast
        left = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *left)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        if (!same_shape(res, *left)) {      // Re-check compatibility
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_MAXIMUM, res, *left, rhs);
    return res;
}

template <typename TO, typename TL, typename TR>
multi_array<TO>& bh_maximum (multi_array<TO>& res, const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_MAXIMUM, TO, TL, TR>();

    multi_array<TR>* right = &rhs;

    if (res.getRank() < right->getRank()) { // Check for unbroadcastable situation
        std::stringstream s;
        s << "Incompatible shapes " << "bh_maximum: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!same_shape(res, *right)) {         // Broadcast
        right = &Runtime::instance().temp_view(rhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        if (!same_shape(res, *right)) {     // Re-check compatibility
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_MAXIMUM, res, lhs, *right);
    return res;
}

//
// Implicit temporary result array
template <typename TL, typename TR>
multi_array<TL>& bh_maximum (multi_array<TL> &lhs, multi_array<TR> &rhs)
{
    Runtime::instance().typecheck<BH_MAXIMUM, TL, TL, TR>();

    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
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
    multi_array<TL>* result = &Runtime::instance().temp<TL, TR>(*left);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_MAXIMUM, *result, *left, *right);
    result->setTemp(true);
    return *result;
}

template <typename TL, typename TR>
multi_array<TL>& bh_maximum (multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_MAXIMUM, TL, TL, TR>();

    multi_array<TL>* result = &Runtime::instance().temp<TL, TR>(lhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_MAXIMUM, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}

template <typename TL, typename TR>
multi_array<TL>& bh_maximum (const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_MAXIMUM, TL, TL, TR>();

    multi_array<TL>* result = &Runtime::instance().temp<TL, TR>(rhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_MAXIMUM, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}


//
// bh_minimum - BH_MINIMUM - runtime.binary - 3
//

//
// Explicit result array
template <typename TO, typename TL, typename TR>
multi_array<TO>& bh_minimum (multi_array<TO>& res, multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_MINIMUM, TO, TL, TR>();

    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(res, *left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(res, *right, *left)) {
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

template <typename TO, typename TL, typename TR>
multi_array<TO>& bh_minimum (multi_array<TO>& res, multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_MINIMUM, TO, TL, TR>();

    multi_array<TL>* left = &lhs;

    if (res.getRank() < left->getRank()) {  // Check for unbroadcastable situation
        std::stringstream s;
        s << "Incompatible shapes " << "bh_minimum: " << res.getRank() << ", " << left->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!same_shape(res, *left)) {          // Broadcast
        left = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *left)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        if (!same_shape(res, *left)) {      // Re-check compatibility
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_MINIMUM, res, *left, rhs);
    return res;
}

template <typename TO, typename TL, typename TR>
multi_array<TO>& bh_minimum (multi_array<TO>& res, const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_MINIMUM, TO, TL, TR>();

    multi_array<TR>* right = &rhs;

    if (res.getRank() < right->getRank()) { // Check for unbroadcastable situation
        std::stringstream s;
        s << "Incompatible shapes " << "bh_minimum: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!same_shape(res, *right)) {         // Broadcast
        right = &Runtime::instance().temp_view(rhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        if (!same_shape(res, *right)) {     // Re-check compatibility
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_MINIMUM, res, lhs, *right);
    return res;
}

//
// Implicit temporary result array
template <typename TL, typename TR>
multi_array<TL>& bh_minimum (multi_array<TL> &lhs, multi_array<TR> &rhs)
{
    Runtime::instance().typecheck<BH_MINIMUM, TL, TL, TR>();

    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
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
    multi_array<TL>* result = &Runtime::instance().temp<TL, TR>(*left);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_MINIMUM, *result, *left, *right);
    result->setTemp(true);
    return *result;
}

template <typename TL, typename TR>
multi_array<TL>& bh_minimum (multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_MINIMUM, TL, TL, TR>();

    multi_array<TL>* result = &Runtime::instance().temp<TL, TR>(lhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_MINIMUM, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}

template <typename TL, typename TR>
multi_array<TL>& bh_minimum (const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_MINIMUM, TL, TL, TR>();

    multi_array<TL>* result = &Runtime::instance().temp<TL, TR>(rhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_MINIMUM, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}


//
// bh_arctan2 - BH_ARCTAN2 - runtime.binary - 3
//

//
// Explicit result array
template <typename TO, typename TL, typename TR>
multi_array<TO>& bh_arctan2 (multi_array<TO>& res, multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_ARCTAN2, TO, TL, TR>();

    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
        left    = &Runtime::instance().temp_view(lhs);
        right   = &Runtime::instance().temp_view(rhs);

        if (lhs.getRank() < rhs.getRank()) {    // Left-handside has lowest rank
            if (!broadcast(res, *left, *right)) {
                throw std::runtime_error("Failed broadcasting.");
            }
        } else {                                // Right-handside has lowest rank
            if (!broadcast(res, *right, *left)) {
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

    Runtime::instance().enqueue((bh_opcode)BH_ARCTAN2, res, *left, *right);

    return res;
}

template <typename TO, typename TL, typename TR>
multi_array<TO>& bh_arctan2 (multi_array<TO>& res, multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_ARCTAN2, TO, TL, TR>();

    multi_array<TL>* left = &lhs;

    if (res.getRank() < left->getRank()) {  // Check for unbroadcastable situation
        std::stringstream s;
        s << "Incompatible shapes " << "bh_arctan2: " << res.getRank() << ", " << left->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!same_shape(res, *left)) {          // Broadcast
        left = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *left)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        if (!same_shape(res, *left)) {      // Re-check compatibility
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_ARCTAN2, res, *left, rhs);
    return res;
}

template <typename TO, typename TL, typename TR>
multi_array<TO>& bh_arctan2 (multi_array<TO>& res, const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_ARCTAN2, TO, TL, TR>();

    multi_array<TR>* right = &rhs;

    if (res.getRank() < right->getRank()) { // Check for unbroadcastable situation
        std::stringstream s;
        s << "Incompatible shapes " << "bh_arctan2: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!same_shape(res, *right)) {         // Broadcast
        right = &Runtime::instance().temp_view(rhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        if (!same_shape(res, *right)) {     // Re-check compatibility
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_ARCTAN2, res, lhs, *right);
    return res;
}

//
// Implicit temporary result array
template <typename TL, typename TR>
multi_array<TL>& bh_arctan2 (multi_array<TL> &lhs, multi_array<TR> &rhs)
{
    Runtime::instance().typecheck<BH_ARCTAN2, TL, TL, TR>();

    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
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
    multi_array<TL>* result = &Runtime::instance().temp<TL, TR>(*left);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_ARCTAN2, *result, *left, *right);
    result->setTemp(true);
    return *result;
}

template <typename TL, typename TR>
multi_array<TL>& bh_arctan2 (multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_ARCTAN2, TL, TL, TR>();

    multi_array<TL>* result = &Runtime::instance().temp<TL, TR>(lhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_ARCTAN2, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}

template <typename TL, typename TR>
multi_array<TL>& bh_arctan2 (const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_ARCTAN2, TL, TL, TR>();

    multi_array<TL>* result = &Runtime::instance().temp<TL, TR>(rhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_ARCTAN2, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}


//
// bh_scatter - BH_SCATTER - runtime.binary - 3
//

//
// Explicit result array
template <typename TO, typename TL, typename TR>
multi_array<TO>& bh_scatter (multi_array<TO>& res, multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_SCATTER, TO, TL, TR>();

    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    

    Runtime::instance().enqueue((bh_opcode)BH_SCATTER, res, *left, *right);

    return res;
}



//
// Implicit temporary result array
template <typename TL, typename TR>
multi_array<TL>& bh_scatter (multi_array<TL> &lhs, multi_array<TR> &rhs)
{
    Runtime::instance().typecheck<BH_SCATTER, TL, TL, TR>();

    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    

    // Construct output / result array
    multi_array<TL>* result = &Runtime::instance().temp<TL, TR>(*left);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_SCATTER, *result, *left, *right);
    result->setTemp(true);
    return *result;
}




//
// bh_gather - BH_GATHER - runtime.binary - 3
//

//
// Explicit result array
template <typename TO, typename TL, typename TR>
multi_array<TO>& bh_gather (multi_array<TO>& res, multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_GATHER, TO, TL, TR>();

    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    

    Runtime::instance().enqueue((bh_opcode)BH_GATHER, res, *left, *right);

    return res;
}



//
// Implicit temporary result array
template <typename TL, typename TR>
multi_array<TL>& bh_gather (multi_array<TL> &lhs, multi_array<TR> &rhs)
{
    Runtime::instance().typecheck<BH_GATHER, TL, TL, TR>();

    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    

    // Construct output / result array
    multi_array<TL>* result = &Runtime::instance().temp<TL, TR>(*left);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_GATHER, *result, *left, *right);
    result->setTemp(true);
    return *result;
}




//
// bh_matmul - BH_MATMUL - runtime.binary - 3
//

//
// Explicit result array
template <typename TO, typename TL, typename TR>
multi_array<TO>& bh_matmul (multi_array<TO>& res, multi_array<TL>& lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_MATMUL, TO, TL, TR>();

    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    

    Runtime::instance().enqueue((bh_opcode)BH_MATMUL, res, *left, *right);

    return res;
}

template <typename TO, typename TL, typename TR>
multi_array<TO>& bh_matmul (multi_array<TO>& res, multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_MATMUL, TO, TL, TR>();

    multi_array<TL>* left = &lhs;


    Runtime::instance().enqueue((bh_opcode)BH_MATMUL, res, *left, rhs);
    return res;
}

template <typename TO, typename TL, typename TR>
multi_array<TO>& bh_matmul (multi_array<TO>& res, const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_MATMUL, TO, TL, TR>();

    multi_array<TR>* right = &rhs;


    Runtime::instance().enqueue((bh_opcode)BH_MATMUL, res, lhs, *right);
    return res;
}

//
// Implicit temporary result array
template <typename TL, typename TR>
multi_array<TL>& bh_matmul (multi_array<TL> &lhs, multi_array<TR> &rhs)
{
    Runtime::instance().typecheck<BH_MATMUL, TL, TL, TR>();

    multi_array<TL>* left    = &lhs;
    multi_array<TR>* right   = &rhs;
    

    // Construct output / result array
    multi_array<TL>* result = &Runtime::instance().temp<TL, TR>(*left);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_MATMUL, *result, *left, *right);
    result->setTemp(true);
    return *result;
}

template <typename TL, typename TR>
multi_array<TL>& bh_matmul (multi_array<TL>& lhs, const TR rhs)
{
    Runtime::instance().typecheck<BH_MATMUL, TL, TL, TR>();

    multi_array<TL>* result = &Runtime::instance().temp<TL, TR>(lhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_MATMUL, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}

template <typename TL, typename TR>
multi_array<TL>& bh_matmul (const TL lhs, multi_array<TR>& rhs)
{
    Runtime::instance().typecheck<BH_MATMUL, TL, TL, TR>();

    multi_array<TL>* result = &Runtime::instance().temp<TL, TR>(rhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_MATMUL, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}




//
// bh_equal - BH_EQUAL - runtime.binary.bool - 3
//

//
// Explicit result array
template <typename OutT, typename InT>
multi_array<OutT>& bh_equal (multi_array<OutT>& res, multi_array<InT>& lhs, multi_array<InT>& rhs)
{
    Runtime::instance().typecheck<BH_EQUAL, OutT, InT, InT>();

    multi_array<InT>* left    = &lhs;
    multi_array<InT>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
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
    Runtime::instance().typecheck<BH_EQUAL, OutT, InT, InT>();

    multi_array<InT>* left = &lhs;

    if (res.getRank() < left->getRank()) {  // Check for unbroadcastable situation
        std::stringstream s;
        s << "Incompatible shapes " << "bh_equal: " << res.getRank() << ", " << left->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!same_shape(res, *left)) {          // Broadcast
        left = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *left)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        if (!same_shape(res, *left)) {      // Re-check compatibility
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, res, *left, rhs);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_equal (multi_array<OutT>& res, const InT lhs, multi_array<InT>& rhs)
{
    Runtime::instance().typecheck<BH_EQUAL, OutT, InT, InT>();

    multi_array<InT>* right = &rhs;

    if (res.getRank() < right->getRank()) { // Check for unbroadcastable situation
        std::stringstream s;
        s << "Incompatible shapes " << "bh_equal: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!same_shape(res, *right)) {         // Broadcast
        right = &Runtime::instance().temp_view(rhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        if (!same_shape(res, *right)) {     // Re-check compatibility
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, res, lhs, *right);
    return res;
}

//
// Implicit temporary result array
template <typename T>
multi_array<bool>& bh_equal (multi_array<T> &lhs, multi_array<T> &rhs)
{
    Runtime::instance().typecheck<BH_EQUAL, bool, T, T>();

    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
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
    multi_array<bool>* result = &Runtime::instance().temp<bool, T>(*left);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, *result, *left, *right);
    result->setTemp(true);
    return *result;
}

template <typename T>
multi_array<bool>& bh_equal (multi_array<T>& lhs, const T rhs)
{
    Runtime::instance().typecheck<BH_EQUAL, bool, T, T>();

    multi_array<bool>* result = &Runtime::instance().temp<bool, T>(lhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}

template <typename T>
multi_array<bool>& bh_equal (const T lhs, multi_array<T>& rhs)
{
    Runtime::instance().typecheck<BH_EQUAL, bool, T, T>();

    multi_array<bool>* result = &Runtime::instance().temp<bool, T>(rhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_EQUAL, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}


//
// bh_not_equal - BH_NOT_EQUAL - runtime.binary.bool - 3
//

//
// Explicit result array
template <typename OutT, typename InT>
multi_array<OutT>& bh_not_equal (multi_array<OutT>& res, multi_array<InT>& lhs, multi_array<InT>& rhs)
{
    Runtime::instance().typecheck<BH_NOT_EQUAL, OutT, InT, InT>();

    multi_array<InT>* left    = &lhs;
    multi_array<InT>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
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
    Runtime::instance().typecheck<BH_NOT_EQUAL, OutT, InT, InT>();

    multi_array<InT>* left = &lhs;

    if (res.getRank() < left->getRank()) {  // Check for unbroadcastable situation
        std::stringstream s;
        s << "Incompatible shapes " << "bh_not_equal: " << res.getRank() << ", " << left->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!same_shape(res, *left)) {          // Broadcast
        left = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *left)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        if (!same_shape(res, *left)) {      // Re-check compatibility
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, res, *left, rhs);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_not_equal (multi_array<OutT>& res, const InT lhs, multi_array<InT>& rhs)
{
    Runtime::instance().typecheck<BH_NOT_EQUAL, OutT, InT, InT>();

    multi_array<InT>* right = &rhs;

    if (res.getRank() < right->getRank()) { // Check for unbroadcastable situation
        std::stringstream s;
        s << "Incompatible shapes " << "bh_not_equal: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!same_shape(res, *right)) {         // Broadcast
        right = &Runtime::instance().temp_view(rhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        if (!same_shape(res, *right)) {     // Re-check compatibility
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, res, lhs, *right);
    return res;
}

//
// Implicit temporary result array
template <typename T>
multi_array<bool>& bh_not_equal (multi_array<T> &lhs, multi_array<T> &rhs)
{
    Runtime::instance().typecheck<BH_NOT_EQUAL, bool, T, T>();

    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
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
    multi_array<bool>* result = &Runtime::instance().temp<bool, T>(*left);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, *result, *left, *right);
    result->setTemp(true);
    return *result;
}

template <typename T>
multi_array<bool>& bh_not_equal (multi_array<T>& lhs, const T rhs)
{
    Runtime::instance().typecheck<BH_NOT_EQUAL, bool, T, T>();

    multi_array<bool>* result = &Runtime::instance().temp<bool, T>(lhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}

template <typename T>
multi_array<bool>& bh_not_equal (const T lhs, multi_array<T>& rhs)
{
    Runtime::instance().typecheck<BH_NOT_EQUAL, bool, T, T>();

    multi_array<bool>* result = &Runtime::instance().temp<bool, T>(rhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_NOT_EQUAL, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}


//
// bh_greater - BH_GREATER - runtime.binary.bool - 3
//

//
// Explicit result array
template <typename OutT, typename InT>
multi_array<OutT>& bh_greater (multi_array<OutT>& res, multi_array<InT>& lhs, multi_array<InT>& rhs)
{
    Runtime::instance().typecheck<BH_GREATER, OutT, InT, InT>();

    multi_array<InT>* left    = &lhs;
    multi_array<InT>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
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
    Runtime::instance().typecheck<BH_GREATER, OutT, InT, InT>();

    multi_array<InT>* left = &lhs;

    if (res.getRank() < left->getRank()) {  // Check for unbroadcastable situation
        std::stringstream s;
        s << "Incompatible shapes " << "bh_greater: " << res.getRank() << ", " << left->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!same_shape(res, *left)) {          // Broadcast
        left = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *left)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        if (!same_shape(res, *left)) {      // Re-check compatibility
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_GREATER, res, *left, rhs);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_greater (multi_array<OutT>& res, const InT lhs, multi_array<InT>& rhs)
{
    Runtime::instance().typecheck<BH_GREATER, OutT, InT, InT>();

    multi_array<InT>* right = &rhs;

    if (res.getRank() < right->getRank()) { // Check for unbroadcastable situation
        std::stringstream s;
        s << "Incompatible shapes " << "bh_greater: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!same_shape(res, *right)) {         // Broadcast
        right = &Runtime::instance().temp_view(rhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        if (!same_shape(res, *right)) {     // Re-check compatibility
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_GREATER, res, lhs, *right);
    return res;
}

//
// Implicit temporary result array
template <typename T>
multi_array<bool>& bh_greater (multi_array<T> &lhs, multi_array<T> &rhs)
{
    Runtime::instance().typecheck<BH_GREATER, bool, T, T>();

    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
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
    multi_array<bool>* result = &Runtime::instance().temp<bool, T>(*left);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_GREATER, *result, *left, *right);
    result->setTemp(true);
    return *result;
}

template <typename T>
multi_array<bool>& bh_greater (multi_array<T>& lhs, const T rhs)
{
    Runtime::instance().typecheck<BH_GREATER, bool, T, T>();

    multi_array<bool>* result = &Runtime::instance().temp<bool, T>(lhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_GREATER, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}

template <typename T>
multi_array<bool>& bh_greater (const T lhs, multi_array<T>& rhs)
{
    Runtime::instance().typecheck<BH_GREATER, bool, T, T>();

    multi_array<bool>* result = &Runtime::instance().temp<bool, T>(rhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_GREATER, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}


//
// bh_greater_equal - BH_GREATER_EQUAL - runtime.binary.bool - 3
//

//
// Explicit result array
template <typename OutT, typename InT>
multi_array<OutT>& bh_greater_equal (multi_array<OutT>& res, multi_array<InT>& lhs, multi_array<InT>& rhs)
{
    Runtime::instance().typecheck<BH_GREATER_EQUAL, OutT, InT, InT>();

    multi_array<InT>* left    = &lhs;
    multi_array<InT>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
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
multi_array<OutT>& bh_greater_equal (multi_array<OutT>& res, multi_array<InT>& lhs, const InT rhs)
{
    Runtime::instance().typecheck<BH_GREATER_EQUAL, OutT, InT, InT>();

    multi_array<InT>* left = &lhs;

    if (res.getRank() < left->getRank()) {  // Check for unbroadcastable situation
        std::stringstream s;
        s << "Incompatible shapes " << "bh_greater_equal: " << res.getRank() << ", " << left->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!same_shape(res, *left)) {          // Broadcast
        left = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *left)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        if (!same_shape(res, *left)) {      // Re-check compatibility
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_GREATER_EQUAL, res, *left, rhs);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_greater_equal (multi_array<OutT>& res, const InT lhs, multi_array<InT>& rhs)
{
    Runtime::instance().typecheck<BH_GREATER_EQUAL, OutT, InT, InT>();

    multi_array<InT>* right = &rhs;

    if (res.getRank() < right->getRank()) { // Check for unbroadcastable situation
        std::stringstream s;
        s << "Incompatible shapes " << "bh_greater_equal: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!same_shape(res, *right)) {         // Broadcast
        right = &Runtime::instance().temp_view(rhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        if (!same_shape(res, *right)) {     // Re-check compatibility
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_GREATER_EQUAL, res, lhs, *right);
    return res;
}

//
// Implicit temporary result array
template <typename T>
multi_array<bool>& bh_greater_equal (multi_array<T> &lhs, multi_array<T> &rhs)
{
    Runtime::instance().typecheck<BH_GREATER_EQUAL, bool, T, T>();

    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
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
    multi_array<bool>* result = &Runtime::instance().temp<bool, T>(*left);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_GREATER_EQUAL, *result, *left, *right);
    result->setTemp(true);
    return *result;
}

template <typename T>
multi_array<bool>& bh_greater_equal (multi_array<T>& lhs, const T rhs)
{
    Runtime::instance().typecheck<BH_GREATER_EQUAL, bool, T, T>();

    multi_array<bool>* result = &Runtime::instance().temp<bool, T>(lhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_GREATER_EQUAL, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}

template <typename T>
multi_array<bool>& bh_greater_equal (const T lhs, multi_array<T>& rhs)
{
    Runtime::instance().typecheck<BH_GREATER_EQUAL, bool, T, T>();

    multi_array<bool>* result = &Runtime::instance().temp<bool, T>(rhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_GREATER_EQUAL, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}


//
// bh_less - BH_LESS - runtime.binary.bool - 3
//

//
// Explicit result array
template <typename OutT, typename InT>
multi_array<OutT>& bh_less (multi_array<OutT>& res, multi_array<InT>& lhs, multi_array<InT>& rhs)
{
    Runtime::instance().typecheck<BH_LESS, OutT, InT, InT>();

    multi_array<InT>* left    = &lhs;
    multi_array<InT>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
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
    Runtime::instance().typecheck<BH_LESS, OutT, InT, InT>();

    multi_array<InT>* left = &lhs;

    if (res.getRank() < left->getRank()) {  // Check for unbroadcastable situation
        std::stringstream s;
        s << "Incompatible shapes " << "bh_less: " << res.getRank() << ", " << left->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!same_shape(res, *left)) {          // Broadcast
        left = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *left)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        if (!same_shape(res, *left)) {      // Re-check compatibility
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_LESS, res, *left, rhs);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_less (multi_array<OutT>& res, const InT lhs, multi_array<InT>& rhs)
{
    Runtime::instance().typecheck<BH_LESS, OutT, InT, InT>();

    multi_array<InT>* right = &rhs;

    if (res.getRank() < right->getRank()) { // Check for unbroadcastable situation
        std::stringstream s;
        s << "Incompatible shapes " << "bh_less: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!same_shape(res, *right)) {         // Broadcast
        right = &Runtime::instance().temp_view(rhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        if (!same_shape(res, *right)) {     // Re-check compatibility
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_LESS, res, lhs, *right);
    return res;
}

//
// Implicit temporary result array
template <typename T>
multi_array<bool>& bh_less (multi_array<T> &lhs, multi_array<T> &rhs)
{
    Runtime::instance().typecheck<BH_LESS, bool, T, T>();

    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
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
    multi_array<bool>* result = &Runtime::instance().temp<bool, T>(*left);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_LESS, *result, *left, *right);
    result->setTemp(true);
    return *result;
}

template <typename T>
multi_array<bool>& bh_less (multi_array<T>& lhs, const T rhs)
{
    Runtime::instance().typecheck<BH_LESS, bool, T, T>();

    multi_array<bool>* result = &Runtime::instance().temp<bool, T>(lhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_LESS, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}

template <typename T>
multi_array<bool>& bh_less (const T lhs, multi_array<T>& rhs)
{
    Runtime::instance().typecheck<BH_LESS, bool, T, T>();

    multi_array<bool>* result = &Runtime::instance().temp<bool, T>(rhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_LESS, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}


//
// bh_less_equal - BH_LESS_EQUAL - runtime.binary.bool - 3
//

//
// Explicit result array
template <typename OutT, typename InT>
multi_array<OutT>& bh_less_equal (multi_array<OutT>& res, multi_array<InT>& lhs, multi_array<InT>& rhs)
{
    Runtime::instance().typecheck<BH_LESS_EQUAL, OutT, InT, InT>();

    multi_array<InT>* left    = &lhs;
    multi_array<InT>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
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
multi_array<OutT>& bh_less_equal (multi_array<OutT>& res, multi_array<InT>& lhs, const InT rhs)
{
    Runtime::instance().typecheck<BH_LESS_EQUAL, OutT, InT, InT>();

    multi_array<InT>* left = &lhs;

    if (res.getRank() < left->getRank()) {  // Check for unbroadcastable situation
        std::stringstream s;
        s << "Incompatible shapes " << "bh_less_equal: " << res.getRank() << ", " << left->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!same_shape(res, *left)) {          // Broadcast
        left = &Runtime::instance().temp_view(lhs);
        
        if (!broadcast_right(res, *left)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        if (!same_shape(res, *left)) {      // Re-check compatibility
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_LESS_EQUAL, res, *left, rhs);
    return res;
}

template <typename OutT, typename InT>
multi_array<OutT>& bh_less_equal (multi_array<OutT>& res, const InT lhs, multi_array<InT>& rhs)
{
    Runtime::instance().typecheck<BH_LESS_EQUAL, OutT, InT, InT>();

    multi_array<InT>* right = &rhs;

    if (res.getRank() < right->getRank()) { // Check for unbroadcastable situation
        std::stringstream s;
        s << "Incompatible shapes " << "bh_less_equal: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
        throw std::runtime_error(s.str());
    }

    if (!same_shape(res, *right)) {         // Broadcast
        right = &Runtime::instance().temp_view(rhs);
        
        if (!broadcast_right(res, *right)) {
            throw std::runtime_error("LHS is not broadcastable.");
        }
        
        if (!same_shape(res, *right)) {     // Re-check compatibility
            throw std::runtime_error("Incompatable shapes after attempted broadcast.");
        }
    }

    Runtime::instance().enqueue((bh_opcode)BH_LESS_EQUAL, res, lhs, *right);
    return res;
}

//
// Implicit temporary result array
template <typename T>
multi_array<bool>& bh_less_equal (multi_array<T> &lhs, multi_array<T> &rhs)
{
    Runtime::instance().typecheck<BH_LESS_EQUAL, bool, T, T>();

    multi_array<T>* left    = &lhs;
    multi_array<T>* right   = &rhs;
    
    if (!same_shape(*left, *right)) {           // Broadcast
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
    multi_array<bool>* result = &Runtime::instance().temp<bool, T>(*left);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_LESS_EQUAL, *result, *left, *right);
    result->setTemp(true);
    return *result;
}

template <typename T>
multi_array<bool>& bh_less_equal (multi_array<T>& lhs, const T rhs)
{
    Runtime::instance().typecheck<BH_LESS_EQUAL, bool, T, T>();

    multi_array<bool>* result = &Runtime::instance().temp<bool, T>(lhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_LESS_EQUAL, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}

template <typename T>
multi_array<bool>& bh_less_equal (const T lhs, multi_array<T>& rhs)
{
    Runtime::instance().typecheck<BH_LESS_EQUAL, bool, T, T>();

    multi_array<bool>* result = &Runtime::instance().temp<bool, T>(rhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_LESS_EQUAL, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}



//
// bh_identity - BH_IDENTITY - runtime.unary - 2
//

//
//  Explicit result array
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_identity (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_IDENTITY, OutT, InT>();

    multi_array<InT>* right = &rhs;
    
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
inline
multi_array<OutT>& bh_identity (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_IDENTITY, OutT, InT>();

    Runtime::instance().enqueue((bh_opcode)BH_IDENTITY, res, rhs);
    return res;
}

//
//  Implicit result array
template <typename T>
inline
multi_array<T>& bh_identity (multi_array<T>& rhs)
{
    Runtime::instance().typecheck<BH_IDENTITY, T, T>();

    multi_array<T>* result = &Runtime::instance().temp<T, T>(rhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_IDENTITY, *result, rhs);
    result->setTemp(true);
    return *result;
}

//
// bh_logical_not - BH_LOGICAL_NOT - runtime.unary - 2
//

//
//  Explicit result array
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_logical_not (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_LOGICAL_NOT, OutT, InT>();

    multi_array<InT>* right = &rhs;
    
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
inline
multi_array<OutT>& bh_logical_not (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_LOGICAL_NOT, OutT, InT>();

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_NOT, res, rhs);
    return res;
}

//
//  Implicit result array
template <typename T>
inline
multi_array<T>& bh_logical_not (multi_array<T>& rhs)
{
    Runtime::instance().typecheck<BH_LOGICAL_NOT, T, T>();

    multi_array<T>* result = &Runtime::instance().temp<T, T>(rhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_NOT, *result, rhs);
    result->setTemp(true);
    return *result;
}

//
// bh_invert - BH_INVERT - runtime.unary - 2
//

//
//  Explicit result array
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_invert (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_INVERT, OutT, InT>();

    multi_array<InT>* right = &rhs;
    
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
inline
multi_array<OutT>& bh_invert (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_INVERT, OutT, InT>();

    Runtime::instance().enqueue((bh_opcode)BH_INVERT, res, rhs);
    return res;
}

//
//  Implicit result array
template <typename T>
inline
multi_array<T>& bh_invert (multi_array<T>& rhs)
{
    Runtime::instance().typecheck<BH_INVERT, T, T>();

    multi_array<T>* result = &Runtime::instance().temp<T, T>(rhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_INVERT, *result, rhs);
    result->setTemp(true);
    return *result;
}

//
// bh_imag - BH_IMAG - runtime.unary - 2
//

//
//  Explicit result array
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_imag (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_IMAG, OutT, InT>();

    multi_array<InT>* right = &rhs;
    
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
inline
multi_array<OutT>& bh_imag (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_IMAG, OutT, InT>();

    Runtime::instance().enqueue((bh_opcode)BH_IMAG, res, rhs);
    return res;
}

//
//  Implicit result array
template <typename T>
inline
multi_array<T>& bh_imag (multi_array<T>& rhs)
{
    Runtime::instance().typecheck<BH_IMAG, T, T>();

    multi_array<T>* result = &Runtime::instance().temp<T, T>(rhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_IMAG, *result, rhs);
    result->setTemp(true);
    return *result;
}

//
// bh_real - BH_REAL - runtime.unary - 2
//

//
//  Explicit result array
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_real (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_REAL, OutT, InT>();

    multi_array<InT>* right = &rhs;
    
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
inline
multi_array<OutT>& bh_real (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_REAL, OutT, InT>();

    Runtime::instance().enqueue((bh_opcode)BH_REAL, res, rhs);
    return res;
}

//
//  Implicit result array
template <typename T>
inline
multi_array<T>& bh_real (multi_array<T>& rhs)
{
    Runtime::instance().typecheck<BH_REAL, T, T>();

    multi_array<T>* result = &Runtime::instance().temp<T, T>(rhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_REAL, *result, rhs);
    result->setTemp(true);
    return *result;
}

//
// bh_absolute - BH_ABSOLUTE - runtime.unary - 2
//

//
//  Explicit result array
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_absolute (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_ABSOLUTE, OutT, InT>();

    multi_array<InT>* right = &rhs;
    
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
inline
multi_array<OutT>& bh_absolute (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_ABSOLUTE, OutT, InT>();

    Runtime::instance().enqueue((bh_opcode)BH_ABSOLUTE, res, rhs);
    return res;
}

//
//  Implicit result array
template <typename T>
inline
multi_array<T>& bh_absolute (multi_array<T>& rhs)
{
    Runtime::instance().typecheck<BH_ABSOLUTE, T, T>();

    multi_array<T>* result = &Runtime::instance().temp<T, T>(rhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_ABSOLUTE, *result, rhs);
    result->setTemp(true);
    return *result;
}

//
// bh_sin - BH_SIN - runtime.unary - 2
//

//
//  Explicit result array
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_sin (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_SIN, OutT, InT>();

    multi_array<InT>* right = &rhs;
    
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
inline
multi_array<OutT>& bh_sin (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_SIN, OutT, InT>();

    Runtime::instance().enqueue((bh_opcode)BH_SIN, res, rhs);
    return res;
}

//
//  Implicit result array
template <typename T>
inline
multi_array<T>& bh_sin (multi_array<T>& rhs)
{
    Runtime::instance().typecheck<BH_SIN, T, T>();

    multi_array<T>* result = &Runtime::instance().temp<T, T>(rhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_SIN, *result, rhs);
    result->setTemp(true);
    return *result;
}

//
// bh_cos - BH_COS - runtime.unary - 2
//

//
//  Explicit result array
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_cos (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_COS, OutT, InT>();

    multi_array<InT>* right = &rhs;
    
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
inline
multi_array<OutT>& bh_cos (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_COS, OutT, InT>();

    Runtime::instance().enqueue((bh_opcode)BH_COS, res, rhs);
    return res;
}

//
//  Implicit result array
template <typename T>
inline
multi_array<T>& bh_cos (multi_array<T>& rhs)
{
    Runtime::instance().typecheck<BH_COS, T, T>();

    multi_array<T>* result = &Runtime::instance().temp<T, T>(rhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_COS, *result, rhs);
    result->setTemp(true);
    return *result;
}

//
// bh_tan - BH_TAN - runtime.unary - 2
//

//
//  Explicit result array
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_tan (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_TAN, OutT, InT>();

    multi_array<InT>* right = &rhs;
    
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
inline
multi_array<OutT>& bh_tan (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_TAN, OutT, InT>();

    Runtime::instance().enqueue((bh_opcode)BH_TAN, res, rhs);
    return res;
}

//
//  Implicit result array
template <typename T>
inline
multi_array<T>& bh_tan (multi_array<T>& rhs)
{
    Runtime::instance().typecheck<BH_TAN, T, T>();

    multi_array<T>* result = &Runtime::instance().temp<T, T>(rhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_TAN, *result, rhs);
    result->setTemp(true);
    return *result;
}

//
// bh_sinh - BH_SINH - runtime.unary - 2
//

//
//  Explicit result array
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_sinh (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_SINH, OutT, InT>();

    multi_array<InT>* right = &rhs;
    
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
inline
multi_array<OutT>& bh_sinh (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_SINH, OutT, InT>();

    Runtime::instance().enqueue((bh_opcode)BH_SINH, res, rhs);
    return res;
}

//
//  Implicit result array
template <typename T>
inline
multi_array<T>& bh_sinh (multi_array<T>& rhs)
{
    Runtime::instance().typecheck<BH_SINH, T, T>();

    multi_array<T>* result = &Runtime::instance().temp<T, T>(rhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_SINH, *result, rhs);
    result->setTemp(true);
    return *result;
}

//
// bh_cosh - BH_COSH - runtime.unary - 2
//

//
//  Explicit result array
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_cosh (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_COSH, OutT, InT>();

    multi_array<InT>* right = &rhs;
    
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
inline
multi_array<OutT>& bh_cosh (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_COSH, OutT, InT>();

    Runtime::instance().enqueue((bh_opcode)BH_COSH, res, rhs);
    return res;
}

//
//  Implicit result array
template <typename T>
inline
multi_array<T>& bh_cosh (multi_array<T>& rhs)
{
    Runtime::instance().typecheck<BH_COSH, T, T>();

    multi_array<T>* result = &Runtime::instance().temp<T, T>(rhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_COSH, *result, rhs);
    result->setTemp(true);
    return *result;
}

//
// bh_tanh - BH_TANH - runtime.unary - 2
//

//
//  Explicit result array
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_tanh (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_TANH, OutT, InT>();

    multi_array<InT>* right = &rhs;
    
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
inline
multi_array<OutT>& bh_tanh (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_TANH, OutT, InT>();

    Runtime::instance().enqueue((bh_opcode)BH_TANH, res, rhs);
    return res;
}

//
//  Implicit result array
template <typename T>
inline
multi_array<T>& bh_tanh (multi_array<T>& rhs)
{
    Runtime::instance().typecheck<BH_TANH, T, T>();

    multi_array<T>* result = &Runtime::instance().temp<T, T>(rhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_TANH, *result, rhs);
    result->setTemp(true);
    return *result;
}

//
// bh_arcsin - BH_ARCSIN - runtime.unary - 2
//

//
//  Explicit result array
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_arcsin (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_ARCSIN, OutT, InT>();

    multi_array<InT>* right = &rhs;
    
    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_arcsin: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
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
inline
multi_array<OutT>& bh_arcsin (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_ARCSIN, OutT, InT>();

    Runtime::instance().enqueue((bh_opcode)BH_ARCSIN, res, rhs);
    return res;
}

//
//  Implicit result array
template <typename T>
inline
multi_array<T>& bh_arcsin (multi_array<T>& rhs)
{
    Runtime::instance().typecheck<BH_ARCSIN, T, T>();

    multi_array<T>* result = &Runtime::instance().temp<T, T>(rhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_ARCSIN, *result, rhs);
    result->setTemp(true);
    return *result;
}

//
// bh_arccos - BH_ARCCOS - runtime.unary - 2
//

//
//  Explicit result array
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_arccos (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_ARCCOS, OutT, InT>();

    multi_array<InT>* right = &rhs;
    
    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_arccos: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
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
inline
multi_array<OutT>& bh_arccos (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_ARCCOS, OutT, InT>();

    Runtime::instance().enqueue((bh_opcode)BH_ARCCOS, res, rhs);
    return res;
}

//
//  Implicit result array
template <typename T>
inline
multi_array<T>& bh_arccos (multi_array<T>& rhs)
{
    Runtime::instance().typecheck<BH_ARCCOS, T, T>();

    multi_array<T>* result = &Runtime::instance().temp<T, T>(rhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_ARCCOS, *result, rhs);
    result->setTemp(true);
    return *result;
}

//
// bh_arctan - BH_ARCTAN - runtime.unary - 2
//

//
//  Explicit result array
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_arctan (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_ARCTAN, OutT, InT>();

    multi_array<InT>* right = &rhs;
    
    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_arctan: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
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
inline
multi_array<OutT>& bh_arctan (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_ARCTAN, OutT, InT>();

    Runtime::instance().enqueue((bh_opcode)BH_ARCTAN, res, rhs);
    return res;
}

//
//  Implicit result array
template <typename T>
inline
multi_array<T>& bh_arctan (multi_array<T>& rhs)
{
    Runtime::instance().typecheck<BH_ARCTAN, T, T>();

    multi_array<T>* result = &Runtime::instance().temp<T, T>(rhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_ARCTAN, *result, rhs);
    result->setTemp(true);
    return *result;
}

//
// bh_arcsinh - BH_ARCSINH - runtime.unary - 2
//

//
//  Explicit result array
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_arcsinh (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_ARCSINH, OutT, InT>();

    multi_array<InT>* right = &rhs;
    
    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_arcsinh: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
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
inline
multi_array<OutT>& bh_arcsinh (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_ARCSINH, OutT, InT>();

    Runtime::instance().enqueue((bh_opcode)BH_ARCSINH, res, rhs);
    return res;
}

//
//  Implicit result array
template <typename T>
inline
multi_array<T>& bh_arcsinh (multi_array<T>& rhs)
{
    Runtime::instance().typecheck<BH_ARCSINH, T, T>();

    multi_array<T>* result = &Runtime::instance().temp<T, T>(rhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_ARCSINH, *result, rhs);
    result->setTemp(true);
    return *result;
}

//
// bh_arccosh - BH_ARCCOSH - runtime.unary - 2
//

//
//  Explicit result array
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_arccosh (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_ARCCOSH, OutT, InT>();

    multi_array<InT>* right = &rhs;
    
    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_arccosh: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
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
inline
multi_array<OutT>& bh_arccosh (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_ARCCOSH, OutT, InT>();

    Runtime::instance().enqueue((bh_opcode)BH_ARCCOSH, res, rhs);
    return res;
}

//
//  Implicit result array
template <typename T>
inline
multi_array<T>& bh_arccosh (multi_array<T>& rhs)
{
    Runtime::instance().typecheck<BH_ARCCOSH, T, T>();

    multi_array<T>* result = &Runtime::instance().temp<T, T>(rhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_ARCCOSH, *result, rhs);
    result->setTemp(true);
    return *result;
}

//
// bh_arctanh - BH_ARCTANH - runtime.unary - 2
//

//
//  Explicit result array
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_arctanh (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_ARCTANH, OutT, InT>();

    multi_array<InT>* right = &rhs;
    
    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_arctanh: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
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
inline
multi_array<OutT>& bh_arctanh (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_ARCTANH, OutT, InT>();

    Runtime::instance().enqueue((bh_opcode)BH_ARCTANH, res, rhs);
    return res;
}

//
//  Implicit result array
template <typename T>
inline
multi_array<T>& bh_arctanh (multi_array<T>& rhs)
{
    Runtime::instance().typecheck<BH_ARCTANH, T, T>();

    multi_array<T>* result = &Runtime::instance().temp<T, T>(rhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_ARCTANH, *result, rhs);
    result->setTemp(true);
    return *result;
}

//
// bh_exp - BH_EXP - runtime.unary - 2
//

//
//  Explicit result array
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_exp (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_EXP, OutT, InT>();

    multi_array<InT>* right = &rhs;
    
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
inline
multi_array<OutT>& bh_exp (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_EXP, OutT, InT>();

    Runtime::instance().enqueue((bh_opcode)BH_EXP, res, rhs);
    return res;
}

//
//  Implicit result array
template <typename T>
inline
multi_array<T>& bh_exp (multi_array<T>& rhs)
{
    Runtime::instance().typecheck<BH_EXP, T, T>();

    multi_array<T>* result = &Runtime::instance().temp<T, T>(rhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_EXP, *result, rhs);
    result->setTemp(true);
    return *result;
}

//
// bh_exp2 - BH_EXP2 - runtime.unary - 2
//

//
//  Explicit result array
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_exp2 (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_EXP2, OutT, InT>();

    multi_array<InT>* right = &rhs;
    
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
inline
multi_array<OutT>& bh_exp2 (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_EXP2, OutT, InT>();

    Runtime::instance().enqueue((bh_opcode)BH_EXP2, res, rhs);
    return res;
}

//
//  Implicit result array
template <typename T>
inline
multi_array<T>& bh_exp2 (multi_array<T>& rhs)
{
    Runtime::instance().typecheck<BH_EXP2, T, T>();

    multi_array<T>* result = &Runtime::instance().temp<T, T>(rhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_EXP2, *result, rhs);
    result->setTemp(true);
    return *result;
}

//
// bh_expm1 - BH_EXPM1 - runtime.unary - 2
//

//
//  Explicit result array
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_expm1 (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_EXPM1, OutT, InT>();

    multi_array<InT>* right = &rhs;
    
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
inline
multi_array<OutT>& bh_expm1 (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_EXPM1, OutT, InT>();

    Runtime::instance().enqueue((bh_opcode)BH_EXPM1, res, rhs);
    return res;
}

//
//  Implicit result array
template <typename T>
inline
multi_array<T>& bh_expm1 (multi_array<T>& rhs)
{
    Runtime::instance().typecheck<BH_EXPM1, T, T>();

    multi_array<T>* result = &Runtime::instance().temp<T, T>(rhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_EXPM1, *result, rhs);
    result->setTemp(true);
    return *result;
}

//
// bh_log - BH_LOG - runtime.unary - 2
//

//
//  Explicit result array
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_log (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_LOG, OutT, InT>();

    multi_array<InT>* right = &rhs;
    
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
inline
multi_array<OutT>& bh_log (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_LOG, OutT, InT>();

    Runtime::instance().enqueue((bh_opcode)BH_LOG, res, rhs);
    return res;
}

//
//  Implicit result array
template <typename T>
inline
multi_array<T>& bh_log (multi_array<T>& rhs)
{
    Runtime::instance().typecheck<BH_LOG, T, T>();

    multi_array<T>* result = &Runtime::instance().temp<T, T>(rhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_LOG, *result, rhs);
    result->setTemp(true);
    return *result;
}

//
// bh_log2 - BH_LOG2 - runtime.unary - 2
//

//
//  Explicit result array
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_log2 (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_LOG2, OutT, InT>();

    multi_array<InT>* right = &rhs;
    
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
inline
multi_array<OutT>& bh_log2 (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_LOG2, OutT, InT>();

    Runtime::instance().enqueue((bh_opcode)BH_LOG2, res, rhs);
    return res;
}

//
//  Implicit result array
template <typename T>
inline
multi_array<T>& bh_log2 (multi_array<T>& rhs)
{
    Runtime::instance().typecheck<BH_LOG2, T, T>();

    multi_array<T>* result = &Runtime::instance().temp<T, T>(rhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_LOG2, *result, rhs);
    result->setTemp(true);
    return *result;
}

//
// bh_log10 - BH_LOG10 - runtime.unary - 2
//

//
//  Explicit result array
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_log10 (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_LOG10, OutT, InT>();

    multi_array<InT>* right = &rhs;
    
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
inline
multi_array<OutT>& bh_log10 (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_LOG10, OutT, InT>();

    Runtime::instance().enqueue((bh_opcode)BH_LOG10, res, rhs);
    return res;
}

//
//  Implicit result array
template <typename T>
inline
multi_array<T>& bh_log10 (multi_array<T>& rhs)
{
    Runtime::instance().typecheck<BH_LOG10, T, T>();

    multi_array<T>* result = &Runtime::instance().temp<T, T>(rhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_LOG10, *result, rhs);
    result->setTemp(true);
    return *result;
}

//
// bh_log1p - BH_LOG1P - runtime.unary - 2
//

//
//  Explicit result array
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_log1p (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_LOG1P, OutT, InT>();

    multi_array<InT>* right = &rhs;
    
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
inline
multi_array<OutT>& bh_log1p (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_LOG1P, OutT, InT>();

    Runtime::instance().enqueue((bh_opcode)BH_LOG1P, res, rhs);
    return res;
}

//
//  Implicit result array
template <typename T>
inline
multi_array<T>& bh_log1p (multi_array<T>& rhs)
{
    Runtime::instance().typecheck<BH_LOG1P, T, T>();

    multi_array<T>* result = &Runtime::instance().temp<T, T>(rhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_LOG1P, *result, rhs);
    result->setTemp(true);
    return *result;
}

//
// bh_sqrt - BH_SQRT - runtime.unary - 2
//

//
//  Explicit result array
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_sqrt (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_SQRT, OutT, InT>();

    multi_array<InT>* right = &rhs;
    
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
inline
multi_array<OutT>& bh_sqrt (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_SQRT, OutT, InT>();

    Runtime::instance().enqueue((bh_opcode)BH_SQRT, res, rhs);
    return res;
}

//
//  Implicit result array
template <typename T>
inline
multi_array<T>& bh_sqrt (multi_array<T>& rhs)
{
    Runtime::instance().typecheck<BH_SQRT, T, T>();

    multi_array<T>* result = &Runtime::instance().temp<T, T>(rhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_SQRT, *result, rhs);
    result->setTemp(true);
    return *result;
}

//
// bh_ceil - BH_CEIL - runtime.unary - 2
//

//
//  Explicit result array
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_ceil (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_CEIL, OutT, InT>();

    multi_array<InT>* right = &rhs;
    
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
inline
multi_array<OutT>& bh_ceil (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_CEIL, OutT, InT>();

    Runtime::instance().enqueue((bh_opcode)BH_CEIL, res, rhs);
    return res;
}

//
//  Implicit result array
template <typename T>
inline
multi_array<T>& bh_ceil (multi_array<T>& rhs)
{
    Runtime::instance().typecheck<BH_CEIL, T, T>();

    multi_array<T>* result = &Runtime::instance().temp<T, T>(rhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_CEIL, *result, rhs);
    result->setTemp(true);
    return *result;
}

//
// bh_trunc - BH_TRUNC - runtime.unary - 2
//

//
//  Explicit result array
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_trunc (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_TRUNC, OutT, InT>();

    multi_array<InT>* right = &rhs;
    
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
inline
multi_array<OutT>& bh_trunc (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_TRUNC, OutT, InT>();

    Runtime::instance().enqueue((bh_opcode)BH_TRUNC, res, rhs);
    return res;
}

//
//  Implicit result array
template <typename T>
inline
multi_array<T>& bh_trunc (multi_array<T>& rhs)
{
    Runtime::instance().typecheck<BH_TRUNC, T, T>();

    multi_array<T>* result = &Runtime::instance().temp<T, T>(rhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_TRUNC, *result, rhs);
    result->setTemp(true);
    return *result;
}

//
// bh_floor - BH_FLOOR - runtime.unary - 2
//

//
//  Explicit result array
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_floor (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_FLOOR, OutT, InT>();

    multi_array<InT>* right = &rhs;
    
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
inline
multi_array<OutT>& bh_floor (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_FLOOR, OutT, InT>();

    Runtime::instance().enqueue((bh_opcode)BH_FLOOR, res, rhs);
    return res;
}

//
//  Implicit result array
template <typename T>
inline
multi_array<T>& bh_floor (multi_array<T>& rhs)
{
    Runtime::instance().typecheck<BH_FLOOR, T, T>();

    multi_array<T>* result = &Runtime::instance().temp<T, T>(rhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_FLOOR, *result, rhs);
    result->setTemp(true);
    return *result;
}

//
// bh_rint - BH_RINT - runtime.unary - 2
//

//
//  Explicit result array
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_rint (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_RINT, OutT, InT>();

    multi_array<InT>* right = &rhs;
    
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
inline
multi_array<OutT>& bh_rint (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_RINT, OutT, InT>();

    Runtime::instance().enqueue((bh_opcode)BH_RINT, res, rhs);
    return res;
}

//
//  Implicit result array
template <typename T>
inline
multi_array<T>& bh_rint (multi_array<T>& rhs)
{
    Runtime::instance().typecheck<BH_RINT, T, T>();

    multi_array<T>* result = &Runtime::instance().temp<T, T>(rhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_RINT, *result, rhs);
    result->setTemp(true);
    return *result;
}

//
// bh_sign - BH_SIGN - runtime.unary - 2
//

//
//  Explicit result array
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_sign (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_SIGN, OutT, InT>();

    multi_array<InT>* right = &rhs;
    
    // Check for unbroadcastable situation
    if (res.getRank() < right->getRank()) {
        std::stringstream s;
        s << "Incompatible shapes " << "bh_sign: " << res.getRank() << ", " << right->getRank() << "." << std::endl;
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

    Runtime::instance().enqueue((bh_opcode)BH_SIGN, res, *right);
    return res;
}

template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_sign (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_SIGN, OutT, InT>();

    Runtime::instance().enqueue((bh_opcode)BH_SIGN, res, rhs);
    return res;
}

//
//  Implicit result array
template <typename T>
inline
multi_array<T>& bh_sign (multi_array<T>& rhs)
{
    Runtime::instance().typecheck<BH_SIGN, T, T>();

    multi_array<T>* result = &Runtime::instance().temp<T, T>(rhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_SIGN, *result, rhs);
    result->setTemp(true);
    return *result;
}



//
// bh_isnan - BH_ISNAN - runtime.unary.bool - 2
//

//
//  Explicit result array
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_isnan (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_ISNAN, OutT, InT>();

    multi_array<InT>* right = &rhs;
    
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
inline
multi_array<OutT>& bh_isnan (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_ISNAN, OutT, InT>();

    Runtime::instance().enqueue((bh_opcode)BH_ISNAN, res, rhs);
    return res;
}

//
//  Implicit result array
template <typename T>
inline
multi_array<bool>& bh_isnan (multi_array<T>& rhs)
{
    Runtime::instance().typecheck<BH_ISNAN, bool, T>();

    multi_array<bool>* result = &Runtime::instance().temp<bool, T>(rhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_ISNAN, *result, rhs);
    result->setTemp(true);
    return *result;
}

//
// bh_isinf - BH_ISINF - runtime.unary.bool - 2
//

//
//  Explicit result array
template <typename OutT, typename InT>
inline
multi_array<OutT>& bh_isinf (multi_array<OutT>& res, multi_array<InT> &rhs)
{
    Runtime::instance().typecheck<BH_ISINF, OutT, InT>();

    multi_array<InT>* right = &rhs;
    
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
inline
multi_array<OutT>& bh_isinf (multi_array<OutT>& res, const InT rhs)
{
    Runtime::instance().typecheck<BH_ISINF, OutT, InT>();

    Runtime::instance().enqueue((bh_opcode)BH_ISINF, res, rhs);
    return res;
}

//
//  Implicit result array
template <typename T>
inline
multi_array<bool>& bh_isinf (multi_array<T>& rhs)
{
    Runtime::instance().typecheck<BH_ISINF, bool, T>();

    multi_array<bool>* result = &Runtime::instance().temp<bool, T>(rhs);
    result->link();
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_ISINF, *result, rhs);
    result->setTemp(true);
    return *result;
}




template <typename T>
multi_array<T>& bh_range (multi_array<T>& res)
{
    Runtime::instance().enqueue((bh_opcode)BH_RANGE, res);
    return res;
}


template <typename T>
multi_array<T>& bh_free (multi_array<T>& res)
{
    Runtime::instance().enqueue((bh_opcode)BH_FREE, res);
    return res;
}


template <typename T>
multi_array<T>& bh_sync (multi_array<T>& res)
{
    Runtime::instance().enqueue((bh_opcode)BH_SYNC, res);
    return res;
}


template <typename T>
multi_array<T>& bh_discard (multi_array<T>& res)
{
    Runtime::instance().enqueue((bh_opcode)BH_DISCARD, res);
    return res;
}




template <typename T>
multi_array<T>& bh_random (multi_array<T>& res, uint64_t in1, uint64_t in2)
{
    Runtime::instance().typecheck<BH_RANDOM, T, uint64_t, uint64_t>();

    Runtime::instance().enqueue((bh_opcode)BH_RANDOM, res, in1, in2);

    return res;
}



template <typename T>
multi_array<T>& bh_add_accumulate (multi_array<T>& res, multi_array<T> &lhs, int64_t rhs)
{
    Runtime::instance().typecheck<BH_ADD_ACCUMULATE, T, T, int64_t>();

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

template <typename T>
multi_array<T>& bh_add_accumulate (multi_array<T> &lhs, int64_t rhs)
{
    Runtime::instance().typecheck<BH_ADD_ACCUMULATE, T, T, int64_t>();

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
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_ADD_ACCUMULATE, *result, lhs, rhs);
    result->setTemp(true);

    return *result;
}


template <typename T>
multi_array<T>& bh_multiply_accumulate (multi_array<T>& res, multi_array<T> &lhs, int64_t rhs)
{
    Runtime::instance().typecheck<BH_MULTIPLY_ACCUMULATE, T, T, int64_t>();

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

template <typename T>
multi_array<T>& bh_multiply_accumulate (multi_array<T> &lhs, int64_t rhs)
{
    Runtime::instance().typecheck<BH_MULTIPLY_ACCUMULATE, T, T, int64_t>();

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
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY_ACCUMULATE, *result, lhs, rhs);
    result->setTemp(true);

    return *result;
}




//
// Reduction with explicitly provided result array
template <typename T>
multi_array<T>& bh_add_reduce (multi_array<T>& res, multi_array<T> &lhs, int64_t rhs)
{
    // TODO: Shape-check
    Runtime::instance().typecheck<BH_ADD_REDUCE, T, T, int64_t>();

    if (rhs<0) {
        rhs = lhs.getRank()+rhs;
    }
    if (rhs >= (int64_t)lhs.getRank()) {
        throw std::runtime_error("Error: Axis out of bounds in reduction.\n");
    }

    Runtime::instance().enqueue((bh_opcode)BH_ADD_REDUCE, res, lhs, rhs);

    return res;
}

//
// Reduction with implicit construction of result array
template <typename T>
multi_array<T>& bh_add_reduce (multi_array<T> &lhs, int64_t rhs)
{
    // TODO: Shape-check
    Runtime::instance().typecheck<BH_ADD_REDUCE, T, T, int64_t>();

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
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_ADD_REDUCE, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}


//
// Reduction with explicitly provided result array
template <typename T>
multi_array<T>& bh_multiply_reduce (multi_array<T>& res, multi_array<T> &lhs, int64_t rhs)
{
    // TODO: Shape-check
    Runtime::instance().typecheck<BH_MULTIPLY_REDUCE, T, T, int64_t>();

    if (rhs<0) {
        rhs = lhs.getRank()+rhs;
    }
    if (rhs >= (int64_t)lhs.getRank()) {
        throw std::runtime_error("Error: Axis out of bounds in reduction.\n");
    }

    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY_REDUCE, res, lhs, rhs);

    return res;
}

//
// Reduction with implicit construction of result array
template <typename T>
multi_array<T>& bh_multiply_reduce (multi_array<T> &lhs, int64_t rhs)
{
    // TODO: Shape-check
    Runtime::instance().typecheck<BH_MULTIPLY_REDUCE, T, T, int64_t>();

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
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_MULTIPLY_REDUCE, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}


//
// Reduction with explicitly provided result array
template <typename T>
multi_array<T>& bh_minimum_reduce (multi_array<T>& res, multi_array<T> &lhs, int64_t rhs)
{
    // TODO: Shape-check
    Runtime::instance().typecheck<BH_MINIMUM_REDUCE, T, T, int64_t>();

    if (rhs<0) {
        rhs = lhs.getRank()+rhs;
    }
    if (rhs >= (int64_t)lhs.getRank()) {
        throw std::runtime_error("Error: Axis out of bounds in reduction.\n");
    }

    Runtime::instance().enqueue((bh_opcode)BH_MINIMUM_REDUCE, res, lhs, rhs);

    return res;
}

//
// Reduction with implicit construction of result array
template <typename T>
multi_array<T>& bh_minimum_reduce (multi_array<T> &lhs, int64_t rhs)
{
    // TODO: Shape-check
    Runtime::instance().typecheck<BH_MINIMUM_REDUCE, T, T, int64_t>();

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
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_MINIMUM_REDUCE, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}


//
// Reduction with explicitly provided result array
template <typename T>
multi_array<T>& bh_maximum_reduce (multi_array<T>& res, multi_array<T> &lhs, int64_t rhs)
{
    // TODO: Shape-check
    Runtime::instance().typecheck<BH_MAXIMUM_REDUCE, T, T, int64_t>();

    if (rhs<0) {
        rhs = lhs.getRank()+rhs;
    }
    if (rhs >= (int64_t)lhs.getRank()) {
        throw std::runtime_error("Error: Axis out of bounds in reduction.\n");
    }

    Runtime::instance().enqueue((bh_opcode)BH_MAXIMUM_REDUCE, res, lhs, rhs);

    return res;
}

//
// Reduction with implicit construction of result array
template <typename T>
multi_array<T>& bh_maximum_reduce (multi_array<T> &lhs, int64_t rhs)
{
    // TODO: Shape-check
    Runtime::instance().typecheck<BH_MAXIMUM_REDUCE, T, T, int64_t>();

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
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_MAXIMUM_REDUCE, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}


//
// Reduction with explicitly provided result array
template <typename T>
multi_array<T>& bh_logical_and_reduce (multi_array<T>& res, multi_array<T> &lhs, int64_t rhs)
{
    // TODO: Shape-check
    Runtime::instance().typecheck<BH_LOGICAL_AND_REDUCE, T, T, int64_t>();

    if (rhs<0) {
        rhs = lhs.getRank()+rhs;
    }
    if (rhs >= (int64_t)lhs.getRank()) {
        throw std::runtime_error("Error: Axis out of bounds in reduction.\n");
    }

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_AND_REDUCE, res, lhs, rhs);

    return res;
}

//
// Reduction with implicit construction of result array
template <typename T>
multi_array<T>& bh_logical_and_reduce (multi_array<T> &lhs, int64_t rhs)
{
    // TODO: Shape-check
    Runtime::instance().typecheck<BH_LOGICAL_AND_REDUCE, T, T, int64_t>();

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
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_AND_REDUCE, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}


//
// Reduction with explicitly provided result array
template <typename T>
multi_array<T>& bh_logical_or_reduce (multi_array<T>& res, multi_array<T> &lhs, int64_t rhs)
{
    // TODO: Shape-check
    Runtime::instance().typecheck<BH_LOGICAL_OR_REDUCE, T, T, int64_t>();

    if (rhs<0) {
        rhs = lhs.getRank()+rhs;
    }
    if (rhs >= (int64_t)lhs.getRank()) {
        throw std::runtime_error("Error: Axis out of bounds in reduction.\n");
    }

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_OR_REDUCE, res, lhs, rhs);

    return res;
}

//
// Reduction with implicit construction of result array
template <typename T>
multi_array<T>& bh_logical_or_reduce (multi_array<T> &lhs, int64_t rhs)
{
    // TODO: Shape-check
    Runtime::instance().typecheck<BH_LOGICAL_OR_REDUCE, T, T, int64_t>();

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
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_OR_REDUCE, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}


//
// Reduction with explicitly provided result array
template <typename T>
multi_array<T>& bh_logical_xor_reduce (multi_array<T>& res, multi_array<T> &lhs, int64_t rhs)
{
    // TODO: Shape-check
    Runtime::instance().typecheck<BH_LOGICAL_XOR_REDUCE, T, T, int64_t>();

    if (rhs<0) {
        rhs = lhs.getRank()+rhs;
    }
    if (rhs >= (int64_t)lhs.getRank()) {
        throw std::runtime_error("Error: Axis out of bounds in reduction.\n");
    }

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_XOR_REDUCE, res, lhs, rhs);

    return res;
}

//
// Reduction with implicit construction of result array
template <typename T>
multi_array<T>& bh_logical_xor_reduce (multi_array<T> &lhs, int64_t rhs)
{
    // TODO: Shape-check
    Runtime::instance().typecheck<BH_LOGICAL_XOR_REDUCE, T, T, int64_t>();

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
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_LOGICAL_XOR_REDUCE, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}


//
// Reduction with explicitly provided result array
template <typename T>
multi_array<T>& bh_bitwise_and_reduce (multi_array<T>& res, multi_array<T> &lhs, int64_t rhs)
{
    // TODO: Shape-check
    Runtime::instance().typecheck<BH_BITWISE_AND_REDUCE, T, T, int64_t>();

    if (rhs<0) {
        rhs = lhs.getRank()+rhs;
    }
    if (rhs >= (int64_t)lhs.getRank()) {
        throw std::runtime_error("Error: Axis out of bounds in reduction.\n");
    }

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_AND_REDUCE, res, lhs, rhs);

    return res;
}

//
// Reduction with implicit construction of result array
template <typename T>
multi_array<T>& bh_bitwise_and_reduce (multi_array<T> &lhs, int64_t rhs)
{
    // TODO: Shape-check
    Runtime::instance().typecheck<BH_BITWISE_AND_REDUCE, T, T, int64_t>();

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
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_AND_REDUCE, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}


//
// Reduction with explicitly provided result array
template <typename T>
multi_array<T>& bh_bitwise_or_reduce (multi_array<T>& res, multi_array<T> &lhs, int64_t rhs)
{
    // TODO: Shape-check
    Runtime::instance().typecheck<BH_BITWISE_OR_REDUCE, T, T, int64_t>();

    if (rhs<0) {
        rhs = lhs.getRank()+rhs;
    }
    if (rhs >= (int64_t)lhs.getRank()) {
        throw std::runtime_error("Error: Axis out of bounds in reduction.\n");
    }

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_OR_REDUCE, res, lhs, rhs);

    return res;
}

//
// Reduction with implicit construction of result array
template <typename T>
multi_array<T>& bh_bitwise_or_reduce (multi_array<T> &lhs, int64_t rhs)
{
    // TODO: Shape-check
    Runtime::instance().typecheck<BH_BITWISE_OR_REDUCE, T, T, int64_t>();

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
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_OR_REDUCE, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}


//
// Reduction with explicitly provided result array
template <typename T>
multi_array<T>& bh_bitwise_xor_reduce (multi_array<T>& res, multi_array<T> &lhs, int64_t rhs)
{
    // TODO: Shape-check
    Runtime::instance().typecheck<BH_BITWISE_XOR_REDUCE, T, T, int64_t>();

    if (rhs<0) {
        rhs = lhs.getRank()+rhs;
    }
    if (rhs >= (int64_t)lhs.getRank()) {
        throw std::runtime_error("Error: Axis out of bounds in reduction.\n");
    }

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_XOR_REDUCE, res, lhs, rhs);

    return res;
}

//
// Reduction with implicit construction of result array
template <typename T>
multi_array<T>& bh_bitwise_xor_reduce (multi_array<T> &lhs, int64_t rhs)
{
    // TODO: Shape-check
    Runtime::instance().typecheck<BH_BITWISE_XOR_REDUCE, T, T, int64_t>();

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
    result->setTemp(false);

    Runtime::instance().enqueue((bh_opcode)BH_BITWISE_XOR_REDUCE, *result, lhs, rhs);
    result->setTemp(true);
    return *result;
}


}
#endif
