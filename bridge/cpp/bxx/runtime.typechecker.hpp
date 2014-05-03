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
#ifndef __BOHRIUM_BRIDGE_CPP_RUNTIME_TYPECHECKER
#define __BOHRIUM_BRIDGE_CPP_RUNTIME_TYPECHECKER
#include <iostream>
#include <sstream>
#include <typeinfo>

namespace bxx {

//
//  Default to deny
//
template <size_t Opcode, typename Out, typename In1, typename In2>
inline
bool Runtime::typecheck(void)
{
    std::stringstream ss;
    ss << "Bytecode(" << Opcode << ") instruction";
    ss << " has invalid type signature: ";
    ss << typeid(Out).name();
    ss << ",";
    ss << typeid(In1).name();
    ss << ",";
    ss << typeid(In2).name();
    ss << ".";

    throw std::runtime_error(ss.str());
    return false;
}

template <size_t Opcode, typename Out, typename In1>
inline
bool Runtime::typecheck(void)
{
    std::stringstream ss;
    ss << "Bytecode(" << Opcode << ") instruction";
    ss << " has invalid type signature: ";
    ss << typeid(Out).name();
    ss << ",";
    ss << typeid(In1).name();
    ss << ",";
    ss << typeid(In2).name();
    ss << ".";

    throw std::runtime_error(ss.str());
    return false;
}

template <size_t Opcode, typename Out>
inline
bool Runtime::typecheck(void)
{
    std::stringstream ss;
    ss << "Bytecode(" << Opcode << ") instruction";
    ss << " has invalid type signature: ";
    ss << typeid(Out).name();
    ss << ",";
    ss << typeid(In1).name();
    ss << ",";
    ss << typeid(In2).name();
    ss << ".";

    throw std::runtime_error(ss.str());
    return false;
}

//
//  Allowed types.
//


template <>
inline
bool Runtime::typecheck<BH_ABSOLUTE, bool, bool>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ABSOLUTE, float, float>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ABSOLUTE, double, double>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ABSOLUTE, int16_t, int16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ABSOLUTE, int32_t, int32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ABSOLUTE, int64_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ABSOLUTE, int8_t, int8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ABSOLUTE, uint16_t, uint16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ABSOLUTE, uint32_t, uint32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ABSOLUTE, uint64_t, uint64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ABSOLUTE, uint8_t, uint8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ARCCOS, float, float>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ARCCOS, double, double>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ARCCOSH, float, float>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ARCCOSH, double, double>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ADD, bool, bool, bool>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ADD, std::complex<double> , std::complex<double> , std::complex<double> >(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ADD, std::complex<float> , std::complex<float> , std::complex<float> >(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ADD, float, float, float>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ADD, double, double, double>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ADD, int16_t, int16_t, int16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ADD, int32_t, int32_t, int32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ADD, int64_t, int64_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ADD, int8_t, int8_t, int8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ADD, uint16_t, uint16_t, uint16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ADD, uint32_t, uint32_t, uint32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ADD, uint64_t, uint64_t, uint64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ADD, uint8_t, uint8_t, uint8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ARCSIN, float, float>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ARCSIN, double, double>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ARCSINH, float, float>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ARCSINH, double, double>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ARCTAN, float, float>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ARCTAN, double, double>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ARCTAN2, float, float, float>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ARCTAN2, double, double, double>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ARCTANH, float, float>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ARCTANH, double, double>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ADD_ACCUMULATE, bool, bool, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ADD_ACCUMULATE, std::complex<double> , std::complex<double> , int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ADD_ACCUMULATE, std::complex<float> , std::complex<float> , int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ADD_ACCUMULATE, float, float, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ADD_ACCUMULATE, double, double, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ADD_ACCUMULATE, int16_t, int16_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ADD_ACCUMULATE, int32_t, int32_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ADD_ACCUMULATE, int64_t, int64_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ADD_ACCUMULATE, int8_t, int8_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ADD_ACCUMULATE, uint16_t, uint16_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ADD_ACCUMULATE, uint32_t, uint32_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ADD_ACCUMULATE, uint64_t, uint64_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ADD_ACCUMULATE, uint8_t, uint8_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ADD_REDUCE, bool, bool, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ADD_REDUCE, std::complex<double> , std::complex<double> , int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ADD_REDUCE, std::complex<float> , std::complex<float> , int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ADD_REDUCE, float, float, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ADD_REDUCE, double, double, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ADD_REDUCE, int16_t, int16_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ADD_REDUCE, int32_t, int32_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ADD_REDUCE, int64_t, int64_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ADD_REDUCE, int8_t, int8_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ADD_REDUCE, uint16_t, uint16_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ADD_REDUCE, uint32_t, uint32_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ADD_REDUCE, uint64_t, uint64_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ADD_REDUCE, uint8_t, uint8_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_BITWISE_AND, bool, bool, bool>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_BITWISE_AND, int16_t, int16_t, int16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_BITWISE_AND, int32_t, int32_t, int32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_BITWISE_AND, int64_t, int64_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_BITWISE_AND, int8_t, int8_t, int8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_BITWISE_AND, uint16_t, uint16_t, uint16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_BITWISE_AND, uint32_t, uint32_t, uint32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_BITWISE_AND, uint64_t, uint64_t, uint64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_BITWISE_AND, uint8_t, uint8_t, uint8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_BITWISE_AND_REDUCE, bool, bool, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_BITWISE_AND_REDUCE, int16_t, int16_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_BITWISE_AND_REDUCE, int32_t, int32_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_BITWISE_AND_REDUCE, int64_t, int64_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_BITWISE_AND_REDUCE, int8_t, int8_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_BITWISE_AND_REDUCE, uint16_t, uint16_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_BITWISE_AND_REDUCE, uint32_t, uint32_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_BITWISE_AND_REDUCE, uint64_t, uint64_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_BITWISE_AND_REDUCE, uint8_t, uint8_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_BITWISE_OR, bool, bool, bool>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_BITWISE_OR, int16_t, int16_t, int16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_BITWISE_OR, int32_t, int32_t, int32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_BITWISE_OR, int64_t, int64_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_BITWISE_OR, int8_t, int8_t, int8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_BITWISE_OR, uint16_t, uint16_t, uint16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_BITWISE_OR, uint32_t, uint32_t, uint32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_BITWISE_OR, uint64_t, uint64_t, uint64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_BITWISE_OR, uint8_t, uint8_t, uint8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_BITWISE_OR_REDUCE, bool, bool, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_BITWISE_OR_REDUCE, int16_t, int16_t, int16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_BITWISE_OR_REDUCE, int32_t, int32_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_BITWISE_OR_REDUCE, int64_t, int64_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_BITWISE_OR_REDUCE, int8_t, int8_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_BITWISE_OR_REDUCE, uint16_t, uint16_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_BITWISE_OR_REDUCE, uint32_t, uint32_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_BITWISE_OR_REDUCE, uint64_t, uint64_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_BITWISE_OR_REDUCE, uint8_t, uint8_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_BITWISE_XOR, bool, bool, bool>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_BITWISE_XOR, int16_t, int16_t, int16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_BITWISE_XOR, int32_t, int32_t, int32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_BITWISE_XOR, int64_t, int64_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_BITWISE_XOR, int8_t, int8_t, int8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_BITWISE_XOR, uint16_t, uint16_t, uint16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_BITWISE_XOR, uint32_t, uint32_t, uint32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_BITWISE_XOR, uint64_t, uint64_t, uint64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_BITWISE_XOR, uint8_t, uint8_t, uint8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_BITWISE_XOR_REDUCE, bool, bool, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_BITWISE_XOR_REDUCE, int16_t, int16_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_BITWISE_XOR_REDUCE, int32_t, int32_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_BITWISE_XOR_REDUCE, int64_t, int64_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_BITWISE_XOR_REDUCE, int8_t, int8_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_BITWISE_XOR_REDUCE, uint16_t, uint16_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_BITWISE_XOR_REDUCE, uint32_t, uint32_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_BITWISE_XOR_REDUCE, uint64_t, uint64_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_BITWISE_XOR_REDUCE, uint8_t, uint8_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_CEIL, float, float>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_CEIL, double, double>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_COS, std::complex<double> , std::complex<double> >(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_COS, std::complex<float> , std::complex<float> >(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_COS, float, float>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_COS, double, double>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_COSH, std::complex<double> , std::complex<double> >(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_COSH, std::complex<float> , std::complex<float> >(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_COSH, float, float>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_COSH, double, double>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_DIVIDE, std::complex<double> , std::complex<double> , std::complex<double> >(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_DIVIDE, std::complex<float> , std::complex<float> , std::complex<float> >(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_DIVIDE, float, float, float>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_DIVIDE, double, double, double>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_DIVIDE, int16_t, int16_t, int16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_DIVIDE, int32_t, int32_t, int32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_DIVIDE, int64_t, int64_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_DIVIDE, int8_t, int8_t, int8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_DIVIDE, uint16_t, uint16_t, uint16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_DIVIDE, uint32_t, uint32_t, uint32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_DIVIDE, uint64_t, uint64_t, uint64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_DIVIDE, uint8_t, uint8_t, uint8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_EQUAL, bool, bool, bool>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_EQUAL, bool, std::complex<double> , std::complex<double> >(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_EQUAL, bool, std::complex<float> , std::complex<float> >(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_EQUAL, bool, float, float>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_EQUAL, bool, double, double>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_EQUAL, bool, int16_t, int16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_EQUAL, bool, int32_t, int32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_EQUAL, bool, int64_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_EQUAL, bool, int8_t, int8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_EQUAL, bool, uint16_t, uint16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_EQUAL, bool, uint32_t, uint32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_EQUAL, bool, uint64_t, uint64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_EQUAL, bool, uint8_t, uint8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_EXP, std::complex<double> , std::complex<double> >(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_EXP, std::complex<float> , std::complex<float> >(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_EXP, float, float>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_EXP, double, double>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_EXP2, float, float>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_EXP2, double, double>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_EXPM1, float, float>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_EXPM1, double, double>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_FLOOR, float, float>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_FLOOR, double, double>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_GREATER, bool, bool, bool>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_GREATER, bool, float, float>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_GREATER, bool, double, double>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_GREATER, bool, int16_t, int16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_GREATER, bool, int32_t, int32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_GREATER, bool, int64_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_GREATER, bool, int8_t, int8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_GREATER, bool, uint16_t, uint16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_GREATER, bool, uint32_t, uint32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_GREATER, bool, uint64_t, uint64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_GREATER, bool, uint8_t, uint8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_GREATER_EQUAL, bool, bool, bool>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_GREATER_EQUAL, bool, float, float>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_GREATER_EQUAL, bool, double, double>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_GREATER_EQUAL, bool, int16_t, int16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_GREATER_EQUAL, bool, int32_t, int32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_GREATER_EQUAL, bool, int64_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_GREATER_EQUAL, bool, int8_t, int8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_GREATER_EQUAL, bool, uint16_t, uint16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_GREATER_EQUAL, bool, uint32_t, uint32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_GREATER_EQUAL, bool, uint64_t, uint64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_GREATER_EQUAL, bool, uint8_t, uint8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, bool, bool>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, bool, float>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, bool, double>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, bool, int16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, bool, int32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, bool, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, bool, int8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, bool, uint16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, bool, uint32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, bool, uint64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, bool, uint8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, std::complex<double> , bool>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, std::complex<double> , std::complex<double> >(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, std::complex<double> , std::complex<float> >(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, std::complex<double> , float>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, std::complex<double> , double>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, std::complex<double> , int16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, std::complex<double> , int32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, std::complex<double> , int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, std::complex<double> , int8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, std::complex<double> , uint16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, std::complex<double> , uint32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, std::complex<double> , uint64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, std::complex<double> , uint8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, std::complex<float> , bool>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, std::complex<float> , std::complex<double> >(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, std::complex<float> , std::complex<float> >(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, std::complex<float> , float>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, std::complex<float> , double>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, std::complex<float> , int16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, std::complex<float> , int32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, std::complex<float> , int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, std::complex<float> , int8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, std::complex<float> , uint16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, std::complex<float> , uint32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, std::complex<float> , uint64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, std::complex<float> , uint8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, float, bool>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, float, float>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, float, double>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, float, int16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, float, int32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, float, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, float, int8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, float, uint16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, float, uint32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, float, uint64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, float, uint8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, double, bool>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, double, float>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, double, double>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, double, int16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, double, int32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, double, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, double, int8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, double, uint16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, double, uint32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, double, uint64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, double, uint8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, int16_t, bool>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, int16_t, float>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, int16_t, double>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, int16_t, int16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, int16_t, int32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, int16_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, int16_t, int8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, int16_t, uint16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, int16_t, uint32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, int16_t, uint64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, int16_t, uint8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, int32_t, bool>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, int32_t, float>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, int32_t, double>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, int32_t, int16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, int32_t, int32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, int32_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, int32_t, int8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, int32_t, uint16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, int32_t, uint32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, int32_t, uint64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, int32_t, uint8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, int64_t, bool>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, int64_t, float>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, int64_t, double>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, int64_t, int16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, int64_t, int32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, int64_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, int64_t, int8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, int64_t, uint16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, int64_t, uint32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, int64_t, uint64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, int64_t, uint8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, int8_t, bool>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, int8_t, float>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, int8_t, double>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, int8_t, int16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, int8_t, int32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, int8_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, int8_t, int8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, int8_t, uint16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, int8_t, uint32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, int8_t, uint64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, int8_t, uint8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, uint16_t, bool>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, uint16_t, float>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, uint16_t, double>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, uint16_t, int16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, uint16_t, int32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, uint16_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, uint16_t, int8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, uint16_t, uint16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, uint16_t, uint32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, uint16_t, uint64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, uint16_t, uint8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, uint32_t, bool>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, uint32_t, float>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, uint32_t, double>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, uint32_t, int16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, uint32_t, int32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, uint32_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, uint32_t, int8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, uint32_t, uint16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, uint32_t, uint32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, uint32_t, uint64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, uint32_t, uint8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, uint64_t, bool>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, uint64_t, float>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, uint64_t, double>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, uint64_t, int16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, uint64_t, int32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, uint64_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, uint64_t, int8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, uint64_t, uint16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, uint64_t, uint32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, uint64_t, uint64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, uint64_t, uint8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, uint8_t, bool>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, uint8_t, float>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, uint8_t, double>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, uint8_t, int16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, uint8_t, int32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, uint8_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, uint8_t, int8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, uint8_t, uint16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, uint8_t, uint32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, uint8_t, uint64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IDENTITY, uint8_t, uint8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IMAG, double, std::complex<double> >(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_IMAG, float, std::complex<float> >(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_INVERT, bool, bool>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_INVERT, int16_t, int16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_INVERT, int32_t, int32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_INVERT, int64_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_INVERT, int8_t, int8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_INVERT, uint16_t, uint16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_INVERT, uint32_t, uint32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_INVERT, uint64_t, uint64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_INVERT, uint8_t, uint8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ISINF, bool, float>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ISINF, bool, double>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ISNAN, bool, float>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_ISNAN, bool, double>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_LEFT_SHIFT, int16_t, int16_t, int16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_LEFT_SHIFT, int32_t, int32_t, int32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_LEFT_SHIFT, int64_t, int64_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_LEFT_SHIFT, int8_t, int8_t, int8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_LEFT_SHIFT, uint16_t, uint16_t, uint16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_LEFT_SHIFT, uint32_t, uint32_t, uint32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_LEFT_SHIFT, uint64_t, uint64_t, uint64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_LEFT_SHIFT, uint8_t, uint8_t, uint8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_LESS, bool, bool, bool>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_LESS, bool, float, float>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_LESS, bool, double, double>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_LESS, bool, int16_t, int16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_LESS, bool, int32_t, int32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_LESS, bool, int64_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_LESS, bool, int8_t, int8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_LESS, bool, uint16_t, uint16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_LESS, bool, uint32_t, uint32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_LESS, bool, uint64_t, uint64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_LESS, bool, uint8_t, uint8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_LESS_EQUAL, bool, bool, bool>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_LESS_EQUAL, bool, float, float>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_LESS_EQUAL, bool, double, double>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_LESS_EQUAL, bool, int16_t, int16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_LESS_EQUAL, bool, int32_t, int32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_LESS_EQUAL, bool, int64_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_LESS_EQUAL, bool, int8_t, int8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_LESS_EQUAL, bool, uint16_t, uint16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_LESS_EQUAL, bool, uint32_t, uint32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_LESS_EQUAL, bool, uint64_t, uint64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_LESS_EQUAL, bool, uint8_t, uint8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_LOG, std::complex<double> , std::complex<double> >(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_LOG, std::complex<float> , std::complex<float> >(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_LOG, float, float>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_LOG, double, double>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_LOG10, std::complex<double> , std::complex<double> >(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_LOG10, std::complex<float> , std::complex<float> >(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_LOG10, float, float>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_LOG10, double, double>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_LOG1P, float, float>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_LOG1P, double, double>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_LOG2, float, float>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_LOG2, double, double>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_LOGICAL_AND, bool, bool, bool>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_LOGICAL_AND_REDUCE, bool, bool, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_LOGICAL_NOT, bool, bool>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_LOGICAL_OR, bool, bool, bool>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_LOGICAL_OR_REDUCE, bool, bool, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_LOGICAL_XOR, bool, bool, bool>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_LOGICAL_XOR_REDUCE, bool, bool, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MAXIMUM, bool, bool, bool>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MAXIMUM, float, float, float>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MAXIMUM, double, double, double>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MAXIMUM, int16_t, int16_t, int16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MAXIMUM, int32_t, int32_t, int32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MAXIMUM, int64_t, int64_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MAXIMUM, int8_t, int8_t, int8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MAXIMUM, uint16_t, uint16_t, uint16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MAXIMUM, uint32_t, uint32_t, uint32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MAXIMUM, uint64_t, uint64_t, uint64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MAXIMUM, uint8_t, uint8_t, uint8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MAXIMUM_REDUCE, bool, bool, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MAXIMUM_REDUCE, float, float, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MAXIMUM_REDUCE, double, double, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MAXIMUM_REDUCE, int16_t, int16_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MAXIMUM_REDUCE, int32_t, int32_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MAXIMUM_REDUCE, int64_t, int64_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MAXIMUM_REDUCE, int8_t, int8_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MAXIMUM_REDUCE, uint16_t, uint16_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MAXIMUM_REDUCE, uint32_t, uint32_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MAXIMUM_REDUCE, uint64_t, uint64_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MAXIMUM_REDUCE, uint8_t, uint8_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MINIMUM, bool, bool, bool>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MINIMUM, float, float, float>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MINIMUM, double, double, double>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MINIMUM, int16_t, int16_t, int16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MINIMUM, int32_t, int32_t, int32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MINIMUM, int64_t, int64_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MINIMUM, int8_t, int8_t, int8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MINIMUM, uint16_t, uint16_t, uint16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MINIMUM, uint32_t, uint32_t, uint32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MINIMUM, uint64_t, uint64_t, uint64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MINIMUM, uint8_t, uint8_t, uint8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MINIMUM_REDUCE, bool, bool, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MINIMUM_REDUCE, float, float, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MINIMUM_REDUCE, double, double, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MINIMUM_REDUCE, int16_t, int16_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MINIMUM_REDUCE, int32_t, int32_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MINIMUM_REDUCE, int64_t, int64_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MINIMUM_REDUCE, int8_t, int8_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MINIMUM_REDUCE, uint16_t, uint16_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MINIMUM_REDUCE, uint32_t, uint32_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MINIMUM_REDUCE, uint64_t, uint64_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MINIMUM_REDUCE, uint8_t, uint8_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MOD, float, float, float>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MOD, double, double, double>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MOD, int16_t, int16_t, int16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MOD, int32_t, int32_t, int32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MOD, int64_t, int64_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MOD, int8_t, int8_t, int8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MOD, uint16_t, uint16_t, uint16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MOD, uint32_t, uint32_t, uint32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MOD, uint64_t, uint64_t, uint64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MOD, uint8_t, uint8_t, uint8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MULTIPLY, bool, bool, bool>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MULTIPLY, std::complex<double> , std::complex<double> , std::complex<double> >(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MULTIPLY, std::complex<float> , std::complex<float> , std::complex<float> >(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MULTIPLY, float, float, float>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MULTIPLY, double, double, double>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MULTIPLY, int16_t, int16_t, int16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MULTIPLY, int32_t, int32_t, int32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MULTIPLY, int64_t, int64_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MULTIPLY, int8_t, int8_t, int8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MULTIPLY, uint16_t, uint16_t, uint16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MULTIPLY, uint32_t, uint32_t, uint32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MULTIPLY, uint64_t, uint64_t, uint64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MULTIPLY, uint8_t, uint8_t, uint8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MULTIPLY_ACCUMULATE, bool, bool, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MULTIPLY_ACCUMULATE, std::complex<double> , std::complex<double> , int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MULTIPLY_ACCUMULATE, std::complex<float> , std::complex<float> , int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MULTIPLY_ACCUMULATE, float, float, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MULTIPLY_ACCUMULATE, double, double, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MULTIPLY_ACCUMULATE, int16_t, int16_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MULTIPLY_ACCUMULATE, int32_t, int32_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MULTIPLY_ACCUMULATE, int64_t, int64_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MULTIPLY_ACCUMULATE, int8_t, int8_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MULTIPLY_ACCUMULATE, uint16_t, uint16_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MULTIPLY_ACCUMULATE, uint32_t, uint32_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MULTIPLY_ACCUMULATE, uint64_t, uint64_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MULTIPLY_ACCUMULATE, uint8_t, uint8_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MULTIPLY_REDUCE, bool, bool, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MULTIPLY_REDUCE, std::complex<double> , std::complex<double> , int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MULTIPLY_REDUCE, std::complex<float> , std::complex<float> , int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MULTIPLY_REDUCE, float, float, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MULTIPLY_REDUCE, double, double, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MULTIPLY_REDUCE, int16_t, int16_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MULTIPLY_REDUCE, int32_t, int32_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MULTIPLY_REDUCE, int64_t, int64_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MULTIPLY_REDUCE, int8_t, int8_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MULTIPLY_REDUCE, uint16_t, uint16_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MULTIPLY_REDUCE, uint32_t, uint32_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MULTIPLY_REDUCE, uint64_t, uint64_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_MULTIPLY_REDUCE, uint8_t, uint8_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_NOT_EQUAL, bool, bool, bool>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_NOT_EQUAL, bool, std::complex<double> , std::complex<double> >(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_NOT_EQUAL, bool, std::complex<float> , std::complex<float> >(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_NOT_EQUAL, bool, float, float>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_NOT_EQUAL, bool, double, double>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_NOT_EQUAL, bool, int16_t, int16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_NOT_EQUAL, bool, int32_t, int32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_NOT_EQUAL, bool, int64_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_NOT_EQUAL, bool, int8_t, int8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_NOT_EQUAL, bool, uint16_t, uint16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_NOT_EQUAL, bool, uint32_t, uint32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_NOT_EQUAL, bool, uint64_t, uint64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_NOT_EQUAL, bool, uint8_t, uint8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_POWER, std::complex<double> , std::complex<double> , std::complex<double> >(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_POWER, std::complex<float> , std::complex<float> , std::complex<float> >(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_POWER, float, float, float>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_POWER, double, double, double>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_POWER, int16_t, int16_t, int16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_POWER, int32_t, int32_t, int32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_POWER, int64_t, int64_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_POWER, int8_t, int8_t, int8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_POWER, uint16_t, uint16_t, uint16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_POWER, uint32_t, uint32_t, uint32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_POWER, uint64_t, uint64_t, uint64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_POWER, uint8_t, uint8_t, uint8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_RANDOM, uint64_t, uint64_t, uint64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_RANGE, uint32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_RANGE, uint64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_REAL, double, std::complex<double> >(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_REAL, float, std::complex<float> >(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_RIGHT_SHIFT, int16_t, int16_t, int16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_RIGHT_SHIFT, int32_t, int32_t, int32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_RIGHT_SHIFT, int64_t, int64_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_RIGHT_SHIFT, int8_t, int8_t, int8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_RIGHT_SHIFT, uint16_t, uint16_t, uint16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_RIGHT_SHIFT, uint32_t, uint32_t, uint32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_RIGHT_SHIFT, uint64_t, uint64_t, uint64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_RIGHT_SHIFT, uint8_t, uint8_t, uint8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_RINT, float, float>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_RINT, double, double>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_SIN, std::complex<double> , std::complex<double> >(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_SIN, std::complex<float> , std::complex<float> >(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_SIN, float, float>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_SIN, double, double>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_SINH, std::complex<double> , std::complex<double> >(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_SINH, std::complex<float> , std::complex<float> >(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_SINH, float, float>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_SINH, double, double>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_SQRT, std::complex<double> , std::complex<double> >(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_SQRT, std::complex<float> , std::complex<float> >(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_SQRT, float, float>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_SQRT, double, double>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_SUBTRACT, bool, bool, bool>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_SUBTRACT, std::complex<double> , std::complex<double> , std::complex<double> >(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_SUBTRACT, std::complex<float> , std::complex<float> , std::complex<float> >(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_SUBTRACT, float, float, float>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_SUBTRACT, double, double, double>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_SUBTRACT, int16_t, int16_t, int16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_SUBTRACT, int32_t, int32_t, int32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_SUBTRACT, int64_t, int64_t, int64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_SUBTRACT, int8_t, int8_t, int8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_SUBTRACT, uint16_t, uint16_t, uint16_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_SUBTRACT, uint32_t, uint32_t, uint32_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_SUBTRACT, uint64_t, uint64_t, uint64_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_SUBTRACT, uint8_t, uint8_t, uint8_t>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_TAN, std::complex<double> , std::complex<double> >(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_TAN, std::complex<float> , std::complex<float> >(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_TAN, float, float>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_TAN, double, double>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_TANH, std::complex<double> , std::complex<double> >(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_TANH, std::complex<float> , std::complex<float> >(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_TANH, float, float>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_TANH, double, double>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_TRUNC, float, float>(void) { return true; }


template <>
inline
bool Runtime::typecheck<BH_TRUNC, double, double>(void) { return true; }

}
#endif

