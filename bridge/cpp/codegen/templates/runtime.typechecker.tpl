@!license!@

#ifndef __BOHRIUM_BRIDGE_CPP_RUNTIME_TYPECHECKER
#define __BOHRIUM_BRIDGE_CPP_RUNTIME_TYPECHECKER
#include <iostream>
#include <sstream>
#include <typeinfo>

namespace bxx {

//
//  Default to deny
//
/*
template <size_t Opcode, typename Out, typename In1, typename In2>
inline
void Runtime::typecheck(void)
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
}

template <size_t Opcode, typename Out, typename In1>
inline
void Runtime::typecheck(void)
{
    std::stringstream ss;
    ss << "Bytecode(" << Opcode << ") instruction";
    ss << " has invalid type signature: ";
    ss << typeid(Out).name();
    ss << ",";
    ss << typeid(In1).name();
    ss << ".";

    throw std::runtime_error(ss.str());
}

template <size_t Opcode, typename Out>
inline
void Runtime::typecheck(void)
{
    std::stringstream ss;
    ss << "Bytecode(" << Opcode << ") instruction";
    ss << " has invalid type signature: ";
    ss << typeid(Out).name();
    ss << ".";

    throw std::runtime_error(ss.str());
}
*/

template <size_t Opcode>
struct dependent_false { enum { value = false }; };

template <size_t Opcode, typename Out, typename In1, typename In2, typename In3>
void Runtime::typecheck(void)
{
    static_assert(dependent_false<Opcode>::value, "ArrayOperation has unsupported type-signature.");
}

template <size_t Opcode, typename Out, typename In1, typename In2>
void Runtime::typecheck(void)
{
    static_assert(dependent_false<Opcode>::value, "ArrayOperation has unsupported type-signature.");
}

template <size_t Opcode, typename Out, typename In1>
void Runtime::typecheck(void)
{
    static_assert(dependent_false<Opcode>::value, "ArrayOperation has unsupported type-signature.");
}

template <size_t Opcode, typename Out>
void Runtime::typecheck(void)
{
    static_assert(dependent_false<Opcode>::value, "ArrayOperation has unsupported type-signature.");
}

//
//  Allowed types.
//
<!--(for _op, opcode, _optype, opcount, typesigs, _layouts, _broadcast in data)-->
    <!--(for typesig in typesigs)-->
        <!--(if opcount == 4)-->
        template <>
        inline
        void Runtime::typecheck<@!opcode!@, @!typesig[0]!@, @!typesig[1]!@, @!typesig[2]!@, @!typesig[3]!@>(void) { }
        <!--(elif opcount == 3)-->
        template <>
        inline
        void Runtime::typecheck<@!opcode!@, @!typesig[0]!@, @!typesig[1]!@, @!typesig[2]!@>(void) { }
        <!--(elif opcount == 2)-->
        template <>
        inline
        void Runtime::typecheck<@!opcode!@, @!typesig[0]!@, @!typesig[1]!@>(void) { }
        <!--(elif opcount == 1)-->
        template <>
        inline
        void Runtime::typecheck<@!opcode!@, @!typesig[0]!@>(void) { }
        <!--(end)-->
    <!--(end)-->
<!--(end)-->

}
#endif
