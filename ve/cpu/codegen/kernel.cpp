#include <sstream>
#include <string>
#include "utils.hpp"
#include "codegen.hpp"

using namespace std;
using namespace bohrium::core;

namespace bohrium{
namespace engine{
namespace cpu{
namespace codegen{

Kernel::Kernel(Plaid& plaid, Block& block) : plaid_(plaid), block_(block) {}

string Kernel::unpack_arguments(void)
{
    stringstream ss;
    for(size_t oidx=0; oidx<block_.noperands(); ++oidx) {
        ss << unpack_argument(oidx);
    }
    return ss.str();
}

string Kernel::args(void)
{
    return "args";
}

string Kernel::iterspace(void)
{
    return "iterspace";
}

string Kernel::generate_source(void)
{
    std::map<string, string> subjects;

    if (block_.narray_tacs()>1) {
        subjects["MODE"] = "FUSED";
    } else {
        subjects["MODE"] = "SIJ";
    }
    subjects["LAYOUT"] = layout_text(block_.iterspace().layout);
    subjects["NINSTR"] = to_string(block_.ntacs());
    subjects["NARRAY_INSTR"] = to_string(block_.narray_tacs());
    subjects["NARGS"] = to_string(block_.noperands());
    subjects["NARRAY_ARGS"] = "?";
    subjects["SYMBOL_TEXT"] = block_.symbol_text();
    subjects["SYMBOL"] = block_.symbol();

    string unpacked_args = unpack_arguments();  // Indent arguments
    plaid_.indent(unpacked_args, 4);
    subjects["ARGUMENTS"] = unpacked_args;

    string declared_operands = declare_operands();
    plaid_.indent(declared_operands, 4);
    subjects["OPERATIONS"] = declared_operands;

    return plaid_.fill("kernel", subjects);
}

string Kernel::operand_walk_forward(uint32_t id)
{
    Operand operand(block_.operand(id), id);    // Grab the operand
    stringstream ss;

    return "";
}

string Kernel::operand_walk_forward(uint32_t id, uint32_t dim)
{
    Operand operand(block_.operand(id), id);    // Grab the operand
    stringstream ss;
    
    return "";
}

string Kernel::declare_operands(void)
{
    stringstream ss;
    for(size_t oidx=0; oidx<block_.noperands(); ++oidx) {
        ss << declare_operand(oidx);
    }
    return ss.str();
}

string Kernel::declare_operand(uint32_t id)
{
    Operand operand(block_.operand(id), id);    // Grab the operand
    stringstream ss;
    ss << "// Argument " << operand.name() << " [" << operand.layout() << "]" << endl;
    switch(operand.operand_.layout) {
        case STRIDED:       
        case SPARSE:
        case CONTIGUOUS:
            ss
            << _declare(
                _ptr(operand.etype()),
                operand.current(),
                _deref(operand.first())
            );
            break;

        case SCALAR:
            ss
             << _declare(
                operand.etype(),
                operand.current(),
                _deref(operand.first())
            );
            break;
        case SCALAR_CONST:
            ss
            << _declare(
                _const(operand.etype()),
                operand.current(),
                _deref(operand.first())
            );

            break;
        case SCALAR_TEMP:
            ss
            << _declare(
                operand.etype(),
                operand.current()
            );
            break;

        default:
            break;
    }
    ss << endl;
    return ss.str();
}

string Kernel::unpack_argument(uint32_t id)
{
    Operand operand(block_.operand(id), id);    // Grab the operand
    stringstream ss;
    ss << "// Argument " << operand.name() << " [" << operand.layout() << "]" << endl;
    switch(operand.operand_.layout) {
        case STRIDED:       
        case SPARSE:        // start, nelem, ndim, shape, stride
            ss
            << _declare(
                _const(_int64()), operand.start(),
                _access_ptr(_index(args(), id), "start")
            ) 
            << _declare(
                _const(_int64()), operand.nelem(),
                _access_ptr(_index(args(), id), "nelem")
            )
            << _declare(
                _const(_int64()), operand.ndim(),
                _access_ptr(_index(args(), id), "ndim")
            )
            << _declare(
                _ptr_const(_int64()), operand.shape(),
                _access_ptr(_index(args(), id), "shape")
            )
            << _declare(
                _ptr_const(_int64()), operand.stride(),
                _access_ptr(_index(args(), id), "stride")
            );

        case CONTIGUOUS:    // "first" = operand_t.data + operand_t.start
            ss
            << _declare(
                _ptr_const(operand.etype()), operand.first(),
                _add(
                    _cast(
                        _ptr(operand.etype()),
                        _deref(_access_ptr(_index(args(), id), "data"))
                    ),
                    _access_ptr(_index(args(), id), "start")
                )
            ) 
            << _assert_not_null(operand.first()) << _end();
            break;

        case SCALAR:
        case SCALAR_CONST:  // "first" = operand_t.data
            ss << _declare(
                _ptr_const(operand.etype()), operand.first(),
                _cast(
                    _ptr(operand.etype()),
                    _deref(_access_ptr(_index(args(), id), "data"))
                )
            )
            << _assert_not_null(operand.first()) << _end();
            break;
        case SCALAR_TEMP:   // Data pointer is never used.
        default:
            break;
    }
    ss << endl;
    return ss.str();
}

}}}}
