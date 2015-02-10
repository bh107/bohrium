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

Kernel::Kernel(Block& block) : block_(block) {}

string Kernel::unpack_operands(void)
{
    stringstream ss;
    for(size_t oidx=0; oidx<block_.noperands(); ++oidx) {
        ss << unpack_operand(oidx);
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

string Kernel::head(void)
{
    stringstream ss;
    
    ss
    << "/*" << endl
    << "KERNEL-DESCRIPTION {" << endl
    << " MODE           = "                                           << "," << endl
    << " LAYOUT         = " << layout_text(block_.iterspace().layout) << "," << endl
    << " NINSTR         = " << block_.ntacs()                         << "," << endl
    << " NARRAY_INSTR   = " << block_.narray_tacs()                   << "," << endl
    << " NARGS          = " << block_.noperands()                     << "," << endl
    << " NARRAY_ARGS    = "                                           << "," << endl
    << " SYMBOL_TEXT    = " << block_.symbol_text()                          << endl
    << "}" << endl
    << "*/" << endl 
    << "void KRN_" << block_.symbol() << "("
    << "operand_t** " << args() << ", "
    << "iterspace_t* " << iterspace() << ") {" << endl;
    return ss.str();
}

string Kernel::body(void)
{
    stringstream ss;
    ss << unpack_operands();
    return ss.str();
}

string Kernel::foot(void)
{
    stringstream ss;
    ss << "}";
    return ss.str();
}

string Kernel::generate_source(void)
{
    stringstream ss;
    ss
    << head()
    << body()
    << foot()
    ;
    return ss.str();
}

string Kernel::unpack_operand(uint32_t id)
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
