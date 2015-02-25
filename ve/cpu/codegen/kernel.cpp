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

Kernel::Kernel(Plaid& plaid, Block& block) : plaid_(plaid), block_(block), iterspace_(block.iterspace()) {

    for(size_t tac_idx=0; tac_idx<block_.ntacs(); ++tac_idx) {
        tac_t& tac = block_.tac(tac_idx);
        if (not ((tac.op & (ARRAY_OPS))>0)) {   // Only interested in array ops
            continue;
        }
        cout << tac_text(tac) << endl;
        tacs_.push_back(&tac);
        switch(tac_noperands(tac)) {
            case 3:
                operands_[tac.in2] = Operand(
                    &block_.operand(block_.global_to_local(tac.in2)),
                    block_.global_to_local(tac.in2)
                );
            case 2:
                operands_[tac.in1] = Operand(
                    &block_.operand(block_.global_to_local(tac.in1)),
                    block_.global_to_local(tac.in1)
                );
            case 1:
                operands_[tac.out] = Operand(
                    &block_.operand(block_.global_to_local(tac.out)),
                    block_.global_to_local(tac.out)
                );
            default:
                break;
        }
    }
}

string Kernel::args(void)
{
    return "args";
}

Iterspace& Kernel::iterspace(void)
{
    return iterspace_;
}

uint64_t Kernel::noperands(void)
{
    return tacs_.size();
}

Operand& Kernel::operand_glb(uint64_t gidx)
{
    return operands_[gidx];
}

Operand& Kernel::operand_lcl(uint64_t lidx)
{
    return operands_[block_.local_to_global(lidx)];
}

kernel_operand_iter Kernel::operands_begin(void)
{
    return operands_.begin();
}

kernel_operand_iter Kernel::operands_end(void)
{
    return operands_.end();
}

uint32_t Kernel::omask(void)
{
    return block_.omask();
}

uint64_t Kernel::ntacs(void)
{
    return tacs_.size();
}

tac_t& Kernel::tac(uint64_t tidx)
{
    return *tacs_[tidx];
}

kernel_tac_iter Kernel::tacs_begin(void)
{
    return tacs_.begin();
}

kernel_tac_iter Kernel::tacs_end(void)
{
    return tacs_.end();
}

string Kernel::generate_source(void)
{
    std::map<string, string> subjects;
    Walker walker(plaid_, *this);

    if (block_.narray_tacs()>1) {
        subjects["MODE"] = "FUSED";
    } else {
        subjects["MODE"] = "SIJ";
    }
    subjects["LAYOUT"]          = layout_text(block_.iterspace().layout);
    subjects["NINSTR"]          = to_string(block_.ntacs());
    subjects["NARRAY_INSTR"]    = to_string(block_.narray_tacs());
    subjects["NARGS"]           = to_string(block_.noperands());
    subjects["NARRAY_ARGS"]     = to_string(operands_.size());
    subjects["OMASK"]           = omask_text(omask());
    subjects["SYMBOL_TEXT"]     = block_.symbol_text();
    subjects["SYMBOL"]          = block_.symbol();
    subjects["ARGUMENTS"]       = unpack_arguments();
    subjects["WALKER"]          = walker.generate_source();

    return plaid_.fill("kernel", subjects);
}

string Kernel::unpack_arguments(void)
{
    stringstream ss;
    for(kernel_operand_iter oit=operands_begin(); oit != operands_end(); ++oit) {
        Operand& operand = oit->second;
        uint64_t id = operand.local_id();
        ss << "// Argument " << operand.name() << " [" << operand.layout() << "]" << endl;
        switch(operand.meta().layout) {
            case STRIDED:       
            case SPARSE:        // ndim, shape, stride
                ss
                << _declare_init(
                    _const(_int64()),
                    operand.ndim(),
                    _access_ptr(_index(args(), id), "ndim")
                )
                << _end()
                << _declare_init(
                    _ptr_const(_int64()),
                    operand.shape(),
                    _access_ptr(_index(args(), id), "shape")
                )
                << _end()
                << _declare_init(
                    _ptr_const(_int64()),
                    operand.stride(),
                    _access_ptr(_index(args(), id), "stride")
                )
                << _end();

            case CONTIGUOUS:    // "first" = operand_t->data + operand_t->start
                ss
                << _declare_init(
                    _ptr_const(operand.etype()),
                    operand.first(),
                    _add(
                        _cast(
                            _ptr(operand.etype()),
                            _deref(_access_ptr(_index(args(), id), "data"))
                        ),
                        _access_ptr(_index(args(), id), "start")
                    )
                )
                << _end() 
                << _assert_not_null(operand.first())
                << _end();
                break;

            case SCALAR:
            case SCALAR_CONST:  // "first" = operand_t->data
                ss << _declare_init(
                    _ptr_const(operand.etype()),
                    operand.first(),
                    _cast(
                        _ptr(operand.etype()),
                        _deref(_access_ptr(_index(args(), id), "data"))
                    )
                )
                << _end() 
                << _assert_not_null(operand.first()) << _end();
                break;
            case SCALAR_TEMP:   // Data pointer is never used.
            default:
                break;
        }
        ss << endl;

    }
    return ss.str();
}

}}}}
