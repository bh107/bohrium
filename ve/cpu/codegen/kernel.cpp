#include <sstream>
#include "utils.hpp"
#include "codegen.hpp"

using namespace std;
using namespace kp::core;

namespace kp{
namespace engine{
namespace codegen{

Kernel::Kernel(Plaid& plaid, Block& block) : plaid_(plaid), block_(block), iterspace_(block.iterspace()) {

    for(size_t tac_idx=0; tac_idx<block_.ntacs(); ++tac_idx) {
        kp_tac & tac = block_.tac(tac_idx);
        if (not ((tac.op & (KP_ARRAY_OPS))>0)) {   // Only interested in array ops
            continue;
        }
        tacs_.push_back(&tac);
        switch(tac_noperands(tac)) {
            case 3:
                add_operand(tac.in2);
            case 2:
                add_operand(tac.in1);
            case 1:
                add_operand(tac.out);
            default:
                break;
        }
    }
}

string Kernel::text(void)
{
    stringstream ss;
    ss << block_.text() << endl;
    return ss.str();
}

void Kernel::add_operand(uint64_t global_idx)
{
    uint64_t local_idx = block_.global_to_local(global_idx);

    kp_operand & operand = block_.operand(local_idx);
    
    Buffer* buffer = NULL;  // Associate a Buffer instance
    if ((operand.base) && ((operand.layout & KP_DYNALLOC_LAYOUT)>0)) {
        size_t buffer_id = block_.resolve_buffer(operand.base);
        buffer = new Buffer(operand.base, buffer_id);
        buffers_[buffer_id] = *buffer;
    }

    operands_[global_idx] = Operand(
        &operand,
        local_idx,
        buffer
    );
}

string Kernel::buffers(void)
{
    return "buffers";
}

string Kernel::args(void)
{
    return "args";
}

Iterspace& Kernel::iterspace(void)
{
    return iterspace_;
}

uint64_t Kernel::base_refcount(uint64_t gidx)
{
    return block_.buffer_refcount(operand_glb(gidx).meta().base);
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

kernel_buffer_iter Kernel::buffers_begin(void)
{
    return buffers_.begin();
}

kernel_buffer_iter Kernel::buffers_end(void)
{
    return buffers_.end();
}

uint32_t Kernel::omask(void)
{
    return block_.omask();
}

uint64_t Kernel::ntacs(void)
{
    return tacs_.size();
}

kp_tac & Kernel::tac(uint64_t tidx)
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

string Kernel::generate_source(bool offload)
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
    subjects["BUFFERS"]         = unpack_buffers();
    subjects["ARGUMENTS"]       = unpack_arguments();
    subjects["ITERSPACE"]       = unpack_iterspace();
    subjects["WALKER"]          = walker.generate_source(offload);

    return plaid_.fill("kernel", subjects);
}

string Kernel::unpack_iterspace(void)
{
    stringstream ss;
    ss << _declare_init(
        "KP_LAYOUT",
        iterspace().layout(),
        _access_ptr(iterspace().name(), "layout")
    )
    << _end();
    ss << _declare_init(
        _const(_int64()),
        iterspace().ndim(),
        _access_ptr(iterspace().name(), "ndim")
    )
    << _end();
    ss << _declare_init(
        _ptr(_int64()),
        iterspace().shape(),
        _access_ptr(iterspace().name(), "shape")
    )
    << _end();
    ss << _declare_init(
        _const(_int64()),
        iterspace().nelem(),
        _access_ptr(iterspace().name(), "nelem")
    )
    << _end();

    return ss.str();
}

string Kernel::unpack_buffers(void)
{
    stringstream ss;
    for(int64_t bid=0; bid< block_.nbuffers(); ++bid) {
        Buffer buffer(&block_.buffer(bid), bid);
        ss << endl;
        ss << "// Buffer " << buffer.name() << endl;
        ss << _declare_init(
            _ptr(buffer.etype()),
            buffer.data(),
            _access_ptr(
                _index("buffers", bid),
                "data"
            )
        ) << _end();
        ss << _declare_init(
            _int64(),
            buffer.nelem(),
            _access_ptr(
                _index("buffers", bid),
                "nelem"
            )
        ) << _end();
        ss << _assert_not_null(buffer.data()) << _end();
    }
    return ss.str();
}

string Kernel::unpack_arguments(void)
{
    stringstream ss;
    for(kernel_operand_iter oit=operands_begin(); oit != operands_end(); ++oit) {
        Operand& operand = oit->second;
        uint64_t id = operand.local_id();
        ss << endl;
        ss << "// Argument " << operand.name() << " [" << operand.layout() << "]" << endl;
        switch(operand.meta().layout) {
            case KP_STRIDED:
            case KP_CONSECUTIVE:
            case KP_CONTIGUOUS:
            case KP_SCALAR:
                ss
                << _declare_init(
                    _ptr_const(_int64()),
                    operand.strides(),
                    _access_ptr(_index(args(), id), "stride")
                )
                << _end();
                ss
                << _declare_init(
                    _const(_int64()),
                    operand.start(),
                    _access_ptr(_index(args(), id), "start")
                )
                << _end();
                ss
                << _declare_init(
                    _const(_int64()),
                    operand.nelem(),
                    _access_ptr(_index(args(), id), "nelem")
                )
                << _end();
                break;

            case KP_SCALAR_CONST:
                ss
                << _declare_init(
                    _const(operand.etype()),
                    operand.walker(),
                    _deref(_cast(
                        _ptr(operand.etype()),
                        _access_ptr(_index(args(), id), "const_data")
                    ))
                )
                << _end();
                break;

            case KP_SCALAR_TEMP:
            case KP_CONTRACTABLE:  // Data pointer is never used.
                ss << _comment("No unpacking needed.") << endl;
                break;

            case KP_SPARSE:
                ss << _beef("Unpacking not implemented for KP_LAYOUT!");
                break;
        }
    }
    return ss.str();
}

}}}
