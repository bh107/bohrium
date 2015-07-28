#include <sstream>
#include <string>
#include "codegen.hpp"

using namespace std;
using namespace bohrium::core;

namespace bohrium{
namespace engine{
namespace cpu{
namespace codegen{

Walker::Walker(Plaid& plaid, Kernel& kernel) : plaid_(plaid), kernel_(kernel) {}

string Walker::declare_operands(void)
{
    stringstream ss;

    for(kernel_operand_iter oit=kernel_.operands_begin();
        oit != kernel_.operands_end();
        ++oit) {
        Operand& operand = oit->second;
        bool restrictable = kernel_.base_refcount(oit->first)==1;
        switch(operand.meta().layout) {
            case SCALAR_CONST:
                break;

            case SCALAR:
                ss
                << _declare_init(
                    operand.etype(),
                    operand.walker(),
                    _deref(_add(operand.buffer_data(), operand.start()))
                );
                break;

            case SCALAR_TEMP:
            case CONTRACTABLE:
                ss
                << _declare(
                    operand.etype(),
                    operand.walker()
                );
                break;

            case CONTIGUOUS:
            case CONSECUTIVE:
            case STRIDED:
                if (restrictable) {
                    ss
                    << _declare_init(
                        _restrict(_ptr(operand.etype())),
                        operand.walker(),
                        _add(operand.buffer_data(), operand.start())
                    );
                } else {
                    ss
                    << _declare_init(
                        _ptr(operand.etype()),
                        operand.walker(),
                        _add(operand.buffer_data(), operand.start())
                    );
                }
                break;

            case SPARSE:
				ss
				<< _beef("Unimplemented LAYOUT.");
				break;
        }
        ss << _end(operand.layout());
    }
    return ss.str();
}

string Walker::offload_leo(void)
{
    stringstream ss;

    ss << "#pragma offload \\" << endl;
    ss << "        target(mic:offload_devid)   \\" << endl;
    ss << "        in(iterspace_layout) \\" << endl;
    ss << "        in(iterspace_ndim) \\" << endl;
    ss << "        in(iterspace_shape:length(CPU_MAXDIM)) \\" << endl;
    ss << "        in(iterspace_nelem) \\" << endl;
    for(kernel_operand_iter oit=kernel_.operands_begin();
        oit != kernel_.operands_end();
        ++oit) {
        Operand& operand = oit->second;
        switch(operand.meta().layout) {
            case SCALAR_CONST:
            case SCALAR_TEMP:
            case CONTRACTABLE:
                break;

            case CONTIGUOUS:
            case CONSECUTIVE:
            case STRIDED:
            case SCALAR:
                ss << "in(" << operand.strides() << ":length(CPU_MAXDIM) alloc_if(1) free_if(1)) \\" << endl;
                break;

            case SPARSE:
				ss << _beef("Unimplemented LAYOUT.");
				break;
        }
    }
    for(kernel_buffer_iter bit=kernel_.buffers_begin();
        bit != kernel_.buffers_end();
        ++bit) { 
        Buffer& buffer = bit->second;
        ss << "nocopy(" << buffer.data() << ":length(" << buffer.nelem() << ") alloc_if(0) free_if(0)) \\" << endl;
    }
    ss << "if(offload)";
    return ss.str();
}


string Walker::assign_collapsed_offset(uint32_t rank, uint64_t oidx)
{
    stringstream ss;
    LAYOUT ispace_layout = kernel_.iterspace().meta().layout;
    Operand& operand = kernel_.operand_glb(oidx);
    switch(operand.meta().layout) {
        case SCALAR_TEMP:
        case SCALAR_CONST:
        case SCALAR:
        case CONTRACTABLE:
            break;
        
        case CONTIGUOUS:
            // CONT COMPATIBLE iteration construct
            // or specialized
            // STRIDED construct for rank=1
            if (((ispace_layout & COLLAPSIBLE)>0) or (rank==1)) {
                ss << _add_assign(
                    operand.walker(),
                    "work_offset"
                ) << _end();
            // STRIDED iteration construct with rank>1
            } else {
                ss << _add_assign(
                    operand.walker(),
                    _mul("work_offset", _index("weight", 0))
                ) << _end();
            }
            break;

        case CONSECUTIVE:
            // CONT COMPATIBLE iteration construct
            // or specialized
            // STRIDED construct for rank=1
            if (((ispace_layout & COLLAPSIBLE)>0) or (rank==1)) {
                ss << _add_assign(
                    operand.walker(),
                    _mul("work_offset", operand.stride_inner())
                ) << _end();
            // STRIDED iteration construct with rank>1
            } else {
                ss << _add_assign(
                    operand.walker(),
                    _mul("work_offset", _index("weight", 0))
                ) << _end();
            }
            break;

        case STRIDED:       
            switch(rank) {
                case 3:
                case 2:
                case 1:
                    ss << _add_assign(
                        operand.walker(),
                        _mul("work_offset", _index(operand.strides(), 0))
                    )
                    << _end();
                    break;
                default:
                    // TODO: implement ND-case
                    break;
            }
            break;

        case SPARSE:
            ss << _beef("Non-implemented LAYOUT.");
            break;
    }
    return ss.str();
}

string Walker::assign_collapsed_offset(uint32_t rank)
{
    stringstream ss;
    for(kernel_operand_iter oit=kernel_.operands_begin();
        oit != kernel_.operands_end();
        ++oit) {
        ss << assign_collapsed_offset(rank, oit->first);
        
    }
    return ss.str();
}

string Walker::assign_offset_outer_2D(uint64_t oidx)
{
    stringstream ss;
    Operand& operand = kernel_.operand_glb(oidx);
    switch(operand.meta().layout) {
        case SCALAR_TEMP:
        case SCALAR_CONST:
        case SCALAR:
        case CONTRACTABLE:
            break;
        
        case CONTIGUOUS:
        case CONSECUTIVE:
        case STRIDED:       
            ss << _add_assign(
                operand.walker(),
                _mul("work_offset", _index(operand.strides(), 0))
            )
            << _end();
            break;

        case SPARSE:
            ss << _beef("Non-implemented LAYOUT.");
            break;
    }
    return ss.str();
}

string Walker::assign_offset_outer_2D(void)
{
    stringstream ss;
    for(kernel_operand_iter oit=kernel_.operands_begin();
        oit != kernel_.operands_end();
        ++oit) {
        ss << assign_offset_outer_2D(oit->first);
        
    }
    return ss.str();
}

string Walker::declare_stride_inner(uint64_t oidx)
{
    stringstream ss;
    Operand& operand = kernel_.operand_glb(oidx);
    switch(operand.meta().layout) {
        case SCALAR_TEMP:
        case SCALAR_CONST:
        case SCALAR:
        case CONTRACTABLE:
        case CONTIGUOUS:
            ss << "// " << operand.name() << " " << operand.layout() << endl;
            break;

        case CONSECUTIVE:
        case STRIDED:
            ss << _declare_init(
                _const(_int64()),
                operand.stride_inner(),
                _index(operand.strides(), "inner_dim")
            )
            << _end(operand.layout());
            break;

        case SPARSE:
            ss << _beef("Non-implemented LAYOUT.");
			break;
    }
    return ss.str();
}

string Walker::declare_stride_inner(void)
{
    stringstream ss;

    for(kernel_operand_iter oit=kernel_.operands_begin();
        oit != kernel_.operands_end();
        ++oit) {
        ss << declare_stride_inner(oit->first);
    }
    return ss.str();
}

string Walker::declare_stride_outer_2D(uint64_t oidx)
{
    stringstream ss;
    Operand& operand = kernel_.operand_glb(oidx);
    if (2 == operand.meta().ndim) {
        switch(operand.meta().layout) {
            case SCALAR_TEMP:
            case SCALAR_CONST:
            case SCALAR:
            case CONTRACTABLE:
            case CONTIGUOUS:
                ss << "// " << operand.name() << " " << operand.layout() << endl;
                break;

            case CONSECUTIVE:
            case STRIDED:
                ss <<
                _declare_init(
                    _const(_int64()),
                    operand.stride_outer(),
                    _sub(
                        _index(operand.strides(), "outer_dim"),
                        _mul(
                            _index(operand.strides(), "inner_dim"),
                            "chunksize"
                        )
                    )
                )
                << _end(operand.layout());
                break;

            case SPARSE:
                ss << _beef("Non-implemented LAYOUT.");
                break;
        }
    } else {
        switch(operand.meta().layout) {
            case SCALAR_TEMP:
            case SCALAR_CONST:
            case SCALAR:
            case CONTRACTABLE:
                ss << "// " << operand.name() << " " << operand.layout() << endl;
                break;

            case CONSECUTIVE:
            case STRIDED:
            case CONTIGUOUS:
                ss <<
                _declare_init(
                    _const(_int64()),
                    operand.stride_outer(),
                    _index(operand.strides(), "0")
                ) << _end(operand.layout());
                break;

            case SPARSE:
                ss << _beef("Non-implemented LAYOUT.");
                break;
        }
    }
    return ss.str();
}

string Walker::declare_stride_outer_2D(void)
{
    stringstream ss;

    for(kernel_operand_iter oit=kernel_.operands_begin();
        oit != kernel_.operands_end();
        ++oit) {
        ss << declare_stride_outer_2D(oit->first);
    }
    return ss.str();
}

string Walker::declare_stride_axis(uint64_t oidx)
{
    stringstream ss;
    Operand& operand = kernel_.operand_glb(oidx);
    switch(operand.meta().layout) {
        case SCALAR_TEMP:
        case SCALAR_CONST:
        case SCALAR:
        case CONTRACTABLE:
            ss << "// " << operand.name() << " " << operand.layout() << endl;
            break;

        case CONTIGUOUS:
        case CONSECUTIVE:
        case STRIDED:
            ss << _declare_init(
                _const(_int64()),
                operand.stride_axis(),
                _index(operand.strides(), "axis_dim")
            )
            << _end(operand.layout());
            break;

        case SPARSE:
            ss << _beef("Non-implemented LAYOUT.");
			break;
    }
    return ss.str();
}

string Walker::declare_stride_axis(void)
{
    stringstream ss;

    for(kernel_operand_iter oit=kernel_.operands_begin();
        oit != kernel_.operands_end();
        ++oit) {
        ss << declare_stride_axis(oit->first);
    }
    return ss.str();
}

string Walker::step_fwd_outer(uint64_t glb_idx)
{
    stringstream ss;

    Operand& operand = kernel_.operand_glb(glb_idx);
    switch(operand.meta().layout) {
        case SPARSE:
        case STRIDED:
        case CONTIGUOUS:
        case CONSECUTIVE:
            ss <<
            _add_assign(
                operand.walker(),
                _mul("coord", _index(operand.strides(), "dim"))
            ) << _end(operand.layout());
            break;

        case SCALAR_TEMP:
        case SCALAR_CONST:
        case SCALAR:        // No stepping for these
        case CONTRACTABLE:
            ss << "// " << operand.name() << " " << operand.layout() << endl;
            break;
    }
 
    return ss.str();
}

string Walker::step_fwd_outer(void)
{
    stringstream ss;
    for(set<uint64_t>::iterator it=outer_opds_.begin(); it!=outer_opds_.end(); it++) {
        ss << step_fwd_outer(*it);
    }
    return ss.str();
}

string Walker::step_fwd_outer_2D(uint64_t glb_idx)
{
    stringstream ss;

    Operand& operand = kernel_.operand_glb(glb_idx);
    switch(operand.meta().layout) {
        case CONTIGUOUS:
            if (2 == operand.meta().ndim) {
                break;
            }
        case SPARSE:
        case STRIDED:
        case CONSECUTIVE:
            ss <<
            _add_assign(
                operand.walker(),
                operand.stride_outer()
            ) << _end(operand.layout());
            break;

        case SCALAR_TEMP:
        case SCALAR_CONST:
        case SCALAR:        // No stepping for these
        case CONTRACTABLE:
            ss << "// " << operand.name() << " " << operand.layout() << endl;
            break;
    }
 
    return ss.str();
}

string Walker::step_fwd_outer_2D(void)
{
    stringstream ss;
    for(set<uint64_t>::iterator it=outer_opds_.begin(); it!=outer_opds_.end(); it++) {
        ss << step_fwd_outer_2D(*it);
    }
    return ss.str();
}

string Walker::step_fwd_inner(uint64_t glb_idx)
{
    stringstream ss;

    Operand& operand = kernel_.operand_glb(glb_idx);
    switch(operand.meta().layout) {
        case SCALAR_TEMP:
        case SCALAR_CONST:
        case SCALAR:        
        case CONTRACTABLE:
            ss << "// " << operand.name() << " " << operand.layout() << endl;
            break;

        case STRIDED:
        case CONSECUTIVE:
            ss
            << _add_assign(
                operand.walker(),
                operand.stride_inner()
            ) << _end(operand.layout());
            break;

        case CONTIGUOUS:
            ss <<
            _inc(operand.walker()) << _end(operand.layout());
            break;

        case SPARSE:
            ss << _beef("Non-implemented layout.");
            break;
    }

    return ss.str();
}

string Walker::step_fwd_inner(void)
{
    stringstream ss;
    for(set<uint64_t>::iterator it=inner_opds_.begin(); it!=inner_opds_.end(); it++) {
        ss << step_fwd_inner(*it);
    }
    return ss.str();
}

string Walker::step_fwd_other(uint64_t glb_idx, string dimvar)
{
    stringstream ss;

    Operand& operand = kernel_.operand_glb(glb_idx);
    switch(operand.meta().layout) {
        case SPARSE:
        case STRIDED:
        case CONTIGUOUS:
        case CONSECUTIVE:
            ss <<
            _add_assign(
                operand.walker(),
                _mul("coord", _index(operand.strides(), dimvar))
            ) << _end(operand.layout());
            break;

        case SCALAR_TEMP:
        case SCALAR_CONST:
        case SCALAR:        // No stepping for these
        case CONTRACTABLE:
            ss << "// " << operand.name() << " " << operand.layout() << endl;
            break;
    }
 
    return ss.str();
}

string Walker::step_fwd_other(void)
{
    stringstream ss;
    for(set<uint64_t>::iterator it=inner_opds_.begin(); it!=inner_opds_.end(); it++) {
        ss << step_fwd_other(*it, "dim");
    }

    set<uint64_t> others;
    set_difference(outer_opds_.begin(), outer_opds_.end(), inner_opds_.begin(), inner_opds_.end(), inserter(others, others.end()));

    for(set<uint64_t>::iterator it=others.begin(); it!=others.end(); it++) {
        ss << step_fwd_other(*it, "other_dim");
    }
    return ss.str();
}

string Walker::step_fwd_axis(uint64_t glb_idx)
{
    stringstream ss;

    Operand& operand = kernel_.operand_glb(glb_idx);
    switch(operand.meta().layout) {
        case SCALAR_TEMP:
        case SCALAR_CONST:
        case SCALAR:        
        case CONTRACTABLE:
            ss << "// " << operand.name() << " " << operand.layout() << endl;
            break;

        case STRIDED:
        case CONSECUTIVE:
        case CONTIGUOUS:
            ss
            << _add_assign(
                operand.walker(),
                operand.stride_axis()
            ) << _end(operand.layout());
            break;

        case SPARSE:
            ss << _beef("Non-implemented layout.");
            break;
    }

    return ss.str();
}

string Walker::step_fwd_axis(void)
{
    stringstream ss;
    for(set<uint64_t>::iterator it=inner_opds_.begin(); it!=inner_opds_.end(); it++) {
        ss << step_fwd_axis(*it);
    }
    return ss.str();
}

string Walker::operations(void)
{
    stringstream ss;
    
    for(kernel_tac_iter tit=kernel_.tacs_begin();
        tit!=kernel_.tacs_end();
        ++tit) {
        tac_t& tac = **tit;
        ETYPE etype;
        if (ABSOLUTE == tac.oper) {
            etype = kernel_.operand_glb(tac.in1).meta().etype;
        } else {
            etype = kernel_.operand_glb(tac.out).meta().etype;
        }

        string out = "ERROR_OUT", in1 = "ERROR_IN1", in2 = "ERROR_IN2";
        switch(tac.op) {
            case MAP:
            case ZIP:
            case GENERATE:
                switch(core::tac_noperands(tac)) {
                    case 3:
                        inner_opds_.insert(tac.in2);
                        outer_opds_.insert(tac.in2);

                        in2 = kernel_.operand_glb(tac.in2).walker_val();
                    case 2:
                        inner_opds_.insert(tac.in1);
                        outer_opds_.insert(tac.in1);

                        in1 = kernel_.operand_glb(tac.in1).walker_val();
                    case 1:
                        inner_opds_.insert(tac.out);
                        outer_opds_.insert(tac.out);

                        out = kernel_.operand_glb(tac.out).walker_val();
                    default:
                        break;
                }
                ss << _assign(
                    out,
                    oper(tac.oper, etype, in1, in2)
                ) << _end(oper_description(tac));
                break;

            case REDUCE_COMPLETE:
            case REDUCE_PARTIAL:
                inner_opds_.insert(tac.in1);

                outer_opds_.insert(tac.in1);
                outer_opds_.insert(tac.out);

                ss << _assign(
                    kernel_.operand_glb(tac.out).accu(),
                    oper(
                        tac.oper,
                        kernel_.operand_glb(tac.out).meta().etype,
                        kernel_.operand_glb(tac.out).accu(),
                        kernel_.operand_glb(tac.in1).walker_val()
                    )
                ) << _end();
                break;

            case SCAN:
                inner_opds_.insert(tac.in1);
                outer_opds_.insert(tac.in1);
                
                inner_opds_.insert(tac.out);
                outer_opds_.insert(tac.out);

                in1 = kernel_.operand_glb(tac.in1).walker_val();

                ss << _assign(
                    kernel_.operand_glb(tac.out).accu(),
                    oper(tac.oper, etype, kernel_.operand_glb(tac.out).accu(), in1)
                ) << _end();
                ss << _assign(
                    kernel_.operand_glb(tac.out).walker_val(),
                    kernel_.operand_glb(tac.out).accu()
                ) << _end();
                break;

            case INDEX:
                switch(tac.oper) {
                    case GATHER:
                        inner_opds_.insert(tac.out);
                        inner_opds_.insert(tac.in2);
                        out = kernel_.operand_glb(tac.out).walker_val();
                        in1 = _add(
                            kernel_.operand_glb(tac.in1).buffer_data(),
                            kernel_.operand_glb(tac.in1).start()
                        );
                        in2 = kernel_.operand_glb(tac.in2).walker_val();

                        ss << _assign(
                            out,
                            _deref(_add(in1, in2))
                        ) << _end();
                        break;

                    case SCATTER:
                        inner_opds_.insert(tac.in1);
                        inner_opds_.insert(tac.in2);
                        out = _add(
                            kernel_.operand_glb(tac.out).buffer_data(),
                            kernel_.operand_glb(tac.out).start()
                        );
                        in1 = kernel_.operand_glb(tac.in1).walker_val();
                        in2 = kernel_.operand_glb(tac.in2).walker_val();

                        ss << _assign(
                            _deref(_add(out, in2)),
                            in1
                        ) << _end();
                        break;
                    default:
                        ss << "UNSUPPORTED_INDEX_OPERATION";
                        break;
                }
                break;

            default:
                ss << "UNSUPPORTED_OPERATION["<< operation_text(tac.op) <<"]_AT_EMITTER_STAGE";
                break;
        }
    }
    return ss.str();
}

string Walker::write_expanded_scalars(void)
{
    stringstream ss;
    set<uint64_t> written;

    for(kernel_tac_iter tit=kernel_.tacs_begin();
        tit!=kernel_.tacs_end();
        ++tit) {
        tac_t& tac = **tit;

        Operand& opd = kernel_.operand_glb(tac.out);
        if (((tac.op & (MAP|ZIP|GENERATE))>0) and \
            ((opd.meta().layout & SCALAR)>0) and \
            (written.find(tac.out)==written.end())) {
            ss << _line(_assign(
                _deref(_add(opd.buffer_data(), opd.start())),
                opd.walker_val()
            ));
            written.insert(tac.out);
        }
    }

    return ss.str();
}

string Walker::generate_source(bool offload)
{
    std::map<string, string> subjects;
    string plaid;

    if ((kernel_.omask() & ARRAY_OPS)==0) { // There must be at lest one array operation
        cerr << kernel_.text() << endl;
        throw runtime_error("No array operations!");
    }
    if (kernel_.omask() & EXTENSION) {      // Extensions are not allowed.
        cerr << kernel_.text() << endl;
        throw runtime_error("EXTENSION in kernel");
    }

    // Experimental offload pragma
    if (offload) {
        subjects["OFFLOAD"] = offload_leo();
    }

    // These are used by all kernels.
    const uint32_t rank = kernel_.iterspace().meta().ndim;
    subjects["WALKER_DECLARATION"]      = declare_operands();
    subjects["OPERATIONS"]              = operations();
    subjects["WRITE_EXPANDED_SCALARS"]  = write_expanded_scalars();

    // Kernel contains nothing but operations on SCALARs
    if ((kernel_.iterspace().meta().layout & SCALAR)>0) {
        // A couple of sanitization checks
        if ((kernel_.omask() & ACCUMULATION)>0) {
            cerr << kernel_.text() << endl;
            throw runtime_error("Accumulation in SCALAR kernel.");
        }
        plaid = "walker.scalar";
        return plaid_.fill(plaid, subjects);
    }

    // Note: start of crappy code used by reduction.
    tac_t* tac = NULL;
    Operand* out = NULL;
    Operand* in1 = NULL;
    Operand* in2 = NULL;

    // To detect special-case of partial-reduction rank > 1
    bool partial_reduction_on_innermost = false;

    if ((kernel_.omask() & ACCUMULATION)>0) {
        for(kernel_tac_iter tit=kernel_.tacs_begin();
            tit != kernel_.tacs_end();
            ++tit) {
            if ((((*tit)->op) & (ACCUMULATION))>0) {
                tac = *tit;
            }
        }
        if (tac) {
            out = &kernel_.operand_glb(tac->out);
            in1 = &kernel_.operand_glb(tac->in1);
            in2 = &kernel_.operand_glb(tac->in2);

            partial_reduction_on_innermost = *((uint64_t*)(in2->meta().const_data)) == (rank-1);
    
            // Construct reduction-type string for reduction-sync
            subjects["OPD_OUT"] = out->name();
            subjects["OPD_IN1"] = in1->name();
            subjects["OPD_IN2"] = in2->name();

            subjects["BUF_OUT"] = out->buffer_name();
            subjects["BUF_IN1"] = in1->buffer_name();

            subjects["NEUTRAL_ELEMENT"] = oper_neutral_element(tac->oper, in1->meta().etype);
            subjects["ETYPE"] = out->etype();
            subjects["ATYPE"] = in2->etype();

            // Declare local accumulator var, REDUCE_[COMPLETE|PARTIAL]|SCAN
            if ((kernel_.omask() & REDUCE_COMPLETE)>0) {
                subjects["ACCU_LOCAL_DECLARE_COMPLETE"] = _line(_declare_init(
                    in1->etype(),
                    out->accu(),
                    oper_neutral_element(tac->oper, in1->meta().etype)
                ));
            } else if ((kernel_.omask() & REDUCE_PARTIAL)>0) {
                subjects["ACCU_LOCAL_DECLARE_PARTIAL"] = _line(_declare_init(
                    in1->etype(),
                    out->accu(),
                    oper_neutral_element(tac->oper, in1->meta().etype)
                ));
            } else if ((kernel_.omask() & SCAN)>0) {
                subjects["ACCU_LOCAL_DECLARE"] = _line(_declare_init(
                    in1->etype(),
                    out->accu(),
                    oper_neutral_element(tac->oper, in1->meta().etype)
                ));
            }
        }
    }
    // Note: end of crappy code used by reductions / scan

    // MAP | ZIP | GENERATE | INDEX | REDUCE_COMPLETE on COLLAPSIBLE LAYOUT of any RANK
    // and
    // REDUCE_PARTIAL on with RANK == 1
    if (((kernel_.iterspace().meta().layout & COLLAPSIBLE)>0)       \
    and ((kernel_.omask() & SCAN)==0)                               \
    and (not((rank>1) and ((kernel_.omask() & REDUCE_PARTIAL)>0)))) {
        plaid = "walker.collapsed";

        subjects["WALKER_INNER_DIM"]    = _declare_init(
            _const(_int64()),
            "inner_dim",
            _sub(kernel_.iterspace().ndim(), "1")
        ) + _end();
        subjects["WALKER_OFFSET"]       = assign_collapsed_offset(rank);
        subjects["WALKER_STRIDE_INNER"] = declare_stride_inner();
        subjects["WALKER_STEP_INNER"]   = step_fwd_inner();

        // Reduction specfics
        if ((kernel_.omask() & REDUCTION)>0) {
            if ((out->meta().layout & (SCALAR|CONTIGUOUS|CONSECUTIVE|STRIDED))>0) {
                // Initialize the accumulator 
                subjects["ACCU_OPD_INIT"] = _line(_assign(
                    _deref(_add(out->buffer_data(), out->start())),
                    oper_neutral_element(tac->oper, in1->meta().etype)
                ));
                if ((kernel_.omask() & REDUCE_COMPLETE)>0) {
                    // Syncronize accumulator and local accumulator var
                    subjects["ACCU_OPD_SYNC_COMPLETE"] = _line(synced_oper(
                        tac->oper,
                        in1->meta().etype,
                        _deref(_add(out->buffer_data(), out->start())),
                        _deref(_add(out->buffer_data(), out->start())),
                        out->accu()
                    ));
                } else {
                    // Syncronize accumulator and local accumulator var
                    subjects["ACCU_OPD_SYNC_PARTIAL"] = _line(synced_oper(
                        tac->oper,
                        in1->meta().etype,
                        _deref(_add(out->buffer_data(), out->start())),
                        _deref(_add(out->buffer_data(), out->start())),
                        out->accu()
                    ));
                }
            }
        }

    // MAP | ZIP | REDUCE_COMPLETE | REDUCE_PARTIAL_INNER on ANY LAYOUT, RANK > 1
    } else if (((kernel_.omask() & (EWISE|REDUCE_COMPLETE|REDUCE_PARTIAL))>0) and \
                (!(((kernel_.omask() & (REDUCE_PARTIAL))>0) and \
                   (!partial_reduction_on_innermost)))) {

        subjects["WALKER_INNER_DIM"]    = _declare_init(
            _const(_int64()),
            "inner_dim",
            _sub(kernel_.iterspace().ndim(), "1")
        ) + _end();
        subjects["WALKER_STRIDE_INNER"] = declare_stride_inner();
        if (2 == rank) {
            plaid = "walker.inner.2d";
            subjects["WALKER_STRIDE_OUTER"] = declare_stride_outer_2D(); 
            subjects["WALKER_OFFSET"]       = assign_offset_outer_2D();
            subjects["WALKER_STEP_OUTER"]   = step_fwd_outer_2D();
        } else {
            plaid = "walker.inner.nd";
            subjects["WALKER_STEP_OUTER"]   = step_fwd_outer();
        }
        subjects["WALKER_STEP_INNER"]   = step_fwd_inner();

        // Reduction specfics
        if (((kernel_.omask() & (REDUCE_PARTIAL|REDUCE_COMPLETE))>0) and \
            ((out->meta().layout & (SCALAR|CONTIGUOUS|CONSECUTIVE|STRIDED))>0)) {
            // Initialize the accumulator 
            subjects["ACCU_OPD_INIT"] = _line(_assign(
                _deref(_add(out->buffer_data(), out->start())),
                oper_neutral_element(tac->oper, in1->meta().etype)
            ));

            // Syncronize accumulator and local accumulator var
            if ((kernel_.omask() & (REDUCE_PARTIAL))>0) {
                subjects["ACCU_OPD_SYNC_PARTIAL"] = _line(_assign(
                    out->walker_val(),
                    out->accu()
                ));
            } else if ((kernel_.omask() & (REDUCE_COMPLETE))>0) {
                subjects["ACCU_OPD_SYNC_COMPLETE"] = _line(synced_oper(
                    tac->oper,
                    in1->meta().etype,
                    _deref(_add(out->buffer_data(), out->start())),
                    _deref(_add(out->buffer_data(), out->start())),
                    out->accu()
                ));
            } else {
                // CRAP
                cerr << kernel_.text() << endl;
                throw runtime_error("Unexpected omask.");
            }
        }

    // MAP | ZIP | REDUCE_PARTIAL_AXIS on ANY LAYOUT, RANK > 1
    } else if ((kernel_.omask() & (EWISE|REDUCE_PARTIAL))>0) {

        plaid = "walker.axis.nd";

        subjects["WALKER_AXIS_DIM"] = _line(_declare_init(
            _const(_int64()),
            "axis_dim",
            in2->walker()
        ));
        subjects["WALKER_STRIDE_AXIS"]  = declare_stride_axis();
        subjects["WALKER_STEP_OTHER"]   = step_fwd_other();
        subjects["WALKER_STEP_AXIS"]    = step_fwd_axis();

        // Reduction specfics
        if ((kernel_.omask() & REDUCE_PARTIAL)>0) {
            if (out->meta().layout == SCALAR) {
                subjects["ACCU_OPD_SYNC_PARTIAL"]= _line(_assign(
                    _deref(_add(out->buffer_data(), out->start())),
                    out->accu()
                ));
            } else {
                subjects["ACCU_OPD_SYNC_PARTIAL"]= _line(_assign(
                    out->walker_val(),
                    out->accu()
                ));
            }
        }

    // SCAN on STRIDED LAYOUT of any RANK
    } else if ((kernel_.omask() & SCAN)>0) {
        switch(rank) {
            case 1:
                plaid = "scan.1d";

                subjects["WALKER_INNER_DIM"]    = _declare_init(
                    _const(_int64()),
                    "inner_dim",
                    _sub(kernel_.iterspace().ndim(), "1")
                ) + _end();
                subjects["WALKER_STRIDE_INNER"] = declare_stride_inner();
                subjects["WALKER_STEP_INNER"]   = step_fwd_inner();

                break;

            default:
                plaid = "scan.nd";
                break;
        }
    } else {
        cerr << kernel_.text() << endl;
        throw runtime_error("Unexpected omask.");
    }

    return plaid_.fill(plaid, subjects);
}

}}}}
