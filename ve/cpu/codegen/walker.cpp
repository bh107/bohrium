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

string Walker::simd_pragma(void)
{
    stringstream ss;
    for(kernel_operand_iter oit=kernel_.operands_begin();
        oit != kernel_.operands_end();
        ++oit) {
        Operand& operand = oit->second;
        bool restrictable = kernel_.base_refcount(oit->first)==1;
        switch(operand.meta().layout) {
            case STRIDED:       
            case SPARSE:
            case CONTIGUOUS:
                if (restrictable) {
                    ss
                    << _declare_init(
                        _restrict(_ptr(operand.etype())),
                        operand.walker(),
                        operand.first()
                    );
                } else {
                    ss
                    << _declare_init(
                        _ptr(operand.etype()),
                        operand.walker(),
                        operand.first()
                    );
                }
                break;

            case SCALAR:
                ss
                 << _declare_init(
                    operand.etype(),
                    operand.walker(),
                    _deref(operand.first())
                );
                break;
            case SCALAR_CONST:
                ss
                << _declare_init(
                    _const(operand.etype()),
                    operand.walker(),
                    _deref(operand.first())
                );

                break;
            case SCALAR_TEMP:
                ss
                << _declare(
                    operand.etype(),
                    operand.walker()
                );
                break;

            default:
                break;
        }
    }
    
    return ss.str();
}

string Walker::declare_operands(void)
{
    stringstream ss;

    for(kernel_operand_iter oit=kernel_.operands_begin();
        oit != kernel_.operands_end();
        ++oit) {
        Operand& operand = oit->second;
        bool restrictable = kernel_.base_refcount(oit->first)==1;
        switch(operand.meta().layout) {
            case STRIDED:       
            case SPARSE:
            case CONTIGUOUS:
                if (restrictable) {
                    ss
                    << _declare_init(
                        _restrict(_ptr(operand.etype())),
                        operand.walker(),
                        operand.first()
                    );
                } else {
                    ss
                    << _declare_init(
                        _ptr(operand.etype()),
                        operand.walker(),
                        operand.first()
                    );
                }
                break;

            case SCALAR:
                ss
                 << _declare_init(
                    operand.etype(),
                    operand.walker(),
                    _deref(operand.first())
                );
                break;
            case SCALAR_CONST:
                ss
                << _declare_init(
                    _const(operand.etype()),
                    operand.walker(),
                    _deref(operand.first())
                );

                break;
            case SCALAR_TEMP:
                ss
                << _declare(
                    operand.etype(),
                    operand.walker()
                );
                break;

            default:
                break;
        }
        ss << _end(operand.layout());
    }
    return ss.str();
}

string Walker::ewise_assign_offset(uint32_t rank, uint64_t oidx)
{
    stringstream ss;
    LAYOUT ispace_layout = kernel_.iterspace().meta().layout;
    Operand& operand = kernel_.operand_glb(oidx);
    switch(operand.meta().layout) {
        case STRIDED:       
        case SPARSE:
            switch(rank) {
                case 3:
                case 2:
                case 1:
                    ss << _add_assign(
                        operand.walker(),
                        _mul("work_offset", _index(operand.stride(), 0))
                    )
                    << _end();
                    break;
                default:
                    // TODO: implement ND-case
                    break;
            }
            break;
        case CONTIGUOUS:
            // CONT COMPATIBLE iteration construct
            // or specialized
            // STRIDED construct for rank=1
            if (((ispace_layout & CONT_COMPATIBLE)>0) or (rank==1)) {
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
        default:
            break;
    }
    return ss.str();
}

string Walker::ewise_assign_offset(uint32_t rank)
{
    stringstream ss;
    for(kernel_operand_iter oit=kernel_.operands_begin();
        oit != kernel_.operands_end();
        ++oit) {
        ss << ewise_assign_offset(rank, oit->first);
        
    }
    return ss.str();
}

string Walker::declare_stridesize(uint64_t oidx)
{
    stringstream ss;
    Operand& operand = kernel_.operand_glb(oidx);
    const uint32_t rank = operand.meta().ndim;
    switch(operand.meta().layout) {
        case SPARSE:
        case STRIDED:
            switch(rank) {
                case 3:
                    ss << _declare_init(
                        _const(_uint64()),
                        operand.stridevar(rank-3),
                        _index(operand.stride(), rank-3)
                    )
                    << _end();
                case 2:
                    ss << _declare_init(
                        _const(_uint64()),
                        operand.stridevar(rank-2),
                        _index(operand.stride(), rank-2)
                    )
                    << _end();
                case 1:
                    ss << _declare_init(
                        _const(_uint64()),
                        operand.stridevar(rank-1),
                        _index(operand.stride(), rank-1)
                    )
                    << _end(operand.layout());
                    break;

                default:    // ND only declare stride of the innermost
                    ss << _declare_init(
                        _const(_uint64()),
                        operand.stridevar(rank-1),
                        _index(operand.stride(), "last_dim")
                    )
                    << _end(operand.layout());
                    break;
            }
            break;
        default:
            ss << "// " << operand.name() << " " << operand.layout() << endl;
            break;
    }
    return ss.str();
}

string Walker::declare_stridesizes(void)
{
    stringstream ss;

    for(kernel_operand_iter oit=kernel_.operands_begin();
        oit != kernel_.operands_end();
        ++oit) {
        ss << declare_stridesize(oit->first);
    }
    return ss.str();
}

string Walker::ewise_declare_stepsizes(uint32_t rank)
{
    stringstream ss;
    Iterspace& iterspace = kernel_.iterspace();

    for(kernel_operand_iter oit=kernel_.operands_begin();
        oit != kernel_.operands_end();
        ++oit) {
        Operand& operand = oit->second;
        switch(operand.meta().layout) {
            case SPARSE:
            case STRIDED:
                switch(rank) {
                    case 3:
                        ss << _declare_init(
                            _const(_uint64()),
                            operand.stepsize(rank-3),
                            _sub(
                                _index(operand.stride(), rank-3),
                                _mul(iterspace.shape(rank-2), _index(operand.stride(), rank-2))
                            )
                        )
                        << _end();
                    case 2:
                        ss << _declare_init(
                            _const(_uint64()),
                            operand.stepsize(rank-2),
                            _sub(
                                _index(operand.stride(), rank-2),
                                _mul(iterspace.shape(rank-1), _index(operand.stride(), rank-1))
                            )
                        )
                        << _end();
                    case 1:
                        ss << _declare_init(
                            _const(_uint64()),
                            operand.stepsize(rank-1),
                            _index(operand.stride(), rank-1)
                        )
                        << _end(operand.layout());
                        break;

                    default:    // ND only declare stepsize of the innermost
                        ss << _declare_init(
                            _const(_uint64()),
                            operand.stepsize(rank-1),
                            _index(operand.stride(), "last_dim")
                        )
                        << _end(operand.layout());
                        break;
                }
                break;
            default:
                ss << "// " << operand.name() << " " << operand.layout() << endl;
                break;
        }
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
            ss <<
            _add_assign(
                operand.walker(),
                _mul("coord", _index(operand.stride(), "dim"))
            ) << _end(operand.layout());
            break;
        case CONTIGUOUS:
            ss <<
            _add_assign(
                operand.walker(),
                _mul("coord", _index("weight", "dim"))
            ) << _end(operand.layout());
            break;
        default:
            ss << "// " << operand.name() << " " << operand.layout() << endl;
            break;
    }
 
    return ss.str();
}

string Walker::step_fwd_outer(void)
{
    stringstream ss;

    for(kernel_operand_iter oit=kernel_.operands_begin();
        oit != kernel_.operands_end();
        ++oit) {
        ss << step_fwd_outer(oit->first);
    }
    return ss.str();
}

string Walker::step_fwd_inner(uint64_t glb_idx)
{
    stringstream ss;

    Operand& operand = kernel_.operand_glb(glb_idx);
    switch(operand.meta().layout) {
        case SPARSE:
        case STRIDED:
            ss
            << _add_assign(
                operand.walker(),
                operand.stepsize(operand.meta().ndim-1)
            ) << _end(operand.layout());
            break;
        case CONTIGUOUS:
            ss <<
            _inc(operand.walker()) << _end(operand.layout());
            break;
        default:
            ss << "// " << operand.name() << " " << operand.layout() << endl;
            break;
    }

    return ss.str();
}

string Walker::step_fwd_inner(void)
{
    stringstream ss;

    for(kernel_operand_iter oit=kernel_.operands_begin();
        oit != kernel_.operands_end();
        ++oit) {
        ss << step_fwd_inner(oit->first);
    }
    return ss.str();
}

string Walker::step_fwd(uint32_t dim, uint64_t oidx)
{
    stringstream ss;

    Operand& operand = kernel_.operand_glb(oidx);

    const int64_t rank = operand.meta().ndim;
    const int64_t last_dim = rank-1;
    const bool innermost = (last_dim == dim);
    switch(operand.meta().layout) {
        case SPARSE:
        case STRIDED:
            if ((rank > 3) and (!innermost)) {          // ND-outer
                ss
                << _add_assign(
                    operand.walker(),
                    _mul("coord", _index(operand.stride(), "dim"))
                ) << _end(operand.layout());
            } else {                                    // ND-inner, 1D, 2D, and 3D.
                ss
                << _add_assign(
                    operand.walker(),
                    operand.stepsize(dim)
                ) << _end(operand.layout());
            } 
            break;

        case CONTIGUOUS:
            if ((rank > 3) and (!innermost)) {          // ND-outer
                ss
                << _add_assign(
                    operand.walker(),
                    _mul("coord", _index("weight", "dim"))
                ) << _end(operand.layout());
            } else if (innermost) {                     // ND-inner, 1D, 2D, and 3D.
                ss
                << _inc(operand.walker()) << _end(operand.layout());
            }
            break;
        default:
            ss << "// " << operand.name() << " " << operand.layout() << endl;
            break;
    }
    return ss.str();
}

string Walker::step_fwd(uint32_t dim)
{
    stringstream ss;

    for(kernel_operand_iter oit=kernel_.operands_begin();
        oit != kernel_.operands_end();
        ++oit) {
        ss << step_fwd(dim, oit->first);
    }
    return ss.str();
}

string Walker::ewise_operations(void)
{
    stringstream ss;
    
    for(kernel_tac_iter tit=kernel_.tacs_begin();
        tit!=kernel_.tacs_end();
        ++tit) {
        tac_t& tac = **tit;
        ETYPE etype = kernel_.operand_glb(tac.out).meta().etype;

        string out = "ERROR", in1 = "ERROR", in2 = "ERROR";
        switch(core::tac_noperands(tac)) {
            case 3:
                in2 = kernel_.operand_glb(tac.in2).walker_val();
            case 2:
                in1 = kernel_.operand_glb(tac.in1).walker_val();
            case 1:
                out = kernel_.operand_glb(tac.out).walker_val();
            default:
                break;
        }

        ss << _assign(
            out,
            oper(tac.oper, etype, in1, in2)
        ) << _end(oper_description(tac));
    }
    return ss.str();
}

string Walker::scan_operations(void)
{
    stringstream ss;
    
    for(kernel_tac_iter tit=kernel_.tacs_begin();
        tit!=kernel_.tacs_end();
        ++tit) {
        tac_t& tac = **tit;
        ETYPE etype = kernel_.operand_glb(tac.out).meta().etype;
        string in1 = kernel_.operand_glb(tac.in1).walker_val();

        ss << _assign(
            "accu",
            oper(tac.oper, etype, "accu", in1)
        ) << _end();
        ss << _assign(
            kernel_.operand_glb(tac.out).walker_val(),
            "accu"
        ) << _end();
    }
    return ss.str();
}

string Walker::generate_source(void)
{
    std::map<string, string> subjects;
    string plaid;

    subjects["WALKER_DECLARATION"]  = declare_operands();
    
    if ((kernel_.omask() & EWISE)>0) {  // Element-wise operations

        const uint32_t rank = kernel_.iterspace().meta().ndim;
        subjects["WALKER_STEPSIZE"]     = ewise_declare_stepsizes(rank);
        subjects["WALKER_OFFSET"]       = ewise_assign_offset(rank);
        subjects["OPERATIONS"]          = ewise_operations();
        
        if ((kernel_.iterspace().meta().layout & CONT_COMPATIBLE)>0) {
            subjects["WALKER_STEP_LD"]  = step_fwd_inner();
            subjects["PRAGMA_SIMD"]     = "#pragma omp simd";
            plaid = "ewise.1d";
        } else {
            switch(rank) {
                case 1:     // 1D specialization
                    subjects["WALKER_STEP_LD"]  = step_fwd_inner();
                    plaid = "ewise.1d";
                    break;
                default:    // ND generic
                    subjects["WALKER_STEP_OUTER"]   = step_fwd_outer();
                    subjects["WALKER_STEP_INNER"]   = step_fwd_inner();
                    plaid = "ewise.nd";
                    break;
            }
        }
    } else if ((kernel_.omask() & (REDUCE|SCAN))>0) {   // Reductions

        // Note: start of crappy code...
        tac_t* tac = NULL;
        Operand* out = NULL;
        Operand* in1 = NULL;
        Operand* in2 = NULL;
        for(kernel_tac_iter tit=kernel_.tacs_begin();
            tit != kernel_.tacs_end();
            ++tit) {
            if ((((*tit)->op) & (REDUCE|SCAN))>0) {
                tac = *tit;
            }
        }

        out = &kernel_.operand_glb(tac->out);
        in1 = &kernel_.operand_glb(tac->in1);
        in2 = &kernel_.operand_glb(tac->in2);

        const uint32_t rank = in1->meta().ndim;
        // Note: end of crappy code...

        subjects["NEUTRAL_ELEMENT"] = oper_neutral_element(tac->oper, in1->meta().etype);
        subjects["ATYPE"]           = in2->etype();
        subjects["ETYPE"]           = out->etype();
        subjects["OPD_OUT"]         = out->name();
        subjects["OPD_IN1"]         = in1->name();
        subjects["OPD_IN2"]         = in2->name();

        if ((kernel_.omask() & REDUCE)>0) {
            subjects["REDUCE_OPER"] = _assign(
                "accu",
                oper(tac->oper, in1->meta().etype, "accu", in1->walker_val())
            )+_end();

            switch(tac->oper) {
                case MAXIMUM:
                case MINIMUM:
                    subjects["REDUCE_SYNC"] = "#pragma omp critical";
                    break;
                default:
                    subjects["REDUCE_SYNC"] = "#pragma omp atomic";
                    break;
            }
            subjects["REDUCE_OPER_COMBINE"] = _assign(
                out->walker_val(),
                oper(tac->oper, in1->meta().etype, out->walker_val(), "accu")
            )+_end();
            
            switch(rank) {
                case 1:
                    subjects["WALKER_STEPSIZE"] = ewise_declare_stepsizes(rank);
                    subjects["WALKER_OFFSET"]   = ewise_assign_offset(rank, tac->in1);
                    subjects["WALKER_STEP_LD"]  = step_fwd(0, tac->in1);
                    plaid = "reduce.1d";
                    break;
                case 2:
                    subjects["WALKER_STRIDES"]  = declare_stridesizes();
                    subjects["WALKER_OFFSET"]   = ewise_assign_offset(rank, tac->in1);
                    subjects["WALKER_STEP_LD"]  = step_fwd(0, tac->in1);
                    subjects["WALKER_STEP_SL"]  = step_fwd(1, tac->in1);

                    plaid = "reduce.2d";
                    break;
                case 3:
                    subjects["WALKER_STRIDES"]  = declare_stridesizes();
                    subjects["WALKER_STEP_LD"]  = step_fwd(0, tac->in1);
                    subjects["WALKER_STEP_SL"]  = step_fwd(1, tac->in1);
                    subjects["WALKER_STEP_TL"]  = step_fwd(2, tac->in1);

                    plaid = "reduce.3d";
                    break;
                default:
                    subjects["WALKER_STRIDES"]  = declare_stridesizes();
                    plaid = "reduce.nd";
                    break;
            }
        } else {

            subjects["PAR_OPERATIONS"]  = scan_operations();
            switch(rank) {
                case 1:
                    subjects["WALKER_STEPSIZE"] = ewise_declare_stepsizes(rank);
                    subjects["WALKER_OFFSET"]   = ewise_assign_offset(rank, tac->in1);
                    subjects["WALKER_STEP_LD"]  = step_fwd(0, tac->in1);
                    plaid = "scan.1d";
                    break;
                default:
                    subjects["WALKER_STRIDES"]  = declare_stridesizes();
                    plaid = "scan.nd";

                    break;
            }
        }
    }
    return plaid_.fill(plaid, subjects);
}

}}}}
