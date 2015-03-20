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
            case STRIDED:       
            case SPARSE:
            case CONTIGUOUS:
            case CONSECUTIVE:
            case SCALAR:
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

            case SCALAR_CONST:
                ss
                << _declare_init(
                    _const(operand.etype()),
                    operand.walker(),
                    _deref(operand.first())
                );

                break;
            case CONTRACTABLE:
                ss
                << _declare(
                    operand.etype(),
                    operand.walker()
                );
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
        case SPARSE:
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

        case SCALAR:
        case SCALAR_CONST:
        case CONTRACTABLE:
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

string Walker::declare_stride_inner(uint64_t oidx)
{
    stringstream ss;
    Operand& operand = kernel_.operand_glb(oidx);
    switch(operand.meta().layout) {
        case SPARSE:
        case STRIDED:
        case CONSECUTIVE:
            ss << _declare_init(
                _const(_int64()),
                operand.stride_inner(),
                _index(operand.strides(), "inner_dim")
            )
            << _end(operand.layout());
            break;

        case CONTIGUOUS:
        case CONTRACTABLE:
        case SCALAR_CONST:
        case SCALAR:
            ss << "// " << operand.name() << " " << operand.layout() << endl;
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

        case SCALAR:        // No stepping for these
        case CONTRACTABLE:
        case SCALAR_CONST:
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

string Walker::step_fwd_inner(uint64_t glb_idx)
{
    stringstream ss;

    Operand& operand = kernel_.operand_glb(glb_idx);
    switch(operand.meta().layout) {
        case SPARSE:
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

        case SCALAR:        
        case SCALAR_CONST:
        case CONTRACTABLE:
            ss << "// " << operand.name() << " " << operand.layout() << endl;
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

string Walker::operations(void)
{
    stringstream ss;
    
    for(kernel_tac_iter tit=kernel_.tacs_begin();
        tit!=kernel_.tacs_end();
        ++tit) {
        tac_t& tac = **tit;
        ETYPE etype = kernel_.operand_glb(tac.out).meta().etype;

        string out = "ERROR", in1 = "ERROR", in2 = "ERROR";
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

            case REDUCE:
                inner_opds_.insert(tac.in1);

                outer_opds_.insert(tac.in1);
                outer_opds_.insert(tac.out);

                ss << _assign(
                    "accu",
                    oper(
                        tac.oper,
                        kernel_.operand_glb(tac.in1).meta().etype,
                        "accu",
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
                    "accu",
                    oper(tac.oper, etype, "accu", in1)
                ) << _end();
                ss << _assign(
                    kernel_.operand_glb(tac.out).walker_val(),
                    "accu"
                ) << _end();
                break;

            default:
                ss << "UNSUPPORTED_OPERATION["<< operation_text(tac.op) <<"]_AT_EMITTER_STAGE";
                break;
        }
    }
    return ss.str();
}

string Walker::generate_source(void)
{
    std::map<string, string> subjects;
    string plaid;

    const uint32_t rank = kernel_.iterspace().meta().ndim;
    subjects["WALKER_DECLARATION"]  = declare_operands();

    // Note: start of crappy code used by reductions / scan.
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
    if (tac) {
        out = &kernel_.operand_glb(tac->out);
        in1 = &kernel_.operand_glb(tac->in1);
        in2 = &kernel_.operand_glb(tac->in2);

        subjects["OPD_OUT"] = out->name();
        subjects["OPD_IN1"] = in1->name();
        subjects["OPD_IN2"] = in2->name();

        subjects["NEUTRAL_ELEMENT"] = oper_neutral_element(tac->oper, in1->meta().etype);
        subjects["ETYPE"] = out->etype();
        subjects["ATYPE"] = in2->etype();
    }
    // Note: end of crappy code used by reductions / scan

    subjects["OPERATIONS"] = operations();

    // MAP | ZIP | GENERATE on COLLAPSIBLE LAYOUT of any RANK
    // and
    // REDUCE on COLLAPSIBLE LAYOUT with RANK == 1
    if (((kernel_.iterspace().meta().layout & COLLAPSIBLE)>0) \
        and ((kernel_.omask() & SCAN)==0)                     \
        and (not((rank>1) and ((kernel_.omask() & REDUCE)>0)))) {
        
        plaid = "walker.collapsed";

        subjects["WALKER_INNER_DIM"]    = _declare_init(
            _const(_int64()),
            "inner_dim",
            _sub(kernel_.iterspace().ndim(), "1")
        ) + _end();
        subjects["WALKER_STRIDE_INNER"] = declare_stride_inner();
        subjects["WALKER_STEP_INNER"]   = step_fwd_inner();
        subjects["WALKER_OFFSET"]       = ewise_assign_offset(rank);

        // Reduction specfics
        if ((kernel_.omask() & REDUCE)>0) {
            // Initialize the accumulator 
            subjects["ACCU_OPD_INIT"] = _line(_assign(
                _deref(out->first()),
                oper_neutral_element(tac->oper, in1->meta().etype)
            ));
            // Declare local accumulator var
            subjects["ACCU_LOCAL_DECLARE"] = _line(_declare_init(
                in1->etype(),
                "accu",
                oper_neutral_element(tac->oper, in1->meta().etype)
            ));
            // Syncronize accumulator and local accumulator var
            subjects["ACCU_OPD_SYNC"] = _line(synced_oper(
                tac->oper,
                in1->meta().etype,
                _deref(out->first()),
                _deref(out->first()),
                "accu"
            ));
        }

    // MAP | ZIP | REDUCE on STRIDED LAYOUT and RANK > 1
    } else if ((kernel_.omask() & SCAN) == 0) {

        // MAP | ZIP
        if ((kernel_.omask() & REDUCE)==0) {
            plaid = "walker.inner";

            subjects["WALKER_INNER_DIM"]    = _declare_init(
                _const(_int64()),
                "inner_dim",
                _sub(kernel_.iterspace().ndim(), "1")
            ) + _end();
            subjects["WALKER_STRIDE_INNER"] = declare_stride_inner();
            subjects["WALKER_OFFSET"]       = ewise_assign_offset(rank);

            subjects["WALKER_STEP_OUTER"] = step_fwd_outer();
            subjects["WALKER_STEP_INNER"] = step_fwd_inner();

        // REDUCE
        } else {
            plaid = "walker.axis";
        }

    // SCAN on STRIDED LAYOUT of any RANK
    } else {
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
    }

    return plaid_.fill(plaid, subjects);
}

}}}}
