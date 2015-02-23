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
        switch(operand.meta().layout) {
            case STRIDED:       
            case SPARSE:
            case CONTIGUOUS:
                ss
                << _declare_init(
                    _ptr(operand.etype()),
                    operand.walker(),
                    operand.first()
                );
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

string Walker::ewise_assign_offset(uint32_t rank)
{
    stringstream ss;

    for(kernel_operand_iter oit=kernel_.operands_begin();
        oit != kernel_.operands_end();
        ++oit) {
        Operand& operand = oit->second;

        switch(operand.meta().layout) {
            case STRIDED:       
            case SPARSE:
                switch(rank) {
                    case 3:
                    case 2:
                    case 1:
                        ss << _add_assign(
                            operand.walker(),
                            _mul("work_offset", _index(operand.stride(), rank-1))
                        )
                        << _end();
                        break;
                    default:
                        // TODO: implement ND-case
                        break;
                }
                break;
            case CONTIGUOUS:
                ss << _add_assign(
                    operand.walker(),
                    "work_offset"
                ) << _end();
                break;
            default:
                break;
        }
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

string Walker::ewise_step_fwd(uint32_t dim)
{
    stringstream ss;

    for(kernel_operand_iter oit=kernel_.operands_begin();
        oit != kernel_.operands_end();
        ++oit) {
        Operand& operand = oit->second;

        const int64_t rank = operand.meta().ndim;
        const int64_t last_dim = rank-1;
        const bool innermost = (last_dim == dim);
        switch(operand.meta().layout) {
            case SPARSE:
            case STRIDED:
                if ((rank > 3) and innermost) {             // ND-inner
                    ss
                    << _add_assign(
                        operand.walker(),
                        operand.stepsize(dim)
                    ) << _end(operand.layout());
                } else if ((rank > 3) and (!innermost)) {   // ND-outer
                    ss
                    << _add_assign(
                        operand.walker(),
                        _mul("coord", _index(operand.stride(), "dim"))
                    ) << _end(operand.layout());
                } else {                                    // 1D, 2D, and 3D.
                    ss
                    << _add_assign(
                        operand.walker(),
                        operand.stepsize(dim)
                    ) << _end(operand.layout());
                }
                break;
            case CONTIGUOUS:    
                if (innermost) {    // Only step forward in the innermost loop
                    ss << _inc(operand.walker()) << _end(operand.layout());
                } else {
                    ss << "// " << operand.name() << " " << operand.layout() << endl;
                }
                break;
            default:
                ss << "// " << operand.name() << " " << operand.layout() << endl;
                break;
        }
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
        cout << tac_text(tac) << endl; 
        for(kernel_operand_iter oit=kernel_.operands_begin();
            oit!= kernel_.operands_end();
            ++oit) {
            cout << "global=" << oit->first << ", local=" << (oit->second.local_id()) << endl;
        }
        Operand& out = kernel_.operand_glb(tac.out);
        ss << _assign(
            out.walker_val(),
            oper(tac)
        ) << _end(oper_description(tac));
    }
    return ss.str();
}

string Walker::generate_source(void)
{
    std::map<string, string> subjects;
    string plaid;
    const uint32_t rank = kernel_.iterspace().meta().ndim;

    subjects["WALKER_DECLARATION"]  = declare_operands();
    subjects["WALKER_STEPSIZE"]     = ewise_declare_stepsizes(rank);
    subjects["WALKER_OFFSET"]       = ewise_assign_offset(rank);
    subjects["OPERATIONS"]          = ewise_operations();

    switch(rank) {
        case 1:     // 1D specialization
            subjects["WALKER_STEP_LD"]  = ewise_step_fwd(0);
            plaid = "ewise.1d";
            break;
        case 2:     // 2D specialization
            subjects["WALKER_STEP_LD"]  = ewise_step_fwd(1);
            subjects["WALKER_STEP_SLD"] = ewise_step_fwd(0);
            plaid = "ewise.2d";
            break;
        case 3:     // 3D specialization
            subjects["WALKER_STEP_LD"]  = ewise_step_fwd(2);
            subjects["WALKER_STEP_SLD"] = ewise_step_fwd(1);
            subjects["WALKER_STEP_TLD"] = ewise_step_fwd(0);
            plaid = "ewise.3d";
            break;
        default:    // ND
            subjects["WALKER_STEP_OUTER"]   = ewise_step_fwd(0);
            subjects["WALKER_STEP_INNER"]   = ewise_step_fwd(rank-1);
            plaid = "ewise.nd";
            break;
    }
    return plaid_.fill(plaid, subjects);
}

}}}}
