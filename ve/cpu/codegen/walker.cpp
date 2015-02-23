#include <sstream>
#include <string>
#include "codegen.hpp"

using namespace std;
using namespace bohrium::core;

namespace bohrium{
namespace engine{
namespace cpu{
namespace codegen{

Walker::Walker(Plaid& plaid, bohrium::core::Block& block) : plaid_(plaid), block_(block) {}

string Walker::declare_operands(void)
{
    stringstream ss;
    for(size_t oidx=0; oidx<block_.noperands(); ++oidx) {
        Operand operand(&block_.operand(oidx), oidx);    // Grab the operand
        switch(operand.operand_->layout) {
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
    for(size_t oidx=0; oidx<block_.noperands(); ++oidx) {
        Operand operand(&block_.operand(oidx), oidx);    // Grab the operand
        switch(operand.operand_->layout) {
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
    Iterspace iterspace;
    for(size_t oidx=0; oidx<block_.noperands(); ++oidx) {
        Operand operand(&block_.operand(oidx), oidx);    // Grab the operand
        switch(operand.operand_->layout) {
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
                    default:
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
    for(size_t oidx=0; oidx<block_.noperands(); ++oidx) {
        Operand operand(&block_.operand(oidx), oidx);    // Grab the operand
        bool innermost = ((operand.operand_->ndim-1) == dim);
        switch(operand.operand_->layout) {
            case SPARSE:
            case STRIDED:
                ss
                << _add_assign(
                    operand.walker(),
                    operand.stepsize(dim)
                ) << _end(operand.layout());
                break;
            case CONTIGUOUS:    // Only step forward in the innermost loop
                if (innermost) {
                    ss << _inc(operand.walker()) << _end(operand.layout());
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
    for(size_t tac_idx=0; tac_idx<block_.narray_tacs(); ++tac_idx) {
        tac_t& tac = block_.array_tac(tac_idx);

        Operand out = Operand(
            &block_.operand(block_.global_to_local(tac.out)),
            block_.global_to_local(tac.out)
        );
        ss << _assign(out.walker_val(), oper(tac)) << _end(oper_description(tac));
    }
    return ss.str();
}

string Walker::generate_source(void)
{
    std::map<string, string> subjects;
    string plaid;
    uint32_t rank = block_.iterspace().ndim;

    subjects["WALKER_DECLARATION"]  = declare_operands();
    subjects["WALKER_STEPSIZE"]     = ewise_declare_stepsizes(rank);
    subjects["WALKER_OFFSET"]       = ewise_assign_offset(rank);
    subjects["OPERATIONS"]          = ewise_operations();

    switch(block_.iterspace().ndim) {
        case 1:
            subjects["WALKER_STEP_LD"]  = ewise_step_fwd(0);
            plaid = "ewise.1d";
            break;
        case 2:
            subjects["WALKER_STEP_LD"]  = ewise_step_fwd(1);
            subjects["WALKER_STEP_SLD"] = ewise_step_fwd(0);
            plaid = "ewise.2d";
            break;
        case 3:
            subjects["WALKER_STEP_LD"]  = ewise_step_fwd(2);
            subjects["WALKER_STEP_SLD"] = ewise_step_fwd(1);
            subjects["WALKER_STEP_TLD"] = ewise_step_fwd(0);
            plaid = "ewise.3d";
            break;
        default:
            subjects["WALKER_STEP_INNER"]   = ewise_step_fwd(rank-1);
            subjects["WALKER_STEP_OUTER"]   = "TODO";
            plaid = "ewise.nd";
            break;
    }
    return plaid_.fill(plaid, subjects);
}

}}}}
