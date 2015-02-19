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

string Walker::ewise_1d_assign_offset(void)
{
    stringstream ss;
    for(size_t oidx=0; oidx<block_.noperands(); ++oidx) {
        Operand operand(&block_.operand(oidx), oidx);    // Grab the operand
        switch(operand.operand_->layout) {
            case STRIDED:       
            case SPARSE:
                ss << _add_assign(
                    operand.walker(),
                    _index(operand.stride(), 0)
                )
                << _end();
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

string Walker::ewise_2d_assign_offset(void)
{
    return ewise_1d_assign_offset();
}

string Walker::ewise_3d_assign_offset(void)
{
    return ewise_1d_assign_offset();
}

string Walker::ewise_nd_assign_offset(void)
{
    return ewise_1d_assign_offset();
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
        ss << _assign(out.walker_val(), oper(tac)) << _end();
    }
    return ss.str();
}

string Walker::generate_source(void)
{
    std::map<string, string> subjects;
    string plaid;

    subjects["WALKER_DECLARATION"]  = declare_operands();
    subjects["OPERATIONS"]          = ewise_operations();
    switch(block_.iterspace().ndim) {
        case 1:
            subjects["WALKER_OFFSET"]   = ewise_1d_assign_offset();
            subjects["WALKER_STEP_LD"]  = ewise_1d_step_fwd();
            plaid = "ewise.1d";
            break;
        case 2:
            subjects["WALKER_OFFSET"]   = ewise_2d_assign_offset();
            subjects["WALKER_STEP_LD"]  = ewise_2d_step_fwd(1);
            subjects["WALKER_STEP_SLD"] = ewise_2d_step_fwd(0);
            plaid = "ewise.2d";
            break;
        case 3:
            subjects["WALKER_OFFSET"]   = ewise_3d_assign_offset();
            subjects["WALKER_STEP_LD"]  = ewise_2d_step_fwd(2);
            subjects["WALKER_STEP_SLD"] = ewise_2d_step_fwd(1);
            subjects["WALKER_STEP_TLD"] = ewise_2d_step_fwd(0);

            plaid = "ewise.3d";
            break;
        default:
            subjects["WALKER_OFFSET"]       = ewise_nd_assign_offset();
            subjects["WALKER_STEP_INNER"]   = ewise_nd_step_fwd(0);
            subjects["WALKER_STEP_OUTER"]   = ewise_nd_step_fwd(1);
            plaid = "ewise.nd";
            break;
    }
    return plaid_.fill(plaid, subjects);
}

}}}}
