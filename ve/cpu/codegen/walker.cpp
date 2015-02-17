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
        ss << declare_operand(oidx);
    }
    return ss.str();
}

string Walker::declare_operand(uint32_t id)
{
    Operand operand(block_.operand(id), id);    // Grab the operand
    stringstream ss;
    switch(operand.operand_.layout) {
        case STRIDED:       
        case SPARSE:
        case CONTIGUOUS:
            ss
            << _declare(
                _ptr(operand.etype()),
                operand.walker(),
                _deref(operand.first())
            );
            break;

        case SCALAR:
            ss
             << _declare(
                operand.etype(),
                operand.walker(),
                _deref(operand.first())
            );
            break;
        case SCALAR_CONST:
            ss
            << _declare(
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
    return ss.str();
}

string Walker::step_forward(void)
{
    stringstream ss;
    for(size_t oidx=0; oidx<block_.noperands(); ++oidx) {
        ss << step_forward(oidx);
    }
    return ss.str();
}

string Walker::step_forward(unsigned int id)
{
    Operand operand(block_.operand(id), id);    // Grab the operand
    stringstream ss;

    switch(operand.operand_.layout) {
        case STRIDED:       
        case SPARSE:
            ss << _add_assign(
                operand.walker(),
                _index(operand.stride(), 0)
            )
            << _end();
            break;

        case CONTIGUOUS:
            ss << _inc(operand.walker()) << _end();
            break;

        case SCALAR:
        case SCALAR_CONST:
        case SCALAR_TEMP:
        default:
            break;
    }

    return ss.str();

}

string Walker::step_forward_last(void)
{
    stringstream ss;

    return ss.str();
}

string Walker::step_forward_2nd_last(void)
{
    stringstream ss;
    /*
    Operand operand(block_.operand(id), id);    // Grab the operand
    ss << _add_assign(operand.walker(), index(operand.stride, 0)) << _endl();*/

    return ss.str();
}

string Walker::step_forward_3rd_last(void)
{
    stringstream ss;
    /*
    Operand operand(block_.operand(id), id);    // Grab the operand
    ss << _add_assign(operand.walker(), index(operand.stride, 0)) << _endl();*/

    return ss.str();
}


string Walker::ewise_operations(void)
{
    stringstream ss;
    for(size_t tac_idx=0; tac_idx<block_.narray_tacs(); ++tac_idx) {
        tac_t& tac = block_.array_tac(tac_idx);
        Operand out = Operand(
            block_.operand(block_.global_to_local(tac.out)),
            block_.global_to_local(tac.out)
        );

        ss << _assign(out.walker_val(), oper(tac)) << _end();
    }
    return ss.str();
}

string Walker::generate_source(void)
{
    std::map<string, string> subjects;
    subjects["WALKER_DECLARATION"] = declare_operands();
    subjects["WALKER_OFFSET"] = "//TODO: Offset...";
    subjects["WALKER_STEP"] = step_forward();
    subjects["OPERATIONS"] = ewise_operations();

    return plaid_.fill("ewise.cont.nd", subjects);
}

}}}}
