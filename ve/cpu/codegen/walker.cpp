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
    Operand operand(&block_.operand(id), id);    // Grab the operand
    stringstream ss;
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
    return ss.str();
}


string Walker::ewise_cont_step(unsigned int id)
{
    Operand operand(&block_.operand(id), id);    // Grab the operand
    stringstream ss;

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

string Walker::ewise_cont_step(void)
{
    stringstream ss;
    for(size_t oidx=0; oidx<block_.noperands(); ++oidx) {
        ss << ewise_cont_step(oidx);
    }
    return ss.str();
}

string Walker::ewise_cont_offset(uint32_t oidx)
{
    stringstream ss;
    Operand opd(&block_.operand(oidx), oidx);

    switch(opd.operand_->layout) {
        case STRIDED:       
        case SPARSE:
        case CONTIGUOUS:
            ss
            << _add_assign(
                opd.walker(),
                "work_offset"
            ) << _end();
            break;

        case SCALAR:
        case SCALAR_CONST:
        case SCALAR_TEMP:
        default:
            break;
    }

    return ss.str();
}

string Walker::ewise_cont_offset(void)
{
    stringstream ss;
    for(size_t oidx=0; oidx<block_.noperands(); ++oidx) {
        ss << ewise_cont_offset(oidx);
    }
    return ss.str();
}

string Walker::ewise_strided_1d_offset(uint32_t oidx)
{
    stringstream ss;
    Operand opd(&block_.operand(oidx), oidx);

    switch(opd.operand_->layout) {
        case STRIDED:       
        case SPARSE:
        case CONTIGUOUS:
            ss
            << _add_assign(
                opd.walker(),
                _mul(
                    "work_offset",
                    _index(opd.stride(), 0)
                )
            ) << _end();
            break;

        case SCALAR:
        case SCALAR_CONST:
        case SCALAR_TEMP:
        default:
            break;
    }

    return ss.str();
}

string Walker::ewise_strided_1d_offset(void)
{
    stringstream ss;
    for(size_t oidx=0; oidx<block_.noperands(); ++oidx) {
        ss << ewise_strided_1d_offset(oidx);
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
        ss << _assign(out.walker_val(), oper(tac)) << _end();
    }
    return ss.str();
}

string Walker::generate_source(void)
{
    std::map<string, string> subjects;
    subjects["WALKER_DECLARATION"]  = declare_operands();
    subjects["WALKER_OFFSET"]       = ewise_cont_offset();
    subjects["WALKER_STEP"]         = ewise_cont_step();
    subjects["OPERATIONS"]          = ewise_operations();

    return plaid_.fill("ewise.cont.nd", subjects);
}

}}}}
