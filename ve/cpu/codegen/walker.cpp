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

string Walker::ewise_cont_step(void)
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
                ss << _inc(operand.walker()) << _end();
                break;

            case SCALAR:
            case SCALAR_CONST:
            case SCALAR_TEMP:
            default:
                break;
        }
    }
    return ss.str();
}

string Walker::ewise_strided_1d_step(void)
{
    stringstream ss;
    for(size_t oidx=0; oidx<block_.noperands(); ++oidx) {
        Operand operand(&block_.operand(oidx), oidx);
        switch(operand.operand_->layout) {
            case STRIDED:       
            case SPARSE:
                break;

            case CONTIGUOUS:
                break;

            case SCALAR:
            case SCALAR_CONST:
            case SCALAR_TEMP:
            default:
                break;
        }
    }
    return ss.str();
}

string Walker::ewise_strided_2d_step(void)
{
    stringstream ss;
    for(size_t oidx=0; oidx<block_.noperands(); ++oidx) {
        Operand operand(&block_.operand(oidx), oidx);
        switch(operand.operand_->layout) {
            case STRIDED:       
            case SPARSE:
                break;

            case CONTIGUOUS:
                break;

            case SCALAR:
            case SCALAR_CONST:
            case SCALAR_TEMP:
            default:
                break;
        }
    }
    return ss.str();
}

string Walker::ewise_strided_3d_step(void)
{
    stringstream ss;
    for(size_t oidx=0; oidx<block_.noperands(); ++oidx) {
        Operand operand(&block_.operand(oidx), oidx);
        switch(operand.operand_->layout) {
            case STRIDED:       
            case SPARSE:
                break;

            case CONTIGUOUS:
                break;

            case SCALAR:
            case SCALAR_CONST:
            case SCALAR_TEMP:
            default:
                break;
        }
    }
    return ss.str();
}

string Walker::ewise_strided_nd_step(void)
{
    stringstream ss;
    for(size_t oidx=0; oidx<block_.noperands(); ++oidx) {
        Operand operand(&block_.operand(oidx), oidx);
        switch(operand.operand_->layout) {
            case STRIDED:       
            case SPARSE:
                break;

            case CONTIGUOUS:
                break;

            case SCALAR:
            case SCALAR_CONST:
            case SCALAR_TEMP:
            default:
                break;
        }
    }
    return ss.str();
}

string Walker::ewise_cont_offset(void)
{
    stringstream ss;
    for(size_t oidx=0; oidx<block_.noperands(); ++oidx) {
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

    }
    return ss.str();
}

/*
string Walker::ewise_strided_step(void)
{
    return "";
}

string Walker::ewise_strided_step(uint32_t dim)
{
    stringstream ss;
    bool innermost = (block_.iterspace().ndim-1 - dim) == 0;
    for(size_t oidx=0; oidx<block_.noperands(); ++oidx) {
        Operand opd(&block_.operand(oidx), oidx);
        switch(opd.operand_->layout) {
            case STRIDED:       
            case SPARSE:
                if (innermost) {
                    ss << _add_assign(
                        operand.walker,
                        _index(operand.stride, dim)
                    ) << _end();
                } else 
                break;
            case CONTIGUOUS:
                if (innermost) {
                    ss << _add_assign(
                        operand.walker,
                        _index(operand.stride, dim)
                    ) << _end();
                }
                break;

            case SCALAR:
            case SCALAR_CONST:
            case SCALAR_TEMP:
            default:
                break;
        }
    }
    return ss.str();
}*/

string Walker::ewise_strided_1d_offset(void)
{
    stringstream ss;
    for(size_t oidx=0; oidx<block_.noperands(); ++oidx) {
        Operand opd(&block_.operand(oidx), oidx);
        switch(opd.operand_->layout) {
            case STRIDED:       
            case SPARSE:
            case CONTIGUOUS:
                break;

            case SCALAR:
            case SCALAR_CONST:
            case SCALAR_TEMP:
            default:
                break;
        }
    }
    return ss.str();
}

string Walker::ewise_strided_2d_offset(void)
{
    stringstream ss;
    for(size_t oidx=0; oidx<block_.noperands(); ++oidx) {
        Operand opd(&block_.operand(oidx), oidx);
        switch(opd.operand_->layout) {
            case STRIDED:       
            case SPARSE:
            case CONTIGUOUS:
                break;

            case SCALAR:
            case SCALAR_CONST:
            case SCALAR_TEMP:
            default:
                break;
        }
    }
    return ss.str();
}

string Walker::ewise_strided_3d_offset(void)
{
    stringstream ss;
    for(size_t oidx=0; oidx<block_.noperands(); ++oidx) {
        Operand opd(&block_.operand(oidx), oidx);
        switch(opd.operand_->layout) {
            case STRIDED:       
            case SPARSE:
            case CONTIGUOUS:
                break;

            case SCALAR:
            case SCALAR_CONST:
            case SCALAR_TEMP:
            default:
                break;
        }
    }
    return ss.str();
}

string Walker::ewise_strided_nd_offset(void)
{
    stringstream ss;
    for(size_t oidx=0; oidx<block_.noperands(); ++oidx) {
        Operand opd(&block_.operand(oidx), oidx);
        switch(opd.operand_->layout) {
            case STRIDED:       
            case SPARSE:
            case CONTIGUOUS:
                break;

            case SCALAR:
            case SCALAR_CONST:
            case SCALAR_TEMP:
            default:
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
        ss << _assign(out.walker_val(), oper(tac)) << _end();
    }
    return ss.str();
}

string Walker::ewise_cont_nd(void)
{
    std::map<string, string> subjects;

    subjects["WALKER_DECLARATION"]  = declare_operands();
    subjects["WALKER_OFFSET"]       = ewise_cont_offset();
    subjects["WALKER_STEP"]         = ewise_cont_step();
    subjects["OPERATIONS"]          = ewise_operations();

    return plaid_.fill("ewise.1d", subjects);
}

string Walker::ewise_strided_1d(void)
{
    std::map<string, string> subjects;

    subjects["WALKER_DECLARATION"]  = declare_operands();
    subjects["WALKER_OFFSET"]       = ewise_strided_1d_offset();
    subjects["WALKER_STEP"]         = ewise_strided_1d_step();
    subjects["OPERATIONS"]          = ewise_operations();

    return plaid_.fill("ewise.1d", subjects);
}

string Walker::ewise_strided_2d(void)
{
    std::map<string, string> subjects;

    subjects["WALKER_DECLARATION"]  = declare_operands();
    subjects["WALKER_OFFSET"]       = ewise_strided_2d_offset();
    subjects["WALKER_STEP"]         = ewise_strided_2d_step();
    subjects["OPERATIONS"]          = ewise_operations();

    return plaid_.fill("ewise.2d", subjects);
}

string Walker::ewise_strided_3d(void)
{
    std::map<string, string> subjects;

    subjects["WALKER_DECLARATION"]  = declare_operands();
    subjects["WALKER_OFFSET"]       = ewise_strided_3d_offset();
    subjects["WALKER_STEP"]         = ewise_strided_3d_step();
    subjects["OPERATIONS"]          = ewise_operations();

    return plaid_.fill("ewise.3d", subjects);
}

string Walker::ewise_strided_nd(void)
{
    std::map<string, string> subjects;

    subjects["WALKER_DECLARATION"]  = declare_operands();
    subjects["WALKER_OFFSET"]       = ewise_strided_nd_offset();
    subjects["WALKER_STEP"]         = ewise_strided_nd_step();
    subjects["OPERATIONS"]          = ewise_operations();

    return plaid_.fill("ewise.nd", subjects);
}

string Walker::generate_source(void)
{
    std::map<string, string> subjects;

    subjects["WALKER_DECLARATION"]  = declare_operands();
    // Switch here on layout
    subjects["WALKER_OFFSET"]       = ewise_cont_offset();
    subjects["WALKER_STEP"]         = ewise_cont_step();
    subjects["OPERATIONS"]          = ewise_operations();

    return plaid_.fill("ewise.1d", subjects);
}

}}}}
