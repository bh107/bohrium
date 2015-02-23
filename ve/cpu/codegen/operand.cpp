#include <sstream>
#include <string>
#include "utils.hpp"
#include "codegen.hpp"

using namespace std;
using namespace bohrium::core;

namespace bohrium{
namespace engine{
namespace cpu{
namespace codegen{

Operand::Operand(void) : operand_(NULL), local_id_(0) {}
Operand::Operand(operand_t* operand, uint32_t local_id) : operand_(operand), local_id_(local_id) {}

string Operand::name(void)
{
    stringstream ss;
    ss << "opd" << local_id_;
    return ss.str();
}

string Operand::first(void)
{
    stringstream ss;
    ss << name() << "_first";
    return ss.str();
}

string Operand::stepsize(uint32_t dim)
{
    stringstream ss;
    ss << name() << "_stepsize";
    switch(operand_->ndim -1 - dim) {
        case 2:
            ss << "_tld";
            break;
        case 1:
            ss << "_sld";
            break;
        case 0:
            ss << "_ld";
            break;

        default:
            ss << "__ND stepsize is not constant but variable__";
            break;
    }

    return ss.str();
}

string Operand::walker(void)
{
    stringstream ss;
    ss << name();
    return ss.str();
}

string Operand::walker_val(void)
{
    stringstream ss;
    switch(operand_->layout) {
        case SCALAR:
        case SCALAR_CONST:
        case SCALAR_TEMP:
            ss << walker();
            break;

        case CONTIGUOUS:
        case STRIDED:
        case SPARSE:
            ss << _deref(walker());
            break;
        default:    // TOOD: Throw an exception here...
            break;
    }
    return ss.str();
}

string Operand::ndim(void)
{
    stringstream ss;
    ss << name() << "_ndim";
    return ss.str();
}

string Operand::shape(void)
{
    stringstream ss;
    ss << name() << "_shape";
    return ss.str();
}

string Operand::stride(void)
{
    stringstream ss;
    ss << name() << "_stride";
    return ss.str();
}

string Operand::layout(void) {
    return layout_text(operand_->layout);    
}

string Operand::etype(void)
{
    return etype_to_ctype_text(operand_->etype);
}

}}}}
