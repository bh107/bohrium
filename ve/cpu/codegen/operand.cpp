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

Operand::Operand(void) : operand_(NULL), id_(0) {}
Operand::Operand(operand_t* operand, uint32_t id) : operand_(operand), id_(id) {}

string Operand::name(void)
{
    stringstream ss;
    ss << "opd" << id_;
    return ss.str();
}

string Operand::first(void)
{
    stringstream ss;
    ss << name() << "_first";
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
