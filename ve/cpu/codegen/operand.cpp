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
Operand::Operand(operand_t* operand, uint32_t local_id) : operand_(operand), local_id_(local_id) {
    if (NULL == operand_) {
        throw runtime_error("Constructing a NULL operand_, when expecting to have one");
    }
}

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

string Operand::layout(void) {
    return layout_text(meta().layout);    
}

string Operand::etype(void)
{
    return etype_to_ctype_text(meta().etype);
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

string Operand::strides(void)
{
    stringstream ss;
    ss << name() << "_strides";
    return ss.str();
}

string Operand::stride_inner(void)
{
    stringstream ss;
    ss << name() << "_stride_inner";
    return ss.str();
}

string Operand::stride_axis(void)
{
    stringstream ss;
    ss << name() << "_stride_axis";
    return ss.str();
}

string Operand::accu(void)
{
    stringstream ss;
    ss << name() << "_accu";
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

    switch(meta().layout) {
        case SCALAR_CONST:
        case SCALAR:
        case CONTRACTABLE:
            ss << walker();
            break;

        case CONSECUTIVE:
        case CONTIGUOUS:
        case STRIDED:
            ss << _deref(walker());
            break;

        case SPARSE:
            ss << _beef("Non-implemented LAYOUT.");
            break;
    }
    return ss.str();
}

operand_t& Operand::meta(void)
{
    return *operand_;
}

uint64_t Operand::local_id(void)
{
    return local_id_;
}

}}}}
