#include <sstream>
#include <string>
#include "utils.hpp"
#include "codegen.hpp"

using namespace std;
using namespace kp::core;

namespace kp{
namespace engine{
namespace codegen{

Operand::Operand(void) : local_id_(0), operand_(NULL), buffer_(NULL) {}
Operand::Operand(kp_operand * operand, uint32_t local_id, Buffer* buffer) : local_id_(local_id), operand_(operand), buffer_(buffer) {
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

string Operand::data(void)
{
    stringstream ss;
    ss << name() << "_data";
    return ss.str();
}

string Operand::nelem(void)
{
    stringstream ss;
    ss << name() << "_nelem";
    return ss.str();
}

string Operand::start(void)
{
    stringstream ss;
    ss << name() << "_start";
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

string Operand::stride_outer(void)
{
    stringstream ss;
    ss << name() << "_stride_outer";
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

string Operand::accu_shared(void)
{
    stringstream ss;
    ss << name() << "_accu_shared";
    return ss.str();
}

string Operand::accu_private(void)
{
    stringstream ss;
    ss << name() << "_accu_private";
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
        case KP_SCALAR_TEMP:
        case KP_SCALAR_CONST:
        case KP_SCALAR:
        case KP_CONTRACTABLE:
            ss << walker();
            break;

        case KP_CONSECUTIVE:
        case KP_CONTIGUOUS:
        case KP_STRIDED:
            ss << _deref(walker());
            break;

        case KP_SPARSE:
            ss << _beef("Non-implemented KP_LAYOUT.");
            break;
    }
    return ss.str();
}

string Operand::walker_subscript_val(void)
{
    stringstream ss;

    switch(meta().layout) {
        case KP_SCALAR_TEMP:
        case KP_SCALAR_CONST:
        case KP_SCALAR:
        case KP_CONTRACTABLE:
            ss << walker();
            break;

        case KP_CONSECUTIVE:
        case KP_CONTIGUOUS:
        case KP_STRIDED:
            ss << _index(walker(), "eidx");
            break;

        case KP_SPARSE:
            ss << _beef("Non-implemented KP_LAYOUT.");
            break;
    }
    return ss.str();
}

kp_operand & Operand::meta(void)
{
    return *operand_;
}

string Operand::buffer_name(void)
{
    stringstream ss;
    if (buffer_) {
        ss << buffer_->name();
    } else {
        ss << name();
    }
    return ss.str();
}

string Operand::buffer_data(void)
{
    stringstream ss;
    if (buffer_) {
        ss << buffer_->data();
    } else {
        ss << data();
    }
    return ss.str();
}

string Operand::buffer_nelem(void)
{
    stringstream ss;
    if (buffer_) {
        ss << buffer_->nelem();
    } else {
        ss << "1";
    }
    return ss.str();
}

string Operand::buffer_etype(void)
{
    stringstream ss;
    if (buffer_) {
        ss << buffer_->etype();
    } else {
        ss << etype();
    }
    return ss.str();
}

kp_buffer * Operand::buffer_meta(void)
{
    if (buffer_) {
        return &buffer_->meta();
    } else {
        return NULL;
    }
}

uint64_t Operand::local_id(void)
{
    return local_id_;
}

}}}

