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

Operand::Operand(operand_t operand, uint32_t id) : operand_(operand), id_(id) {}

string Operand::name(void)
{
    stringstream ss;
    ss << "a" << id_;
    return ss.str();
}

string Operand::first(void)
{
    stringstream ss;
    ss << name() << "_first";
    return ss.str();
}

string Operand::current(void)
{
    stringstream ss;
    ss << name() << "_current";
    return ss.str();
}

string Operand::nelem(void)
{
    stringstream ss;
    ss << name() << "_nelem";
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
    // operand_.layout
    return "";
}

string Operand::etype(void)
{
    //operand_.etype
    return "";
}

}}}}
