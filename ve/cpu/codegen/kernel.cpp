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

Kernel::Kernel() {}

void Kernel::add_operand(Operand& operand, uint32_t id)
{
    operands_[id] = &operand;
}

string Kernel::unpack_operand(uint32_t id)
{
    Operand& operand = *operands_[id];
    stringstream ss;
    ss << "// Argument " << id;
    ss << "[" << operand.layout() << "]" << endl;
    switch(operand.operand_.layout) {
        case STRIDED:
        case SPARSE:
            ss << _const("int64_t") << operand.start();
            ss << _const("int64_t") << operand.nelem();
            ss << _const("int64_t") << operand.ndim();
            ss << _ptr("int64_t") << operand.shape();
            ss << _ptr("int64_t") << operand.stride();

        case CONTIGUOUS:    // We only use the data-pointer
            ss << _declare(_ptr(operand.etype()), operand_first);
            ss << " = ARGS_DPTR(" << id << ");" << endl;

            ss << assert_not_null(operand.first()) << endl;
            break;

        case SCALAR:
        case SCALAR_CONST:
            ss << ptr_type(operand.etype()) << " ";
            ss << operand.first();
            ss << " = ARGS_DPTR(" << id << ");" << endl;
            ss << assert_not_null(operand.first()) << endl;
            break;
        case SCALAR_TEMP:   // Data pointer is never used.
        default:
            break;
    }
    return ss.str();
}

}}}}
