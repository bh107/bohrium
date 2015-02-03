#include "codegen.hpp"

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
    ss << "[" << layout_text(operand.layout()) << "]" << endl;
    switch(operand.layout()) {
        case CONTIGUOUS:
        case STRIDED:
        case SPARSE:
            ss << ptr_type(etype_to_ctype_text(operand.etype()));
            ss << operand.first();
            ss << " = ARGS_DPTR(" << id << ");" << endl;

            ss << const_type("const int64_t") << operand.name() << "_start";
            ss << const_type("int64_t") << operand.name() << "_nelem";
            ss << const_type("int64_t") << operand.name() << "_ndim";
            ss << ptr_type("int64_t") << operand.name() << "_shape";
            ss << ptr_type("int64_t") << operand.name() << "_stride";

            ss << assert_not_null(operand.first()) << endl;
            break;

        case SCALAR:
        case SCALAR_CONST:
            ss << ptr_type(etype_to_ctype_text(operand.etype()));
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
