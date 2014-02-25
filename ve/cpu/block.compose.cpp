#include "block.hpp"

using namespace std;
namespace bohrium{
namespace engine{
namespace cpu{

/**
 *  Compose a block based on the instruction-nodes within a dag.
 */
bool Block::compose()
{
    for (int i=0; i< this->dag.nnode; ++i) {
        this->instr[i] = &this->ir.instr_list[dag.node_map[i]];
        bh_instruction& instr = *this->instr[i];

        uint32_t out=0, in1=0, in2=0;

        //
        // Program packing: output argument
        // NOTE: All but BH_NONE has an output which is an array
        if (instr.opcode != BH_NONE) {
            out = this->add_operand(instr, 0);
        }

        //
        // Program packing; operator, operand and input argument(s).
        switch (instr.opcode) {    // [OPCODE_SWITCH]

            case BH_ABSOLUTE:
                in1 = this->add_operand(instr, 1);

                this->program[i].op    = MAP;  // TAC
                this->program[i].oper  = ABSOLUTE;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= MAP;    // Operationmask
                break;
            case BH_ARCCOS:
                in1 = this->add_operand(instr, 1);

                this->program[i].op    = MAP;  // TAC
                this->program[i].oper  = ARCCOS;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= MAP;    // Operationmask
                break;
            case BH_ARCCOSH:
                in1 = this->add_operand(instr, 1);

                this->program[i].op    = MAP;  // TAC
                this->program[i].oper  = ARCCOSH;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= MAP;    // Operationmask
                break;
            case BH_ARCSIN:
                in1 = this->add_operand(instr, 1);

                this->program[i].op    = MAP;  // TAC
                this->program[i].oper  = ARCSIN;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= MAP;    // Operationmask
                break;
            case BH_ARCSINH:
                in1 = this->add_operand(instr, 1);

                this->program[i].op    = MAP;  // TAC
                this->program[i].oper  = ARCSINH;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= MAP;    // Operationmask
                break;
            case BH_ARCTAN:
                in1 = this->add_operand(instr, 1);

                this->program[i].op    = MAP;  // TAC
                this->program[i].oper  = ARCTAN;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= MAP;    // Operationmask
                break;
            case BH_ARCTANH:
                in1 = this->add_operand(instr, 1);

                this->program[i].op    = MAP;  // TAC
                this->program[i].oper  = ARCTANH;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= MAP;    // Operationmask
                break;
            case BH_CEIL:
                in1 = this->add_operand(instr, 1);

                this->program[i].op    = MAP;  // TAC
                this->program[i].oper  = CEIL;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= MAP;    // Operationmask
                break;
            case BH_COS:
                in1 = this->add_operand(instr, 1);

                this->program[i].op    = MAP;  // TAC
                this->program[i].oper  = COS;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= MAP;    // Operationmask
                break;
            case BH_COSH:
                in1 = this->add_operand(instr, 1);

                this->program[i].op    = MAP;  // TAC
                this->program[i].oper  = COSH;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= MAP;    // Operationmask
                break;
            case BH_EXP:
                in1 = this->add_operand(instr, 1);

                this->program[i].op    = MAP;  // TAC
                this->program[i].oper  = EXP;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= MAP;    // Operationmask
                break;
            case BH_EXP2:
                in1 = this->add_operand(instr, 1);

                this->program[i].op    = MAP;  // TAC
                this->program[i].oper  = EXP2;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= MAP;    // Operationmask
                break;
            case BH_EXPM1:
                in1 = this->add_operand(instr, 1);

                this->program[i].op    = MAP;  // TAC
                this->program[i].oper  = EXPM1;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= MAP;    // Operationmask
                break;
            case BH_FLOOR:
                in1 = this->add_operand(instr, 1);

                this->program[i].op    = MAP;  // TAC
                this->program[i].oper  = FLOOR;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= MAP;    // Operationmask
                break;
            case BH_IDENTITY:
                in1 = this->add_operand(instr, 1);

                this->program[i].op    = MAP;  // TAC
                this->program[i].oper  = IDENTITY;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= MAP;    // Operationmask
                break;
            case BH_IMAG:
                in1 = this->add_operand(instr, 1);

                this->program[i].op    = MAP;  // TAC
                this->program[i].oper  = IMAG;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= MAP;    // Operationmask
                break;
            case BH_INVERT:
                in1 = this->add_operand(instr, 1);

                this->program[i].op    = MAP;  // TAC
                this->program[i].oper  = INVERT;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= MAP;    // Operationmask
                break;
            case BH_ISINF:
                in1 = this->add_operand(instr, 1);

                this->program[i].op    = MAP;  // TAC
                this->program[i].oper  = ISINF;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= MAP;    // Operationmask
                break;
            case BH_ISNAN:
                in1 = this->add_operand(instr, 1);

                this->program[i].op    = MAP;  // TAC
                this->program[i].oper  = ISNAN;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= MAP;    // Operationmask
                break;
            case BH_LOG:
                in1 = this->add_operand(instr, 1);

                this->program[i].op    = MAP;  // TAC
                this->program[i].oper  = LOG;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= MAP;    // Operationmask
                break;
            case BH_LOG10:
                in1 = this->add_operand(instr, 1);

                this->program[i].op    = MAP;  // TAC
                this->program[i].oper  = LOG10;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= MAP;    // Operationmask
                break;
            case BH_LOG1P:
                in1 = this->add_operand(instr, 1);

                this->program[i].op    = MAP;  // TAC
                this->program[i].oper  = LOG1P;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= MAP;    // Operationmask
                break;
            case BH_LOG2:
                in1 = this->add_operand(instr, 1);

                this->program[i].op    = MAP;  // TAC
                this->program[i].oper  = LOG2;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= MAP;    // Operationmask
                break;
            case BH_LOGICAL_NOT:
                in1 = this->add_operand(instr, 1);

                this->program[i].op    = MAP;  // TAC
                this->program[i].oper  = LOGICAL_NOT;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= MAP;    // Operationmask
                break;
            case BH_REAL:
                in1 = this->add_operand(instr, 1);

                this->program[i].op    = MAP;  // TAC
                this->program[i].oper  = REAL;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= MAP;    // Operationmask
                break;
            case BH_RINT:
                in1 = this->add_operand(instr, 1);

                this->program[i].op    = MAP;  // TAC
                this->program[i].oper  = RINT;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= MAP;    // Operationmask
                break;
            case BH_SIN:
                in1 = this->add_operand(instr, 1);

                this->program[i].op    = MAP;  // TAC
                this->program[i].oper  = SIN;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= MAP;    // Operationmask
                break;
            case BH_SINH:
                in1 = this->add_operand(instr, 1);

                this->program[i].op    = MAP;  // TAC
                this->program[i].oper  = SINH;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= MAP;    // Operationmask
                break;
            case BH_SQRT:
                in1 = this->add_operand(instr, 1);

                this->program[i].op    = MAP;  // TAC
                this->program[i].oper  = SQRT;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= MAP;    // Operationmask
                break;
            case BH_TAN:
                in1 = this->add_operand(instr, 1);

                this->program[i].op    = MAP;  // TAC
                this->program[i].oper  = TAN;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= MAP;    // Operationmask
                break;
            case BH_TANH:
                in1 = this->add_operand(instr, 1);

                this->program[i].op    = MAP;  // TAC
                this->program[i].oper  = TANH;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= MAP;    // Operationmask
                break;
            case BH_TRUNC:
                in1 = this->add_operand(instr, 1);

                this->program[i].op    = MAP;  // TAC
                this->program[i].oper  = TRUNC;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= MAP;    // Operationmask
                break;
            case BH_ADD:
                in1 = this->add_operand(instr, 1);
                in2 = this->add_operand(instr, 2);

                this->program[i].op    = ZIP;  // TAC
                this->program[i].oper  = ADD;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= ZIP;    // Operationmask
                break;
            case BH_ARCTAN2:
                in1 = this->add_operand(instr, 1);
                in2 = this->add_operand(instr, 2);

                this->program[i].op    = ZIP;  // TAC
                this->program[i].oper  = ARCTAN2;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= ZIP;    // Operationmask
                break;
            case BH_BITWISE_AND:
                in1 = this->add_operand(instr, 1);
                in2 = this->add_operand(instr, 2);

                this->program[i].op    = ZIP;  // TAC
                this->program[i].oper  = BITWISE_AND;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= ZIP;    // Operationmask
                break;
            case BH_BITWISE_OR:
                in1 = this->add_operand(instr, 1);
                in2 = this->add_operand(instr, 2);

                this->program[i].op    = ZIP;  // TAC
                this->program[i].oper  = BITWISE_OR;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= ZIP;    // Operationmask
                break;
            case BH_BITWISE_XOR:
                in1 = this->add_operand(instr, 1);
                in2 = this->add_operand(instr, 2);

                this->program[i].op    = ZIP;  // TAC
                this->program[i].oper  = BITWISE_XOR;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= ZIP;    // Operationmask
                break;
            case BH_DIVIDE:
                in1 = this->add_operand(instr, 1);
                in2 = this->add_operand(instr, 2);

                this->program[i].op    = ZIP;  // TAC
                this->program[i].oper  = DIVIDE;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= ZIP;    // Operationmask
                break;
            case BH_EQUAL:
                in1 = this->add_operand(instr, 1);
                in2 = this->add_operand(instr, 2);

                this->program[i].op    = ZIP;  // TAC
                this->program[i].oper  = EQUAL;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= ZIP;    // Operationmask
                break;
            case BH_GREATER:
                in1 = this->add_operand(instr, 1);
                in2 = this->add_operand(instr, 2);

                this->program[i].op    = ZIP;  // TAC
                this->program[i].oper  = GREATER;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= ZIP;    // Operationmask
                break;
            case BH_GREATER_EQUAL:
                in1 = this->add_operand(instr, 1);
                in2 = this->add_operand(instr, 2);

                this->program[i].op    = ZIP;  // TAC
                this->program[i].oper  = GREATER_EQUAL;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= ZIP;    // Operationmask
                break;
            case BH_LEFT_SHIFT:
                in1 = this->add_operand(instr, 1);
                in2 = this->add_operand(instr, 2);

                this->program[i].op    = ZIP;  // TAC
                this->program[i].oper  = LEFT_SHIFT;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= ZIP;    // Operationmask
                break;
            case BH_LESS:
                in1 = this->add_operand(instr, 1);
                in2 = this->add_operand(instr, 2);

                this->program[i].op    = ZIP;  // TAC
                this->program[i].oper  = LESS;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= ZIP;    // Operationmask
                break;
            case BH_LESS_EQUAL:
                in1 = this->add_operand(instr, 1);
                in2 = this->add_operand(instr, 2);

                this->program[i].op    = ZIP;  // TAC
                this->program[i].oper  = LESS_EQUAL;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= ZIP;    // Operationmask
                break;
            case BH_LOGICAL_AND:
                in1 = this->add_operand(instr, 1);
                in2 = this->add_operand(instr, 2);

                this->program[i].op    = ZIP;  // TAC
                this->program[i].oper  = LOGICAL_AND;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= ZIP;    // Operationmask
                break;
            case BH_LOGICAL_OR:
                in1 = this->add_operand(instr, 1);
                in2 = this->add_operand(instr, 2);

                this->program[i].op    = ZIP;  // TAC
                this->program[i].oper  = LOGICAL_OR;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= ZIP;    // Operationmask
                break;
            case BH_LOGICAL_XOR:
                in1 = this->add_operand(instr, 1);
                in2 = this->add_operand(instr, 2);

                this->program[i].op    = ZIP;  // TAC
                this->program[i].oper  = LOGICAL_XOR;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= ZIP;    // Operationmask
                break;
            case BH_MAXIMUM:
                in1 = this->add_operand(instr, 1);
                in2 = this->add_operand(instr, 2);

                this->program[i].op    = ZIP;  // TAC
                this->program[i].oper  = MAXIMUM;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= ZIP;    // Operationmask
                break;
            case BH_MINIMUM:
                in1 = this->add_operand(instr, 1);
                in2 = this->add_operand(instr, 2);

                this->program[i].op    = ZIP;  // TAC
                this->program[i].oper  = MINIMUM;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= ZIP;    // Operationmask
                break;
            case BH_MOD:
                in1 = this->add_operand(instr, 1);
                in2 = this->add_operand(instr, 2);

                this->program[i].op    = ZIP;  // TAC
                this->program[i].oper  = MOD;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= ZIP;    // Operationmask
                break;
            case BH_MULTIPLY:
                in1 = this->add_operand(instr, 1);
                in2 = this->add_operand(instr, 2);

                this->program[i].op    = ZIP;  // TAC
                this->program[i].oper  = MULTIPLY;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= ZIP;    // Operationmask
                break;
            case BH_NOT_EQUAL:
                in1 = this->add_operand(instr, 1);
                in2 = this->add_operand(instr, 2);

                this->program[i].op    = ZIP;  // TAC
                this->program[i].oper  = NOT_EQUAL;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= ZIP;    // Operationmask
                break;
            case BH_POWER:
                in1 = this->add_operand(instr, 1);
                in2 = this->add_operand(instr, 2);

                this->program[i].op    = ZIP;  // TAC
                this->program[i].oper  = POWER;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= ZIP;    // Operationmask
                break;
            case BH_RIGHT_SHIFT:
                in1 = this->add_operand(instr, 1);
                in2 = this->add_operand(instr, 2);

                this->program[i].op    = ZIP;  // TAC
                this->program[i].oper  = RIGHT_SHIFT;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= ZIP;    // Operationmask
                break;
            case BH_SUBTRACT:
                in1 = this->add_operand(instr, 1);
                in2 = this->add_operand(instr, 2);

                this->program[i].op    = ZIP;  // TAC
                this->program[i].oper  = SUBTRACT;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= ZIP;    // Operationmask
                break;
            case BH_ADD_REDUCE:
                in1 = this->add_operand(instr, 1);

                in2 = ++(this->noperands);
                this->scope[in2].layout    = CONSTANT;
                this->scope[in2].data      = &(instr.constant.value.r123.key);
                this->scope[in2].type      = UINT64;
                this->scope[in2].nelem     = 1;

                this->program[i].op    = REDUCE;  // TAC
                this->program[i].oper  = ADD;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= REDUCE;    // Operationmask
                break;
            case BH_BITWISE_AND_REDUCE:
                in1 = this->add_operand(instr, 1);

                in2 = ++(this->noperands);
                this->scope[in2].layout    = CONSTANT;
                this->scope[in2].data      = &(instr.constant.value.r123.key);
                this->scope[in2].type      = UINT64;
                this->scope[in2].nelem     = 1;

                this->program[i].op    = REDUCE;  // TAC
                this->program[i].oper  = BITWISE_AND;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= REDUCE;    // Operationmask
                break;
            case BH_BITWISE_OR_REDUCE:
                in1 = this->add_operand(instr, 1);

                in2 = ++(this->noperands);
                this->scope[in2].layout    = CONSTANT;
                this->scope[in2].data      = &(instr.constant.value.r123.key);
                this->scope[in2].type      = UINT64;
                this->scope[in2].nelem     = 1;

                this->program[i].op    = REDUCE;  // TAC
                this->program[i].oper  = BITWISE_OR;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= REDUCE;    // Operationmask
                break;
            case BH_BITWISE_XOR_REDUCE:
                in1 = this->add_operand(instr, 1);

                in2 = ++(this->noperands);
                this->scope[in2].layout    = CONSTANT;
                this->scope[in2].data      = &(instr.constant.value.r123.key);
                this->scope[in2].type      = UINT64;
                this->scope[in2].nelem     = 1;

                this->program[i].op    = REDUCE;  // TAC
                this->program[i].oper  = BITWISE_XOR;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= REDUCE;    // Operationmask
                break;
            case BH_LOGICAL_AND_REDUCE:
                in1 = this->add_operand(instr, 1);

                in2 = ++(this->noperands);
                this->scope[in2].layout    = CONSTANT;
                this->scope[in2].data      = &(instr.constant.value.r123.key);
                this->scope[in2].type      = UINT64;
                this->scope[in2].nelem     = 1;

                this->program[i].op    = REDUCE;  // TAC
                this->program[i].oper  = LOGICAL_AND;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= REDUCE;    // Operationmask
                break;
            case BH_LOGICAL_OR_REDUCE:
                in1 = this->add_operand(instr, 1);

                in2 = ++(this->noperands);
                this->scope[in2].layout    = CONSTANT;
                this->scope[in2].data      = &(instr.constant.value.r123.key);
                this->scope[in2].type      = UINT64;
                this->scope[in2].nelem     = 1;

                this->program[i].op    = REDUCE;  // TAC
                this->program[i].oper  = LOGICAL_OR;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= REDUCE;    // Operationmask
                break;
            case BH_LOGICAL_XOR_REDUCE:
                in1 = this->add_operand(instr, 1);

                in2 = ++(this->noperands);
                this->scope[in2].layout    = CONSTANT;
                this->scope[in2].data      = &(instr.constant.value.r123.key);
                this->scope[in2].type      = UINT64;
                this->scope[in2].nelem     = 1;

                this->program[i].op    = REDUCE;  // TAC
                this->program[i].oper  = LOGICAL_XOR;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= REDUCE;    // Operationmask
                break;
            case BH_MAXIMUM_REDUCE:
                in1 = this->add_operand(instr, 1);

                in2 = ++(this->noperands);
                this->scope[in2].layout    = CONSTANT;
                this->scope[in2].data      = &(instr.constant.value.r123.key);
                this->scope[in2].type      = UINT64;
                this->scope[in2].nelem     = 1;

                this->program[i].op    = REDUCE;  // TAC
                this->program[i].oper  = MAXIMUM;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= REDUCE;    // Operationmask
                break;
            case BH_MINIMUM_REDUCE:
                in1 = this->add_operand(instr, 1);

                in2 = ++(this->noperands);
                this->scope[in2].layout    = CONSTANT;
                this->scope[in2].data      = &(instr.constant.value.r123.key);
                this->scope[in2].type      = UINT64;
                this->scope[in2].nelem     = 1;

                this->program[i].op    = REDUCE;  // TAC
                this->program[i].oper  = MINIMUM;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= REDUCE;    // Operationmask
                break;
            case BH_MULTIPLY_REDUCE:
                in1 = this->add_operand(instr, 1);

                in2 = ++(this->noperands);
                this->scope[in2].layout    = CONSTANT;
                this->scope[in2].data      = &(instr.constant.value.r123.key);
                this->scope[in2].type      = UINT64;
                this->scope[in2].nelem     = 1;

                this->program[i].op    = REDUCE;  // TAC
                this->program[i].oper  = MULTIPLY;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= REDUCE;    // Operationmask
                break;
            case BH_ADD_ACCUMULATE:
                in1 = this->add_operand(instr, 1);

                in2 = ++(this->noperands);
                this->scope[in2].layout    = CONSTANT;
                this->scope[in2].data      = &(instr.constant.value.r123.key);
                this->scope[in2].type      = UINT64;
                this->scope[in2].nelem     = 1;

                this->program[i].op    = SCAN;  // TAC
                this->program[i].oper  = ADD;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= SCAN;    // Operationmask
                break;
            case BH_MULTIPLY_ACCUMULATE:
                in1 = this->add_operand(instr, 1);

                in2 = ++(this->noperands);
                this->scope[in2].layout    = CONSTANT;
                this->scope[in2].data      = &(instr.constant.value.r123.key);
                this->scope[in2].type      = UINT64;
                this->scope[in2].nelem     = 1;

                this->program[i].op    = SCAN;  // TAC
                this->program[i].oper  = MULTIPLY;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= SCAN;    // Operationmask
                break;
            case BH_RANDOM:
                // This one requires special-handling... what a beaty...
                in1 = ++(this->noperands);                // Input
                this->scope[in1].layout    = CONSTANT;
                this->scope[in1].data      = &(instr.constant.value.r123.start);
                this->scope[in1].type      = UINT64;
                this->scope[in1].nelem     = 1;

                in2 = ++(this->noperands);
                this->scope[in2].layout    = CONSTANT;
                this->scope[in2].data      = &(instr.constant.value.r123.key);
                this->scope[in2].type      = BH_UINT64;
                this->scope[in2].nelem     = 1;

                this->program[i].op    = GENERATE;  // TAC
                this->program[i].oper  = RANDOM;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= GENERATE;    // Operationmask
                break;
            case BH_RANGE:

                this->program[i].op    = GENERATE;  // TAC
                this->program[i].oper  = RANGE;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= GENERATE;    // Operationmask
                break;
            case BH_DISCARD:
                in1 = this->add_operand(instr, 1);

                this->program[i].op    = SYSTEM;  // TAC
                this->program[i].oper  = DISCARD;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= SYSTEM;    // Operationmask
                break;
            case BH_FREE:

                this->program[i].op    = SYSTEM;  // TAC
                this->program[i].oper  = FREE;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= SYSTEM;    // Operationmask
                break;
            case BH_NONE:
                in1 = this->add_operand(instr, 1);

                this->program[i].op    = SYSTEM;  // TAC
                this->program[i].oper  = NONE;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= SYSTEM;    // Operationmask
                break;
            case BH_SYNC:
                in1 = this->add_operand(instr, 1);

                this->program[i].op    = SYSTEM;  // TAC
                this->program[i].oper  = SYNC;
                this->program[i].out   = out;
                this->program[i].in1   = in1;
                this->program[i].in2   = in2;
            
                this->omask |= SYSTEM;    // Operationmask
                break;

            default:
                if (instr.opcode>=BH_MAX_OPCODE_ID) {   // Handle extensions here

                    this->program[i].op   = EXTENSION; // TODO: Be clever about it
                    this->program[i].oper = EXT_OFFSET;
                    this->program[i].out  = 0;
                    this->program[i].in1  = 0;
                    this->program[i].in2  = 0;

                    cout << "Extension method." << endl;
                } else {
                    in1 = 1;
                    in2 = 2;
                    printf("compose: Err=[Unsupported instruction] {\n");
                    bh_pprint_instr(&instr);
                    printf("}\n");
                    return BH_ERROR;
                }
        }
    }
    return BH_SUCCESS;
}

}}}