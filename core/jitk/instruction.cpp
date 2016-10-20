/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium
team <http://www.bh107.org>.

Bohrium is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3
of the License, or (at your option) any later version.

Bohrium is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the
GNU Lesser General Public License along with Bohrium.

If not, see <http://www.gnu.org/licenses/>.
*/

#include <sstream>

#include <bh_instruction.hpp>
#include <jitk/block.hpp>
#include <jitk/type.hpp>
#include <jitk/instruction.hpp>
#include <jitk/base_db.hpp>

using namespace std;

namespace bohrium {
namespace jitk {

void write_array_subscription(const bh_view &view, stringstream &out, int hidden_axis, const pair<int, int> axis_offset) {
    assert(view.base != NULL); // Not a constant
    bool empty_subscription = true;
    if (view.start > 0) {
        out << "[" << view.start;
        empty_subscription = false;
    } else {
        out << "[";
    }
    if (not bh_is_scalar(&view)) { // NB: this optimization is required when reducing a vector to a scalar!
        for (int i = 0; i < view.ndim; ++i) {
            int t = i;
            if (i >= hidden_axis)
                ++t;
            if (view.stride[i] > 0) {
                if (axis_offset.first == t) {
                    out << " +(i" << t << "+(" << axis_offset.second << ")) ";
                } else {
                    out << " +i" << t;
                }
                if (view.stride[i] != 1) {
                    out << "*" << view.stride[i];
                }
                empty_subscription = false;
            }
        }
    }
    if (empty_subscription)
        out << "0";
    out << "]";
}

void write_system_operation(const BaseDB &base_ids, const bh_instruction &instr, stringstream &out) {

    switch (instr.opcode) {
        case BH_FREE:
            out << "// FREE a" << base_ids[instr.operand[0].base];
            break;
        case BH_SYNC:
            out << "// SYNC a" << base_ids[instr.operand[0].base];
            break;
        case BH_NONE:
            out << "// NONE ";
            break;
        case BH_TALLY:
            out << "// TALLY";
            break;
        case BH_REPEAT:
            out << "// REPEAT";
            break;
        default:
            std::cerr << "Instruction \"" << bh_opcode_text(instr.opcode) << "\" (" << instr.opcode <<
                      ") not supported for non complex operations." << endl;
            throw std::runtime_error("Instruction not supported.");
    }
    out << endl;
}

// Write the sign function ((x > 0) - (0 > x)) to 'out'
void write_sign_function(const string &operand, stringstream &out) {

    out << "((" << operand << " > 0) - (0 > " << operand << "))";
}

void write_operation(const bh_instruction &instr, const vector<string> &operands, stringstream &out, bool opencl) {

    switch (instr.opcode) {
        case BH_ADD:
            out << operands[0] << " = " << operands[1] << " + " << operands[2] << ";" << endl;
            break;
        case BH_ADD_REDUCE:
            out << operands[0] << " += " << operands[1] << ";" << endl;
            break;
        case BH_ADD_ACCUMULATE:
            out << operands[0] << " = " << operands[1] << " + " << operands[2] << ";" << endl;
            break;
        case BH_SUBTRACT:
            out << operands[0] << " = " << operands[1] << " - " << operands[2] << ";" << endl;
            break;
        case BH_MULTIPLY:
            out << operands[0] << " = " << operands[1] << " * " << operands[2] << ";" << endl;
            break;
        case BH_MULTIPLY_REDUCE:
            out << operands[0] << " *= " << operands[1] << ";" << endl;
            break;
        case BH_MULTIPLY_ACCUMULATE:
            out << operands[0] << " = " << operands[1] << " * " << operands[2] << ";" << endl;
            break;
        case BH_DIVIDE:
            out << operands[0] << " = " << operands[1] << " / " << operands[2] << ";" << endl;
            break;
        case BH_POWER:
            out << operands[0] << " = pow(" << operands[1] << ", " << operands[2] << ");" << endl;
            break;
        case BH_MOD:
            if (bh_type_is_float(instr.operand[0].base->type))
                out << operands[0] << " = fmod(" << operands[1] << ", " << operands[2] << ");" << endl;
            else
                out << operands[0] << " = " << operands[1] << " % " << operands[2] << ";" << endl;
            break;
        case BH_ABSOLUTE:
            if (bh_type_is_float(instr.operand[0].base->type))
                out << operands[0] << " = fabs(" << operands[1] << ");" << endl;
            else
                out << operands[0] << " = abs(" << operands[1] << ");" << endl;
            break;
        case BH_RINT:
            out << operands[0] << " = rint(" << operands[1] << ");" << endl;
            break;
        case BH_EXP:
            out << operands[0] << " = exp(" << operands[1] << ");" << endl;
            break;
        case BH_EXP2:
            out << operands[0] << " = exp2(" << operands[1] << ");" << endl;
            break;
        case BH_LOG:
            out << operands[0] << " = log(" << operands[1] << ");" << endl;
            break;
        case BH_LOG10:
            // C99 does not have log10 for complex, so we use the formula: clog(z) = log(z)/log(10)
            if (bh_type_is_complex(instr.operand[0].base->type))
                out << operands[0] << " = clog(" << operands[1] << ") / log(10);" << endl;
            else
                out << operands[0] << " = log10(" << operands[1] << ");" << endl;
            break;
        case BH_EXPM1:
            out << operands[0] << " = expm1(" << operands[1] << ");" << endl;
            break;
        case BH_LOG1P:
            out << operands[0] << " = log1p(" << operands[1] << ");" << endl;
            break;
        case BH_SQRT:
            out << operands[0] << " = sqrt(" << operands[1] << ");" << endl;
            break;
        case BH_SIN:
            out << operands[0] << " = sin(" << operands[1] << ");" << endl;
            break;
        case BH_COS:
            out << operands[0] << " = cos(" << operands[1] << ");" << endl;
            break;
        case BH_TAN:
            out << operands[0] << " = tan(" << operands[1] << ");" << endl;
            break;
        case BH_ARCSIN:
            out << operands[0] << " = asin(" << operands[1] << ");" << endl;
            break;
        case BH_ARCCOS:
            out << operands[0] << " = acos(" << operands[1] << ");" << endl;
            break;
        case BH_ARCTAN:
            out << operands[0] << " = atan(" << operands[1] << ");" << endl;
            break;
        case BH_ARCTAN2:
            out << operands[0] << " = atan2(" << operands[1] << ", " << operands[2] << ");" << endl;
            break;
        case BH_SINH:
            out << operands[0] << " = sinh(" << operands[1] << ");" << endl;
            break;
        case BH_COSH:
            out << operands[0] << " = cosh(" << operands[1] << ");" << endl;
            break;
        case BH_TANH:
            out << operands[0] << " = tanh(" << operands[1] << ");" << endl;
            break;
        case BH_ARCSINH:
            out << operands[0] << " = asinh(" << operands[1] << ");" << endl;
            break;
        case BH_ARCCOSH:
            out << operands[0] << " = acosh(" << operands[1] << ");" << endl;
            break;
        case BH_ARCTANH:
            out << operands[0] << " = atanh(" << operands[1] << ");" << endl;
            break;
        case BH_BITWISE_AND:
            out << operands[0] << " = " << operands[1] << " & " << operands[2] << ";" << endl;
            break;
        case BH_BITWISE_AND_REDUCE:
            out << operands[0] << " = " << operands[0] << " & " << operands[1] << ";" << endl;
            break;
        case BH_BITWISE_OR:
            out << operands[0] << " = " << operands[1] << " | " << operands[2] << ";" << endl;
            break;
        case BH_BITWISE_OR_REDUCE:
            out << operands[0] << " = " << operands[0] << " | " << operands[1] << ";" << endl;
            break;
        case BH_BITWISE_XOR:
            out << operands[0] << " = " << operands[1] << " ^ " << operands[2] << ";" << endl;
            break;
        case BH_BITWISE_XOR_REDUCE:
            out << operands[0] << " = " << operands[0] << " ^ " << operands[1] << ";" << endl;
            break;
        case BH_LOGICAL_NOT:
            out << operands[0] << " = !" << operands[1] << ";" << endl;
            break;
        case BH_LOGICAL_OR:
            out << operands[0] << " = " << operands[1] << " || " << operands[2] << ";" << endl;
            break;
        case BH_LOGICAL_OR_REDUCE:
            out << operands[0] << " = " << operands[0] << " || " << operands[1] << ";" << endl;
            break;
        case BH_LOGICAL_AND:
            out << operands[0] << " = " << operands[1] << " && " << operands[2] << ";" << endl;
            break;
        case BH_LOGICAL_AND_REDUCE:
            out << operands[0] << " = " << operands[0] << " && " << operands[1] << ";" << endl;
            break;
        case BH_LOGICAL_XOR:
            out << operands[0] << " = !" << operands[1] << " != !" << operands[2] << ";" << endl;
            break;
        case BH_LOGICAL_XOR_REDUCE:
            out << operands[0] << " = !" << operands[0] << " != !" << operands[1] << ";" << endl;
            break;
        case BH_INVERT:
            if (instr.operand[0].base->type == BH_BOOL)
                out << operands[0] << " = !" << operands[1] << ";" << endl;
            else
                out << operands[0] << " = ~" << operands[1] << ";" << endl;
            break;
        case BH_LEFT_SHIFT:
            out << operands[0] << " = " << operands[1] << " << " << operands[2] << ";" << endl;
            break;
        case BH_RIGHT_SHIFT:
            out << operands[0] << " = " << operands[1] << " >> " << operands[2] << ";" << endl;
            break;
        case BH_GREATER:
            out << operands[0] << " = " << operands[1] << " > " << operands[2] << ";" << endl;
            break;
        case BH_GREATER_EQUAL:
            out << operands[0] << " = " << operands[1] << " >= " << operands[2] << ";" << endl;
            break;
        case BH_LESS:
            out << operands[0] << " = " << operands[1] << " < " << operands[2] << ";" << endl;
            break;
        case BH_LESS_EQUAL:
            out << operands[0] << " = " << operands[1] << " <= " << operands[2] << ";" << endl;
            break;
        case BH_NOT_EQUAL:
            out << operands[0] << " = " << operands[1] << " != " << operands[2] << ";" << endl;
            break;
        case BH_EQUAL:
            out << operands[0] << " = " << operands[1] << " == " << operands[2] << ";" << endl;
            break;
        case BH_MAXIMUM:
            out << operands[0] << " = " << operands[1] << " > " << operands[2] << " ? " << operands[1] << " : "
                << operands[2] << ";" << endl;
            break;
        case BH_MAXIMUM_REDUCE:
            out << operands[0] << " = " << operands[0] << " > " << operands[1] << " ? " << operands[0] << " : "
                << operands[1] << ";" << endl;
            break;
        case BH_MINIMUM:
            out << operands[0] << " = " << operands[1] << " < " << operands[2] << " ? " << operands[1] << " : "
                << operands[2] << ";" << endl;
            break;
        case BH_MINIMUM_REDUCE:
            out << operands[0] << " = " << operands[0] << " < " << operands[1] << " ? " << operands[0] << " : "
                << operands[1] << ";" << endl;
            break;
        case BH_IDENTITY: {
            out << operands[0] << " = ";
            const bh_type t0 = instr.operand_type(0);
            const bh_type t1 = instr.operand_type(1);
            // In OpenCL we have to do explicit conversion of complex numbers
            if (opencl and t0 == BH_COMPLEX64 and t1 == BH_COMPLEX128) {
                out << "convert_float2(" << operands[1] << ")";
            } else if (opencl and t0 == BH_COMPLEX128 and t1 == BH_COMPLEX64) {
                out << "convert_double2(" << operands[1] << ")";
            } else if (opencl and bh_type_is_complex(t0) and not bh_type_is_complex(t1)){
                out << "(" << (t0 == BH_COMPLEX64?"float2":"double2") << ")(" << operands[1] << ", 0.0)";
            } else {
                out << operands[1];
            }
            out << ";" << endl;
            break;
        }
        case BH_FLOOR:
            out << operands[0] << " = floor(" << operands[1] << ");" << endl;
            break;
        case BH_CEIL:
            out << operands[0] << " = ceil(" << operands[1] << ");" << endl;
            break;
        case BH_TRUNC:
            out << operands[0] << " = trunc(" << operands[1] << ");" << endl;
            break;
        case BH_LOG2:
            out << operands[0] << " = log2(" << operands[1] << ");" << endl;
            break;
        case BH_ISNAN:
            out << operands[0] << " = isnan(" << operands[1] << ");" << endl;
            break;
        case BH_ISINF:
            out << operands[0] << " = isinf(" << operands[1] << ");" << endl;
            break;
        case BH_RANGE:
            out << operands[0] << " = " << operands[1] << ";" << endl;
            break;
        case BH_RANDOM:
            out << operands[0] << " = " << operands[1] << ";" << endl;
            break;
        case BH_REAL:
            out << operands[0] << " = creal(" << operands[1] << ");" << endl;
            break;
        case BH_IMAG:
            out << operands[0] << " = cimag(" << operands[1] << ");" << endl;
            break;
        case BH_SIGN: {
            /* NB: there are different ways to define sign of complex numbers.
             *     We uses the same definition as in NumPy
             *     <http://docs.scipy.org/doc/numpy-dev/reference/generated/numpy.sign.html>
             */
            bh_type t = instr.operand[0].base->type;
            if (bh_type_is_complex(t)) {
                if (t == BH_COMPLEX64) {
                    out << "float real = creal(" << operands[1] << "); ";
                    out << "float imag = cimag(" << operands[1] << "); ";
                } else {
                    out << "double real = creal(" << operands[1] << "); ";
                    out << "double imag = cimag(" << operands[1] << "); ";
                }
                out << operands[0] << " = real != 0 ? ";
                write_sign_function("real", out);
                out << " : ";
                write_sign_function("imag", out);
                out << ";" << endl;

            } else {
                out << operands[0] << " = ";
                write_sign_function(operands[1], out);
                out << ";" << endl;
            }
            break;
        }
        default:
            cerr << "Instruction \"" << instr << "\" not supported" << endl;
            throw runtime_error("Instruction not supported.");
    }
}

int sweep_axis(const bh_instruction &instr) {
    if (bh_opcode_is_sweep(instr.opcode)) {
        assert(bh_noperands(instr.opcode) == 3);
        assert(bh_is_constant(&instr.operand[2]));
        return static_cast<int>(instr.constant.get_int64());
    }
    return BH_MAXDIM;
}

void write_instr(const BaseDB &base_ids, const bh_instruction &instr, stringstream &out, bool opencl) {
    if (bh_opcode_is_system(instr.opcode)) {
        write_system_operation(base_ids, instr, out);
        return;
    }
    if (instr.opcode == BH_RANGE) {
        assert(instr.operand[0].ndim == 1); //TODO: support multidimensional output
        vector<string> operands;
        if (base_ids.isTmp(instr.operand[0].base)) {
            stringstream ss;
            ss << "t" << base_ids[instr.operand[0].base];
            operands.push_back(ss.str());
        } else {
            stringstream ss;
            ss << "a" << base_ids[instr.operand[0].base];
            write_array_subscription(instr.operand[0], ss);
            operands.push_back(ss.str());
        }
        operands.push_back("i0");
        write_operation(instr, operands, out, opencl);
        return;
    }
    if (instr.opcode == BH_RANDOM) {
        assert(instr.operand[0].ndim == 1); //TODO: support multidimensional output
        vector<string> operands;
        // Write output operand
        if (base_ids.isTmp(instr.operand[0].base)) {
            stringstream ss;
            ss << "t" << base_ids[instr.operand[0].base];
            operands.push_back(ss.str());
        } else {
            stringstream ss;
            ss << "a" << base_ids[instr.operand[0].base];
            write_array_subscription(instr.operand[0], ss);
            operands.push_back(ss.str());
        }
        // Write the random generation
        {
            stringstream ss;
            ss << "random123(" << instr.constant.value.r123.start \
               << ", " << instr.constant.value.r123.key << ", i0)";
            operands.push_back(ss.str());
        }
        write_operation(instr, operands, out, opencl);
        return;
    }
    if (bh_opcode_is_accumulate(instr.opcode)) {
        vector<string> operands;
        // Write output operand
        if (base_ids.isTmp(instr.operand[0].base)) {
            stringstream ss;
            ss << "t" << base_ids[instr.operand[0].base];
            operands.push_back(ss.str());
        } else {
            stringstream ss;
            ss << "a" << base_ids[instr.operand[0].base];
            write_array_subscription(instr.operand[0], ss);
            operands.push_back(ss.str());
        }
        // Write the previous element access, NB: this works because of loop peeling
        {
            stringstream ss;
            ss << "a" << base_ids[instr.operand[0].base];
            write_array_subscription(instr.operand[0], ss, BH_MAXDIM, make_pair(sweep_axis(instr), -1));
            operands.push_back(ss.str());
        }
        // Write the current element access
        {
            stringstream ss;
            ss << "a" << base_ids[instr.operand[1].base];
            write_array_subscription(instr.operand[1], ss);
            operands.push_back(ss.str());
        }
        write_operation(instr, operands, out, opencl);
        return;
    }
    vector<string> operands;
    for (int o = 0; o < bh_noperands(instr.opcode); ++o) {
        const bh_view &view = instr.operand[o];
        stringstream ss;
        if (bh_is_constant(&view)) {
            instr.constant.pprint(ss, opencl);
        } else {
            if (base_ids.isTmp(view.base)) {
                ss << "t" << base_ids[view.base];
            } else if (base_ids.isScalarReplaced(view.base)) {
                ss << "s" << base_ids[view.base];
            } else {
                ss << "a" << base_ids[view.base];
                if (o == 0 and bh_opcode_is_reduction(instr.opcode) and instr.operand[1].ndim > 1) {
                    // If 'instr' is a reduction we have to ignore the reduced axis of the output array when
                    // reducing to a non-scalar
                    write_array_subscription(view, ss, sweep_axis(instr));
                } else {
                    write_array_subscription(view, ss);
                }
            }
        }
        operands.push_back(ss.str());
    }
    write_operation(instr, operands, out, opencl);
}


} // jitk
} // bohrium

