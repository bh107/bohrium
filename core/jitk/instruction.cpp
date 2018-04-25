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
#include <jitk/instruction.hpp>
#include <jitk/symbol_table.hpp>
#include <jitk/view.hpp>

using namespace std;

namespace bohrium {
namespace jitk {

namespace { // We need some help functions

// Write the sign function ((x > 0) - (0 > x)) to 'out'
void write_sign_function(const string &operand, stringstream &out) {
    out << "((" << operand << " > 0) - (0 > " << operand << "))";
}

// Write opcodes that uses a different complex functions when targeting OpenCL
void write_opcodes_with_special_opencl_complex(const bh_instruction &instr, const vector<string> &ops,
                                               stringstream &out, int opencl, const char *fname,
                                               const char *fname_complex) {
    const bh_type t0 = instr.operand_type(0);
    if (opencl and bh_type_is_complex(t0)) {
        out << fname_complex << "(" << (t0 == bh_type::COMPLEX64 ? "float" : "double") << ", " << ops[0] \
 << ", " << ops[1] << ");";
    } else {
        out << ops[0] << " = " << fname << "(" << ops[1] << ");";
    }
    out << "\n";
}

// Write the 'instr' using the string in 'ops' as ops
void write_operation(const bh_instruction &instr, const vector<string> &ops, stringstream &out, bool opencl) {
    switch (instr.opcode) {
        // Opcodes that are Complex/OpenCL agnostic
        case BH_BITWISE_AND:
            out << ops[0] << " = " << ops[1] << " & " << ops[2] << ";";
            break;
        case BH_BITWISE_AND_REDUCE:
            out << ops[0] << " = " << ops[0] << " & " << ops[1] << ";";
            break;
        case BH_BITWISE_OR:
            out << ops[0] << " = " << ops[1] << " | " << ops[2] << ";";
            break;
        case BH_BITWISE_OR_REDUCE:
            out << ops[0] << " = " << ops[0] << " | " << ops[1] << ";";
            break;
        case BH_BITWISE_XOR:
            out << ops[0] << " = " << ops[1] << " ^ " << ops[2] << ";";
            break;
        case BH_BITWISE_XOR_REDUCE:
            out << ops[0] << " = " << ops[0] << " ^ " << ops[1] << ";";
            break;
        case BH_LOGICAL_NOT:
            out << ops[0] << " = !" << ops[1] << ";";
            break;
        case BH_LOGICAL_OR:
            out << ops[0] << " = " << ops[1] << " || " << ops[2] << ";";
            break;
        case BH_LOGICAL_OR_REDUCE:
            out << ops[0] << " = " << ops[0] << " || " << ops[1] << ";";
            break;
        case BH_LOGICAL_AND:
            out << ops[0] << " = " << ops[1] << " && " << ops[2] << ";";
            break;
        case BH_LOGICAL_AND_REDUCE:
            out << ops[0] << " = " << ops[0] << " && " << ops[1] << ";";
            break;
        case BH_LOGICAL_XOR:
            out << ops[0] << " = !" << ops[1] << " != !" << ops[2] << ";";
            break;
        case BH_LOGICAL_XOR_REDUCE:
            out << ops[0] << " = !" << ops[0] << " != !" << ops[1] << ";";
            break;
        case BH_LEFT_SHIFT:
            out << ops[0] << " = " << ops[1] << " << " << ops[2] << ";";
            break;
        case BH_RIGHT_SHIFT:
            out << ops[0] << " = " << ops[1] << " >> " << ops[2] << ";";
            break;
        case BH_GREATER:
            out << ops[0] << " = " << ops[1] << " > " << ops[2] << ";";
            break;
        case BH_GREATER_EQUAL:
            out << ops[0] << " = " << ops[1] << " >= " << ops[2] << ";";
            break;
        case BH_LESS:
            out << ops[0] << " = " << ops[1] << " < " << ops[2] << ";";
            break;
        case BH_LESS_EQUAL:
            out << ops[0] << " = " << ops[1] << " <= " << ops[2] << ";";
            break;
        case BH_MAXIMUM:
            out << ops[0] << " = " << ops[1] << " > " << ops[2] << " ? " << ops[1] << " : "
                << ops[2] << ";";
            break;
        case BH_MAXIMUM_REDUCE:
            out << ops[0] << " = " << ops[0] << " > " << ops[1] << " ? " << ops[0] << " : "
                << ops[1] << ";";
            break;
        case BH_MINIMUM:
            out << ops[0] << " = " << ops[1] << " < " << ops[2] << " ? " << ops[1] << " : "
                << ops[2] << ";";
            break;
        case BH_MINIMUM_REDUCE:
            out << ops[0] << " = " << ops[0] << " < " << ops[1] << " ? " << ops[0] << " : "
                << ops[1] << ";";
            break;
        case BH_INVERT:
            if (instr.operand[0].base->type == bh_type::BOOL)
                out << ops[0] << " = !" << ops[1] << ";";
            else
                out << ops[0] << " = ~" << ops[1] << ";";
            break;
        case BH_MOD:
            if (bh_type_is_float(instr.operand[0].base->type))
                out << ops[0] << " = fmod(" << ops[1] << ", " << ops[2] << ");";
            else
                out << ops[0] << " = " << ops[1] << " % " << ops[2] << ";";
            break;
        case BH_REMAINDER:
            if (bh_type_is_float(instr.operand[0].base->type)) {
                out << ops[0] << " = " << ops[1] << " - floor(" << ops[1] <<  " / " << ops[2] << ") * " << ops[2] << ";";
            } else if (bh_type_is_unsigned_integer(instr.operand[0].base->type)) {
                out << ops[0] << " = " << ops[1] << " % " << ops[2] << ";";
            } else {
                /* The Python/NumPy implementation of remainder on signed integers
                    const @type@ rem = in1 % in2;
                    if ((in1 > 0) == (in2 > 0) || rem == 0) {
                        *((@type@ *)op1) = rem;
                    }
                    else {
                        *((@type@ *)op1) = rem + in2;
                    }
                */
                out << ops[0] << " = ((" << ops[1] << " > 0) == (" << ops[2] << " > 0) || "
                                          "(" << ops[1] <<  " % " << ops[2] << ") == 0) ? "
                                          "(" << ops[1] <<  " % " << ops[2] << ") : "
                                          "(" << ops[1] <<  " % " << ops[2] << ") + " << ops[2] << ";";
            }
            break;
        case BH_RINT:
            out << ops[0] << " = rint(" << ops[1] << ");";
            break;
        case BH_EXP2:
            out << ops[0] << " = exp2(" << ops[1] << ");";
            break;
        case BH_EXPM1:
            out << ops[0] << " = expm1(" << ops[1] << ");";
            break;
        case BH_LOG1P:
            out << ops[0] << " = log1p(" << ops[1] << ");";
            break;
        case BH_ARCSIN:
            out << ops[0] << " = asin(" << ops[1] << ");";
            break;
        case BH_ARCCOS:
            out << ops[0] << " = acos(" << ops[1] << ");";
            break;
        case BH_ARCTAN:
            out << ops[0] << " = atan(" << ops[1] << ");";
            break;
        case BH_ARCTAN2:
            out << ops[0] << " = atan2(" << ops[1] << ", " << ops[2] << ");";
            break;
        case BH_ARCSINH:
            out << ops[0] << " = asinh(" << ops[1] << ");";
            break;
        case BH_ARCCOSH:
            out << ops[0] << " = acosh(" << ops[1] << ");";
            break;
        case BH_ARCTANH:
            out << ops[0] << " = atanh(" << ops[1] << ");";
            break;
        case BH_FLOOR:
            out << ops[0] << " = floor(" << ops[1] << ");";
            break;
        case BH_CEIL:
            out << ops[0] << " = ceil(" << ops[1] << ");";
            break;
        case BH_TRUNC:
            out << ops[0] << " = trunc(" << ops[1] << ");";
            break;
        case BH_LOG2:
            out << ops[0] << " = log2(" << ops[1] << ");";
            break;
        case BH_ISNAN: {
            const bh_type t0 = instr.operand_type(1);

            if (bh_type_is_complex(t0)) {
                if (opencl) {
                    out << ops[0] << " = isnan(" << ops[1] << ".x);";
                } else {
                    out << ops[0] << " = isnan(creal(" << ops[1] << "));";
                }
            } else if (bh_type_is_float(t0)) {
                out << ops[0] << " = isnan(" << ops[1] << ");";
            } else {
                out << ops[0] << " = false;";
            }
            break;
        }
        case BH_ISINF: {
            const bh_type t0 = instr.operand_type(1);

            if (bh_type_is_complex(t0)) {
                if (opencl) {
                    out << ops[0] << " = isinf(" << ops[1] << ".x);";
                } else {
                    out << ops[0] << " = isinf(creal(" << ops[1] << "));";
                }
            } else if(bh_type_is_float(t0)) {
                out << ops[0] << " = isinf(" << ops[1] << ");";
            } else {
                out << ops[0] << " = false;";
            }
            break;
        }
        case BH_ISFINITE: {
            const bh_type t0 = instr.operand_type(1);

            if (bh_type_is_complex(t0)) {
                if (opencl) {
                    out << ops[0] << " = isfinite(" << ops[1] << ".x);";
                } else {
                    out << ops[0] << " = isfinite(creal(" << ops[1] << "));";
                }
            } else if (bh_type_is_float(t0)) {
                out << ops[0] << " = isfinite(" << ops[1] << ");";
            } else {
                out << ops[0] << " = true;";
            }
            break;
        }
        case BH_CONJ:
            if (opencl) {
                out << ops[0] << " = " << ops[1] << ";";
                out << ops[0] << ".y *= -1;";
            } else {
                out << ops[0] << " = conj(" << ops[1] << ");";
            }
            break;
        case BH_RANGE:
            out << ops[0] << " = " << ops[1] << ";";
            break;
        case BH_RANDOM:
            out << ops[0] << " = " << ops[1] << ";";
            break;

        // Opcodes that uses a different function name in OpenCL
        case BH_SIN:
            write_opcodes_with_special_opencl_complex(instr, ops, out, opencl, "sin", "CSIN");
            break;
        case BH_COS:
            write_opcodes_with_special_opencl_complex(instr, ops, out, opencl, "cos", "CCOS");
            break;
        case BH_TAN:
            write_opcodes_with_special_opencl_complex(instr, ops, out, opencl, "tan", "CTAN");
            break;
        case BH_SINH:
            write_opcodes_with_special_opencl_complex(instr, ops, out, opencl, "sinh", "CSINH");
            break;
        case BH_COSH:
            write_opcodes_with_special_opencl_complex(instr, ops, out, opencl, "cosh", "CCOSH");
            break;
        case BH_TANH:
            write_opcodes_with_special_opencl_complex(instr, ops, out, opencl, "tanh", "CTANH");
            break;
        case BH_EXP:
            write_opcodes_with_special_opencl_complex(instr, ops, out, opencl, "exp", "CEXP");
            break;
        case BH_ABSOLUTE: {
            const bh_type t0 = instr.operand_type(1);

            if (t0 == bh_type::BOOL) {
                out << ops[0] << " = true;";
            } else if (bh_type_is_unsigned_integer(t0)) {
                out << ops[0] << " = " << ops[1] << ";"; // no-op
            } else if (opencl and bh_type_is_complex(t0)) {
                out << "CABS(" << ops[0] << ", " << ops[1] << ");";
            } else if (bh_type_is_float(t0)) {
                out << ops[0] << " = fabs(" << ops[1] << ");";
            } else if (!opencl and t0 == bh_type::INT64) {
                out << ops[0] << " = llabs(" << ops[1] << ");";
            } else {
                out << ops[0] << " = abs((int)" << ops[1] << ");";
            }
            break;
        }
        case BH_SQRT:
            if (opencl and bh_type_is_complex(instr.operand_type(0))) {
                out << "CSQRT(" << ops[0] << ", " << ops[1] << ");";
            } else {
                out << ops[0] << " = sqrt(" << ops[1] << ");";
            }
            break;
        case BH_LOG:
            if (opencl and bh_type_is_complex(instr.operand_type(0))) {
                out << "CLOG(" << ops[0] << ", " << ops[1] << ");";
            } else {
                out << ops[0] << " = log(" << ops[1] << ");";
            }
            break;
        case BH_NOT_EQUAL:
            if (opencl and bh_type_is_complex(instr.operand_type(1))) {
                out << "CNEQ(" << ops[0] << ", " << ops[1] << ", " << ops[2] << ");";
            } else {
                out << ops[0] << " = " << ops[1] << " != " << ops[2] << ";";
            }
            break;
        case BH_EQUAL:
            if (opencl and bh_type_is_complex(instr.operand_type(1))) {
                out << "CEQ(" << ops[0] << ", " << ops[1] << ", " << ops[2] << ");";
            } else {
                out << ops[0] << " = " << ops[1] << " == " << ops[2] << ";";
            }
            break;
        case BH_POWER: {
            const bh_type t0 = instr.operand_type(0);
            if (opencl and bh_type_is_complex(t0)) {
                out << "CPOW(" << (t0 == bh_type::COMPLEX64 ? "float" : "double") << ", " \
                    << ops[0] << ", " << ops[1] << ", " << ops[2] << ");";
            } else if (opencl and bh_type_is_integer(t0)) {
                out << "IPOW(" << ops[0] << ", " << ops[1] << ", " << ops[2] << ");";
            } else {
                out << ops[0] << " = pow(" << ops[1] << ", " << ops[2] << ");";
            }
            break;
        }


        // The operations that has to be handled differently in OpenCL
        case BH_ADD:
            if (opencl and bh_type_is_complex(instr.operand_type(0))) {
                out << "CADD(" << ops[0] << ", " << ops[1] << ", " << ops[2] << ");";
            } else {
                out << ops[0] << " = " << ops[1] << " + " << ops[2] << ";";
            }
            break;
        case BH_ADD_REDUCE:
            if (opencl and bh_type_is_complex(instr.operand_type(0))) {
                out << "CADD(" << ops[0] << ", " << ops[1] << ", " << ops[1] << ");";
            } else {
                out << ops[0] << " += " << ops[1] << ";";
            }
            break;
        case BH_ADD_ACCUMULATE:
            if (opencl and bh_type_is_complex(instr.operand_type(0))) {
                out << "CADD(" << ops[0] << ", " << ops[1] << ", " << ops[2] << ");";
            } else {
                out << ops[0] << " = " << ops[1] << " + " << ops[2] << ";";
            }
            break;
        case BH_SUBTRACT:
            if (opencl and bh_type_is_complex(instr.operand_type(0))) {
                out << "CSUB(" << ops[0] << ", " << ops[1] << ", " << ops[2] << ");";
            } else {
                out << ops[0] << " = " << ops[1] << " - " << ops[2] << ";";
            }
            break;
        case BH_MULTIPLY:
            if (opencl and bh_type_is_complex(instr.operand_type(0))) {
                out << "CMUL(" << ops[0] << ", " << ops[1] << ", " << ops[2] << ");";
            } else {
                out << ops[0] << " = " << ops[1] << " * " << ops[2] << ";";
            }
            break;
        case BH_MULTIPLY_REDUCE:
            if (opencl and bh_type_is_complex(instr.operand_type(0))) {
                out << "CMUL(" << ops[0] << ", " << ops[1] << ", " << ops[1] << ");";
            } else {
                out << ops[0] << " *= " << ops[1] << ";";
            }
            break;
        case BH_MULTIPLY_ACCUMULATE:
            if (opencl and bh_type_is_complex(instr.operand_type(0))) {
                out << "CMUL(" << ops[0] << ", " << ops[1] << ", " << ops[2] << ");";
            } else {
                out << ops[0] << " = " << ops[1] << " * " << ops[2] << ";";
            }
            break;
        case BH_DIVIDE: {
            const bh_type t0 = instr.operand_type(0);
            if (opencl and bh_type_is_complex(t0)) {
                out << "CDIV(" << (t0 == bh_type::COMPLEX64 ? "float" : "double") << ", " \
                    << ops[0] << ", " << ops[1] << ", " << ops[2] << ");";
            } else if (bh_type_is_signed_integer(instr.operand[0].base->type)) {
                /* Python/NumPy signed integer division
                    if (in2 == 0 || (in1 == NPY_MIN_@TYPE@ && in2 == -1)) {
                        npy_set_floatstatus_divbyzero();
                        *((@type@ *)op1) = 0;
                    }
                    else if (((in1 > 0) != (in2 > 0)) && (in1 % in2 != 0)) {
                        *((@type@ *)op1) = in1/in2 - 1;
                    }
                    else {
                        *((@type@ *)op1) = in1/in2;
                    }
                */
                out << ops[0] << " = ((" << ops[1] << " > 0) != (" << ops[2] << " > 0) && "
                                          "(" << ops[1] <<  " % " << ops[2] << ") != 0)?"
                                          "(" << ops[1] <<  " / " << ops[2] << " - 1):"
                                          "(" << ops[1] <<  " / " << ops[2] << ");";
            } else {
                out << ops[0] << " = " << ops[1] << " / " << ops[2] << ";";
            }
            break;
        }

        // In OpenCL we have to do explicit conversion of complex numbers
        case BH_IDENTITY: {
            out << ops[0] << " = ";
            const bh_type t0 = instr.operand_type(0);
            const bh_type t1 = instr.operand_type(1);

            if (opencl and t0 == bh_type::COMPLEX64 and t1 == bh_type::COMPLEX128) {
                out << "make_complex64((float)" << ops[1] << ".x, (float)" << ops[1] << ".y)";
            } else if (opencl and t0 == bh_type::COMPLEX128 and t1 == bh_type::COMPLEX64) {
                out << "make_complex128((double)" << ops[1] << ".x, (double)" << ops[1] << ".y)";
            } else if (opencl and bh_type_is_complex(t0) and not bh_type_is_complex(t1)) {
                out << "make_complex" << (t0 == bh_type::COMPLEX64 ? "64" : "128") << "(" << ops[1] << ", 0.0f)";
            } else if (opencl and t0 == bh_type::BOOL and t1 != bh_type::BOOL) {
                out << "(" << ops[1] << " == 0 ? 0 : 1)";
            } else {
                out << ops[1];
            }
            out << ";";
            break;
        }

        // C99 does not have log10 for complex, so we use the formula: clog(z) = log(z)/log(10)
        case BH_LOG10: {
            const bh_type t0 = instr.operand_type(0);
            if (opencl and bh_type_is_complex(t0)) {
                out << "CLOG(" << ops[0] << ", " << ops[1] << "); CDIV(" << (t0 == bh_type::COMPLEX64 ? "float" : "double") \
                    << ", " << ops[0] << ", " <<  ops[0] << ", make_complex" << (t0 == bh_type::COMPLEX64 ? "64" : "128")
                    << "(log(10.0f), 0.0f));";
            } else if (bh_type_is_complex(t0)) {
                out << ops[0] << " = clog(" << ops[1] << ") / log(10.0f);";
            } else {
                out << ops[0] << " = log10(" << ops[1] << ");";
            }
            break;
        }

        // Extracting the real or imaginary part differ in OpenCL
        case BH_REAL:
            if (opencl) {
                out << ops[0] << " = " << ops[1] << ".x;";
            } else {
                out << ops[0] << " = creal(" << ops[1] << ");";
            }
            break;
        case BH_IMAG:
            if (opencl) {
                out << ops[0] << " = " << ops[1] << ".y;";
            } else {
                out << ops[0] << " = cimag(" << ops[1] << ");";
            }
            break;

        /* NOTE: There are different ways to define sign of complex numbers.
         * We uses the same definition as in NumPy
         * <http://docs.scipy.org/doc/numpy-dev/reference/generated/numpy.sign.html>
         */
        case BH_SIGN: {
            const bh_type t0 = instr.operand_type(0);
            if (bh_type_is_complex(t0)) {
                //              1         if Re(z) > 0
                // csgn(z) = { -1         if Re(z) < 0
                //             sgn(Im(z)) if Re(z) = 0
                const char *ctype = (t0 == bh_type::COMPLEX64 ? "float" : "double");
                if (opencl) {
                    out << ctype << " real = " << ops[1] << ".x; \n";
                    out << ctype << " imag = " << ops[1] << ".y; \n";

                    // Complex sign always have Im(x) = 0
                    out << ops[0] << ".y = 0.0;\n";
                    out << ops[0] << ".x ";
                } else {
                    out << ctype << " real = creal(" << ops[1] << "); \n";
                    out << ctype << " imag = cimag(" << ops[1] << "); \n";
                    out << ops[0] << " ";
                }

                out << "= (real == 0 ? ";

                write_sign_function("imag", out);
                out << " : ";
                write_sign_function("real", out);
                out << ");";
            } else {
                out << ops[0] << " = ";
                write_sign_function(ops[1], out);
                out << ";";
            }
            break;
        }
        case BH_GATHER:
            out << ops[0] << " = " << ops[1] << ";";
            break;
        case BH_SCATTER:
            out << ops[0] << " = " << ops[1] << ";";
            break;
        case BH_COND_SCATTER:
            out << "if (" << ops[2] << ") { " << ops[0] << " = " << ops[1] << "; }";
            break;
        default:
            cerr << "Instruction \"" << instr << "\" not supported\n";
            throw runtime_error("Instruction not supported.");
    }
    out << "\n";
}

// Print the maximum value of 'dtype'
void dtype_max(bh_type dtype, stringstream &out) {
    if (bh_type_is_integer(dtype)) {
        out << bh_type_limit_max_integer(dtype);
        if (not bh_type_is_signed_integer(dtype)) {
            out << "u";
        }
    } else {
        out.precision(std::numeric_limits<double>::max_digits10);
        out << bh_type_limit_max_float(dtype);
    }
}

// Print the minimum value of 'dtype'
void dtype_min(bh_type dtype, stringstream &out) {
    if (bh_type_is_integer(dtype)) {
        out << bh_type_limit_min_integer(dtype) + 1;
    } else {
        out.precision(std::numeric_limits<double>::max_digits10);
        out << bh_type_limit_min_float(dtype);
    }
}

void get_name_and_subscription(const Scope &scope, const bh_view &view, stringstream &out) {
    scope.getName(view, out);
    if (scope.isArray(view)) {
        write_array_subscription(scope, view, out);
    }
}

string get_name_and_subscription(const Scope &scope, const bh_view &view) {
    stringstream ss;
    get_name_and_subscription(scope, view, ss);
    return ss.str();
}

void write_range_instr(const Scope &scope, const bh_instruction &instr, stringstream &out, bool opencl) {
    vector<string> ops;

    // Write output operand
    ops.push_back(get_name_and_subscription(scope, instr.operand[0]));

    // Let's find the flatten index of the output view
    stringstream ss;
    ss << "(";
    write_array_index(scope, instr.operand[0], ss);
    ss << ")";
    ops.push_back(ss.str());
    write_operation(instr, ops, out, opencl);
}

void write_random_instr(const Scope &scope, const bh_instruction &instr, stringstream &out, bool opencl) {
    vector<string> ops;

    // Write output operand
    ops.push_back(get_name_and_subscription(scope, instr.operand[0]));

    // Write the random generation
    stringstream ss;
    // Find the random `start` and `key`
    const int64_t constID = scope.symbols.constID(instr);
    if (constID >= 0) {
        ss << "random123(" << "c" << constID << ".x, " << "c" << constID << ".y, " ;
    } else {
        ss << "random123(" << instr.constant.value.r123.start << ", " << instr.constant.value.r123.key << ", ";
    }
    write_array_index(scope, instr.operand[0], ss);
    ss << ")";
    ops.push_back(ss.str());
    write_operation(instr, ops, out, opencl);
}

void write_gather_instr(const Scope &scope, const bh_instruction &instr, stringstream &out, bool opencl) {
    assert(not bh_is_constant(&instr.operand[1]));

    // Format of GATHER: out[<loop-indexes>] = in1[in1.start + in2[<loop-indexes>]]
    vector<string> ops;
    ops.push_back(get_name_and_subscription(scope, instr.operand[0]));

    stringstream ss;
    scope.getName(instr.operand[1], ss);
    ss << "[" << instr.operand[1].start << " + ";
    get_name_and_subscription(scope, instr.operand[2], ss);
    ss << "]";
    ops.push_back(ss.str());

    write_operation(instr, ops, out, opencl);
}

void write_scatter_instr(const Scope &scope, const bh_instruction &instr, stringstream &out, bool opencl) {
     // Format of SCATTER: out[out.start + in2[<loop-indexes>]] = in1[<loop-indexes>]
    vector<string> ops;

    stringstream ss;
    scope.getName(instr.operand[0], ss);
    ss << "[" << instr.operand[0].start << " + ";
    get_name_and_subscription(scope, instr.operand[2], ss);
    ss << "]";
    ops.push_back(ss.str());

    ops.push_back(get_name_and_subscription(scope, instr.operand[1]));

    if (instr.opcode == BH_COND_SCATTER) { // Add the conditional array (fourth operand)
        ops.push_back(get_name_and_subscription(scope, instr.operand[3]));
    }

    write_operation(instr, ops, out, opencl);
}

void write_accumulate_instr(const Scope &scope, const bh_instruction &instr, stringstream &out, bool opencl) {
    vector<string> ops;

    // Write output operand
    ops.push_back(get_name_and_subscription(scope, instr.operand[0]));

    // Write the previous element access, NB: this works because of loop peeling
    stringstream ss;
    scope.getName(instr.operand[0], ss);
    write_array_subscription(scope, instr.operand[0], ss, true, BH_MAXDIM, make_pair(instr.sweep_axis(), -1));
    ops.push_back(ss.str());

    // Write the current element access
    ops.push_back(get_name_and_subscription(scope, instr.operand[1]));

    write_operation(instr, ops, out, opencl);
}

void write_other_instr(const Scope &scope, const bh_instruction &instr, stringstream &out, bool opencl) {
    vector<string> ops;
    for (size_t o = 0; o < instr.operand.size(); ++o) {
        const bh_view &view = instr.operand[o];
        stringstream ss;
        if (bh_is_constant(&view)) {
            const int64_t constID = scope.symbols.constID(instr);
            if (constID >= 0) {
                ss << "c" << scope.symbols.constID(instr);
            } else {
                instr.constant.pprint(ss, opencl);
            }
        } else {
            scope.getName(view, ss);
            if (scope.isArray(view)) {
                if (o == 0 and bh_opcode_is_reduction(instr.opcode) and instr.operand[1].ndim > 1) {
                    // If 'instr' is a reduction we have to ignore the reduced axis of the output array when
                    // reducing to a non-scalar
                    write_array_subscription(scope, view, ss, true, instr.sweep_axis());
                } else {
                    write_array_subscription(scope, view, ss);
                }
            }
        }
        ops.push_back(ss.str());
    }
    write_operation(instr, ops, out, opencl);
}

} // Anon namespace

void write_instr(const Scope &scope, const bh_instruction &instr, stringstream &out, bool opencl) {
    if (bh_opcode_is_system(instr.opcode)) {
        return;
    }
    switch(instr.opcode) {
        case BH_RANGE:
            write_range_instr(scope, instr, out, opencl);
            break;
        case BH_RANDOM:
            write_random_instr(scope, instr, out, opencl);
            break;
        case BH_GATHER:
            write_gather_instr(scope, instr, out, opencl);
            break;
        case BH_SCATTER:
        case BH_COND_SCATTER:
            write_scatter_instr(scope, instr, out, opencl);
            break;
        default:
            if (bh_opcode_is_accumulate(instr.opcode)) {
                write_accumulate_instr(scope, instr, out, opencl);
            } else {
                write_other_instr(scope, instr, out, opencl);
            }
            break;
    }
}

bh_constant sweep_identity(bh_opcode opcode, bh_type dtype) {
    switch (opcode) {
        case BH_ADD_REDUCE:
        case BH_BITWISE_OR_REDUCE:
        case BH_BITWISE_XOR_REDUCE:
        case BH_LOGICAL_OR_REDUCE:
        case BH_LOGICAL_XOR_REDUCE:
        case BH_ADD_ACCUMULATE:
            return bh_constant(0, dtype);
        case BH_MULTIPLY_REDUCE:
        case BH_MULTIPLY_ACCUMULATE:
            return bh_constant(1, dtype);
        case BH_BITWISE_AND_REDUCE:
        case BH_LOGICAL_AND_REDUCE:
            return bh_constant(~0u, dtype);
        case BH_MAXIMUM_REDUCE:
            if (dtype == bh_type::BOOL) {
                return bh_constant(bh_bool{1});
            } else {
                return bh_constant::get_min(dtype);
            }
        case BH_MINIMUM_REDUCE:
            if (dtype == bh_type::BOOL) {
                return bh_constant(bh_bool{1});
            } else {
                return bh_constant::get_max(dtype);
            }
        default:
            cout << "sweep_identity(): unsupported operation: " << bh_opcode_text(opcode) << endl;
            assert(1 == 2);
            throw runtime_error("sweep_identity(): unsupported operation");
    }
}

void write_reduce_identity(bh_opcode opcode, bh_type dtype, stringstream &out) {
    switch (opcode) {
        case BH_ADD_REDUCE:
        case BH_BITWISE_OR_REDUCE:
        case BH_BITWISE_XOR_REDUCE:
        case BH_LOGICAL_OR_REDUCE:
        case BH_LOGICAL_XOR_REDUCE:
            out << "0";
            break;
        case BH_MULTIPLY_REDUCE:
            out << "1";
            break;
        case BH_BITWISE_AND_REDUCE:
        case BH_LOGICAL_AND_REDUCE:
            out << "~0";
            break;
        case BH_MAXIMUM_REDUCE:
            dtype_min(dtype, out);
            break;
        case BH_MINIMUM_REDUCE:
            dtype_max(dtype, out);
            break;
        default:
            cout << "write_reduce_identity: unsupported operation: " << bh_opcode_text(opcode) << endl;
            throw runtime_error("write_reduce_identity: unsupported operation");
    }
}

vector<bh_instruction*> remove_non_computed_system_instr(vector<bh_instruction> &instr_list, set<bh_base *> &frees) {
    vector<bh_instruction*> ret;
    set<const bh_base*> computes;
    for (bh_instruction &instr: instr_list) {
        if (instr.opcode == BH_FREE and not util::exist(computes, instr.operand[0].base)) {
            frees.insert(instr.operand[0].base);
        } else if (not (instr.opcode == BH_NONE or instr.opcode == BH_TALLY)) {
            set<const bh_base*> bases = instr.get_bases_const();
            computes.insert(bases.begin(), bases.end());
            ret.push_back(&instr);
        }
    }
    return ret;
}

InstrPtr reshape_rank(const InstrPtr &instr, int rank, int64_t size_of_rank_dim) {
    vector<int64_t> shape((size_t) rank + 1);
    // The dimensions up til 'rank' (not including 'rank') are unchanged
    for (int64_t r = 0; r < rank; ++r) {
        shape[r] = instr->operand[0].shape[r];
    }
    int64_t size = 1; // The size of the reshapeable block
    for (int64_t r = rank; r < instr->operand[0].ndim; ++r) {
        size *= instr->operand[0].shape[r];
    }
    assert(size >= size_of_rank_dim);
    shape[rank] = size_of_rank_dim;
    if (size != size_of_rank_dim) { // We might have to add an extra dimension
        if (size % size_of_rank_dim != 0) {
            throw runtime_error("reshape_rank(): shape is not divisible with 'size_of_rank_dim'");
        }
        shape.push_back(size / size_of_rank_dim);
    }
    bh_instruction ret = bh_instruction(*instr);
    ret.reshape(shape);
    return std::make_shared<bh_instruction>(ret);
}

} // jitk
} // bohrium
