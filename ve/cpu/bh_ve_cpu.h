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
#ifndef __BH_VE_CPU_H
#define __BH_VE_CPU_H

#include <bh.h>

typedef enum OPERATION {
    EWISE       = 1,
    REDUCE      = 2,
    SCAN        = 4,
    GENERATOR   = 8,
    SYSTEM      = 16,
    EXTENSION   = 32,
} OPERATION;

typedef enum OPERATOR {
    // Used by elementwise, reduce and scan operations
    ABSOLUTE,
    ARCCOS,
    ARCCOSH,
    ARCSIN,
    ARCSINH,
    ARCTAN,
    ARCTANH,
    CEIL,
    COS,
    COSH,
    EXP,
    EXP2,
    EXPM1,
    FLOOR,
    IDENTITY,
    IMAG,
    INVERT,
    ISINF,
    ISNAN,
    LOG,
    LOG10,
    LOG1P,
    LOG2,
    LOGICAL_NOT,
    REAL,
    RINT,
    SIN,
    SINH,
    SQRT,
    TAN,
    TANH,
    TRUNC,

    // Used by elementwise, reduce and scan operations
    ADD,
    ARCTAN2,
    BITWISE_AND,
    BITWISE_OR,
    BITWISE_XOR,
    DIVIDE,
    EQUAL,
    GREATER,
    GREATER_EQUAL,
    LEFT_SHIFT,
    LESS,
    LESS_EQUAL,
    LOGICAL_AND,
    LOGICAL_OR,
    LOGICAL_XOR,
    MAXIMUM,
    MINIMUM,
    MOD,
    MULTIPLY,
    NOT_EQUAL,
    POWER,
    RIGHT_SHIFT,
    SUBTRACT,

    // Used by system operations
    DISCARD,
    FREE,
    NONE,
    SYNC,

    // Used by generator operations
    FLOOD,
    RANDOM,
    RANGE,

    NBUILTIN,   // Not an operator but a count of built-in operators
    EXT_OFFSET  // Wildcard for extension opcodes

} OPERATOR;

typedef struct bytecode {
    OPERATION op;       // Operation
    OPERATOR  oper;     // Operator
    uint32_t  out;      // Output operand
    uint32_t  in1;      // First input operand
    uint32_t  in2;      // Second input operand
} bytecode_t;

//
// NOTE: Changes to bk_kernel_args_t must be 
//       replicated to "templates/kernel.tpl".
//
typedef struct bh_kernel_arg {
    void*   data;       // Pointer to memory allocated for the array
    int64_t start;      // Offset from memory allocation to start of array
    int64_t nelem;      // Number of elements available in the allocation

    int64_t ndim;       // Number of dimensions of the array
    int64_t* shape;     // Shape of the array
    int64_t* stride;    // Stride in each dimension of the array
} bh_kernel_arg_t;      // Meta-data for a kernel argument

// Layoutmask
#define A0_CONSTANT     (1 << 0)
#define A0_CONTIGUOUS   (1 << 1)
#define A0_STRIDED      (1 << 2)
#define A0_SPARSE       (1 << 3)

#define A1_CONSTANT     (1 << 4)
#define A1_CONTIGUOUS   (1 << 5)
#define A1_STRIDED      (1 << 6)
#define A1_SPARSE       (1 << 7)

#define A2_CONSTANT     (1 << 8)
#define A2_CONTIGUOUS   (1 << 9)
#define A2_STRIDED      (1 << 10)
#define A2_SPARSE       (1 << 11)

#ifdef __cplusplus
extern "C" {
#endif

/* Component interface: init (see bh_component.h) */
DLLEXPORT bh_error bh_ve_cpu_init(const char *name);

/* Component interface: execute (see bh_component.h) */
DLLEXPORT bh_error bh_ve_cpu_execute(bh_ir* bhir);

/* Component interface: shutdown (see bh_component.h) */
DLLEXPORT bh_error bh_ve_cpu_shutdown(void);

/* Component interface: extmethod (see bh_component.h) */
DLLEXPORT bh_error bh_ve_cpu_extmethod(const char *name, bh_opcode opcode);

#ifdef __cplusplus
}
#endif

#endif
