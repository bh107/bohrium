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

// Single-Expression-Jit hash: OPCODE_NDIM_LAYOUT_TYPESIG
#define A0_CONSTANT (1 << 0)
#define A0_DENSE    (1 << 1)
#define A0_STRIDED  (1 << 2)
#define A0_SPARSE   (1 << 3)

#define A1_CONSTANT (1 << 4)
#define A1_DENSE    (1 << 5)
#define A1_STRIDED  (1 << 6)
#define A1_SPARSE   (1 << 7)

#define A2_CONSTANT (1 << 8)
#define A2_DENSE    (1 << 9)
#define A2_STRIDED  (1 << 10)
#define A2_SPARSE   (1 << 11)

#define A0_BOOL         (1<<0)
#define A0_INT8         (1<<1)
#define A0_INT16        (1<<2)
#define A0_INT32        (1<<3)
#define A0_INT64        (1<<4)
#define A0_UINT8        (1<<5)
#define A0_UINT16       (1<<6)
#define A0_UINT32       (1<<7)
#define A0_UINT64       (1<<8)
#define A0_FLOAT16      (1<<9)
#define A0_FLOAT32      (1<<10)
#define A0_FLOAT64      (1<<11)
#define A0_COMPLEX64    (1<<12)
#define A0_COMPLEX128   (1<<13)
#define A0_UNKNOWN      (1<<14)

#define A1_BOOL         (1<<15)
#define A1_INT8         (1<<16)
#define A1_INT16        (1<<17)
#define A1_INT32        (1<<18)
#define A1_INT64        (1<<19)
#define A1_UINT8        (1<<20)
#define A1_UINT16       (1<<21)
#define A1_UINT32       (1<<22)
#define A1_UINT64       (1<<23)
#define A1_FLOAT16      (1<<24)
#define A1_FLOAT32      (1<<25)
#define A1_FLOAT64      (1<<26)
#define A1_COMPLEX64    (1<<27)
#define A1_COMPLEX128   (1<<28)
#define A1_UNKNOWN      (1<<29)

#define A2_BOOL         (1<<30)
#define A2_INT8         (1<<31)
#define A2_INT16        (1<<32)
#define A2_INT32        (1<<33)
#define A2_INT64        (1<<34)
#define A2_UINT8        (1<<35)
#define A2_UINT16       (1<<36)
#define A2_UINT32       (1<<37)
#define A2_UINT64       (1<<38)
#define A2_FLOAT16      (1<<39)
#define A2_FLOAT32      (1<<40)
#define A2_FLOAT64      (1<<41)
#define A2_COMPLEX64    (1<<42)
#define A2_COMPLEX128   (1<<43)
#define A2_UNKNOWN      (1<<44)

#ifdef __cplusplus
extern "C" {
#endif

DLLEXPORT bh_error bh_ve_cpu_init(bh_component *self);

DLLEXPORT bh_error bh_ve_cpu_execute(bh_ir* bhir);

DLLEXPORT bh_error bh_ve_cpu_shutdown(void);

DLLEXPORT bh_error bh_ve_cpu_reg_func(char *fun, bh_intp *id);

#ifdef __cplusplus
}
#endif

#endif
