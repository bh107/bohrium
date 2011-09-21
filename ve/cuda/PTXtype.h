/*
 * Copyright 2011 Troels Blum <troels@blum.dk>
 *
 * This file is part of cphVB <http://code.google.com/p/cphvb/>.
 *
 * cphVB is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * cphVB is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with cphVB. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef __PTXTYPE_HPP
#define __PTXTYPE_HPP

#include <cphvb.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum
{
    PTX_INT,
    PTX_UINT,
    PTX_FLOAT,
    PTX_BITS,
    PTX_BASE_TYPES //Number of base types 
} PTXbaseType;

typedef enum
{
    PTX_INT8,
    PTX_INT16,
    PTX_INT32,
    PTX_INT64,
    PTX_UINT8,
    PTX_UINT16,
    PTX_UINT32,
    PTX_UINT64,
    PTX_FLOAT16,
    PTX_FLOAT32,
    PTX_FLOAT64,
    PTX_BITS8,
    PTX_BITS16,
    PTX_BITS32,
    PTX_BITS64,
    PTX_PRED,
    PTX_TYPES //Number of types 
} PTXtype;
#ifdef __LP64__
#define PTX_POINTER PTX_UINT64
#else
#define PTX_POINTER PTX_UINT32
#endif

PTXtype ptxType(cphvb_type vbtype);
PTXbaseType ptxBaseType(PTXtype type);
const char* ptxTypeStr(PTXtype type);
int ptxTypeSize(PTXtype type);
const char* ptxWideOpStr(PTXtype type);
size_t ptxAlign(PTXtype type);
size_t ptxSizeOf(PTXtype type);
const char* ptxRegPrefix(PTXtype type);

#ifdef __cplusplus
}
#endif

#endif
