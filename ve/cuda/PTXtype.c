/*
 * Copyright 2011 Troels Blum <troels@blum.dk>
 *
 * This file is part of cphVB.
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

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <cphvb.h>
#include "PTXtype.h"

PTXtype ptxType(cphvb_type vbtype)
{
    switch (vbtype)
    {
    case CPHVB_BOOL:
        return PTX_UINT8;
    case CPHVB_INT8:
        return PTX_INT8;
    case CPHVB_INT16:
        return PTX_INT16;
    case CPHVB_INT32:
        return PTX_INT32;
    case CPHVB_INT64:
        return PTX_INT64;
    case CPHVB_UINT8:
        return PTX_UINT8;
    case CPHVB_UINT16:
        return PTX_UINT16;
    case CPHVB_UINT32:
        return PTX_UINT32;
    case CPHVB_UINT64:
        return PTX_UINT64;
    case CPHVB_FLOAT16:
        return PTX_FLOAT16;
    case CPHVB_FLOAT32:
        return PTX_FLOAT32;
    case CPHVB_FLOAT64:
        return PTX_FLOAT64;
    default:
        assert(false);
    }
}

PTXbaseType ptxBaseType(PTXtype type)
{
    switch (type)
    {
    case PTX_INT8:
    case PTX_INT16:
    case PTX_INT32:
    case PTX_INT64:
        return PTX_INT;
    case PTX_UINT8:
    case PTX_UINT16:
    case PTX_UINT32:
    case PTX_UINT64:
        return PTX_UINT;
    case PTX_FLOAT16:
    case PTX_FLOAT32:
    case PTX_FLOAT64:
        return PTX_FLOAT;
    default:
        assert(false);
    }
}

const char* _ptxTypeStr[] =
{
    [PTX_INT8] = ".s8",
    [PTX_INT16] = ".s16",
    [PTX_INT32] = ".s32",
    [PTX_INT64] = ".s64",
    [PTX_UINT8] = ".u8",
    [PTX_UINT16] = ".u16",
    [PTX_UINT32] = ".u32",
    [PTX_UINT64] = ".u64",
    [PTX_FLOAT16] = ".f16",
    [PTX_FLOAT32] = ".f32",
    [PTX_FLOAT64] = ".f64",
    [PTX_BITS8] = ".b8",
    [PTX_BITS16] = ".b16",
    [PTX_BITS32] = ".b32",
    [PTX_BITS64] = ".b64",
    [PTX_PRED] = ".pred"
};

const char* ptxTypeStr(PTXtype type)
{
    return _ptxTypeStr[type];
}

int _ptxTypeSize[] =
{
    [PTX_INT8] = 1,
    [PTX_INT16] = 2,
    [PTX_INT32] = 4,
    [PTX_INT64] = 8,
    [PTX_UINT8] = 1,
    [PTX_UINT16] = 2,
    [PTX_UINT32] = 4,
    [PTX_UINT64] = 8,
    [PTX_FLOAT16] = 2,
    [PTX_FLOAT32] = 4,
    [PTX_FLOAT64] = 8,
    [PTX_BITS8] = 1,
    [PTX_BITS16] = 2,
    [PTX_BITS32] = 4,
    [PTX_BITS64] = 8,
    [PTX_PRED] = -1
};

int ptxTypeSize(PTXtype type)
{
    return _ptxTypeSize[type];
}

const char* ptxWideOpStr(PTXtype type)
{
    switch (type)
    {
    case PTX_INT32:
        return ".s16";
        break;
    case PTX_INT64:
        return ".s32";
        break;
    case PTX_UINT32:
        return ".u16";
        break;
    case PTX_UINT64:
        return ".u32";
        break;
    default:
        assert (false);
    }
}

size_t ptxAlign(PTXtype type)
{
    switch (type)
    {
    case PTX_INT8:
        return __alignof(int8_t);
    case PTX_INT16:
        return __alignof(int16_t);
    case PTX_INT32:
        return __alignof(int32_t);
    case PTX_INT64:
        return __alignof(int64_t);
    case PTX_UINT8:
        return __alignof(uint8_t);
    case PTX_UINT16:
        return __alignof(uint16_t);
    case PTX_UINT32:
        return __alignof(uint32_t);
    case PTX_UINT64:
        return __alignof(uint64_t);
    case PTX_FLOAT32:
        return __alignof(float);
    case PTX_FLOAT64:
        return __alignof(double);
    default:
        assert(false);
    }
}

size_t ptxSizeOf(PTXtype type)
{
    switch (type)
    {
    case PTX_INT8:
        return sizeof(int8_t);
    case PTX_INT16:
        return sizeof(int16_t);
    case PTX_INT32:
        return sizeof(int32_t);
    case PTX_INT64:
        return sizeof(int64_t);
    case PTX_UINT8:
        return sizeof(uint8_t);
    case PTX_UINT16:
        return sizeof(uint16_t);
    case PTX_UINT32:
        return sizeof(uint32_t);
    case PTX_UINT64:
        return sizeof(uint64_t);
    case PTX_FLOAT32:
        return sizeof(float);
    case PTX_FLOAT64:
        return sizeof(double);
    default:
        assert(false);
    }
}

const char* _ptxRegPrefix[] =
{
    [PTX_INT8] = "$sc",
    [PTX_INT16] = "$sh",
    [PTX_INT32] = "$si",
    [PTX_INT64] = "$sd",
    [PTX_UINT8] = "$uc",
    [PTX_UINT16] = "$uh",
    [PTX_UINT32] = "$ui",
    [PTX_UINT64] = "$ud",
    [PTX_FLOAT16] = "$fh",
    [PTX_FLOAT32] = "$f_",
    [PTX_FLOAT64] = "$fd",
    [PTX_BITS8] = "$bc",
    [PTX_BITS16] = "$bh",
    [PTX_BITS32] = "$b_",
    [PTX_BITS64] = "$bd",
    [PTX_PRED] = "$p"
};

const char* ptxRegPrefix(PTXtype type)
{
    return _ptxRegPrefix[type];
}
