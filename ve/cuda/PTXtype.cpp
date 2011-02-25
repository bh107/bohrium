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
 * along with cphVB.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <cassert>
#include <cphvb.h>
#include "PTXtype.hpp"

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

PTXbaseType ptxBaseType(cphvb_type vbtype)
{
    return(ptxBaseType(ptxType(vbtype)));
}

const char* _ptxTypeStr[] =
{
    /*[PTX_INT8] = */".s8",
    /*[PTX_INT16] = */".s16",
    /*[PTX_INT32] = */".s32",
    /*[PTX_INT64] = */".s64",
    /*[PTX_UINT8] = */".u8",
    /*[PTX_UINT16] = */".u16",
    /*[PTX_UINT32] = */".u32",
    /*[PTX_UINT64] = */".u64",
    /*[PTX_FLOAT16] = */".f16",
    /*[PTX_FLOAT32] = */".f32",
    /*[PTX_FLOAT64] = */".f64",
    /*[PTX_BITS8] = */".b8",
    /*[PTX_BITS16] = */".b16",
    /*[PTX_BITS32] = */".b32",
    /*[PTX_BITS64] = */".b64",
    /*[PTX_PRED] = */".pred"
};

const char* ptxTypeStr(PTXtype type)
{
    return _ptxTypeStr[type];
}
