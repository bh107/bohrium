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
#include "PTXoperand.hpp"

PTXtype PTXoperand::ptxType(cphvb_type vbtype)
{
    switch (vbtype)
    {
    case CPHVB_BOOL:
        return PTX_BITS;
    case CPHVB_INT8:
    case CPHVB_INT16:
    case CPHVB_INT32:
    case CPHVB_INT64:
        return PTX_INT;
    case CPHVB_UINT8:
    case CPHVB_UINT16:
    case CPHVB_UINT32:
    case CPHVB_UINT64:
        return PTX_UINT;
    case CPHVB_FLOAT16:
    case CPHVB_FLOAT32:
    case CPHVB_FLOAT64:
        return PTX_FLOAT;
    default:
        assert(false);
    }
}


