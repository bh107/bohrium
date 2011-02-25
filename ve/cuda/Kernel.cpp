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

#include <vector>
#include <cassert>
#include <cuda.h>
#include "PTXtype.hpp"
#include "PTXparameter.hpp"
#include "Kernel.hpp"

#define ALIGN_UP(offset, alignment) \
	(offset) = ((offset) + (alignment) - 1) & ~((alignment) - 1)

void Kernel::setParameters(ParameterList parameters)
{
    int offset = 0;
    assert (parameters.size() == signature.size());
    Signature::iterator siter = signature.begin();
    ParameterList::iterator piter = parameters.begin();
    while (piter != parameters.end())
    {
        assert (piter->type == *siter);
        ALIGN_UP(offset, ptxAlign(piter->type));
        cuParamSetv(entry, offset, &piter->value, ptxSizeOf(piter->type));
        offset += ptxSizeOf(piter->type);
        ++piter; ++siter;
    }
    cuParamSetSize(entry,offset);
}
