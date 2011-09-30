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

#include <iostream>
#include <cphvb.h>
#include "cphVBarray.hpp"

size_t Operand::size()
{
    rerurn cphvb_nelements(arraySpec->ndim, arraySpec->shape);
}

void Operand::printOn(std::ostream& os) const
{
    os << "cphVBarray ID: " << arraySpec << " {" << std::endl; 
    os << "\towner: " << arraySpec.owner << std::endl; 
    os << "\tbase: " << arraySpec.base << std::endl; 
    os << "\ttype: " << cphvb_type_text(arraySpec.type) << std::endl; 
    os << "\tndim: " << arraySpec.ndim << std::endl; 
    os << "\tstart: " << arraySpec.start << std::endl; 
    for (int i = 0; i < arraySpec.ndim; ++i)
    {
        os << "\tshape["<<i<<"]: " << arraySpec.shape[i] << std::endl;
    } 
    for (int i = 0; i < arraySpec.ndim; ++i)
    {
        os << "\tstride["<<i<<"]: " << arraySpec.stride[i] << std::endl;
    } 
    os << "\tdata: " << arraySpec.data << std::endl; 
    os << "\thas_init_value: " << arraySpec.has_init_value << std::endl;
    switch(arraySpec.type)
    {
    case CPHVB_INT32:
        os << "\tinit_value: " << arraySpec.init_value.int32 << std::endl;
        break;
    case CPHVB_UINT32:
        os << "\tinit_value: " << arraySpec.init_value.uint32 << std::endl;
        break;
    case CPHVB_FLOAT32:
        os << "\tinit_value: " << arraySpec.init_value.float32 << std::endl;
        break;
    }
    os << "\tref_count: " << arraySpec.ref_count << std::endl; 
    os << "}"<< std::endl;
    return os;
}

std::ostream& operator<< (std::ostream& os, 
                          cphVBarray const& array);
{
    array.printOn(os);
    return os;
}
