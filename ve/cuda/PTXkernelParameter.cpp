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

#include <cstdio>
#include <stdexcept>
#include "PTXkernelParameter.hpp"

inline void PTXkernelParameter::declareOn(std::ostream& os) const
{
    os << ".param " << ptxTypeStr(type) << " kp" << id;
}

inline void PTXkernelParameter::printOn(std::ostream& os) const
{
    os << "kp" << id;
}

std::ostream& operator<<= (std::ostream& os, 
                                  PTXkernelParameter const& ptxKernelParameter)
{
    ptxKernelParameter.declareOn(os);
    return os;
}

PTXkernelParameter::PTXkernelParameter(PTXtype type_, 
                                            int id_) :
    type(type_),
    id(id_) {}

PTXtype PTXkernelParameter::getType()
{
    return type;
}
