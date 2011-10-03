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

#ifndef __ARRAYOPERAND_HPP
#define __ARRAYOPERAND_HPP

#include <cphvb_array.h>
#include <CL/cl.hpp>
#include "OCLtype.h"

class ArrayOperand
{
private:
protected:
    cphvb_array* spec;
    void printOn(std::ostream& os) const;
    ArrayOperand(cphvb_array* spec);
public:
    size_t size();
    friend std::ostream& operator<< (std::ostream& os, 
                                     ArrayOperand const& array);
    virtual OCLtype type() = 0;
};


#endif
