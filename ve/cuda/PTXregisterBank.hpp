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

#ifndef __PTXREGISTERBANK_HPP
#define __PTXREGISTERBANK_HPP

#include <cphvb.h>
#include <StaticStack.hpp>
#include "PTXregister.hpp"
#include "PTXspecialRegister.hpp"

class PTXregisterBank
{
private:
    StaticStack<PTXregister>* registers;
    int instanceTable[PTX_TYPES];
protected:
    void declareOn(std::ostream& os) const;
public:
    PTXregisterBank();
    void clear();
    PTXregister* next(PTXtype type);
    PTXregister* next(cphvb_type type);
    friend std::ostream& operator<<= (std::ostream& os, 
                                      PTXregisterBank const& ptxRegisterBank);
    // Special registers
    PTXspecialRegister tid_x;
    PTXspecialRegister ntid_x;
    PTXspecialRegister ctaid_x;
};

#endif
