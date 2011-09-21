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

#ifndef __KERNELGENERATOR_HPP
#define __KERNELGENERATOR_HPP

#include <vector>
#include "cphVBinstruction.hpp"

typedef unsigned long int Threads;
typedef std::vector<cphVBinstruction*>::iterator InstructionIterator;


class KernelGenerator
{
public:
    virtual void run(Threads threads,
                     InstructionIterator first,
                     InstructionIterator last) = 0;
};

#endif
