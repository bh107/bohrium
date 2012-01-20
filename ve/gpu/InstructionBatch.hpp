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

#ifndef __INSTRUCTIONBATCH_HPP
#define __INSTRUCTIONBATCH_HPP

#include <vector>
#include <map>
#include <set>
#include <cphvb.h>
#include "BaseArray.hpp"

class InstructionBatch
{
private:
    std::vector<cphvb_index> shape;
    std::vector<cphvb_instruction*> instructions;
    std::map<BaseArray*, cphvb_array*> output;
    std::multimap<BaseArray*, cphvb_array*> input;
public:
    bool match(cphvb_intp ndim, const cphvb_index dims[]);
    bool sameView(const cphvb_array* a, const cphvb_array* b);
};

#endif
