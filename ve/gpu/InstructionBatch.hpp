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

typedef std::map<cphvb_array*, std::pair<BaseArray*,std::string>> ParamMap;
typedef std::map<BaseArray*, cphvb_array*> OutputMap;
typedef std::multimap<BaseArray*, cphvb_array*> InputMap;


class InstructionBatch
{
private:
    std::vector<cphvb_index> shape;
    std::vector<cphvb_instruction*> instructions;
    OutputMap output;
    InputMap input;
    ParamMap kernelParameters; 
    int arraynum = 0;
    static int kernel = 0;
public:
    bool match(cphvb_intp ndim, const cphvb_index dims[]);
    bool sameView(const cphvb_array* a, const cphvb_array* b);
};

#endif
