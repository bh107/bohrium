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
#include <list>
#include <cphvb.h>
#include "BaseArray.hpp"
#include "Kernel.hpp"
#include "StringHasher.hpp"

#ifdef STATS
#include "timing.h"
#endif
class InstructionBatch
{
    typedef std::map<KernelParameter*, std::string> ParameterMap;
    typedef std::map<void*, std::string> VariableMap;
    typedef std::multimap<BaseArray*, cphvb_array*> ArrayMap;
    typedef std::pair<ArrayMap::iterator, ArrayMap::iterator> ArrayRange;
    typedef std::map<size_t, Kernel> KernelMap;
    typedef std::list<KernelParameter*> ParameterList;
    typedef std::list<std::pair<BaseArray*, cphvb_array*> > ArrayList;
private:
    std::vector<cphvb_index> shape;
    std::vector<cphvb_instruction*> instructions;
    ArrayMap output;
    ArrayMap input;
    ParameterMap parameters;
    ParameterList parameterList;
    ArrayList outputList;
    ArrayList inputList;
    VariableMap kernelVariables;
    int arraynum;
    int scalarnum;
    int variablenum;
    bool float16;
    bool float64;
    static KernelMap kernelMap;
#ifdef STATS
    timeval createTime;
#endif
    bool shapeMatch(cphvb_intp ndim, const cphvb_index dims[]);
    bool sameView(const cphvb_array* a, const cphvb_array* b);
    bool disjointView(const cphvb_array* a, const cphvb_array* b);
    std::string generateCode();
public:
    InstructionBatch(cphvb_instruction* inst, const std::vector<KernelParameter*>& operands);
    Kernel generateKernel(ResourceManager* resourceManager);
    void run(ResourceManager* resourceManager);
    void add(cphvb_instruction* inst, const std::vector<KernelParameter*>& operands);
    bool read(BaseArray* array);
    bool write(BaseArray* array);    
    bool access(BaseArray* array);
    bool discard(BaseArray* array);
};

class BatchException 
{
private:
    int code;
public:
    BatchException(int code_) : code(code_) {}
};


#endif
