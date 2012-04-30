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
#ifdef STATS
#include <sys/time.h>
#endif
class InstructionBatch
{
    typedef std::map<KernelParameter*, std::string> ParameterMap;
    typedef std::map<void*, std::string> VariableMap;
    typedef std::map<BaseArray*, cphvb_array*> OutputMap;
    typedef std::multimap<BaseArray*, cphvb_array*> InputMap;
    typedef std::map<size_t, Kernel> KernelMap;
    typedef std::list<KernelParameter*> ParameterList;
    typedef std::list<BaseArray*> OutputList;
    typedef std::list<std::pair<BaseArray*, cphvb_array*>> InputList;
private:
    std::vector<cphvb_index> shape;
    std::vector<cphvb_instruction*> instructions;
    OutputMap output;
    InputMap input;
    ParameterMap parameters;
    ParameterList parameterList;
    OutputList outputList;
    InputList inputList;
    VariableMap kernelVariables;
    int arraynum;
    int scalarnum;
    int variablenum;
    static std::hash<std::string> strHash;
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
