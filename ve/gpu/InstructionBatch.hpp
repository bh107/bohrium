/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium
team <http://www.bh107.org>.

Bohrium is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as 
published by the Free Software Foundation, either version 3 
of the License, or (at your option) any later version.

Bohrium is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the 
GNU Lesser General Public License along with Bohrium. 

If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __INSTRUCTIONBATCH_HPP
#define __INSTRUCTIONBATCH_HPP

#include <vector>
#include <map>
#include <list>
#include <bh.h>
#include "BaseArray.hpp"
#include "Kernel.hpp"
#include "StringHasher.hpp"

#ifdef STATS
#include "timing.h"
#endif

#define SCALAR_OFFSET (1<<16)
class InstructionBatch
{
    typedef std::map<KernelParameter*, std::string> ParameterMap;
    typedef std::map<unsigned int, std::string> VariableMap;
    typedef std::multimap<BaseArray*, unsigned int> ArrayMap;
    typedef std::pair<ArrayMap::iterator, ArrayMap::iterator> ArrayRange;
    typedef std::map<size_t, Kernel> KernelMap;
    typedef std::list<std::pair<bh_instruction*, std::vector<int>>> InstructionList;
    //typedef std::list<KernelParameter*> ParameterList;
    //typedef std::list<std::pair<BaseArray*, bh_base*> > ArrayList;
private:
    std::vector<bh_index> shape;
    InstructionList instructions;
    std::vector<bh_view> views;
    ArrayMap output;
    ArrayMap input;
    ParameterMap parameters;
    //ParameterList parameterList;
    //ArrayList outputList;
    //ArrayList inputList;
    VariableMap kernelVariables;
    int arraynum;
    int scalarnum;
    //int variablenum;
    bool float16;
    bool float64;
    static KernelMap kernelMap;
#ifdef STATS
    timeval createTime;
#endif
    bool shapeMatch(bh_intp ndim, const bh_index dims[]);
    bool sameView(const bh_view& a, const bh_view& b);
    bool disjointView(const bh_view& a, const bh_view& b);
    std::string generateCode();
public:
    InstructionBatch(bh_instruction* inst, const std::vector<KernelParameter*>& operands);
    Kernel generateKernel(ResourceManager* resourceManager);
    void run(ResourceManager* resourceManager);
    void add(bh_instruction* inst, const std::vector<KernelParameter*>& operands);
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
