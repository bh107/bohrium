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
#include "SourceKernelCall.hpp"

#define SCALAR_OFFSET (1<<16)
class InstructionBatch
{
    typedef std::map<KernelParameter*, std::string> ParameterMap;
    typedef std::map<unsigned int, std::string> VariableMap;
    typedef std::multimap<BaseArray*, unsigned int> ArrayMap;
    typedef std::map<unsigned int, BaseArray*> ViewList;
    typedef std::list<std::pair<bh_instruction*, std::vector<int> > > InstructionList;
    typedef std::map<std::string, std::string> ConstantList;
    
private:
    std::vector<bh_index> shape;
    InstructionList instructions;
    std::vector<bh_view> views;
    ArrayMap output;
    ArrayMap input;
    ParameterMap parameters;
    ConstantList constantList;
    VariableMap kernelVariables;
    int arraynum;
    int scalarnum;
    bool float64;
    bool complex;
    bool integer;
    bool random;
#ifdef BH_TIMING
    bh_uint64 createTime;
#endif
    bool shapeMatch(bh_intp ndim, const bh_index dims[]);
    std::string generateFunctionBody(ViewList inputList, ViewList outputList);

public:
    InstructionBatch(bh_instruction* inst, const std::vector<KernelParameter*>& operands);
    SourceKernelCall generateKernel();
    void add(bh_instruction* inst, const std::vector<KernelParameter*>& operands);
    bool read(BaseArray* array);
    bool write(BaseArray* array);    
    bool access(BaseArray* array);
    bool discard(BaseArray* array);
    static void CL_CALLBACK buildDone(cl_program p, void* id);
};

class BatchException 
{
public:
    int code;
    BatchException(int code_) : code(code_) {}
};


#endif
