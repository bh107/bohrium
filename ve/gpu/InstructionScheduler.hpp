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


#ifndef __INSTRUCTIONSCHEDULER_HPP
#define __INSTRUCTIONSCHEDULER_HPP

#include <map>
#include <set>
#include <deque>
#include <mutex>
#include <bh.h>
#include "ResourceManager.hpp"
#include "SourceKernelCall.hpp"

class InstructionScheduler
{
private:
    typedef std::map<bh_base*, BaseArray*> ArrayMap;
    typedef std::map<bh_opcode, bh_extmethod_impl> FunctionMap;

    typedef std::map<KernelID, Kernel> KernelMap;
    typedef std::pair<KernelID, SourceKernelCall> KernelCall;
    typedef std::deque<KernelCall> CallQueue;
    typedef std::map<size_t, BaseArray*> ViewList;

    std::mutex kernelMutex;
    std::map<size_t,size_t> knownKernelID;
    KernelMap kernelMap;
    CallQueue callQueue;

    ArrayMap arrayMap;
    FunctionMap functionMap;
    void compileAndRun(SourceKernelCall sourceKernel);
    void build(KernelID id, const std::string source);
    bh_error extmethod(bh_instruction* inst);
    bh_error call_child(const bh_ir_kernel& kernel);
    SourceKernelCall generateKernel(const bh_ir_kernel& kernel);
    std::string generateFunctionBody(const bh_ir_kernel& kernel, const size_t kdims,
                                     const std::vector<bh_index>& shape,    
                                     const std::vector<std::vector<size_t> >& dimOrders,
                                     bool& float64, bool& complex, bool& integer, bool& random);
    void sync(const std::set<bh_base*>& arrays);
    void discard(const std::set<bh_base*>& arrays);
    void beginDim(std::stringstream& source, 
                  std::stringstream& indentss, 
                  std::vector<std::string>& beforesource, 
                  const size_t dims);
    void endDim(std::stringstream& source, 
                std::stringstream& indentss, 
                std::vector<std::string>& beforesource, 
                std::set<bh_view>& save,
                const size_t dims,
                const size_t kdims,
                const bh_ir_kernel& kernel);
    std::vector<std::vector<size_t> > genDimOrders(const std::map<bh_intp, bh_int64>& sweeps, size_t ndim);
public:
    InstructionScheduler() {}
    void registerFunction(bh_opcode opcode, bh_extmethod_impl extmothod);
    bh_error schedule(const bh_ir* bhir);
};

#endif
