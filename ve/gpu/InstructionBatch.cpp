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
#include <iostream>
#include <functional>
#include <algorithm>
#include <sstream>
#include <cassert>
#include <stdexcept>
#include <bh.h>
#include <bh_timing.hpp>
#include "InstructionBatch.hpp"
#include "GenerateSourceCode.hpp"
#include "Scalar.hpp"
#define MIN(a, b) ((a) < (b) ? (a) : (b))

InstructionBatch::KernelMap InstructionBatch::kernelMap = InstructionBatch::KernelMap();

InstructionBatch::InstructionBatch(bh_instruction* inst, const std::vector<KernelParameter*>& operands)
    : arraynum(0)
    , scalarnum(0)
    , float16(false)
    , float64(false)
    , complex(false)
{
#ifdef BH_TIMING
    createTime = bh::Timer<>::stamp();
#endif
    shape = std::vector<bh_index>(inst->operand[0].shape, inst->operand[0].shape + inst->operand[0].ndim);
    add(inst, operands);
}

bool InstructionBatch::shapeMatch(bh_intp ndim,const bh_index dims[])
{
    int size = shape.size();
    if (ndim == size)
    {
        for (int i = 0; i < ndim; ++i)
        {
            if (dims[i] != shape[i])
                return false;
        }
        return true;
    }
    return false;
}

void InstructionBatch::add(bh_instruction* inst, const std::vector<KernelParameter*>& operands)
{
    assert(!isScalar(operands[0]));

    // Check that the shape matches
    if (!shapeMatch(inst->operand[0].ndim, inst->operand[0].shape))
        throw BatchException(0);

    std::vector<int> opids(operands.size(),-1);
    std::vector<bool> load_store(operands.size(),true); //do we need to load/store variables
    for (size_t op = 0; op < operands.size(); ++op)
    {
        BaseArray* ba = dynamic_cast<BaseArray*>(operands[op]);
        if (ba) // not a scalar
        {
            // If any operand's base is already used as output, it has to be alligned or disjoint
            ArrayRange orange = output.equal_range(ba);
            for (ArrayMap::iterator oit = orange.first ; oit != orange.second; ++oit)
            {
                if (bh_view_identical(&views[oit->second], &inst->operand[op]))
                {
                    // Same view so we use the ID for it
                    opids[op] = oit->second;
                    load_store[op] = false; // Allready exists
                } 
                else if (!bh_view_disjoint(&views[oit->second], &inst->operand[op])) 
                { 
                    throw BatchException(0);
                }
            }
            // If the output operand is allready used as input it has to be alligned or disjoint
            ArrayRange irange = input.equal_range(ba);
            for (ArrayMap::iterator iit = irange.first ; iit != irange.second; ++iit)
            {
                if (bh_view_identical(&views[iit->second], &inst->operand[op]))
                {
                    // Same view so we use the same ID for it
                    opids[op] =  iit->second;
                    if (op > 0) // also input: no need to load again
                        load_store[op] = false;
                } 
                else if (op == 0 && !bh_view_disjoint(&views[iit->second], &inst->operand[0]))
                {
                    throw BatchException(0);
                }
            }
        }
    }


    // OK so we can accept the instruction
    for (size_t op = 0; op < operands.size(); ++op)
    {
        // Assign new view id's
        if (opids[op]<0 && inst->operand[op].base)
        {
            int i = op-1;
            for (; i >= 0; --i)
            {   //catch when same view is used twice within oparation and doesn't allready have an id
                if (inst->operand[op].base == inst->operand[i].base && 
                    bh_view_identical(&inst->operand[op], &inst->operand[i]))
                {
                    opids[op] = opids[i];
                    if (op > 0 && i > 0) // both input: no need to load twice
                        load_store[op] = false;
                    break;
                }
            }
            if (opids[op] < 0)
            {   // Unkwown view
                opids[op] = views.size();
                views.push_back(inst->operand[op]);
            }
        }
        
        // Register new parameters and input / output variables
        if (load_store[op])
        {
            KernelParameter* kp = operands[op];
            BaseArray* ba = dynamic_cast<BaseArray*>(kp);
            if (ba)
            {
                if (op == 0)
                {
                    output.insert(std::make_pair(ba, opids[op]));
                }
                else
                {
                    input.insert(std::make_pair(ba, opids[op]));
                }
                if (parameters.find(kp) == parameters.end())
                {
                    std::stringstream ss;
                    ss << "a" << arraynum++;
                    parameters[kp] = ss.str();
                }
            }
            else //scalar
            {
                std::stringstream ss;
                ss << "s" << scalarnum++;
                opids[op] = SCALAR_OFFSET+scalarnum;
                kernelVariables[opids[op]] = ss.str();
                parameters[kp] = ss.str();
            }
            switch (kp->type())
            {
            case OCL_FLOAT16:
                float16 = true;
                break;
            case OCL_FLOAT64:
                float64 = true;
                break;
            case OCL_COMPLEX64:
                complex = true;
                break;
            case OCL_COMPLEX128:
                float64 = true;
                complex = true;
                break;
            default:
                break;
            }
        }
    }
    instructions.push_back(make_pair(inst, opids));
}

Kernel InstructionBatch::generateKernel(ResourceManager* resourceManager)
{
#ifdef BH_TIMING
    bh_uint64 start = bh::Timer<>::stamp();
#endif
    std::string code = generateCode();
#ifdef BH_TIMING
    resourceManager->codeGen->add({start, bh::Timer<>::stamp()}); 
#endif
    size_t codeHash = string_hasher(code);

    KernelMap::iterator kit = kernelMap.find(codeHash);
    if (kit == kernelMap.end())
    {
        std::stringstream source, kname;
        kname << "kernel" << std::hex << codeHash;
        if (float16)
        {
            source << "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
        }
        if (float64)
        {
            source << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
        }
        if (complex)
        {
            source << "#include <complex.h>\n";
        }
        source << "__kernel void " << kname.str() << code;
        Kernel kernel(resourceManager, MIN(shape.size(),3), source.str(), kname.str());
        kernelMap.insert(std::make_pair(codeHash, kernel));
        return kernel;
    } else {
        return kit->second;
    }
}

void InstructionBatch::run(ResourceManager* resourceManager)
{
#ifdef BH_TIMING
    resourceManager->batchBuild->add({createTime, bh::Timer<>::stamp()}); 
#endif

    if (output.begin() != output.end())
    {
#ifndef STATIC_KERNEL
        for (int i = 0; i < shape.size(); ++i)
        {
            std::stringstream ss;
            ss << "ds" << shape.size() -(i+1);
            parameterList.insert(std::make_pair(ss.str(), new Scalar(shape[i])));
        }
#endif
        for (ArrayMap::iterator iit = input.begin(); iit != input.end(); ++iit)
        {
#ifndef STATIC_KERNEL
            {
                std::stringstream ss;
                ss << "v" << iit->second << "s0";
                parameterList.insert(std::make_pair(ss.str(), new Scalar(views[iit->second].start)));
            }
            bh_intp vndim = views[iit->second].ndim;
            for (bh_intp d = 0; d < vndim; ++d)
            {
                std::stringstream ss;
                ss << "v" << iit->second << "s" << vndim-d;
                parameterList.insert(std::make_pair(ss.str(), new Scalar(views[iit->second].stride[d])));
            }
#endif
            inputList.insert(std::make_pair(iit->second, iit->first));
        }
        for (ArrayMap::iterator oit = output.begin(); oit != output.end(); ++oit)
        {
#ifndef STATIC_KERNEL
            {
                std::stringstream ss;
                ss << "v" << oit->second << "s0";
                parameterList.insert(std::make_pair(ss.str(), new Scalar(views[oit->second].start)));
            }
            bh_intp vndim = views[oit->second].ndim;
            for (bh_intp d = 0; d < vndim; ++d)
            {
                std::stringstream ss;
                ss << "v" << oit->second << "s" << vndim-d;
                parameterList.insert(std::make_pair(ss.str(), new Scalar(views[oit->second].stride[d])));
            }
#endif
            outputList.insert(std::make_pair(oit->second, oit->first));
        }
        for (ParameterMap::iterator pit = parameters.begin(); pit != parameters.end(); ++pit)
            parameterList.insert(std::make_pair(pit->second, pit->first));
        Kernel kernel = generateKernel(resourceManager);
        Kernel::Parameters kernelParameters;
        for (ParameterList::iterator pit = parameterList.begin(); pit != parameterList.end(); ++pit)
        {
            if (output.find(dynamic_cast<BaseArray*>(pit->second)) == output.end())
                kernelParameters.push_back(std::make_pair(pit->second, false));
            else
                kernelParameters.push_back(std::make_pair(pit->second, true));
        }
        std::vector<size_t> globalShape;
        for (int i = shape.size()-1; i>=0 && shape.size()-i < 4; --i)
            globalShape.push_back(shape[i]);
        kernel.call(kernelParameters, globalShape);
    }
}

std::string InstructionBatch::generateCode()
{
    std::stringstream source;
    source << "( ";
    // Add kernel parameters
    ParameterList::iterator pit = parameterList.begin();
    source << *(pit->second) << " " << pit->first;
    for (++pit; pit != parameterList.end(); ++pit)
    {
        source << "\n                     , " << *(pit->second) << " " << pit->first;
    }

    source << ")\n{\n";
    
    generateGIDSource(shape, source);
    std::stringstream indentss;
    indentss << "\t";
    for (size_t d = shape.size()-1; d > 2; --d)
    {
#ifdef STATIC_KERNEL
        source << indentss.str() << "for (int ids" << d << " = 0; ids" << d << " < " << 
            shape[shape.size()-(d+1)] << "; ++ids" << d << ")\n" << indentss.str() << "{\n";
#else
        source << indentss.str() << "for (int ids" << d << " = 0; ids" << d << " < ds" << d << 
            "; ++ids" << d << ")\n" << indentss.str() << "{\n";
#endif
        indentss << "\t";
    }
    std::string indent = indentss.str();
    
    // Load input parameters
    for (ArrayList::iterator iit = inputList.begin(); iit != inputList.end(); ++iit)
    {
        std::stringstream ss;
        ss << "v" << iit->first;
        kernelVariables[iit->first] = ss.str();
        source << indent << oclTypeStr(iit->second->type()) << " " << ss.str() << " = " <<
            parameters[iit->second] << "[";
        generateOffsetSource(views, iit->first, source);
        source << "];\n";
    }

    // Generate code for instructions
    for (InstructionList::iterator iit = instructions.begin(); iit != instructions.end(); ++iit)
    {
        std::vector<std::string> operands;
        std::vector<OCLtype> types;
        // Has the output operand been assigned a variable name?
        VariableMap::iterator kvit = kernelVariables.find(iit->second[0]);
        if (kvit == kernelVariables.end())
        {
            std::stringstream ss;
            ss << "v" << iit->second[0];
            kernelVariables[(iit->second)[0]] = ss.str();
            operands.push_back(ss.str());
            source << indent << oclTypeStr(oclType(iit->first->operand[0].base->type)) << " " << ss.str() << ";\n";
        }
        else
        {
            operands.push_back(kvit->second);
        }
        // find variable names for input operands
        for (int op = 1; op < bh_operands(iit->first->opcode); ++op)
        {
            operands.push_back(kernelVariables[iit->second[op]]);
        }
        for (int op = 0; op < bh_operands(iit->first->opcode); ++op)
        {
            types.push_back(oclType(iit->first->operand[op].base ? 
                                    iit->first->operand[op].base->type : 
                                    iit->first->constant.type));
        } 
        // generate source code for the instruction
        generateInstructionSource(iit->first->opcode, types, operands, indent, source);
    }

    // Save output parameters
    for (ArrayList::iterator oit = outputList.begin(); oit != outputList.end(); ++oit)
    {
        source << indent << parameters[oit->second] << "[";
        generateOffsetSource(views, oit->first, source);
        source << "] = " <<  kernelVariables[oit->first] << ";\n";
    }
    for (size_t d = 3; d < shape.size(); ++d)
        source << indent.substr(d-2) << "}\n";
    source << "}\n";
    return source.str();
}

bool InstructionBatch::read(BaseArray* array)
{
    if (input.find(array) == input.end())
        return false;
    return true;
}

bool InstructionBatch::write(BaseArray* array)
{
    if (output.find(array) == output.end())
        return false;
    return true;
}

bool InstructionBatch::access(BaseArray* array)
{
    return (read(array) || write(array));
}

bool InstructionBatch::discard(BaseArray* array)
{
    output.erase(array);
    bool r =  read(array);
    if (!r)
    {
        parameters.erase(static_cast<KernelParameter*>(array));
    }
    return !r;
}
