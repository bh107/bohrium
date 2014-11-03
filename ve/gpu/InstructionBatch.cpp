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
#include "bh_ve_gpu.h"
#include "InstructionBatch.hpp"
#include "GenerateSourceCode.hpp"
#include "Scalar.hpp"
#include "StringHasher.hpp"
#define MIN(a, b) ((a) < (b) ? (a) : (b))

InstructionBatch::InstructionBatch(bh_instruction* inst, const std::vector<KernelParameter*>& operands)
    : arraynum(0)
    , scalarnum(0)
    , float64(false)
    , complex(false)
    , integer(false)
    , random(false)
{
#ifdef BH_TIMING
    createTime = bh::Timer<>::stamp();
#endif
    shape = std::vector<bh_index>(inst->operand[0].shape, inst->operand[0].shape + inst->operand[0].ndim);
    add(inst, operands);
}

bool InstructionBatch::shapeMatch(bh_intp ndim,const bh_index dims[])
{
    bh_intp size = shape.size();
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
    typedef std::pair<ArrayMap::iterator, ArrayMap::iterator> ArrayRange;

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
            // If any operand's base is already used as output, it has to be aligned or disjoint
            ArrayRange orange = output.equal_range(ba);
            for (ArrayMap::iterator oit = orange.first ; oit != orange.second; ++oit)
            {
                if (bh_view_identical(&views[oit->second], &inst->operand[op]))
                {
                    // Same view so we use the ID for it
                    opids[op] = oit->second;
                    load_store[op] = false; // Already exists
                } 
                else if (!bh_view_disjoint(&views[oit->second], &inst->operand[op])) 
                { 
                    throw BatchException(0);
                }
            }
            // If the output operand is already used as input it has to be aligned or disjoint
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
            case OCL_R123:
                random = true;
                break;
            case OCL_INT8:
            case OCL_INT16:
            case OCL_INT32:
            case OCL_INT64:
            case OCL_UINT8:
            case OCL_UINT16:
            case OCL_UINT32:
            case OCL_UINT64:
                if (inst->opcode == BH_POWER)
                    integer = true;
                break;
            default:
                break;
            }
        }
    }
    instructions.push_back(make_pair(inst, opids));
}

SourceKernelCall InstructionBatch::generateKernel()
{
#ifdef BH_TIMING
    resourceManager->batchBuild->add({createTime, bh::Timer<>::stamp()}); 
    bh_uint64 start = bh::Timer<>::stamp();
#endif
    // If there is no output we just skip execution
    if (output.begin() == output.end())
    {
        throw BatchException(1);
    }
    std::vector<KernelParameter*> sizeParameters;
    std::stringstream defines;
    std::stringstream functionDeclaration;
    
    functionDeclaration << "(";
    Kernel::Parameters kernelParameters;
    typedef std::map<std::string, KernelParameter*> ParameterList; 
    ParameterList parameterList;
    for (ParameterMap::iterator pit = parameters.begin(); pit != parameters.end(); ++pit)
        parameterList.insert(std::make_pair(pit->second, pit->first));
    for (ParameterList::iterator pit = parameterList.begin(); pit != parameterList.end(); ++pit)
    {
        functionDeclaration << "\n\t" << (pit==parameterList.begin()?"  ":", ") << *pit->second << " " << 
            pit->first;
        if (output.find(dynamic_cast<BaseArray*>(pit->second)) == output.end())
            kernelParameters.push_back(std::make_pair(pit->second, false));
         else
             kernelParameters.push_back(std::make_pair(pit->second, true));
    }

    functionDeclaration << "\n#ifndef FIXED_SIZE";
    
    for (size_t i = 0; i < shape.size(); ++i)
    {
        std::stringstream ss;
        ss << "ds" << shape.size() -(i+1);
        Scalar* s = new Scalar(shape[i]);
        (defines << "#define " << ss.str() << " " <<= *s) << "\n";
        sizeParameters.push_back(s);
        functionDeclaration << "\n\t, " << *s << " " << ss.str();
    }

    ViewList inputList, outputList, viewList;
    for (ArrayMap::iterator iit = input.begin(); iit != input.end(); ++iit)
    {
        inputList.insert(std::make_pair(iit->second, iit->first));
        viewList.insert(std::make_pair(iit->second, iit->first));
    }
    for (ArrayMap::iterator oit = output.begin(); oit != output.end(); ++oit)
    {
        outputList.insert(std::make_pair(oit->second, oit->first));
        viewList.insert(std::make_pair(oit->second, oit->first));
    }
    for (ViewList::iterator vit = viewList.begin(); vit != viewList.end(); ++vit)
    {
        {
            std::stringstream ss;
            ss << "v" << vit->first << "s0";
            Scalar* s = new Scalar(views[vit->first].start);
            (defines << "#define " << ss.str() << " " <<= *s) << "\n";
            sizeParameters.push_back(s);
            functionDeclaration << "\n\t, " << *s << " " << ss.str();
        }
        bh_intp vndim = views[vit->first].ndim;
        for (bh_intp d = 0; d < vndim; ++d)
        {
            std::stringstream ss;
            ss << "v" << vit->first << "s" << vndim-d;
            Scalar* s = new Scalar(views[vit->first].stride[d]);
            (defines << "#define " << ss.str() << " " <<= *s) << "\n";
            sizeParameters.push_back(s);
            functionDeclaration << "\n\t, " << *s << " " << ss.str();
        }
    }
    functionDeclaration << "\n#endif\n)\n";
    
    std::string functionBody = generateFunctionBody(inputList, outputList);
    size_t functionID = string_hasher(functionBody);
    size_t literalID = string_hasher(defines.str());
    
    std::vector<size_t> kernelShape;
    for (int i = shape.size()-1; i>=0 && shape.size()-i < 4; --i)
        kernelShape.push_back(shape[i]);
    
    std::stringstream source;
    if (float64)
        source << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
    if (complex)
        source << "#include <ocl_complex.h>\n";
    if (integer)
        source << "#include <ocl_integer.h>\n";
    if (random)
        source << "#include <ocl_random.h>\n";
    source << "#ifdef FIXED_SIZE\n" << defines.str() << "#endif\n" << 
        "__kernel void\n#ifndef FIXED_SIZE\nkernel" << std::hex << functionID <<
        "\n#else\nkernel" << std::hex << functionID << "_\n#endif\n" << functionDeclaration.str() << 
        "\n" << functionBody;
    
#ifdef BH_TIMING
    resourceManager->codeGen->add({start, bh::Timer<>::stamp()}); 
#endif
    return SourceKernelCall(KernelID(functionID, literalID), kernelShape,source.str(),
                            sizeParameters, kernelParameters);
    
}

std::string InstructionBatch::generateFunctionBody(ViewList inputList, ViewList outputList)
{
    std::stringstream source;
    source << "{\n";
    
    generateGIDSource(shape.size(), source);
    std::stringstream indentss;
    indentss << "\t";
    for (size_t d = shape.size()-1; d > 2; --d)
    {
        source << indentss.str() << "for (int ids" << d << " = 0; ids" << d << " < ds" << d << 
            "; ++ids" << d << ")\n" << indentss.str() << "{\n";
        indentss << "\t";
    }
    std::string indent = indentss.str();
    
    // Load input parameters
    for (ViewList::iterator iit = inputList.begin(); iit != inputList.end(); ++iit)
    {
        std::stringstream ss;
        ss << "v" << iit->first;
        kernelVariables[iit->first] = ss.str();
        source << indent << oclTypeStr(iit->second->type()) << " " << ss.str() << " = " <<
            parameters[iit->second] << "[";
        generateOffsetSource(views[iit->first].ndim, iit->first, source);
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
        // HACK to make BH_INVERT on BH_BOOL work correctly TODO Fix!
        if (iit->first->opcode == BH_INVERT && (iit->first->operand[1].base ? 
                                                iit->first->operand[1].base->type : 
                                                iit->first->constant.type) == BH_BOOL)
            generateInstructionSource(BH_LOGICAL_NOT, types, operands, indent, source);
        else
            generateInstructionSource(iit->first->opcode, types, operands, indent, source);
    }

    // Save output parameters
    for (ViewList::iterator oit = outputList.begin(); oit != outputList.end(); ++oit)
    {
        source << indent << parameters[oit->second] << "[";
        generateOffsetSource(views[oit->first].ndim, oit->first, source);
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
