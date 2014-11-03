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
#include <cassert>
#include <stdexcept>
#include <thread>
#include <chrono>
#include <sstream>
#include <bh.h>
#include "InstructionScheduler.hpp"
#include "UserFuncArg.hpp"
#include "Scalar.hpp"
#include "Reduce.hpp"

InstructionScheduler::InstructionScheduler()
    : batch(NULL) {}

bh_error InstructionScheduler::schedule(bh_ir* bhir)
{
    for (bh_intp i = 0; i < bhir->ninstr; ++i)
    {
        bh_instruction* inst = &(bhir->instr_list[i]);
        if (inst->opcode != BH_NONE)
        {
#ifdef DEBUG
            bh_pprint_instr(inst);
#endif
			bh_error res;

            switch (inst->opcode)
            {
            case BH_SYNC:
                sync(inst->operand[0].base);
                res = BH_SUCCESS;
                break;
            case BH_DISCARD:
                discard(inst->operand[0].base);
                res = BH_SUCCESS;
                break;
            case BH_FREE:
                bh_data_free(inst->operand[0].base);
                res = BH_SUCCESS;
                break;
            case BH_ADD_REDUCE:
            case BH_MULTIPLY_REDUCE:
            case BH_MINIMUM_REDUCE:
            case BH_MAXIMUM_REDUCE:
            case BH_LOGICAL_AND_REDUCE:
            case BH_BITWISE_AND_REDUCE:
            case BH_LOGICAL_OR_REDUCE:
            case BH_BITWISE_OR_REDUCE:
            case BH_LOGICAL_XOR_REDUCE:
            case BH_BITWISE_XOR_REDUCE:
            case BH_ADD_ACCUMULATE:
            case BH_MULTIPLY_ACCUMULATE:
                res = reduce(inst);
                break;
            default:
                if (inst->opcode <= BH_MAX_OPCODE_ID)
                    res = ufunc(inst);
                else
                    res = extmethod(inst);
            }

            if (res != BH_SUCCESS)
            {
            	return res;
            }
        }
    }

    /* End of batch cleanup */
    executeBatch();
    return BH_SUCCESS;
}

void InstructionScheduler::compileAndRun(SourceKernelCall sourceKernel)
{
    size_t functionID = sourceKernel.id().first;
    std::map<size_t,size_t>::iterator kidit = knownKernelID.find(functionID);
    if (kidit != knownKernelID.end() && (resourceManager->dynamicSizeKernel() ||
                                         kidit->second == sourceKernel.literalID()))
    {   /*
         * We know the functionID, and if we are only building fixed size kernels 
         * we also know the literalID
         */
        KernelID kernelID(sourceKernel.functionID(),0); 
        if (kidit->second == sourceKernel.literalID())
        {
            kernelID.second = kidit->second;
            assert(resourceManager->fixedSizeKernel());
        } else {
            assert(resourceManager->dynamicSizeKernel());
        }
        kernelMutex.lock();
        if (callQueue.empty())
        {
            KernelMap::iterator kit = kernelMap.find(kernelID);
            if (kit == kernelMap.end())
            {
                callQueue.push_back(std::make_pair(kernelID,sourceKernel));
            } else {
                if (kernelID.second == 0)
                {
                    kit->second.call(sourceKernel.allParameters(), sourceKernel.shape());
                    sourceKernel.deleteBuffers();
                } else {
                    kit->second.call(sourceKernel.valueParameters(), sourceKernel.shape());
                    sourceKernel.deleteBuffers();
                }
            }
        } else {
            callQueue.push_back(std::make_pair(kernelID,sourceKernel));
        }
        kernelMutex.unlock();
    } else { // New Kernel
        KernelID kernelID = sourceKernel.id();
        if (!resourceManager->fixedSizeKernel())
            kernelID.second = 0;
        knownKernelID.insert(kernelID);
        kernelMutex.lock();
        callQueue.push_back(std::make_pair(kernelID,sourceKernel));
        kernelMutex.unlock();
        if (resourceManager->asyncCompile())
        {
            if (resourceManager->fixedSizeKernel())
                std::thread(&InstructionScheduler::build, this, sourceKernel.id(), 
                            sourceKernel.source()).detach();
            if (resourceManager->dynamicSizeKernel())
                std::thread(&InstructionScheduler::build, this, KernelID(functionID,0), 
                            sourceKernel.source()).detach();
        } else {
            if (resourceManager->fixedSizeKernel())
                build(sourceKernel.id(), sourceKernel.source());
            if (resourceManager->dynamicSizeKernel())
                build(KernelID(functionID,0), sourceKernel.source());
        }
    }        
}

void InstructionScheduler::executeBatch()
{
    if (batch)
    {
        try {
            SourceKernelCall sourceKernel = batch->generateKernel();
            sourceKernel.setDiscard(discardSet);
            compileAndRun(sourceKernel);
        }
        catch(BatchException be) 
        {
            if (!discardSet.empty())
            {
                kernelMutex.lock();
                if (!callQueue.empty())
                {
                    SourceKernelCall& sourceKernel = callQueue.back().second;
                    for (BaseArray *ba: discardSet)
                    {
                        sourceKernel.addDiscard(ba);
                    }
                    kernelMutex.unlock();  
                }
                else {
                    kernelMutex.unlock();  
                    for (BaseArray *ba: discardSet)
                    {
                        delete ba;
                    }
                }
            }
        }
        discardSet.clear();
        delete batch;
        batch = 0;
    }
}


void InstructionScheduler::build(KernelID kernelID, const std::string source)
{

    std::stringstream kname;
    kname << "kernel" <<  std::hex << kernelID.first << (kernelID.second==0?"":"_");
    Kernel kernel(source, kname.str(), (kernelID.second==0?"":"-DFIXED_SIZE"));
    kernelMutex.lock();
    kernelMap.insert(std::make_pair(kernelID, kernel));
    while (!callQueue.empty())
    {
        KernelCall kernelCall = callQueue.front();
        KernelMap::iterator kit = kernelMap.find(kernelCall.first);
        if (kit != kernelMap.end())
        {
            kit->second.call((kernelCall.first.second==0?
                              kernelCall.second.allParameters():
                              kernelCall.second.valueParameters()),
                             kernelCall.second.shape());
            kernelCall.second.deleteBuffers();
            callQueue.pop_front();
        }
        else
            break;            
    }
    kernelMutex.unlock(); 
}

void InstructionScheduler::sync(bh_base* base)
{
    //TODO postpone sync
    // We may recieve sync for arrays I don't own
    ArrayMap::iterator it = arrayMap.find(base);
    if  (it == arrayMap.end())
    {
        return;
    }
    if (batch && batch->write(it->second))
    {
        executeBatch();
    }
    while (!callQueue.empty())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    it->second->sync();
}

void InstructionScheduler::discard(bh_base* base)
{
    // We may recieve discard for arrays I don't own
    ArrayMap::iterator it = arrayMap.find(base);
    if  (it == arrayMap.end())
    {
        return;
    }
    if (batch && !batch->discard(it->second))
    {
        discardSet.insert(it->second);
    }
    else
    {
        kernelMutex.lock();
        if (!callQueue.empty())
        {
            callQueue.back().second.addDiscard(it->second);
            kernelMutex.unlock();  
        }
        else {
            kernelMutex.unlock();  
            delete it->second;
        }
    }
    arrayMap.erase(it);
}

std::vector<KernelParameter*> InstructionScheduler::getKernelParameters(bh_instruction* inst)
{
    bh_intp nops = bh_operands(inst->opcode);
    assert(nops > 0);
    std::vector<KernelParameter*> operands(nops);
    for (int i = 0; i < nops; ++i)
    {
        if (bh_is_constant(&(inst->operand[i])))
        {
            operands[i] = new Scalar(inst->constant);
            continue;
        }
        bh_base* base = inst->operand[i].base;
        if (!resourceManager->float64support() && base->type == BH_FLOAT64)
        {
            throw BH_TYPE_NOT_SUPPORTED;
        }
        // Is it a new base array we haven't heard of before?
        ArrayMap::iterator it = arrayMap.find(base);
        if (it == arrayMap.end())
        {
            // Then create it
            BaseArray* ba =  new BaseArray(base, resourceManager);
            arrayMap[base] = ba;
            operands[i] = ba;
        }
        else
        {
            operands[i] = it->second;
        }
    }
    return operands;
}

bh_error InstructionScheduler::ufunc(bh_instruction* inst)
{
    try {
        std::vector<KernelParameter*> operands = getKernelParameters(inst);
        if (batch)
        {
            try {
                batch->add(inst, operands);
            }
            catch (BatchException& be)
            {
                executeBatch();
                batch = new InstructionBatch(inst, operands);
            }
        } else {
            batch = new InstructionBatch(inst, operands);
        }
        return BH_SUCCESS;
    }
    catch (bh_error e)
    {
        return e;
    }
}

bh_error InstructionScheduler::reduce(bh_instruction* inst)
{
    if(inst->operand[1].ndim < 2)
    {
        // TODO these two syncs are a hack. Are we sure this is correct?????
        sync(inst->operand[1].base);
        sync(inst->operand[0].base);
        
        bh_ir bhir;
        bh_error err = bh_ir_create(&bhir, 1, inst);
        if(err != BH_SUCCESS)
            return err;
        return resourceManager->childExecute(&bhir);
    }
    std::vector<KernelParameter*> operands = getKernelParameters(inst);
    if (batch && (batch->access(static_cast<BaseArray*>(operands[0])) ||
                  batch->write(static_cast<BaseArray*>(operands[1]))))
    {
        executeBatch();
    }
    try {
        SourceKernelCall sourceKernel = Reduce::generateKernel(inst, operands);
        compileAndRun(sourceKernel);
    }
    catch (bh_error e)
    {
        return e;
    }
    return BH_SUCCESS;
}

void InstructionScheduler::registerFunction(bh_opcode opcode, bh_extmethod_impl extmethod_impl)
{
    if(functionMap.find(opcode) != functionMap.end())
    {
        std::cerr << "[GPU-VE] Warning, multiple registrations of the same extension method: " <<
            opcode << std::endl;
    }
    functionMap[opcode] = extmethod_impl;
}

bh_error InstructionScheduler::extmethod(bh_instruction* inst)
{
    FunctionMap::iterator fit = functionMap.find(inst->opcode);
    if (fit == functionMap.end())
    {
        return BH_EXTMETHOD_NOT_SUPPORTED;
    }

    try {
        UserFuncArg userFuncArg;
        userFuncArg.resourceManager = resourceManager;
        userFuncArg.operands = getKernelParameters(inst);

        // If the instruction batch accesses any of the output operands it need to be executed first
        BaseArray* ba = dynamic_cast<BaseArray*>(userFuncArg.operands[0]);
        if (batch &&  ba && batch->access(ba))
        {
            executeBatch();
        }
        // If the instruction batch writes to any of the input operands it need to be executed first
        for (int i = 1; i < bh_operands(inst->opcode); ++i)
        {
            BaseArray* ba = dynamic_cast<BaseArray*>(userFuncArg.operands[i]);
            if (batch && ba && batch->write(ba))
            {
                executeBatch();
            }
        }

        // Execute the extension method
        return fit->second(inst, &userFuncArg);
    }
    catch (bh_error e)
    {
        return e;
    }
}
