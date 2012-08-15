/*
This file is part of cphVB and copyright (c) 2012 the cphVB team:
http://cphvb.bitbucket.org

cphVB is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as 
published by the Free Software Foundation, either version 3 
of the License, or (at your option) any later version.

cphVB is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the 
GNU Lesser General Public License along with cphVB. 

If not, see <http://www.gnu.org/licenses/>.
*/

#include <iostream>
#include <cassert>
#include <stdexcept>
#include <cphvb.h>
#include "InstructionScheduler.hpp"
#include "UserFuncArg.hpp"
#include "Scalar.hpp"

InstructionScheduler::InstructionScheduler(ResourceManager* resourceManager_) 
    : resourceManager(resourceManager_) 
    , batch(0)
{}

cphvb_error InstructionScheduler::schedule(cphvb_intp instructionCount,
                                    cphvb_instruction* instructionList)
{
#ifdef DEBUG
    std::cout << "[VE GPU] InstructionScheduler: recieved batch with " << 
        instructionCount << " instructions." << std::endl;
#endif
    for (cphvb_intp i = 0; i < instructionCount; ++i)
    {
        cphvb_instruction* inst = instructionList++;
        if (inst->opcode != CPHVB_NONE && inst->status != CPHVB_SUCCESS)
        {
#ifdef DEBUG
            cphvb_pprint_instr(inst);
#endif
            switch (inst->opcode)
            {
            case CPHVB_SYNC:
                sync(inst->operand[0]);
                inst->status = CPHVB_SUCCESS;
                break;
            case CPHVB_DISCARD:
                if (inst->operand[0]->base == NULL)
                    discard(inst->operand[0]);
                inst->status = CPHVB_SUCCESS;
                break;
            case CPHVB_FREE:
                cphvb_data_free(inst->operand[0]);
                inst->status = CPHVB_SUCCESS;
                break;                
            case CPHVB_USERFUNC:
                inst->status = userdeffunc(inst->userfunc);
                break;
            default:
                inst->status = ufunc(inst);
                break;
            }
            if (inst->status != CPHVB_SUCCESS)
            {
                executeBatch();
                return CPHVB_PARTIAL_SUCCESS;
            }
        }
    }
    
    /* End of batch cleanup */
    executeBatch();
    return CPHVB_SUCCESS;
}

void InstructionScheduler::executeBatch()
{
    if (batch)
    {
        batch->run(resourceManager);
        for (std::set<BaseArray*>::iterator dsit = discardSet.begin(); dsit != discardSet.end(); ++dsit)
        {
            delete *dsit;
        }
        discardSet.clear();
        delete batch;
        batch = 0;
    }
}

void InstructionScheduler::sync(cphvb_array* base)
{
    //TODO postpone sync
    assert(base->base == NULL);
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
    it->second->sync();
}

void InstructionScheduler::discard(cphvb_array* base)
{
    assert(base->base == NULL);
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
        delete it->second;
    }
    arrayMap.erase(it);
}

cphvb_error InstructionScheduler::userdeffunc(cphvb_userfunc* userfunc)
{
    FunctionMap::iterator fit = functionMap.find(userfunc->id);
    if (fit == functionMap.end())
    {
        return CPHVB_USERFUNC_NOT_SUPPORTED;
    }
    cphvb_intp nops = userfunc->nout + userfunc->nin;
    UserFuncArg userFuncArg;
    userFuncArg.resourceManager = resourceManager;
    for (int i = 0; i < nops; ++i)
    {
        cphvb_array* operand = userfunc->operand[i];
        if ((!resourceManager->float64support() && operand->type == CPHVB_FLOAT64)
            || (!resourceManager->float16support() && operand->type == CPHVB_FLOAT16))
        {
            return CPHVB_TYPE_NOT_SUPPORTED;
        }
        cphvb_array* base = cphvb_base_array(operand);
        // Is it a new base array we haven't heard of before?
        ArrayMap::iterator it = arrayMap.find(base);
        if (it == arrayMap.end())
        {
            // Then create it
            BaseArray* ba =  new BaseArray(base, resourceManager);
            arrayMap[base] = ba;
            userFuncArg.operands.push_back(ba);
        }
        else
        {
            userFuncArg.operands.push_back(it->second);
        }
    }

    // If the instruction batch accesses any of the output operands it need to be executed first
    for (int i = 0; i < userfunc->nout; ++i)
    {
        if (batch && batch->access(static_cast<BaseArray*>(userFuncArg.operands[i])))
        {
            executeBatch();
        }
    }
    // If the instruction batch writes to any of the input operands it need to be executed first
    for (int i = userfunc->nout; i < nops; ++i)
    {
        if (batch && batch->write(static_cast<BaseArray*>(userFuncArg.operands[i])))
        {
            executeBatch();
        }
    }

    // Execute the userdefined function
    return fit->second(userfunc, &userFuncArg);
}

cphvb_error InstructionScheduler::ufunc(cphvb_instruction* inst)
{
    //TODO Find out if we support the operation before copying data to device

    cphvb_intp nops = cphvb_operands(inst->opcode);
    assert(nops > 0);
    std::vector<KernelParameter*> operands(nops);
    for (int i = 0; i < nops; ++i)
    {
        cphvb_array* operand = inst->operand[i];
        if (cphvb_is_constant(operand))
        {
            operands[i] = new Scalar(inst->constant);
            continue;
        }
        if ((!resourceManager->float64support() && operand->type == CPHVB_FLOAT64)
            || (!resourceManager->float16support() && operand->type == CPHVB_FLOAT16))
        {
            return CPHVB_TYPE_NOT_SUPPORTED;
        }
        cphvb_array* base = cphvb_base_array(operand);
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
    if (batch)
    {
        try 
        {
            batch->add(inst, operands);
        } 
        catch (BatchException& be)
        {
            executeBatch();
            batch = new InstructionBatch(inst, operands);
        } 
    }
    else
    {
        batch = new InstructionBatch(inst, operands);
    }
    return CPHVB_SUCCESS;
}

void InstructionScheduler::registerFunction(cphvb_intp id, cphvb_userfunc_impl userfunc)
{
    functionMap[id] = userfunc;
}
