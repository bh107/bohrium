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
#include <bh.h>
#include "InstructionScheduler.hpp"
#include "UserFuncArg.hpp"
#include "Scalar.hpp"

InstructionScheduler::InstructionScheduler(ResourceManager* resourceManager_) 
    : resourceManager(resourceManager_) 
    , batch(0)
{}

bh_error InstructionScheduler::schedule(bh_intp instructionCount,
                                    bh_instruction* instructionList)
{
#ifdef DEBUG
    std::cout << "[VE GPU] InstructionScheduler: recieved batch with " << 
        instructionCount << " instructions." << std::endl;
#endif
    for (bh_intp i = 0; i < instructionCount; ++i)
    {
        bh_instruction* inst = instructionList++;
        if (inst->opcode != BH_NONE)
        {
#ifdef DEBUG
            bh_pprint_instr(inst);
#endif
			bh_error res;
			
            switch (inst->opcode)
            {
            case BH_SYNC:
                sync(inst->operand[0]);
                res = BH_SUCCESS;
                break;
            case BH_DISCARD:
                if (inst->operand[0]->base == NULL)
                    discard(inst->operand[0]);
                res = BH_SUCCESS;
                break;
            case BH_FREE:
                bh_data_free(inst->operand[0]);
                res = BH_SUCCESS;
                break;                
            case BH_USERFUNC:
                res = userdeffunc(inst->userfunc);
                break;
            case BH_ADD_REDUCE:
            case BH_MUL_REDUCE:
				bh_reduce_type reduce_data;
            	reduce_data.id = 0;
            	reduce_data.nout = 1;
            	reduce_data.nin = 1;
            	reduce_data.struct_size = sizeof(bh_reduce_type);
            	reduce_data.opcode = inst->opcode == BH_ADD_REDUCE ? BH_ADD : BH_MULTIPLY;
            	reduce_data.operand[0] = inst->operand[0];
            	reduce_data.operand[1] = inst->operand[1];
            	
	            if (inst->constant.type == BH_INT64) {
	            	reduce_data.axis = inst->constant.value.int64;
	            	res = UserFunctionReduce::reduce_impl((bh_userfunc *)&reduce_data, NULL);
	            }
	            else
	            	res = BH_TYPE_NOT_SUPPORTED;

            default:
                res = ufunc(inst);
                break;
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

void InstructionScheduler::sync(bh_array* base)
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

void InstructionScheduler::discard(bh_array* base)
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

bh_error InstructionScheduler::userdeffunc(bh_userfunc* userfunc)
{
    FunctionMap::iterator fit = functionMap.find(userfunc->id);
    if (fit == functionMap.end())
    {
        return BH_USERFUNC_NOT_SUPPORTED;
    }
    bh_intp nops = userfunc->nout + userfunc->nin;
    UserFuncArg userFuncArg;
    userFuncArg.resourceManager = resourceManager;
    for (int i = 0; i < nops; ++i)
    {
        bh_array* operand = userfunc->operand[i];
        if ((!resourceManager->float64support() && operand->type == BH_FLOAT64)
            || (!resourceManager->float16support() && operand->type == BH_FLOAT16))
        {
            return BH_TYPE_NOT_SUPPORTED;
        }
        bh_array* base = bh_base_array(operand);
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

bh_error InstructionScheduler::ufunc(bh_instruction* inst)
{
    //TODO Find out if we support the operation before copying data to device

    bh_intp nops = bh_operands(inst->opcode);
    assert(nops > 0);
    std::vector<KernelParameter*> operands(nops);
    for (int i = 0; i < nops; ++i)
    {
        bh_array* operand = inst->operand[i];
        if (bh_is_constant(operand))
        {
            operands[i] = new Scalar(inst->constant);
            continue;
        }
        if ((!resourceManager->float64support() && operand->type == BH_FLOAT64)
            || (!resourceManager->float16support() && operand->type == BH_FLOAT16))
        {
            return BH_TYPE_NOT_SUPPORTED;
        }
        bh_array* base = bh_base_array(operand);
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
    return BH_SUCCESS;
}

void InstructionScheduler::registerFunction(bh_intp id, bh_userfunc_impl userfunc)
{
    functionMap[id] = userfunc;
}
