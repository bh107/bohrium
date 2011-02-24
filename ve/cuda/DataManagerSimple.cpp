/*
 * Copyright 2011 Troels Blum <troels@blum.dk>
 *
 * This file is part of cphVB.
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
 * along with cphVB.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <cassert>
#include <queue>
#include <map>
#include <cphvb.h>
#include "DataManager.hpp"
#include "MemoryManager.hpp"
#include "InstructionBatch.hpp"
#include "WrapperFunctions.hpp"


typedef std::map<cphVBArray*, InstructionBatch*> WriteLockTable;
typedef std::map<cphVBArray* ,CUdeviceptr> ShadowArrayTable; // <Orig,Shadow>
typedef std::map<cphVBArray*, CUdeviceptr> Base2CudaMap;
typedef std::map<cphVBArray* ,cphVBArray*> Operand2BaseMap;



class DataManagerSimple : public DataManager
{
private:
    MemoryManager* memoryManager;
    WriteLockTable writeLockTable;
    ShadowArrayTable shadowArrayTable;
    Base2CudaMap base2Cuda;
    Operand2BaseMap op2Base;

    InstructionBatch* activeBatch;
    
    CUdeviceptr copyNlock(cphVBArray* operands[], 
                   int nops, 
                   InstructionBatch* batch)
    {
        /* First we need to call justLock, so any flushes needed are 
         * done before copying */
        justLock(operands, nops, batch);
        cphVBArray* baseArray = cphVBBaseArray(operands[0]);
        CUdeviceptr cudaPtr = memoryManager->deviceAlloc(baseArray);
        memoryManager->deviceCopy(cudaPtr, baseArray);
        shadowArrayTable[baseArray] = cudaPtr;
        return cudaPtr;
    }

    CUdeviceptr justLock(cphVBArray* operands[], 
                   int nops, 
                   InstructionBatch* batch)
    {
        assert(nops > 0);
        cphVBArray* baseArray;
        /* We need to _sync all arrays in the operation*/
        for (int i = 0; i < nops; ++i)
        {
           baseArray = op2Base[operands[i]];  
           _sync(baseArray);
        }
        /* Now we can just take the write lock on the array */
        baseArray = op2Base[operands[0]];
        CUdeviceptr resPtr = base2Cuda[baseArray];
        writeLockTable[baseArray] = batch;
        return resPtr;
    }

    void flush(InstructionBatch* batch)
    {
        batch->execute();
        ShadowArrayTable::iterator saiter = shadowArrayTable.begin();
        for (; saiter != shadowArrayTable.end(); ++saiter)
        {
            memoryManager->free(saiter->first);
            saiter->first->cudaPtr = saiter->second;
        }
        shadowArrayTable.clear();
    }

    void _sync(cphVBArray* baseArray)
    {
        WriteLockTable::iterator wliter = writeLockTable.find(baseArray);
        if (wliter != writeLockTable.end())
        {
            flush(activeBatch);
            activeBatch = NULL;
        }
    }

    void initCudaArray(cphVBArray* baseArray)
    {
        assert(baseArray->base == NULL);
        if (baseArray->data == NULL)
        {
            if (baseArray->hasInitValue)
            {
                memoryManager->memset(baseArray);
            }
            else  
            {   //Nothing to init
                return;
            }
        }
        else
        {
            memoryManager->copyToDevice(baseArray);
        }
    }

    /* Map operands to CUDA device pointers via base arrays.
     * Also updates cphVBArray with apropriate info.
     */
    void mapOperands(cphVBArray* operands[],
                int nops)
    {
        assert (nops > 0);
        Operand2BaseMap::iterator oiter;
        Base2CudaMap::iterator biter;
        cphVBArray* baseArray;
        cphVBArray* operand;
        for (int i = 0; i < nops; ++i)
        {
            operand = operands[i];
            oiter = op2Base.find(operand);
            if (oiter == op2Base.end())
            {   //The operand is not mapped to a base array - so we will 
                baseArray = cphVBBaseArray(operand);
                biter = base2Cuda.find(baseArray);
                if (biter == base2Cuda.end())
                {   //The base array is not mapped to a cudaPtr - so we will
                    //We also need to initialize it
                    CUdeviceptr cudaPtr = memoryManager->deviceAlloc(baseArray);
                    base2Cuda[baseArray] = cudaPtr;
                    baseArray->cudaPtr = cudaPtr;
                    initCudaArray(baseArray);
                }
                op2Base[operand] = baseArray;
            }
        }
    }
    
    bool internalConflict(cphVBArray* operands[], 
                          int nops)
    {
        for (int i = 1; i < nops; ++i)
        {
            if (op2Base[operands[0]] == op2Base[operands[i]])
            {
                return true;
            }
        }
        return false;
    }

public:
    DataManagerSimple(MemoryManager* memoryManager_) :
        memoryManager(memoryManager_),
        activeBatch(NULL) {}

    CUdeviceptr lock(cphVBArray* operands[], 
                     int nops, 
                     InstructionBatch* batch)
    {
        if (batch != activeBatch && activeBatch != NULL)
        {
            flush(activeBatch);
            activeBatch = batch;
        }

        CUdeviceptr resPtr;
        assert(nops > 0);
        mapOperands(operands, nops);
        
        if (internalConflict(operands, nops))
        {
            resPtr = copyNlock(operands, nops, batch);
        }
        else
        {
            resPtr = justLock(operands, nops, batch);
        }
        return resPtr;
    }

    void release(cphVBArray* baseArray)
    {
        sync(baseArray);
        discard(baseArray);
    }

    void sync(cphVBArray* baseArray)
    {
        assert(baseArray->base == NULL);
        _sync(baseArray);
        if (baseArray->data == NULL)
        {
            cphvb_data_ptr ptr = memoryManager->hostAlloc(baseArray);
            baseArray->data = ptr;
        }
        memoryManager->copyToHost(baseArray);
    }

    void discard(cphVBArray* baseArray)
    {
        assert(baseArray->base == NULL);
        memoryManager->free(baseArray);
        baseArray->cudaPtr = 0;
        base2Cuda.erase(baseArray);
        Operand2BaseMap::iterator oiter = op2Base.begin();
        while (oiter != op2Base.end())
        {
            if (oiter->second == baseArray)
            {
                op2Base.erase(oiter++);    
            }
            else
            {
                ++oiter;
            }
        }
    }

    void flushAll()
    {
        flush(activeBatch);
        activeBatch = NULL;
    }

};

