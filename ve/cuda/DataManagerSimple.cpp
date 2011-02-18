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
typedef std::multimap<cphVBArray*, InstructionBatch*> ReadLockTable;
typedef std::map<cphVBArray* ,CUdeviceptr> ShadowArrayTable; // <Orig,Shadow>
typedef std::map<cphVBArray*, CUdeviceptr> Base2CudaMap;
typedef std::map<cphVBArray* ,cphVBArray*> Operand2BaseMap;



class DataManagerSimple : public DataManager
{
private:
    MemoryManager* memoryManager;
    WriteLockTable writeLockTable;
    ReadLockTable readLockTable;
    ShadowArrayTable shadowArrayTable;
    Base2CudaMap base2Cuda;
    Operand2BaseMap op2Base;
    
    CUdevicePtr copyNlock(cphVBArray* operands[], 
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

    CUdevicePtr justLock(cphVBArray* operands[], 
                   int nops, 
                   InstructionBatch* batch)
    {
        assert(nops > 0);
        WriteLockTable::iterator wliter;
        cphVBArray* baseArray;
        /* If any of the base arrays for the operators 
         * exist in the writeLockTable we need to flush 
         * operations to the array */
        for (int i = 0; i < nops; ++i)
        {
           baseArray = op2Base[operands[i]];  
            wliter = writeLockTable.find(baseArray);
            if (wliter != writeLockTable.end())
            {
                flush(baseArray);
            }
        }
        /* Now we can just take lock on the array */
        baseArray = op2Base[operands[0]];
        CUdevicePtr resPtr = base2Cuda[baseArray];
        writeLock(baseArray, batch);
        for (int i = 1; i < nops; ++i)
        {
            baseArray = op2Base[operands[i]];
            readLockTable.insert(std::pair<cphVBArray*, InstructionBatch*>
                                 (baseArray,batch));
        }
        return resPtr;
    }
    
    /* Take a write lock while upgrading from read lock if necessary*/
    void writeLock(cphVBArray*baseArray, 
                   InstructionBatch* batch)
    {
        readLockTable.erase(baseArray);
        writeLockTable[baseArray] = batch;
    }

    void run(InstructionBatch* batch)
    {
        batch->execute();
        ShadowArrayTable::iterator saiter = 
shadowArrayTable;        
    }

    void flush(cphVBArray* baseArray)
    {
        /* First we flush readers*/
        ReadLockTable::iterator rliter;
        std::pair<ReadLockTable::iterator, ReadLockTable::iterator> rlrange =
            readLockTable.equal_range(baseArray);
        for (rliter = rlrange.first; rliter != rlrange.second;)
        {
            run(rliter->second);
            readLockTable.erase(rliter++);
        }

        /* And the we flush the writer*/
        WriteLockTable::iterator wliter;
        wliter = writeLockTable.find(baseArray);
        if (wliter != writeLockTable.end())
        {
            run(wliter->second);
            writeLockTable.erase(wliter);
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
                    CUdeviceptr cudaPtr = memoryManager->deviceAlloc(baseArray);
                    base2Cuda[baseArray] = cudaPtr;
                    baseArray->cudaPtr = cudaPtr;
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
        memoryManager(memoryManager_)
    {}

    CUdeviceptr lock(cphVBArray* operands[], 
                     int nops, 
                     InstructionBatch* batch)
    {
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
        flush(baseArray);
        if (baseArray->data == NULL)
        {
            cphvb_data_ptr ptr = memoryManager->hostAlloc(baseArray);
            baseArray->data = ptr;
        }
        memoryManager->copyToHost(baseArray);
    }

    void discard(cphVBArray* baseArray)
    {
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
        /* First we flush readers*/
        ReadLockTable::iterator rliter = readLockTable.begin();
        for (; rliter != readLockTable.end(); ++rliter)
        {
            run(rliter->second);
        }
        readLockTable.clear();

        /* And the we flush writers*/
        WriteLockTable::iterator wliter = writeLockTable.begin();
        for (; wliter != writeLockTable.end(); ++wliter)
        {
            run(wliter->second);
        }
        writeLockTable.clear();
    }

};

