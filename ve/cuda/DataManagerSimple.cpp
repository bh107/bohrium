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

struct LockHolders
{
    InstructionBatch* writer;
    std::queue<InstructionBatch*> readers;
};

typedef std::map<cphVBArray*, LockHolders> LockTable;
typedef std::map<cphVBArray*, CUdeviceptr> Base2CudaMap;
typedef std::map<cphVBArray* ,cphVBArray*> Operand2BaseMap;


class DataManagerSimple : public DataManager
{
private:
    MemoryManager* memoryManager;
    LockTable lockTable;
    Base2CudaMap base2Cuda;
    Operand2BaseMap op2Base;
    
    void copyNlock(cphVBArray* baseArrays[], 
                   int nops, 
                   InstructionBatch* batch)
    {
        //TODO
    }

    void justLock(cphVBArray* baseArrays[], 
                   int nops, 
                   InstructionBatch* batch)
    {
        //TODO
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
            { 
                baseArray = cphVBBaseArray(operand);
                biter = base2Cuda.find(baseArray);
                if (biter == base2Cuda.end())
                {
                    CUdeviceptr cudaPtr = memoryManager->deviceAlloc(baseArray);
                    base2Cuda[baseArray] = cudaPtr;
                    //setCudaStride(baseArray);
                }
                op2Base[operand] = baseArray;
                if (operand != baseArray)
                {
                    //setCudaStride(operand);
                }
            }
        }
    }

public:
    DataManagerSimple(MemoryManager* memoryManager_) :
        memoryManager(memoryManager_)
    {}
    void lock(cphVBArray* operands[], 
              int nops, 
              InstructionBatch* batch)
    {
        assert(nops > 0);
        cphVBArray* baseArrays[CPHVB_MAX_NO_OPERANDS];
        baseArrays[0] = cphVBBaseArray(operands[0]);
        bool internalConflict = false;
        for (int i = 1; i < nops; ++i)
        {
            baseArrays[i] = cphVBBaseArray(operands[i]);
            if (baseArrays[0] == baseArrays[i])
            {
                internalConflict = true;
            }
        }

        mapOperands(operands, nops);
        
        if (internalConflict)
        {
            copyNlock(baseArrays, nops, batch);
        }
        else
        {
            justLock(baseArrays, nops, batch);
        }
    }

    void release(cphVBArray* array)
    {
        //TODO
    }

    void sync(cphVBArray* array)
    {
        //TODO
    }

    void flushAll()
    {
        //TODO
    }

    
};

