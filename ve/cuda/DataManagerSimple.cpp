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

#import "DataManagerSimple.hpp"
#include <cassert>

typedef struct
{
    InstructionBatch* writer;
    std::queue<InstructionBatch*> readers;
} LockHolders;

typedef std::map<cphvb_array*, LockHolders> LockTable;
typedef std::map<cphvb_array*, CUdeviceptr> ArrayMap;

class DataManagerSimple : public DataManager
{
private:
    LockTable lockTable;
    ArrayMap  arrayMap;
    
    void copyNlock(cphvb_array* baseArrays, 
                   int nops, 
                   InstructionBatch* batch)
    {
        //TODO
    }

    void justLock(cphvb_array* baseArrays, 
                   int nops, 
                   InstructionBatch* batch)
    {
        //TODO
    }

    mapArrays(cphVBArray* operands,
              cphvb_array* baseArrays,
              int nops)
    {
        ArrayMap::iterator iter;
        for (int i = 0; i < nops, ++i)
        {
            iter = arrayMap.find(baseArray[i]);
            if (iter != my_map.end())
            {
                operands[i]->cudaPtr = iter->second;
            }
            else
            {
                CUdeviceptr cudaPtr = memoryManager.deviceAlloc(baseArrays[i]);
                arrayMap[baseArrays[i]] = cudaPtr;
                operands[i]->cudaPtr = cudaPtr;
            }
        }
    }

public:
    DataManagerSimple() {}
    void lock(cphVBArray* operands, 
                      int nops, 
                      InstructionBatch* batch)
    {
        assert(nops > 0);
        cphvb_array baseArrays[CPHVB_MAX_NO_OPERANDS];
        baseArrays[0] = cphvb_base_array(operands[0]);
        bool internalConflict = false;
        for (int i = 1; i < nops; ++i)
        {
            baseArrays[i] = cphvb_baseArray(operands[i]);
            if (baseArrays[0] == baseArrays[i])
            {
                internalConflict = true;
            }
        }

        mapArrays(operands, baseArrays, nops);
        
        if (internalConflict)
        {
            copyNlock(baseArrays, nops, batch);
        }
        else
        {
            justLock(base_arrays, nops, batch);
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

    
}
#endif

