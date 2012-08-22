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
#include <sstream>
#include <cassert>
#include <stdexcept>
#include "UserFunctionLu.hpp"
#include "Scalar.hpp"

UserFunctionLu* userFunctionLu = NULL;

cphvb_error cphvb_lu(cphvb_userfunc* arg, void* ve_arg)
{
    cphvb_lu_type* luDef = (cphvb_lu_type*)arg;
    UserFuncArg* userFuncArg = (UserFuncArg*)ve_arg;
    
    //TODO asserts square matrix, dimension match etc. currently also checked in python before this call
    if (userFunctionLu == NULL)
    {
        userFunctionLu = new UserFunctionLu(userFuncArg->resourceManager);
    }
    return userFunctionLu->lu(luDef, userFuncArg);
}

UserFunctionLu::UserFunctionLu(ResourceManager* rm)
    : resourceManager(rm)
{   
    std::vector<std::string> kernelNames;
    kernelNames.push_back("pivot");
    kernelNames.push_back("update_col");
    kernelNames.push_back("update_rest");
    kernelNames.push_back("update_block_col");
    kernelNames.push_back("update_block_row");
    kernelNames.push_back("mm");
    std::vector<cphvb_intp> ndims(6,1);
    ndims[2] = 2;
    ndims[3] = 2;
    ndims[5] = 2;
    std::vector<Kernel> kernels = 
        Kernel::createKernelsFromFile(resourceManager, ndims, 
                                      resourceManager->getKernelPath() + "/Lu.cl", kernelNames);
    kernelMap.insert(std::make_pair("pivot", kernels[0]));
    kernelMap.insert(std::make_pair("update_col", kernels[1]));
    kernelMap.insert(std::make_pair("update_rest", kernels[2]));
    kernelMap.insert(std::make_pair("update_block_col", kernels[3]));
    kernelMap.insert(std::make_pair("update_block_row", kernels[4]));
    kernelMap.insert(std::make_pair("mm", kernels[5]));
}


cphvb_error UserFunctionLu::lu(cphvb_lu_type* luDef, UserFuncArg* userFuncArg)
{
    assert (userFuncArg->resourceManager == resourceManager);
    
    BaseArray* A = static_cast<BaseArray*>(userFuncArg->operands[0]);
    BaseArray* P = static_cast<BaseArray*>(userFuncArg->operands[1]);
    
    int n = luDef->operand[0]->shape[0];
    Scalar N(n);
    int block_size = 32; //needs to match the define in lu.cl
    int full_blocks = n / block_size;
    int rest = n % block_size;
    //the smallest multiple of block_size which is smaller or equal n
    int n_roof = rest == 0 ? n : (full_blocks+1) * block_size;
    
    Kernel::Parameters parameters;
    KernelMap::iterator kit;
    std::vector<size_t> localShape1d(1,1);
    std::vector<size_t> globalShape1d(1,1);
    std::vector<size_t> localShape2d(2,1);
    std::vector<size_t> globalShape2d(2,1);

    //calculate the first rows of the lu factorization, untill the rest can be done in blocks
    if(rest > 0){
      for (int k = 0; k < rest; k++){
        parameters.clear();
        Scalar K(k);
        
        //find pivot
        kit = kernelMap.find("pivot");
        if (kit == kernelMap.end())
          return CPHVB_TYPE_NOT_SUPPORTED; //TODO better error msg?
        
        parameters.push_back(std::make_pair(A, true));
        parameters.push_back(std::make_pair(&K, false));
        parameters.push_back(std::make_pair(P, true));
        parameters.push_back(std::make_pair(&N, false));
        
        localShape1d[0] = 256;
        globalShape1d[0] = 256;
        kit->second.call(parameters, globalShape1d, localShape1d);
        
        
        //update the column under k
        kit = kernelMap.find("update_col");
        if (kit == kernelMap.end())
          return CPHVB_TYPE_NOT_SUPPORTED;
        
        parameters.clear();
        parameters.push_back(std::make_pair(A, true));
        parameters.push_back(std::make_pair(&K, false));
        parameters.push_back(std::make_pair(&N, false));
        
        localShape1d[0] = block_size;
        globalShape1d[0] = n_roof;
        kit->second.call(parameters, globalShape1d, localShape1d);
        
        //update the rest matrix
        kit = kernelMap.find("update_rest");
        if (kit == kernelMap.end())
          return CPHVB_TYPE_NOT_SUPPORTED;
          
        localShape2d[0] = block_size;
        localShape2d[1] = block_size;
        globalShape2d[0] = n_roof;
        globalShape2d[1] = n_roof;
        kit->second.call(parameters, globalShape2d, localShape2d); //same parameters as update col       
        
      }
    }
    
    //perform the rest of the LU factorization on blocks
    for(int block = 0; block < full_blocks; block++){
        for(int k_block = 0; k_block < block_size; k_block++){
            int k = block * block_size + k_block + rest;
            Scalar K(k);
            parameters.clear();        
        
            //find pivot
            kit = kernelMap.find("pivot");
            if (kit == kernelMap.end())
                return CPHVB_TYPE_NOT_SUPPORTED; //TODO better error msg?
        
            parameters.push_back(std::make_pair(A, true));
            parameters.push_back(std::make_pair(&K, false));
            parameters.push_back(std::make_pair(P, true));
            parameters.push_back(std::make_pair(&N, false));
        
            localShape1d[0] = 256;
            globalShape1d[0] = 256;
            kit->second.call(parameters, globalShape1d, localShape1d);
        
                    
            //update column under k
            kit = kernelMap.find("update_col");
                if (kit == kernelMap.end())
                    return CPHVB_TYPE_NOT_SUPPORTED;
        
            parameters.clear();
            parameters.push_back(std::make_pair(A, true));
            parameters.push_back(std::make_pair(&K, false));
            parameters.push_back(std::make_pair(&N, false));
        
            localShape1d[0] = block_size;
            globalShape1d[0] = n_roof;
            kit->second.call(parameters, globalShape1d, localShape1d);
        
            //update rest of the block columns
            if(k_block < block_size-1){
                kit = kernelMap.find("update_block_col");
                if (kit == kernelMap.end())
                    return CPHVB_TYPE_NOT_SUPPORTED;
                
                parameters.clear();                
                parameters.push_back(std::make_pair(A, true));
                parameters.push_back(std::make_pair(&K, false));
                Scalar block_start(block * block_size + rest);
                parameters.push_back(std::make_pair(&block_start, false));
                parameters.push_back(std::make_pair(&N, false));
                
                localShape2d[0] = block_size;
                localShape2d[1] = block_size;
                globalShape2d[0] = (full_blocks - block)*block_size;
                globalShape2d[1] = block_size;
                kit->second.call(parameters, globalShape2d, localShape2d);
        
            }
        
        }
        
        if( block < full_blocks - 1){
            //solve for rows of U
            kit = kernelMap.find("update_block_row");
                if (kit == kernelMap.end())
                    return CPHVB_TYPE_NOT_SUPPORTED;
        
            parameters.clear();
            parameters.push_back(std::make_pair(A, true));
            Scalar L_start(block * block_size + rest);
            parameters.push_back(std::make_pair(&L_start, false));
            parameters.push_back(std::make_pair(&N, false));
        
            localShape1d[0] = block_size;
            globalShape1d[0] = (full_blocks - block - 1)*block_size;
            kit->second.call(parameters, globalShape1d, localShape1d);
            
            //update the rest of the matrix with matmul
            kit = kernelMap.find("mm");
            if (kit == kernelMap.end())
                return CPHVB_TYPE_NOT_SUPPORTED;
                
            parameters.clear();                
            parameters.push_back(std::make_pair(A, true));
            parameters.push_back(std::make_pair(A, true));
            parameters.push_back(std::make_pair(A, true));
            Scalar a_width(block_size);
            parameters.push_back(std::make_pair(&a_width, false));
            parameters.push_back(std::make_pair(&N, false));
            Scalar K(block * block_size + rest);
            parameters.push_back(std::make_pair(&K, false));
            
            
            localShape2d[0] = block_size;
            localShape2d[1] = block_size;
            globalShape2d[0] = (full_blocks - block - 1) * block_size;
            globalShape2d[1] = (full_blocks - block - 1) * block_size;
            kit->second.call(parameters, globalShape2d, localShape2d);
        
        }
        
        
            
    
    }
    
    return CPHVB_SUCCESS;
}


