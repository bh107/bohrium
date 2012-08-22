/*
 * Copyright 2012 Andreas Thorning <thorning@diku.dk>
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
 * along with cphVB. If not, see <http://www.gnu.org/licenses/>.
 */
 
#include <iostream>
#include <sstream>
#include <cassert>
#include <stdexcept>
#include "UserFunctionFft.hpp"
#include "Scalar.hpp"

UserFunctionFft* userFunctionFft = NULL;

cphvb_error cphvb_fft(cphvb_userfunc* arg, void* ve_arg)
{
    cphvb_fft_type* fftDef = (cphvb_fft_type*)arg;
    UserFuncArg* userFuncArg = (UserFuncArg*)ve_arg;
    
    if(fftDef->operand[1]->ndim != 1){
        return CPHVB_TYPE_NOT_SUPPORTED;
    }
    
    int s0 = fftDef->operand[1]->shape[0];
    
    //check power of 2
    if(s0 & (s0 - 1)){
        return CPHVB_TYPE_NOT_SUPPORTED;
    }
    //currently only support simple strides
    if(fftDef->operand[1]->stride[0] != 1){
       return CPHVB_TYPE_NOT_SUPPORTED;
    }
    
    
    if (userFunctionFft == NULL)
    {
        userFunctionFft = new UserFunctionFft(userFuncArg->resourceManager);
    }
    
    return userFunctionFft->fft(fftDef, userFuncArg);
    
}

cphvb_error cphvb_fft2(cphvb_userfunc* arg, void* ve_arg)
{
    cphvb_fft_type* fftDef = (cphvb_fft_type*)arg;
    UserFuncArg* userFuncArg = (UserFuncArg*)ve_arg;
    
    if(fftDef->operand[1]->ndim != 2){
        return CPHVB_TYPE_NOT_SUPPORTED;
    }
    
    int s0 = fftDef->operand[1]->shape[0];
    int s1 = fftDef->operand[1]->shape[1];
    
    //check power of 2
    if((s0 & (s0 - 1)) || (s1 & (s1 - 1))){
        return CPHVB_TYPE_NOT_SUPPORTED;
    }
    //currently only support simple strides
    if(fftDef->operand[1]->stride[1] != 1 || fftDef->operand[1]->stride[0] != fftDef->operand[0]->shape[1]){
       return CPHVB_TYPE_NOT_SUPPORTED;
    }
    
    if (userFunctionFft == NULL)
    {
        userFunctionFft = new UserFunctionFft(userFuncArg->resourceManager);
    }
    
    return userFunctionFft->fft2d(fftDef, userFuncArg);
    
}

UserFunctionFft::UserFunctionFft(ResourceManager* rm)
    : resourceManager(rm)
{   
    std::vector<std::string> kernelNames;
    kernelNames.push_back("fft");
    kernelNames.push_back("fft2d");
    kernelNames.push_back("copy");
    std::vector<cphvb_intp> ndims(3,1);
    ndims[0] = 1;
    ndims[1] = 2;
    ndims[2] = 1;
    std::vector<Kernel> kernels = 
        Kernel::createKernelsFromFile(resourceManager, ndims, 
                                      resourceManager->getKernelPath() + "/Fft.cl", kernelNames);
    kernelMap.insert(std::make_pair("fft", kernels[0]));
    kernelMap.insert(std::make_pair("fft2d", kernels[1]));
    kernelMap.insert(std::make_pair("copy", kernels[2]));
}

int log2( int x )
{
  int res = 0 ;
  while( x>>=1 ) res++;
  return res ;
}



cphvb_error UserFunctionFft::fft(cphvb_fft_type* fftDef, UserFuncArg* userFuncArg)
{
    assert (userFuncArg->resourceManager == resourceManager);
    
    Buffer* out = static_cast<Buffer*>(userFuncArg->operands[0]);
    Buffer* in = static_cast<Buffer*>(userFuncArg->operands[1]);
    
    Kernel::Parameters parameters;
    KernelMap::iterator kit;
    
    int n = fftDef->operand[0]->shape[0];
    int iterations = log2(n);
    bool use_temp = iterations % 2 == 0;
    int p = 1;
    Scalar P(p);
      
    Buffer* temp = new Buffer(n*2*sizeof(cl_float), resourceManager);
      
    kit = kernelMap.find("fft");
    if (kit == kernelMap.end())
        return CPHVB_TYPE_NOT_SUPPORTED; //TODO better error msg?
        
    std::vector<size_t> globalShape(1,n/2);
    size_t local = n/2 >= 256 ? 256 : n/2;
    std::vector<size_t> localShape(1,local);
         
    parameters.push_back(std::make_pair(in, false));
    if(use_temp)
      parameters.push_back(std::make_pair(temp, true));
    else
      parameters.push_back(std::make_pair(out, true));
    parameters.push_back(std::make_pair(&P, false));
      
     
    kit->second.call(parameters, globalShape, localShape);      
    use_temp = !use_temp;
      
    for(p = 2; p < n; p<<=1){
      parameters[0] = use_temp ? std::make_pair(out, false) : std::make_pair(temp, false);
      parameters[1] = use_temp ? std::make_pair(temp, true) : std::make_pair(out, true);
      P = Scalar(p);
      parameters[2] = std::make_pair(&P, false);
      kit->second.call(parameters, globalShape, localShape);      
      use_temp = !use_temp;
    }
      
      
    delete temp;
      
    return CPHVB_SUCCESS;
}

cphvb_error UserFunctionFft::fft2d(cphvb_fft_type* fftDef, UserFuncArg* userFuncArg)
{
    assert (userFuncArg->resourceManager == resourceManager);
    
    
    Buffer* out = static_cast<Buffer*>(userFuncArg->operands[0]);
    Buffer* in = static_cast<Buffer*>(userFuncArg->operands[1]);
    
    Kernel::Parameters parameters;
    KernelMap::iterator kit;
    
    int rows = fftDef->operand[0]->shape[0];
    int cols = fftDef->operand[0]->shape[1];
    
    //along rows
    int iterations = log2(cols);
    bool res_in_out = iterations % 2 == 0;
    int p = 1;
    Scalar P(p);
    Scalar One(1);
    Scalar Rows(rows);
    Scalar Cols(cols);
      
    Buffer* temp = new Buffer(rows*cols*2*sizeof(cl_float), resourceManager);
      
    kit = kernelMap.find("fft2d");
    if (kit == kernelMap.end())
        return CPHVB_TYPE_NOT_SUPPORTED; //TODO better error msg?
    
    
    std::vector<size_t> globalShape(2,cols/2);
    globalShape[1] = rows;
    size_t local = cols/2 >= 32 ? 32 : cols/2;
    size_t local2 = rows >= 32 ? 32 : rows;
    std::vector<size_t> localShape(2,local);
    localShape[1] = local2;
         
    parameters.push_back(std::make_pair(in, false));
    if(res_in_out)
      parameters.push_back(std::make_pair(temp, true));
    else
      parameters.push_back(std::make_pair(out, true));
    parameters.push_back(std::make_pair(&P, false));
    
    //Strides
    parameters.push_back(std::make_pair(&One, false));
    parameters.push_back(std::make_pair(&Cols, false));
      
     
    kit->second.call(parameters, globalShape,localShape);      
    res_in_out = !res_in_out;
    
    for(p = 2; p < cols; p<<=1){
      parameters[0] = res_in_out ? std::make_pair(out, false) : std::make_pair(temp, false);
      parameters[1] = res_in_out ? std::make_pair(temp, true) : std::make_pair(out, true);
      P = Scalar(p);
      parameters[2] = std::make_pair(&P, false);
      kit->second.call(parameters, globalShape, localShape);      
      res_in_out = !res_in_out;
    }
    
    
    //along cols     
              
    globalShape[0] = rows/2;
    globalShape[1] = cols;
    local = rows/2 >= 32 ? 32 : rows/2;
    local2 = cols >= 32 ? 32 : cols;
    localShape[0] = local;
    localShape[1] = local2;
         
    //TODO this should actually be strides
    parameters[3] = std::make_pair(&Cols, false);
    parameters[4] = std::make_pair(&One, false);
      
    for(p = 1; p < rows; p<<=1){
      parameters[0] = res_in_out ? std::make_pair(out, false) : std::make_pair(temp, false);
      parameters[1] = res_in_out ? std::make_pair(temp, true) : std::make_pair(out, true);
      P = Scalar(p);
      parameters[2] = std::make_pair(&P, false);
      kit->second.call(parameters, globalShape,localShape);      
      res_in_out = !res_in_out;
    }
    
    if(!res_in_out){
      kit = kernelMap.find("copy");
      if (kit == kernelMap.end())
        return CPHVB_TYPE_NOT_SUPPORTED; //TODO better error msg?
      
      parameters.clear();
      parameters.push_back(std::make_pair(temp, false));
      parameters.push_back(std::make_pair(out, true));
      
      std::vector<size_t> globalShape1d(1,rows * cols);
      local = rows * cols >= 256 ? 256 : rows * cols;
      std::vector<size_t> localShape1d(1,local);
      
      kit->second.call(parameters, globalShape1d, localShape1d);
    }
      
    delete temp;
      
    return CPHVB_SUCCESS;
}



