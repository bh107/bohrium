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
#include "UserFunctionMatmul.hpp"


cphvb_error cphvb_matmul(cphvb_userfunc* arg, void* ve_arg)
{
    cphvb_matmul_type* matmulDef = (cphvb_matmul_type*)arg;
    UserFuncArg* userFuncArg = (UserFuncArg*)ve_arg;
    UserFunctionMatmul::matmul(matmulDef, userFuncArg);
    return CPHVB_SUCCESS;
}

void UserFunctionMatmul::matmul(cphvb_matmul_type* matmulDef, UserFuncArg* userFuncArg)
{
    Kernel kernel = getKernel(matmulDef, userFuncArg);
    Kernel::Parameters kernelParameters;
    kernelParameters.push_back(std::make_pair(userFuncArg->operands[0], true));
    kernelParameters.push_back(std::make_pair(userFuncArg->operands[1], false));
    kernelParameters.push_back(std::make_pair(userFuncArg->operands[2], false));
    
    std::vector<size_t> globalShape, localShape;
    
    int a_height = matmulDef->operand[1]->shape[0];
    int b_width = matmulDef->operand[2]->shape[1];
    
    //round up to nearst multiple of 32
    a_height = a_height % 32 == 0 ? a_height : (a_height / 32 + 1) * 32;
    b_width = b_width % 32 == 0 ? b_width : (b_width/ 32 + 1) * 32;
    
    globalShape.push_back(b_width);
    globalShape.push_back(a_height);
    localShape.push_back(32);
    localShape.push_back(32);
    
    kernel.call(kernelParameters, globalShape, localShape);
}

Kernel UserFunctionMatmul::getKernel(cphvb_matmul_type* matmulDef, 
                                     UserFuncArg* userFuncArg)
{
#ifdef STATS
    timeval start, end;
    gettimeofday(&start,NULL);
#endif
    std::string code = generateCode(matmulDef, userFuncArg->operands[0]->type());
    std::string defines = generateDefines(matmulDef, userFuncArg->operands[0]->type());
#ifdef STATS
    gettimeofday(&end,NULL);
    userFuncArg->resourceManager->batchSource += 
        (end.tv_sec - start.tv_sec)*1000000.0 + (end.tv_usec - start.tv_usec);
#endif
    size_t codeHash = string_hasher(code + defines);
    KernelMap::iterator kit = kernelMap.find(codeHash);
    if (kit == kernelMap.end())
    {
        std::stringstream source, kname;
        
        kname << "matmul" << std::hex << codeHash;
        source << defines << "__kernel void " << kname.str() << code;
        Kernel kernel(userFuncArg->resourceManager, 2, source.str(), kname.str());
        kernelMap.insert(std::make_pair(codeHash, kernel));
        return kernel;
    } else {
        return kit->second;
    }
}

std::string UserFunctionMatmul::generateDefines(cphvb_matmul_type* matmulDef, 
                                             OCLtype type)
{
  int a_height = matmulDef->operand[1]->shape[0];
  int a_width = matmulDef->operand[1]->shape[1];
  int b_width = matmulDef->operand[2]->shape[1];
  
  int a_start = matmulDef->operand[1]->start;
  int b_start = matmulDef->operand[2]->start;
  int c_start = matmulDef->operand[0]->start;
  
  int a_row_stride = matmulDef->operand[1]->stride[0];
  int b_row_stride = matmulDef->operand[2]->stride[0];
  int c_row_stride = matmulDef->operand[0]->stride[0];
  
  int a_col_stride = matmulDef->operand[1]->stride[1];
  int b_col_stride = matmulDef->operand[2]->stride[1];
  int c_col_stride = matmulDef->operand[0]->stride[1];
  
  
  std::stringstream source;
  if(type == OCL_FLOAT64)
    source << "#pragma OPENCL EXTENSION cl_khr_fp64: enable\n";
    
  source << "#define BLOCK_SIZE 32\n";
  source << "#define A_HEIGHT " <<  a_height << "\n";
  source << "#define A_WIDTH " <<  a_width << "\n";
  source << "#define A_START " <<  a_start << "\n";
  source << "#define A_ROW_STRIDE " <<  a_row_stride << "\n";
  source << "#define A_COL_STRIDE " <<  a_col_stride << "\n";
  source << "#define B_START " <<  b_start << "\n";
  source << "#define B_ROW_STRIDE " <<  b_row_stride << "\n";
  source << "#define B_COL_STRIDE " <<  b_col_stride << "\n";
  source << "#define C_START " <<  c_start << "\n";
  source << "#define C_ROW_STRIDE " <<  c_row_stride << "\n";
  source << "#define C_COL_STRIDE " <<  c_col_stride << "\n";
  source << "#define B_WIDTH " <<  b_width << "\n";
    
  return source.str();
}

std::string UserFunctionMatmul::generateCode(cphvb_matmul_type* matmulDef, 
                                             OCLtype type)
{
    bool good_block_size =    matmulDef->operand[1]->shape[0] % 32 == 0 
                           && matmulDef->operand[1]->shape[1] % 32 == 0 
                           && matmulDef->operand[2]->shape[1] % 32 == 0;
    
    std::stringstream source;
    source << "( __global " << oclTypeStr(type) << "* C\n" 
        "                     , __global const " << oclTypeStr(type) << "* A\n"
        "                     , __global const " << oclTypeStr(type) << "* B)\n{\n";
    source << "unsigned int local_col = get_local_id(0);\n";
    source << "unsigned int local_row = get_local_id(1);\n";
    source << "unsigned int col = get_global_id(0);\n";
    source << "unsigned int row = get_global_id(1);\n";
    source << "__local float A_cache[BLOCK_SIZE*BLOCK_SIZE];\n";
    source << "__local float B_cache[BLOCK_SIZE*BLOCK_SIZE];\n";
    source << "float val = 0;\n";
    source << "for(int k0 = 0; k0 < A_WIDTH; k0 += BLOCK_SIZE){\n";
    if(good_block_size){
      source << "A_cache[local_col + local_row*BLOCK_SIZE] = A[A_START + row * A_ROW_STRIDE + (k0 + local_col) * A_COL_STRIDE];\n";
      source << "B_cache[local_col + local_row*BLOCK_SIZE] = B[B_START + (k0+local_row) * B_ROW_STRIDE + col * B_COL_STRIDE];\n";
    }else{
      source << "A_cache[local_col + local_row*BLOCK_SIZE] = row < A_HEIGHT && k0+local_col < A_WIDTH ? A[A_START + row * A_ROW_STRIDE + (k0 + local_col) * A_COL_STRIDE] : 0;\n";
      source << "B_cache[local_col + local_row*BLOCK_SIZE] = col < B_WIDTH && k0+local_row < A_WIDTH ? B[B_START + (k0+local_row) * B_ROW_STRIDE + col * B_COL_STRIDE] : 0;\n";
    }    
    source << "barrier(CLK_LOCAL_MEM_FENCE);\n";
    source << "for(int k = 0; k < BLOCK_SIZE; k++){\n";
    source << "val += A_cache[k+local_row*BLOCK_SIZE] * B_cache[local_col+k*BLOCK_SIZE];}\n";
    source << "barrier(CLK_LOCAL_MEM_FENCE);}\n";
    if(good_block_size){
      source << "C[C_START + col * C_COL_STRIDE + row*C_ROW_STRIDE] = val;}\n";
    }else{
      source << "if(col < B_WIDTH && row < A_HEIGHT){\n";
      source << "C[C_START + col * C_COL_STRIDE + row*C_ROW_STRIDE] = val;}}\n";
    }    
    return source.str();
}
