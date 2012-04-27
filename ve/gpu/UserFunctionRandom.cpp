/*
 * Copyright 2012 Troels Blum <troels@blum.dk>
 *
 * This file is part of cphVB <http://code.google.com/p/cphvb/>.
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

#include <cassert>
#include <stdexcept>
#include <cstdlib>
#include <ctime>
#include "UserFunctionRandom.hpp"
#include "Scalar.hpp"

cphvb_error cphvb_random(cphvb_userfunc *arg, void* ve_arg)
{
    if (arg == NULL)
        UserFunctionRandom::finalize();
    cphvb_random_type* randomDef = (cphvb_random_type*)arg;
    UserFuncArg* userFuncArg = (UserFuncArg*)ve_arg;
    assert (randomDef->nout == 1);
    assert (randomDef->nin == 0);
    assert (randomDef->operand[0]->base == NULL);
    assert (userFuncArg->operands.size() == 1);
    if (UserFunctionRandom::resourceManager == NULL)
    {
        UserFunctionRandom::resourceManager = userFuncArg->resourceManager;
        UserFunctionRandom::initialize();
    }
    assert (UserFunctionRandom::resourceManager == userFuncArg->resourceManager);
    UserFunctionRandom::run(userFuncArg);
    return CPHVB_SUCCESS;
}

#define TPB 128
#define BPG 128

void UserFunctionRandom::initialize()
{
    cl_uint4* init_data = new cl_uint4[BPG*TPB];
    srandom((unsigned int)std::time(NULL));
    for (int i = 0; i < BPG*TPB; ++i)
    {
        while ((init_data[i].s0 = random())<129);
        while ((init_data[i].s1 = random())<129);
        while ((init_data[i].s2 = random())<129);
        while ((init_data[i].s3 = random())<129);
    }
    
    state = new Buffer(BPG*TPB*sizeof(cl_uint4), resourceManager);
    state->write(init_data);
    cl::Event event = state->getWriteEvent();
    event.setCallback(CL_COMPLETE, &hostDataDelete, init_data);
    std::vector<std::string> kernelNames;
    kernelNames.push_back("htrand_int32");
    kernelNames.push_back("htrand_uint32");
    kernelNames.push_back("htrand_float32");
    std::vector<cphvb_intp> ndims(3,1);
    std::vector<Kernel> kernels = 
        Kernel::createKernelsFromFile(resourceManager, ndims, 
                                      "/opt/cphvb/lib/ocl_source/HybridTaus.cl", kernelNames);
    kernelMap.insert(std::make_pair(OCL_INT32, kernels[0]));
    kernelMap.insert(std::make_pair(OCL_UINT32, kernels[1]));
    kernelMap.insert(std::make_pair(OCL_FLOAT32, kernels[2]));    
}

void CL_CALLBACK UserFunctionRandom::hostDataDelete(cl_event ev, cl_int eventStatus, void* data)
{
    assert(eventStatus == CL_COMPLETE);
    delete [](cl_uint4*)data;
}


void UserFunctionRandom::finalize()
{
    delete state;
}

void UserFunctionRandom::run(UserFuncArg* userFuncArg)
{
    BaseArray* ba = static_cast<BaseArray*>(userFuncArg->operands[0]);
    KernelMap::iterator kit = kernelMap.find(ba->type());
    if (kit == kernelMap.end())
        throw std::runtime_error("Data type not supported for random number generation.");
    Scalar size(ba->size());    
    Kernel::Parameters parameters;
    parameters.push_back(std::make_pair(ba, true));
    parameters.push_back(std::make_pair(&size, false));
    parameters.push_back(std::make_pair(state, false));
    
    std::vector<size_t> localShape(1,TPB);
    std::vector<cphvb_index> globalShape(1,BPG*TPB);
    kit->second.call(parameters, globalShape, localShape);
}
