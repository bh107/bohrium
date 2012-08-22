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

#include <cassert>
#include <stdexcept>
#include <cstdlib>
#include <ctime>
#include "UserFunctionRandom.hpp"
#include "Scalar.hpp"

#ifdef _WIN32
#define srandom srand
#define random rand
#endif

UserFunctionRandom* userFunctionRandom = NULL;

cphvb_error cphvb_random(cphvb_userfunc *arg, void* ve_arg)
{
    cphvb_random_type* randomDef = (cphvb_random_type*)arg;
    UserFuncArg* userFuncArg = (UserFuncArg*)ve_arg;
    assert (randomDef->nout == 1);
    assert (randomDef->nin == 0);
    assert (randomDef->operand[0]->base == NULL);
    assert (userFuncArg->operands.size() == 1);
    if (userFunctionRandom == NULL)
    {
        userFunctionRandom = new UserFunctionRandom(userFuncArg->resourceManager);
    }
    return userFunctionRandom->fill(userFuncArg);
}

#define TPB 128
#define BPG 128

UserFunctionRandom::UserFunctionRandom(ResourceManager* rm)
    : resourceManager(rm)
{
    cl_uint4* init_data = new cl_uint4[BPG*TPB];
    srandom((unsigned int)std::time(NULL));
    for (int i = 0; i < BPG*TPB; ++i)
    {
        while ((init_data[i].s[0] = random())<129);
        while ((init_data[i].s[1] = random())<129);
        while ((init_data[i].s[2] = random())<129);
        while ((init_data[i].s[3] = random())<129);
    }
    
    state = new Buffer(BPG*TPB*sizeof(cl_uint4), resourceManager);
    state->write(init_data);
    cl::Event event = state->getWriteEvent();
    event.setCallback(CL_COMPLETE, &hostDataDelete, init_data);
    std::vector<std::string> kernelNames;
    kernelNames.push_back("htrand_int32");
    kernelNames.push_back("htrand_uint32");
    kernelNames.push_back("htrand_float32");
    kernelNames.push_back("htrand_int64");
    kernelNames.push_back("htrand_uint64");
    if (resourceManager->float64support()) 
        kernelNames.push_back("htrand_float64");
    std::vector<cphvb_intp> ndims(kernelNames.size(),1);
    std::vector<Kernel> kernels = 
        Kernel::createKernelsFromFile(resourceManager, ndims, 
                                      resourceManager->getKernelPath() + "/HybridTaus.cl", kernelNames);
    kernelMap.insert(std::make_pair(OCL_INT32, kernels[0]));
    kernelMap.insert(std::make_pair(OCL_UINT32, kernels[1]));
    kernelMap.insert(std::make_pair(OCL_FLOAT32, kernels[2]));    
    kernelMap.insert(std::make_pair(OCL_INT64, kernels[3]));
    kernelMap.insert(std::make_pair(OCL_UINT64, kernels[4]));
    if (resourceManager->float64support()) 
        kernelMap.insert(std::make_pair(OCL_FLOAT64, kernels[5]));    
}

void CL_CALLBACK UserFunctionRandom::hostDataDelete(cl_event ev, cl_int eventStatus, void* data)
{
    assert(eventStatus == CL_COMPLETE);
    delete [](cl_uint4*)data;
}

cphvb_error UserFunctionRandom::fill(UserFuncArg* userFuncArg)
{
    assert (userFuncArg->resourceManager == resourceManager);
    BaseArray* ba = static_cast<BaseArray*>(userFuncArg->operands[0]);
    KernelMap::iterator kit = kernelMap.find(ba->type());
    if (kit == kernelMap.end())
        return CPHVB_TYPE_NOT_SUPPORTED;
    Scalar size(ba->size());    
    Kernel::Parameters parameters;
    parameters.push_back(std::make_pair(ba, true));
    parameters.push_back(std::make_pair(&size, false));
    parameters.push_back(std::make_pair(state, false));
    
    std::vector<size_t> localShape(1,TPB);
    std::vector<size_t> globalShape(1,BPG*TPB);
    kit->second.call(parameters, globalShape, localShape);
    return CPHVB_SUCCESS;
}
