typedef unsigned int uint;

__device__
uint TausStep(uint* z, int s0, int s1, int s2, uint M)
{
    uint b = (((*z << s0) ^ *z) >> s1);
    return *z = (((*z & M) << s2) ^ b);  
}

__device__
uint LCGStep(uint* z, uint A, uint C)
{
    return *z = (A * *z + C);    
}

__device__
uint HybridTaus(uint* z0, uint* z1, 
                        uint* z2, uint* z3)
{
    return TausStep(z0, 13, 19, 12, 4294967294UL) ^ 
        TausStep(z1, 2, 25, 4, 4294967288UL)      ^
        TausStep(z2, 3, 11, 17, 4294967280UL)     ^
        LCGStep(z3, 1664525, 1013904223UL);
}

__device__
void loadState(uint* state, uint* stateIdx, 
               uint* z0, uint* z1, uint* z2, uint* z3)
{
	*z0 = state[*stateIdx+0];
	*z1 = state[*stateIdx+1];
	*z2 = state[*stateIdx+2];
	*z3 = state[*stateIdx+3];    
}

__device__
void saveState(uint* state, uint* stateIdx, 
               uint* z0, uint* z1, uint* z2, uint* z3)
{
	state[*stateIdx+0] = *z0;
	state[*stateIdx+1] = *z1;
	state[*stateIdx+2] = *z2;
	state[*stateIdx+3] = *z3;
}


extern "C" __global__ 
void htrand_uint32(uint* res, uint elements, uint* state)
{
    const uint nThreads = blockDim.x*gridDim.x;
	uint stateIdx = threadIdx.x + blockIdx.x*blockDim.x;
	uint outIdx, z0, z1, z2, z3;
    loadState(state, &stateIdx, &z0, &z1, &z2, &z3);
	for (outIdx = stateIdx; outIdx < elements; outIdx += nThreads) 
	{
        res[outIdx] = HybridTaus(&z0,&z1,&z2,&z3);
	}
    saveState(state, &stateIdx, &z0, &z1, &z2, &z3);
}

extern "C" __global__ 
void htrand_int32(int *res, uint elements, uint* state)
{
    const uint nThreads = blockDim.x*gridDim.x;
	uint stateIdx = threadIdx.x + blockIdx.x*blockDim.x;
	uint outIdx, z0, z1, z2, z3;
    loadState(state, &stateIdx, &z0, &z1, &z2, &z3);
	for (outIdx = stateIdx; outIdx < elements; outIdx += nThreads) 
	{
        res[outIdx] = HybridTaus(&z0,&z1,&z2,&z3) >> 1;
	}
    saveState(state, &stateIdx, &z0, &z1, &z2, &z3);
}

extern "C" __global__ 
void htrand_float32(float *res, uint elements, uint* state)
{
    const uint nThreads = blockDim.x*gridDim.x;
	uint stateIdx = threadIdx.x + blockIdx.x*blockDim.x;
	uint outIdx, z0, z1, z2, z3;
    loadState(state, &stateIdx, &z0, &z1, &z2, &z3);
	for (outIdx = stateIdx; outIdx < elements; outIdx += nThreads) 
	{
        res[outIdx] = 2.3283064365387e-10 * HybridTaus(&z0,&z1,&z2,&z3);
	}
    saveState(state, &stateIdx, &z0, &z1, &z2, &z3);
}

extern "C" __global__ 
void htrand_uint32_step(uint *res, uint elements, uint step, uint* state)
{
    const uint nThreads = blockDim.x*gridDim.x;
	uint stateIdx = threadIdx.x + blockIdx.x*blockDim.x;
	uint outIdx, z0, z1, z2, z3;
    loadState(state, &stateIdx, &z0, &z1, &z2, &z3);
	for (outIdx = stateIdx * step; outIdx < elements * step; 
         outIdx += nThreads * step) 
	{
        res[outIdx] = HybridTaus(&z0,&z1,&z2,&z3);
	}
    saveState(state, &stateIdx, &z0, &z1, &z2, &z3);
}

extern "C" __global__ 
void htrand_int32_step(int *res, uint elements, uint step, uint* state)
{
    const uint nThreads = blockDim.x*gridDim.x;
	uint stateIdx = threadIdx.x + blockIdx.x*blockDim.x;
	uint outIdx, z0, z1, z2, z3;
    loadState(state, &stateIdx, &z0, &z1, &z2, &z3);
	for (outIdx = stateIdx * step; outIdx < elements * step; 
         outIdx += nThreads * step) 
	{
        res[outIdx] = HybridTaus(&z0,&z1,&z2,&z3) >> 1;
	}
    saveState(state, &stateIdx, &z0, &z1, &z2, &z3);
}

extern "C" __global__ 
void htrand_float32_step(float *res, uint elements, uint step, uint* state)
{
    const uint nThreads = blockDim.x*gridDim.x;
	uint stateIdx = threadIdx.x + blockIdx.x*blockDim.x;
	uint outIdx, z0, z1, z2, z3;
    loadState(state, &stateIdx, &z0, &z1, &z2, &z3);
	for (outIdx = stateIdx * step; outIdx < elements * step; 
         outIdx += nThreads * step) 
	{
        res[outIdx] = 2.3283064365387e-10 * HybridTaus(&z0,&z1,&z2,&z3);
	}
    saveState(state, &stateIdx, &z0, &z1, &z2, &z3);
}
