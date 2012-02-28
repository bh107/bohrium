/*
 * Copyright 2011 Simon A. F. Lund <safl@safl.dk>
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
#include <cphvb.h>
#include "cphvb_ve_score.h"
#include <sys/time.h>

#ifdef _OPENMP
    #include <omp.h>
#endif

// We use the same Mersenne Twister implementation as NumPy
typedef struct
{
    unsigned long key[624];
    int pos;
} rk_state;

/* Magic Mersenne Twister constants */
#define N 624
#define M 397
#define MATRIX_A 0x9908b0dfUL
#define UPPER_MASK 0x80000000UL
#define LOWER_MASK 0x7fffffffUL
#define RK_STATE_LEN 624

/* Slightly optimised reference implementation of the Mersenne Twister */
unsigned long
rk_random(rk_state *state)
{
    unsigned long y;

    if (state->pos == RK_STATE_LEN) {
        int i;

        for (i = 0; i < N - M; i++) {
            y = (state->key[i] & UPPER_MASK) | (state->key[i+1] & LOWER_MASK);
            state->key[i] = state->key[i+M] ^ (y>>1) ^ (-(y & 1) & MATRIX_A);
        }
        for (; i < N - 1; i++) {
            y = (state->key[i] & UPPER_MASK) | (state->key[i+1] & LOWER_MASK);
            state->key[i] = state->key[i+(M-N)] ^ (y>>1) ^ (-(y & 1) & MATRIX_A);
        }
        y = (state->key[N - 1] & UPPER_MASK) | (state->key[0] & LOWER_MASK);
        state->key[N - 1] = state->key[M - 1] ^ (y >> 1) ^ (-(y & 1) & MATRIX_A);

        state->pos = 0;
    }
    y = state->key[state->pos++];

    /* Tempering */
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);

    return y;
}
double
rk_double(rk_state *state)
{
    /* shifts : 67108864 = 0x4000000, 9007199254740992 = 0x20000000000000 */
    long a = rk_random(state) >> 5, b = rk_random(state) >> 6;
    return (a * 67108864.0 + b) / 9007199254740992.0;
}
void
rk_seed(unsigned long seed, rk_state *state)
{
    int pos;
    seed &= 0xffffffffUL;

    /* Knuth's PRNG as used in the Mersenne Twister reference implementation */
    for (pos = 0; pos < RK_STATE_LEN; pos++) {
        state->key[pos] = seed;
        seed = (1812433253UL * (seed ^ (seed >> 30)) + pos + 1) & 0xffffffffUL;
    }
    state->pos = RK_STATE_LEN;
}
/* Thomas Wang 32 bits integer hash function */
unsigned long
rk_hash(unsigned long key)
{
    key += ~(key << 15);
    key ^=  (key >> 10);
    key +=  (key << 3);
    key ^=  (key >> 6);
    key += ~(key << 11);
    key ^=  (key >> 16);
    return key;
}
int
rk_initseed(rk_state *state)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    rk_seed(rk_hash(getpid()) ^ rk_hash(tv.tv_sec) ^ rk_hash(tv.tv_usec), state);
    return 0;
}


//Implementation of the user-defined funtion "random". Note that we
//follows the function signature defined by cphvb_userfunc_impl.
cphvb_error cphvb_random(cphvb_userfunc *arg)
{
    cphvb_random_type *a = (cphvb_random_type *) arg;
    cphvb_array *ary = a->operand[0];
    cphvb_intp size = cphvb_nelements(ary->ndim, ary->shape);
    cphvb_intp nthds = 1;

    //Make sure that the array memory is allocated.
    if(cphvb_data_malloc(ary) != CPHVB_SUCCESS)
        return CPHVB_OUT_OF_MEMORY;
    double *data = (double *) ary->data;

    if(ary->type != CPHVB_FLOAT64)
        return CPHVB_TYPE_NOT_SUPPORTED;

    //Handle the blocks.
    //We will use OpenMP to parallelize the computation.
    //We divide the blocks between the threads.
    #ifdef _OPENMP
        nthds = omp_get_max_threads();
        if(nthds > size)
            nthds = size;
        #pragma omp parallel num_threads(nthds) default(none) \
                shared(nthds,size,data)
    #endif
    {
        #ifdef _OPENMP
            int myid = omp_get_thread_num();
        #else
            int myid = 0;
        #endif
        cphvb_intp length = size / nthds; // Find this thread's length of work.
        cphvb_intp start = myid * length; // Find this thread's start block.
        if(myid == nthds-1)
            length += size % nthds;       // The last thread gets the reminder.

        rk_state state;
        rk_initseed(&state);
        for(cphvb_intp i=start; i<start+length; ++i)
            data[i] = rk_double(&state);
    }

    return CPHVB_SUCCESS;
}
