/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium
team <http://www.bh107.org>.

Bohrium is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as 
published by the Free Software Foundation, either version 3 
of the License, or (at your option) any later version.

Bohrium is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the 
GNU Lesser General Public License along with Bohrium. 

If not, see <http://www.gnu.org/licenses/>.
*/
#include "assert.h"
#include "stdarg.h"
#include "string.h"
#include "stdlib.h"
#include "stdint.h"
#include "stdio.h"
#include "complex.h"
#include "math.h"

#include <sys/time.h>
#include <unistd.h>
#include <limits.h>

#include "omp.h"

//
// We use the same Mersenne Twister implementation as NumPy
//
typedef struct {    
    unsigned long key[624];
    int pos;
} rk_state;

// Mersenne Twister constants
#define N 624
#define M 397
#define MATRIX_A 0x9908b0dfUL
#define UPPER_MASK 0x80000000UL
#define LOWER_MASK 0x7fffffffUL
#define RK_STATE_LEN 624

// Slightly optimised reference implementation of the Mersenne Twister
unsigned long rk_random(rk_state *state)
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

    y ^= (y >> 11);     // Tempering
    y ^= (y <<  7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);

    return y;
}
int8_t rk_b(rk_state *state)
{
    return (int8_t)(rk_random(state) & 0x7f);
}

int16_t rk_s(rk_state *state)
{
    return (int16_t)(rk_random(state) & 0x7fff);
}

int32_t rk_i(rk_state *state)
{
    return (int32_t)rk_random(state) & 0x7fffffff;
}

int64_t rk_l(rk_state *state)
{
    int64_t res = rk_random(state) & 0x7fffffff;
    res = (res << 32) | rk_random(state);
    return res;
}

uint8_t rk_B(rk_state *state)
{
    return (uint8_t)(rk_random(state) & 0xff);
}

uint16_t rk_S(rk_state *state)
{
    return (uint16_t)(rk_random(state) & 0xffff);
}

uint32_t rk_I(rk_state *state)
{
    return rk_random(state);
}

uint64_t rk_L(rk_state *state)
{
    uint64_t res = rk_random(state); 
    res = (res << 32) | rk_random(state);
    return res;
}

//
//  Add support for halfs here... "float16"
//

float rk_f(rk_state *state)
{
    //return rk_random(state) * 2.3283064365387e-10;
    float res =  rk_random(state);
    return res;
}

double rk_d(rk_state *state)
{
    /* shifts : 67108864 = 0x4000000, 9007199254740992 = 0x20000000000000 */
    long    a = rk_random(state) >> 5,
            b = rk_random(state) >> 6;
    return (a * 67108864.0 + b) / 9007199254740992.0;
}

//
//  Add support for complex numbers here...
//

void rk_seed(unsigned long seed, rk_state *state)
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
unsigned long rk_hash(unsigned long key)
{
    key += ~(key << 15);
    key ^=  (key >> 10);
    key +=  (key << 3);
    key ^=  (key >> 6);
    key += ~(key << 11);
    key ^=  (key >> 16);
    return key;
}

int rk_initseed(rk_state *state, int thread_id)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    rk_seed(rk_hash(thread_id) ^ rk_hash(getpid()) ^ rk_hash(tv.tv_sec) ^ rk_hash(tv.tv_usec), state);
    return 0;
}

void {{SYMBOL}}(int tool, ...)
{
    va_list list;
    va_start(list,tool);
    {{TYPE_A0}} *a0_data = va_arg(list, {{TYPE_A0}}*);
    int64_t nelements = va_arg(list, int64_t);
    va_end(list);

    #pragma omp parallel
    {
        int64_t nthreads    = omp_get_num_threads();
        int64_t thread_id   = omp_get_thread_num();
        int64_t my_elements = nelements / nthreads;
        int64_t my_offset   = thread_id * my_elements;
        if ((thread_id == nthreads-1) && (thread_id != 0)) {
            my_elements += nelements % thread_id;
        }

        rk_state state;
        rk_initseed(&state, thread_id);

        int64_t i;
        for(i=my_offset; i<my_elements+my_offset; ++i) {
            a0_data[i] = rk_{{TYPE_A0_SHORTHAND}}(&state);
        }
    }
}



