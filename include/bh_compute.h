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

#ifndef __BH_COMPUTE_H
#define __BH_COMPUTE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct bh_tstate_naive bh_tstate_naive;
struct bh_tstate_naive {
    bh_index coord[BH_MAXDIM];
    bh_index cur_e;
};
void bh_tstate_reset_naive( bh_tstate_naive *state );

typedef struct bh_tstate bh_tstate;
struct bh_tstate {
    bh_index ndim;
    bh_index noperands;
    bh_index shape[BH_MAXDIM];
    void* start[BH_MAX_NO_OPERANDS];
    bh_index stride[BH_MAX_NO_OPERANDS][BH_MAXDIM];
};
void bh_tstate_reset( bh_tstate *state, bh_instruction* instr );

typedef bh_error (*bh_computeloop)( bh_instruction*, bh_tstate* );
typedef bh_error (*bh_computeloop_naive)( bh_instruction*, bh_tstate_naive*, bh_index );

bh_computeloop_naive bh_compute_get_naive( bh_instruction *instr );
bh_error bh_compute_apply_naive( bh_instruction *instr );
bh_error bh_compute_reduce_naive( bh_instruction *instr );

bh_computeloop bh_compute_get( bh_instruction *instr );
bh_error bh_compute_apply( bh_instruction *instr );
bh_error bh_compute_reduce( bh_instruction *instr );
bh_error bh_compute_random(bh_userfunc *arg, void* ve_arg);

#ifdef __cplusplus
}
#endif

#endif
