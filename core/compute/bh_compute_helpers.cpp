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
#include <assert.h>
#include <bh.h>
#include <bh_compute.h>

//This function protects against arrays that have superfluos dimensions,
// which could negatively affect computation speed
void bh_compact_dimensions(bh_tstate* state)
{
	bh_index i, j;

	//As long as the inner dimension has length == 1, we can safely ignore it
	while(state->ndim > 1 && (state->shape[state->ndim - 1] == 1 || state->shape[state->ndim - 1] == 0))
		state->ndim--;

	//We can also remove dimensions inside the seqence
	for(i = state->ndim-2; i >= 0; i--)
	{
		if (state->shape[i] == 1 || state->shape[i] == 0)
		{
			state->ndim--;
			memmove(&state->shape[i], &state->shape[i+1], (state->ndim - i) * sizeof(bh_index));
			for(j = 0; j < state->noperands; j++)
				memmove(&state->stride[j][i], &state->stride[j][i+1], (state->ndim - i) * sizeof(bh_index));
		}
	}
}

void bh_tstate_reset( bh_tstate *state, bh_instruction *instr ) {

    bh_index i, j, blocksize, elsize;
    void* basep;

    state->ndim         = instr->operand[0].ndim;
    state->noperands    = bh_operands(instr->opcode);
    blocksize 			= state->ndim * sizeof(bh_index);

    // As all arrays have the same dimensions, we keep a single shared shape
    memcpy(state->shape, instr->operand[0].shape, blocksize);

    //Prepare strides for compacting
    for(i = 0; i < state->noperands; i++)
    if (!bh_is_constant(&instr->operand[i]))
        memcpy(state->stride[i], instr->operand[i].stride, blocksize);

    bh_compact_dimensions(state);

    // Revisit the strides and pre-calculate them for traversal
    for(i = 0; i < state->noperands; i++)
    {
        if (!bh_is_constant(&instr->operand[i]))
        {
            elsize = bh_type_size(instr->operand[i].base->type);

            // Precalculate the pointer
            basep = instr->operand[i].base->data;
            if(basep == NULL)
            {
                printf("Input array wasn't initiated ");
                bh_pprint_array(&instr->operand[i]);
                assert(basep != NULL);
            }
            state->start[i] = (void*)(((char*)basep) + (instr->operand[i].start * elsize));

            // Precalculate the strides in bytes,
            // relative to the size of the underlying dimension
            for(j = 0; j < state->ndim - 1; j++) {
                state->stride[i][j] = (state->stride[i][j] - (state->stride[i][j+1] * state->shape[j+1])) * elsize;
            }
            state->stride[i][state->ndim - 1] = state->stride[i][state->ndim - 1] * elsize;
        }
    }
}

/**
 * Execute an instruction using the optimization traversal.
 *
 * @param instr Instruction to execute.
 * @return Status of execution
 */
bh_error bh_compute_apply( bh_instruction *instr ) {

    bh_computeloop comp = bh_compute_get( instr );

    if (comp == NULL) {
    	return bh_compute_reduce(instr);
    } else {
		bh_tstate state;
		bh_tstate_reset( &state, instr );
        return comp( instr, &state );
    }

}

//
// Below is usage of the naive traversal.
//
void bh_tstate_reset_naive( bh_tstate_naive *state ) {
    memset(state->coord, 0, BH_MAXDIM * sizeof(bh_index));
    state->cur_e = 0;
}

bh_error bh_compute_apply_naive( bh_instruction *instr ) {

    bh_computeloop_naive comp = bh_compute_get_naive( instr );

    if (comp == NULL) {
    	return bh_compute_reduce_naive(instr);
    } else {
    	bh_tstate_naive state;
	    bh_tstate_reset_naive( &state );
        return comp( instr, &state, 0 );
    }

}

