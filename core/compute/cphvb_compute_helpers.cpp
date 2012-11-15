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
#include <cphvb.h>
#include <cphvb_compute.h>

//This function protects against arrays that have superfluos dimensions,
// which could negatively affect computation speed
void cphvb_compact_dimensions(cphvb_tstate* state)
{
	cphvb_index i, j;
	
	//As long as the inner dimension has length == 1, we can safely ignore it
	while(state->ndim > 1 && (state->shape[state->ndim - 1] == 1 || state->shape[state->ndim - 1] == 0))
		state->ndim--;

	//We can also remove dimensions inside the seqence
	for(i = state->ndim-2; i >= 0; i--) 
	{
		if (state->shape[i] == 1 || state->shape[i] == 0) 
		{
			state->ndim--;
			memmove(&state->shape[i], &state->shape[i+1], (state->ndim - i) * sizeof(cphvb_index));
			for(j = 0; j < state->noperands; j++)
				memmove(&state->stride[j][i], &state->stride[j][i+1], (state->ndim - i) * sizeof(cphvb_index));
		}
	}
}

void cphvb_tstate_reset( cphvb_tstate *state, cphvb_instruction *instr ) {

	// As all arrays have the same dimensions, we keep a single shared shape
	cphvb_index i, blocksize;
	
	state->ndim         = instr->operand[0]->ndim;
	state->noperands    = cphvb_operands(instr->opcode);
	blocksize           = state->ndim * sizeof(cphvb_index);
	memcpy(state->shape, instr->operand[0]->shape, blocksize);

	for(i = 0; i < state->noperands; i++) {
        if (!cphvb_is_constant(instr->operand[i])) {
			memcpy(state->stride[i], instr->operand[i]->stride, blocksize);
			state->start[i] = instr->operand[i]->start;
		}
	}

	cphvb_compact_dimensions(state);
}

/**
 * Execute an instruction using the optimization traversal.
 *
 * @param instr Instruction to execute.
 * @return Status of execution
 */
cphvb_error cphvb_compute_apply( cphvb_instruction *instr ) {

    cphvb_computeloop comp = cphvb_compute_get( instr );
    cphvb_tstate state;
    cphvb_tstate_reset( &state, instr );
    
    if (comp == NULL) {
        return CPHVB_TYPE_NOT_SUPPORTED;
    } else {
        return comp( instr, &state );
    }

}

//
// Below is usage of the naive traversal.
//
void cphvb_tstate_reset_naive( cphvb_tstate_naive *state ) {
    memset(state->coord, 0, CPHVB_MAXDIM * sizeof(cphvb_index));
    state->cur_e = 0;   
}

cphvb_error cphvb_compute_apply_naive( cphvb_instruction *instr ) {

    cphvb_computeloop_naive comp = cphvb_compute_get_naive( instr );
    cphvb_tstate_naive state;
    cphvb_tstate_reset_naive( &state );
    
    if (comp == NULL) {
        return CPHVB_TYPE_NOT_SUPPORTED;
    } else {
        return comp( instr, &state, 0 );
    }

}

