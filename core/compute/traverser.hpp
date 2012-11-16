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
#include <assert.h>

typedef char BYTE;

#define INNER_LOOP_AAA(opcode_func, fulls, remainder, d0, d1, d2, stride0, stride1, stride2) \
{ \
	cphvb_index loop_i; \
	for (loop_i = 0; loop_i < fulls; loop_i++) \
	{ \
		opcode_func( ((T0*)d0), ((T1*)d1), ((T2*)d2) ); \
		d0 += stride0; \
		d1 += stride1; \
		d2 += stride2; \
		opcode_func( ((T0*)d0), ((T1*)d1), ((T2*)d2) ); \
		d0 += stride0; \
		d1 += stride1; \
		d2 += stride2; \
		opcode_func( ((T0*)d0), ((T1*)d1), ((T2*)d2) ); \
		d0 += stride0; \
		d1 += stride1; \
		d2 += stride2; \
		opcode_func( ((T0*)d0), ((T1*)d1), ((T2*)d2) ); \
		d0 += stride0; \
		d1 += stride1; \
		d2 += stride2; \
	} \
 \
	switch (remainder) \
	{ \
		case 3: \
			opcode_func( ((T0*)d0), ((T1*)d1), ((T2*)d2) ); \
			d0 += stride0; \
			d1 += stride1; \
			d2 += stride2; \
		case 2: \
			opcode_func( ((T0*)d0), ((T1*)d1), ((T2*)d2) ); \
			d0 += stride0; \
			d1 += stride1; \
			d2 += stride2; \
		case 1: \
			opcode_func( ((T0*)d0), ((T1*)d1), ((T2*)d2) ); \
			d0 += stride0; \
			d1 += stride1; \
			d2 += stride2; \
	} \
}

#define INNER_LOOP_AAC(opcode_func, fulls, remainder, d0, d1, c, stride0, stride1) \
{ \
	cphvb_index loop_i; \
	for (loop_i = 0; loop_i < fulls; loop_i++) \
	{ \
		opcode_func( ((T0*)d0), ((T1*)d1), c ); \
		d0 += stride0; \
		d1 += stride1; \
		opcode_func( ((T0*)d0), ((T1*)d1), c ); \
		d0 += stride0; \
		d1 += stride1; \
		opcode_func( ((T0*)d0), ((T1*)d1), c ); \
		d0 += stride0; \
		d1 += stride1; \
		opcode_func( ((T0*)d0), ((T1*)d1), c ); \
		d0 += stride0; \
		d1 += stride1; \
	} \
 \
	switch (remainder) \
	{ \
		case 3: \
			opcode_func( ((T0*)d0), ((T1*)d1), c ); \
			d0 += stride0; \
			d1 += stride1; \
		case 2: \
			opcode_func( ((T0*)d0), ((T1*)d1), c ); \
			d0 += stride0; \
			d1 += stride1; \
		case 1: \
			opcode_func( ((T0*)d0), ((T1*)d1), c ); \
			d0 += stride0; \
			d1 += stride1; \
	} \
}

#define INNER_LOOP_ACA(opcode_func, fulls, remainder, d0, c, d2, stride0, stride2) \
{ \
	cphvb_index loop_i; \
	for (loop_i = 0; loop_i < fulls; loop_i++) \
	{ \
		opcode_func( ((T0*)d0), c, ((T2*)d2) ); \
		d0 += stride0; \
		d2 += stride2; \
		opcode_func( ((T0*)d0), c, ((T2*)d2) ); \
		d0 += stride0; \
		d2 += stride2; \
		opcode_func( ((T0*)d0), c, ((T2*)d2) ); \
		d0 += stride0; \
		d2 += stride2; \
		opcode_func( ((T0*)d0), c, ((T2*)d2) ); \
		d0 += stride0; \
		d2 += stride2; \
	} \
 \
	switch (remainder) \
	{ \
		case 3: \
			opcode_func( ((T0*)d0), c, ((T2*)d2) ); \
			d0 += stride0; \
			d2 += stride2; \
		case 2: \
			opcode_func( ((T0*)d0), c, ((T2*)d2) ); \
			d0 += stride0; \
			d2 += stride2; \
		case 1: \
			opcode_func( ((T0*)d0), c, ((T2*)d2) ); \
			d0 += stride0; \
			d2 += stride2; \
	} \
}


#define INNER_LOOP_AA(opcode_func, fulls, remainder, d0, d1, stride0, stride1) \
{ \
	cphvb_index loop_i; \
	for (loop_i = 0; loop_i < fulls; loop_i++) \
	{ \
		opcode_func( ((T0*)d0), ((T1*)d1) ); \
		d0 += stride0; \
		d1 += stride1; \
		opcode_func( ((T0*)d0), ((T1*)d1) ); \
		d0 += stride0; \
		d1 += stride1; \
		opcode_func( ((T0*)d0), ((T1*)d1) ); \
		d0 += stride0; \
		d1 += stride1; \
		opcode_func( ((T0*)d0), ((T1*)d1) ); \
		d0 += stride0; \
		d1 += stride1; \
	} \
 \
	switch (remainder) \
	{ \
		case 3: \
			opcode_func( ((T0*)d0), ((T1*)d1) ); \
			d0 += stride0; \
			d1 += stride1; \
		case 2: \
			opcode_func( ((T0*)d0), ((T1*)d1) ); \
			d0 += stride0; \
			d1 += stride1; \
		case 1: \
			opcode_func( ((T0*)d0), ((T1*)d1) ); \
			d0 += stride0; \
			d1 += stride1; \
	} \
}

#define INNER_LOOP_AC(opcode_func, fulls, remainder, d0, c, stride0) \
{ \
	cphvb_index loop_i; \
	for (loop_i = 0; loop_i < fulls; loop_i++) \
	{ \
		opcode_func( ((T0*)d0), (c) ); \
		d0 += stride0; \
		opcode_func( ((T0*)d0), (c) ); \
		d0 += stride0; \
		opcode_func( ((T0*)d0), (c) ); \
		d0 += stride0; \
		opcode_func( ((T0*)d0), (c) ); \
		d0 += stride0; \
	} \
 \
	switch (remainder) \
	{ \
		case 3: \
			opcode_func( ((T0*)d0), (c) ); \
			d0 += stride0; \
		case 2: \
			opcode_func( ((T0*)d0), (c) ); \
			d0 += stride0; \
		case 1: \
			opcode_func( ((T0*)d0), (c) ); \
			d0 += stride0; \
	} \
}

/**
 *  A optimized implementation of executing an instruction.
 *
 *  @param instr The instruction to execute
 *  @param state State of the iteration
 *  @param nelements the number of elements on which the instruction should be applied.
 *  @return This function always returns CPHVB_SUCCESS unless it raises an exception with assert.
 */
template <typename T0, typename T1, typename T2, typename Instr>
cphvb_error traverse_aaa( cphvb_instruction *instr, cphvb_tstate* state ) {

    Instr opcode_func;                        	// Element-wise functor-pointer

    cphvb_index i, j;                        	// Traversal variables

    BYTE* d0;									// Pointers to start of data elements
    BYTE* d1;
    BYTE* d2;

    size_t elsize0 = sizeof(T0);				// We use the size for explicit multiplication
    size_t elsize1 = sizeof(T1);
    size_t elsize2 = sizeof(T2);

    d0 = (BYTE*) cphvb_base_array(instr->operand[0])->data;
    d1 = (BYTE*) cphvb_base_array(instr->operand[1])->data;
    d2 = (BYTE*) cphvb_base_array(instr->operand[2])->data;

    assert(d0 != NULL);                         // Ensure that data is allocated
    assert(d1 != NULL);
    assert(d2 != NULL);

	d0 += state->start[0] * elsize0;			// Compute offsets
    d1 += state->start[1] * elsize1;
    d2 += state->start[2] * elsize2;
		
	if (state->ndim == 1)
	{
		// Simple 1D loop
		cphvb_index stride0 = state->stride[0][0] * elsize0;
		cphvb_index stride1 = state->stride[1][0] * elsize1;
		cphvb_index stride2 = state->stride[2][0] * elsize2;
		
		cphvb_index total_ops = state->shape[0];

		cphvb_index remainder = total_ops % 4;
		cphvb_index fulls = total_ops / 4;

		//Macro magic time!
        INNER_LOOP_AAA(opcode_func, fulls, remainder, d0, d1, d2, stride0, stride1, stride2);
	}
	else if(state->ndim == 2)
	{
		cphvb_index ops_outer = state->shape[0];
		cphvb_index ops_inner = state->shape[1];
		
		cphvb_index outer_stride0 = state->stride[0][0] * elsize0;
		cphvb_index outer_stride1 = state->stride[1][0] * elsize1;
		cphvb_index outer_stride2 = state->stride[2][0] * elsize2;

		// Basic 2D loop with unrolling
		cphvb_index inner_stride0 = state->stride[0][1] * elsize0;
		cphvb_index inner_stride1 = state->stride[1][1] * elsize1;
		cphvb_index inner_stride2 = state->stride[2][1] * elsize2;

		outer_stride0 -= inner_stride0 * state->shape[1];
		outer_stride1 -= inner_stride1 * state->shape[1];
		outer_stride2 -= inner_stride2 * state->shape[1];

		cphvb_index remainder = ops_inner % 4;
		cphvb_index fulls = ops_inner / 4;

		for (i = 0; i < ops_outer; i++)
		{
			//Macro magic time!
            INNER_LOOP_AAA(opcode_func, fulls, remainder, d0, d1, d2, inner_stride0, inner_stride1, inner_stride2);
			d0 += outer_stride0;
			d1 += outer_stride1;
			d2 += outer_stride2;

		}
	}
	else
	{
		//General case, optimal up to 3D, and almost optimal for 4D
		cphvb_index n = state->ndim - 3;
		cphvb_index counters[CPHVB_MAXDIM - 3];
		memset(&counters, 0, sizeof(cphvb_index) * n);		

		cphvb_index total_ops = 1;
		for(i = 0; i < n; i++)
			total_ops *= state->shape[i];
			
		//This chunk of variables prevents repeated calculations of offsets
		cphvb_index dim_index0 = n + 0;
		cphvb_index dim_index1 = n + 1;
		cphvb_index dim_index2 = n + 2;

		cphvb_index ops_outer = state->shape[dim_index0];
		cphvb_index ops_inner = state->shape[dim_index1];
		cphvb_index ops_inner_inner = state->shape[dim_index2];

		cphvb_index outer_stride0 = state->stride[0][dim_index0] * elsize0;
		cphvb_index outer_stride1 = state->stride[1][dim_index0] * elsize1;
		cphvb_index outer_stride2 = state->stride[2][dim_index0] * elsize2;

		cphvb_index inner_stride0 = state->stride[0][dim_index1] * elsize0;
		cphvb_index inner_stride1 = state->stride[1][dim_index1] * elsize1;
		cphvb_index inner_stride2 = state->stride[2][dim_index1] * elsize2;

		cphvb_index inner_inner_stride0 = state->stride[0][dim_index2] * elsize0;
		cphvb_index inner_inner_stride1 = state->stride[1][dim_index2] * elsize1;
		cphvb_index inner_inner_stride2 = state->stride[2][dim_index2] * elsize2;

		outer_stride0 -= inner_stride0 * state->shape[dim_index1];
		outer_stride1 -= inner_stride1 * state->shape[dim_index1];
		outer_stride2 -= inner_stride2 * state->shape[dim_index1];

		inner_stride0 -= inner_inner_stride0 * state->shape[dim_index2];
		inner_stride1 -= inner_inner_stride1 * state->shape[dim_index2];
		inner_stride2 -= inner_inner_stride2 * state->shape[dim_index2];

		cphvb_index remainder = ops_inner_inner % 4;
		cphvb_index fulls = ops_inner_inner / 4;

		BYTE* d0_orig = d0;
		BYTE* d1_orig = d1;
		BYTE* d2_orig = d2;

		while (total_ops-- > 0)
		{
			for (i = 0; i < ops_outer; i++)
			{
				for (j = 0; j < ops_inner; j++)
				{
					//Macro magic time!
                    INNER_LOOP_AAA(opcode_func, fulls, remainder, d0, d1, d2, inner_inner_stride0, inner_inner_stride1, inner_inner_stride2);
					d0 += inner_stride0;
					d1 += inner_stride1;
					d2 += inner_stride2;
				}

				d0 += outer_stride0;
				d1 += outer_stride1;
				d2 += outer_stride2;
			}

			if (n > 0)
			{
				//Basically a ripple carry adder
				long p = n - 1;

				// Move one in current dimension
				d0_orig += (state->stride[0][p] * elsize0);
				d1_orig += (state->stride[1][p] * elsize1);
				d2_orig += (state->stride[2][p] * elsize2);

				while (++counters[p] == state->shape[p] && p > 0)
				{
					//Update to move in the outer dimension, on carry
					d0_orig += ((state->stride[0][p-1]) - (state->shape[p] * state->stride[0][p])) * elsize0;
					d1_orig += ((state->stride[1][p-1]) - (state->shape[p] * state->stride[1][p])) * elsize1;
					d2_orig += ((state->stride[2][p-1]) - (state->shape[p] * state->stride[2][p])) * elsize2;

					counters[p] = 0;
					p--;
				}
				
				d0 = d0_orig;
				d1 = d1_orig;
				d2 = d2_orig;

			}
		}		
	}

    return CPHVB_SUCCESS;

}

/**
 *  A optimized implementation of executing an instruction.
 *
 *  @param instr The instruction to execute
 *  @param state State of the iteration
 *  @param nelements the number of elements on which the instruction should be applied.
 *  @return This function always returns CPHVB_SUCCESS unless it raises an exception with assert.
 */
template <typename T0, typename T1, typename T2, typename Instr>
cphvb_error traverse_aac( cphvb_instruction *instr, cphvb_tstate* state ) {

    Instr opcode_func;                        	// Element-wise functor-pointer

    cphvb_index i, j;                        	// Traversal variables

    BYTE* d0;									// Pointers to start of data elements
    BYTE* d1;

    size_t elsize0 = sizeof(T0);				// We use the size for explicit multiplication
    size_t elsize1 = sizeof(T1);

    d0 = (BYTE*) cphvb_base_array(instr->operand[0])->data;
    d1 = (BYTE*) cphvb_base_array(instr->operand[1])->data;
    T2* c = (T2*) &(instr->constant.value);

    assert(d0 != NULL);                         // Ensure that data is allocated
    assert(d1 != NULL);
    assert(c != NULL);

	d0 += state->start[0] * elsize0;			// Compute offsets
    d1 += state->start[1] * elsize1;
		
	if (state->ndim == 1)
	{
		// Simple 1D loop
		cphvb_index stride0 = state->stride[0][0] * elsize0;
		cphvb_index stride1 = state->stride[1][0] * elsize1;
		
		cphvb_index total_ops = state->shape[0];

		cphvb_index remainder = total_ops % 4;
		cphvb_index fulls = total_ops / 4;

		//Macro magic time!
        INNER_LOOP_AAC(opcode_func, fulls, remainder, d0, d1, c, stride0, stride1);
	}
	else if(state->ndim == 2)
	{
		cphvb_index ops_outer = state->shape[0];
		cphvb_index ops_inner = state->shape[1];
		
		cphvb_index outer_stride0 = state->stride[0][0] * elsize0;
		cphvb_index outer_stride1 = state->stride[1][0] * elsize1;

		// Basic 2D loop with unrolling
		cphvb_index inner_stride0 = state->stride[0][1] * elsize0;
		cphvb_index inner_stride1 = state->stride[1][1] * elsize1;

		outer_stride0 -= inner_stride0 * state->shape[1];
		outer_stride1 -= inner_stride1 * state->shape[1];

		cphvb_index remainder = ops_inner % 4;
		cphvb_index fulls = ops_inner / 4;

		for (i = 0; i < ops_outer; i++)
		{
			//Macro magic time!
            INNER_LOOP_AAC(opcode_func, fulls, remainder, d0, d1, c, inner_stride0, inner_stride1);
			d0 += outer_stride0;
			d1 += outer_stride1;

		}
	}
	else
	{
		//General case, optimal up to 3D, and almost optimal for 4D
		cphvb_index n = state->ndim - 3;
		cphvb_index counters[CPHVB_MAXDIM - 3];
		memset(&counters, 0, sizeof(cphvb_index) * n);		

		cphvb_index total_ops = 1;
		for(i = 0; i < n; i++)
			total_ops *= state->shape[i];
			
		//This chunk of variables prevents repeated calculations of offsets
		cphvb_index dim_index0 = n + 0;
		cphvb_index dim_index1 = n + 1;
		cphvb_index dim_index2 = n + 2;

		cphvb_index ops_outer = state->shape[dim_index0];
		cphvb_index ops_inner = state->shape[dim_index1];
		cphvb_index ops_inner_inner = state->shape[dim_index2];

		cphvb_index outer_stride0 = state->stride[0][dim_index0] * elsize0;
		cphvb_index outer_stride1 = state->stride[1][dim_index0] * elsize1;

		cphvb_index inner_stride0 = state->stride[0][dim_index1] * elsize0;
		cphvb_index inner_stride1 = state->stride[1][dim_index1] * elsize1;

		cphvb_index inner_inner_stride0 = state->stride[0][dim_index2] * elsize0;
		cphvb_index inner_inner_stride1 = state->stride[1][dim_index2] * elsize1;

		outer_stride0 -= inner_stride0 * state->shape[dim_index1];
		outer_stride1 -= inner_stride1 * state->shape[dim_index1];

		inner_stride0 -= inner_inner_stride0 * state->shape[dim_index2];
		inner_stride1 -= inner_inner_stride1 * state->shape[dim_index2];

		cphvb_index remainder = ops_inner_inner % 4;
		cphvb_index fulls = ops_inner_inner / 4;

		BYTE* d0_orig = d0;
		BYTE* d1_orig = d1;

		while (total_ops-- > 0)
		{
			for (i = 0; i < ops_outer; i++)
			{
				for (j = 0; j < ops_inner; j++)
				{
					//Macro magic time!
                    INNER_LOOP_AAC(opcode_func, fulls, remainder, d0, d1, c, inner_inner_stride0, inner_inner_stride1);
					d0 += inner_stride0;
					d1 += inner_stride1;
				}

				d0 += outer_stride0;
				d1 += outer_stride1;
			}

			if (n > 0)
			{
				//Basically a ripple carry adder
				long p = n - 1;

				// Move one in current dimension
				d0_orig += (state->stride[0][p] * elsize0);
				d1_orig += (state->stride[1][p] * elsize1);

				while (++counters[p] == state->shape[p] && p > 0)
				{
					//Update to move in the outer dimension, on carry
					d0_orig += ((state->stride[0][p-1]) - (state->shape[p] * state->stride[0][p])) * elsize0;
					d1_orig += ((state->stride[1][p-1]) - (state->shape[p] * state->stride[1][p])) * elsize1;

					counters[p] = 0;
					p--;
				}
				
				d0 = d0_orig;
				d1 = d1_orig;

			}
		}		
	}

    return CPHVB_SUCCESS;

}

/**
 *  A optimized implementation of executing an instruction.
 *
 *  @param instr The instruction to execute
 *  @param state State of the iteration
 *  @param nelements the number of elements on which the instruction should be applied.
 *  @return This function always returns CPHVB_SUCCESS unless it raises an exception with assert.
 */
template <typename T0, typename T1, typename T2, typename Instr>
cphvb_error traverse_aca( cphvb_instruction *instr, cphvb_tstate* state ) {

    Instr opcode_func;                        	// Element-wise functor-pointer

    cphvb_index i, j;                        	// Traversal variables

    BYTE* d0;									// Pointers to start of data elements
    BYTE* d2;

    size_t elsize0 = sizeof(T0);				// We use the size for explicit multiplication
    size_t elsize2 = sizeof(T2);

    d0 = (BYTE*) cphvb_base_array(instr->operand[0])->data;
    T1* c = (T1*) &(instr->constant.value);
    d2 = (BYTE*) cphvb_base_array(instr->operand[2])->data;

    assert(d0 != NULL);                         // Ensure that data is allocated
    assert(c != NULL);
    assert(d2 != NULL);

	d0 += state->start[0] * elsize0;			// Compute offsets
    d2 += state->start[2] * elsize2;
		
	if (state->ndim == 1)
	{
		// Simple 1D loop
		cphvb_index stride0 = state->stride[0][0] * elsize0;
		cphvb_index stride2 = state->stride[2][0] * elsize2;
		
		cphvb_index total_ops = state->shape[0];

		cphvb_index remainder = total_ops % 4;
		cphvb_index fulls = total_ops / 4;

		//Macro magic time!
        INNER_LOOP_ACA(opcode_func, fulls, remainder, d0, c, d2, stride0, stride2);
	}
	else if(state->ndim == 2)
	{
		cphvb_index ops_outer = state->shape[0];
		cphvb_index ops_inner = state->shape[1];
		
		cphvb_index outer_stride0 = state->stride[0][0] * elsize0;
		cphvb_index outer_stride2 = state->stride[2][0] * elsize2;

		// Basic 2D loop with unrolling
		cphvb_index inner_stride0 = state->stride[0][1] * elsize0;
		cphvb_index inner_stride2 = state->stride[2][1] * elsize2;

		outer_stride0 -= inner_stride0 * state->shape[1];
		outer_stride2 -= inner_stride2 * state->shape[1];

		cphvb_index remainder = ops_inner % 4;
		cphvb_index fulls = ops_inner / 4;

		for (i = 0; i < ops_outer; i++)
		{
			//Macro magic time!
            INNER_LOOP_ACA(opcode_func, fulls, remainder, d0, c, d2, inner_stride0, inner_stride2);
			d0 += outer_stride0;
			d2 += outer_stride2;

		}
	}
	else
	{
		//General case, optimal up to 3D, and almost optimal for 4D
		cphvb_index n = state->ndim - 3;
		cphvb_index counters[CPHVB_MAXDIM - 3];
		memset(&counters, 0, sizeof(cphvb_index) * n);		

		cphvb_index total_ops = 1;
		for(i = 0; i < n; i++)
			total_ops *= state->shape[i];
			
		//This chunk of variables prevents repeated calculations of offsets
		cphvb_index dim_index0 = n + 0;
		cphvb_index dim_index1 = n + 1;
		cphvb_index dim_index2 = n + 2;

		cphvb_index ops_outer = state->shape[dim_index0];
		cphvb_index ops_inner = state->shape[dim_index1];
		cphvb_index ops_inner_inner = state->shape[dim_index2];

		cphvb_index outer_stride0 = state->stride[0][dim_index0] * elsize0;
		cphvb_index outer_stride2 = state->stride[2][dim_index0] * elsize2;

		cphvb_index inner_stride0 = state->stride[0][dim_index1] * elsize0;
		cphvb_index inner_stride2 = state->stride[2][dim_index1] * elsize2;

		cphvb_index inner_inner_stride0 = state->stride[0][dim_index2] * elsize0;
		cphvb_index inner_inner_stride2 = state->stride[2][dim_index2] * elsize2;

		outer_stride0 -= inner_stride0 * state->shape[dim_index1];
		outer_stride2 -= inner_stride2 * state->shape[dim_index1];

		inner_stride0 -= inner_inner_stride0 * state->shape[dim_index2];
		inner_stride2 -= inner_inner_stride2 * state->shape[dim_index2];

		cphvb_index remainder = ops_inner_inner % 4;
		cphvb_index fulls = ops_inner_inner / 4;

		BYTE* d0_orig = d0;
		BYTE* d2_orig = d2;

		while (total_ops-- > 0)
		{
			for (i = 0; i < ops_outer; i++)
			{
				for (j = 0; j < ops_inner; j++)
				{
					//Macro magic time!
                    INNER_LOOP_ACA(opcode_func, fulls, remainder, d0, c, d2, inner_inner_stride0, inner_inner_stride2);
					d0 += inner_stride0;
					d2 += inner_stride2;
				}

				d0 += outer_stride0;
				d2 += outer_stride2;
			}

			if (n > 0)
			{
				//Basically a ripple carry adder
				long p = n - 1;

				// Move one in current dimension
				d0_orig += (state->stride[0][p] * elsize0);
				d2_orig += (state->stride[2][p] * elsize2);

				while (++counters[p] == state->shape[p] && p > 0)
				{
					//Update to move in the outer dimension, on carry
					d0_orig += ((state->stride[0][p-1]) - (state->shape[p] * state->stride[0][p])) * elsize0;
					d2_orig += ((state->stride[2][p-1]) - (state->shape[p] * state->stride[2][p])) * elsize2;

					counters[p] = 0;
					p--;
				}
				
				d0 = d0_orig;
				d2 = d2_orig;

			}
		}		
	}

    return CPHVB_SUCCESS;

}

/**
 *  A optimized implementation of executing an instruction.
 *
 *  @param instr The instruction to execute
 *  @param state State of the iteration
 *  @param nelements the number of elements on which the instruction should be applied.
 *  @return This function always returns CPHVB_SUCCESS unless it raises an exception with assert.
 */
template <typename T0, typename T1, typename Instr>
cphvb_error traverse_aa( cphvb_instruction *instr, cphvb_tstate* state ) {

    Instr opcode_func;                        	// Element-wise functor-pointer

    cphvb_index i, j;                        	// Traversal variables

    BYTE* d0;									// Pointers to start of data elements
    BYTE* d1;

    size_t elsize0 = sizeof(T0);				// We use the size for explicit multiplication
    size_t elsize1 = sizeof(T1);

    d0 = (BYTE*) cphvb_base_array(instr->operand[0])->data;
    d1 = (BYTE*) cphvb_base_array(instr->operand[1])->data;

    assert(d0 != NULL);                         // Ensure that data is allocated
    assert(d1 != NULL);

	d0 += state->start[0] * elsize0;			// Compute offsets
    d1 += state->start[1] * elsize1;
		
	if (state->ndim == 1)
	{
		// Simple 1D loop
		cphvb_index stride0 = state->stride[0][0] * elsize0;
		cphvb_index stride1 = state->stride[1][0] * elsize1;
		
		cphvb_index total_ops = state->shape[0];

		cphvb_index remainder = total_ops % 4;
		cphvb_index fulls = total_ops / 4;

		//Macro magic time!
        INNER_LOOP_AA(opcode_func, fulls, remainder, d0, d1, stride0, stride1);
	}
	else if(state->ndim == 2)
	{
		cphvb_index ops_outer = state->shape[0];
		cphvb_index ops_inner = state->shape[1];
		
		cphvb_index outer_stride0 = state->stride[0][0] * elsize0;
		cphvb_index outer_stride1 = state->stride[1][0] * elsize1;

		// Basic 2D loop with unrolling
		cphvb_index inner_stride0 = state->stride[0][1] * elsize0;
		cphvb_index inner_stride1 = state->stride[1][1] * elsize1;

		outer_stride0 -= inner_stride0 * state->shape[1];
		outer_stride1 -= inner_stride1 * state->shape[1];

		cphvb_index remainder = ops_inner % 4;
		cphvb_index fulls = ops_inner / 4;

		for (i = 0; i < ops_outer; i++)
		{
			//Macro magic time!
            INNER_LOOP_AA(opcode_func, fulls, remainder, d0, d1, inner_stride0, inner_stride1);
			d0 += outer_stride0;
			d1 += outer_stride1;

		}
	}
	else
	{
		//General case, optimal up to 3D, and almost optimal for 4D
		cphvb_index n = state->ndim - 3;
		cphvb_index counters[CPHVB_MAXDIM - 3];
		memset(&counters, 0, sizeof(cphvb_index) * n);		

		cphvb_index total_ops = 1;
		for(i = 0; i < n; i++)
			total_ops *= state->shape[i];
			
		//This chunk of variables prevents repeated calculations of offsets
		cphvb_index dim_index0 = n + 0;
		cphvb_index dim_index1 = n + 1;
		cphvb_index dim_index2 = n + 2;

		cphvb_index ops_outer = state->shape[dim_index0];
		cphvb_index ops_inner = state->shape[dim_index1];
		cphvb_index ops_inner_inner = state->shape[dim_index2];

		cphvb_index outer_stride0 = state->stride[0][dim_index0] * elsize0;
		cphvb_index outer_stride1 = state->stride[1][dim_index0] * elsize1;

		cphvb_index inner_stride0 = state->stride[0][dim_index1] * elsize0;
		cphvb_index inner_stride1 = state->stride[1][dim_index1] * elsize1;

		cphvb_index inner_inner_stride0 = state->stride[0][dim_index2] * elsize0;
		cphvb_index inner_inner_stride1 = state->stride[1][dim_index2] * elsize1;

		outer_stride0 -= inner_stride0 * state->shape[dim_index1];
		outer_stride1 -= inner_stride1 * state->shape[dim_index1];

		inner_stride0 -= inner_inner_stride0 * state->shape[dim_index2];
		inner_stride1 -= inner_inner_stride1 * state->shape[dim_index2];

		cphvb_index remainder = ops_inner_inner % 4;
		cphvb_index fulls = ops_inner_inner / 4;

		BYTE* d0_orig = d0;
		BYTE* d1_orig = d1;

		while (total_ops-- > 0)
		{
			for (i = 0; i < ops_outer; i++)
			{
				for (j = 0; j < ops_inner; j++)
				{
					//Macro magic time!
                    INNER_LOOP_AA(opcode_func, fulls, remainder, d0, d1, inner_inner_stride0, inner_inner_stride1);
					d0 += inner_stride0;
					d1 += inner_stride1;
				}

				d0 += outer_stride0;
				d1 += outer_stride1;
			}

			if (n > 0)
			{
				//Basically a ripple carry adder
				long p = n - 1;

				// Move one in current dimension
				d0_orig += (state->stride[0][p] * elsize0);
				d1_orig += (state->stride[1][p] * elsize1);

				while (++counters[p] == state->shape[p] && p > 0)
				{
					//Update to move in the outer dimension, on carry
					d0_orig += ((state->stride[0][p-1]) - (state->shape[p] * state->stride[0][p])) * elsize0;
					d1_orig += ((state->stride[1][p-1]) - (state->shape[p] * state->stride[1][p])) * elsize1;

					counters[p] = 0;
					p--;
				}
				
				d0 = d0_orig;
				d1 = d1_orig;

			}
		}		
	}

    return CPHVB_SUCCESS;

}

/**
 *  A optimized implementation of executing an instruction.
 *
 *  @param instr The instruction to execute
 *  @param state State of the iteration
 *  @param nelements the number of elements on which the instruction should be applied.
 *  @return This function always returns CPHVB_SUCCESS unless it raises an exception with assert.
 */
template <typename T0, typename T1, typename Instr>
cphvb_error traverse_ac( cphvb_instruction *instr, cphvb_tstate* state ) {

    Instr opcode_func;                        	// Element-wise functor-pointer

    cphvb_index i, j;                        	// Traversal variables

    BYTE* d0;									// Pointers to start of data elements

    size_t elsize0 = sizeof(T0);				// We use the size for explicit multiplication

    d0 = (BYTE*) cphvb_base_array(instr->operand[0])->data;
    T1* c = (T1*) &(instr->constant.value);

    assert(d0 != NULL);                         // Ensure that data is allocated
    assert(c != NULL);

	d0 += state->start[0] * elsize0;			// Compute offsets
		
	if (state->ndim == 1)
	{
		// Simple 1D loop
		cphvb_index stride0 = state->stride[0][0] * elsize0;
		
		cphvb_index total_ops = state->shape[0];

		cphvb_index remainder = total_ops % 4;
		cphvb_index fulls = total_ops / 4;

		//Macro magic time!
        INNER_LOOP_AC(opcode_func, fulls, remainder, d0, c, stride0);
	}
	else if(state->ndim == 2)
	{
		cphvb_index ops_outer = state->shape[0];
		cphvb_index ops_inner = state->shape[1];
		
		cphvb_index outer_stride0 = state->stride[0][0] * elsize0;

		// Basic 2D loop with unrolling
		cphvb_index inner_stride0 = state->stride[0][1] * elsize0;

		outer_stride0 -= inner_stride0 * state->shape[1];

		cphvb_index remainder = ops_inner % 4;
		cphvb_index fulls = ops_inner / 4;

		for (i = 0; i < ops_outer; i++)
		{
			//Macro magic time!
            INNER_LOOP_AC(opcode_func, fulls, remainder, d0, c, inner_stride0);
			d0 += outer_stride0;

		}
	}
	else
	{
		//General case, optimal up to 3D, and almost optimal for 4D
		cphvb_index n = state->ndim - 3;
		cphvb_index counters[CPHVB_MAXDIM - 3];
		memset(&counters, 0, sizeof(cphvb_index) * n);		

		cphvb_index total_ops = 1;
		for(i = 0; i < n; i++)
			total_ops *= state->shape[i];
			
		//This chunk of variables prevents repeated calculations of offsets
		cphvb_index dim_index0 = n + 0;
		cphvb_index dim_index1 = n + 1;
		cphvb_index dim_index2 = n + 2;

		cphvb_index ops_outer = state->shape[dim_index0];
		cphvb_index ops_inner = state->shape[dim_index1];
		cphvb_index ops_inner_inner = state->shape[dim_index2];

		cphvb_index outer_stride0 = state->stride[0][dim_index0] * elsize0;

		cphvb_index inner_stride0 = state->stride[0][dim_index1] * elsize0;

		cphvb_index inner_inner_stride0 = state->stride[0][dim_index2] * elsize0;

		outer_stride0 -= inner_stride0 * state->shape[dim_index1];

		inner_stride0 -= inner_inner_stride0 * state->shape[dim_index2];

		cphvb_index remainder = ops_inner_inner % 4;
		cphvb_index fulls = ops_inner_inner / 4;

		BYTE* d0_orig = d0;

		while (total_ops-- > 0)
		{
			for (i = 0; i < ops_outer; i++)
			{
				for (j = 0; j < ops_inner; j++)
				{
					//Macro magic time!
                    INNER_LOOP_AC(opcode_func, fulls, remainder, d0, c, inner_inner_stride0);
					d0 += inner_stride0;
				}

				d0 += outer_stride0;
			}

			if (n > 0)
			{
				//Basically a ripple carry adder
				long p = n - 1;

				// Move one in current dimension
				d0_orig += (state->stride[0][p] * elsize0);

				while (++counters[p] == state->shape[p] && p > 0)
				{
					//Update to move in the outer dimension, on carry
					d0_orig += ((state->stride[0][p-1]) - (state->shape[p] * state->stride[0][p])) * elsize0;

					counters[p] = 0;
					p--;
				}
				
				d0 = d0_orig;

			}
		}		
	}

    return CPHVB_SUCCESS;

}



/**
 *  A naive implementation of executing an instruction.
 *
 *  @param instr The instruction to execute
 *  @param state State of the iteration
 *  @param nelements the number of elements on which the instruction should be applied.
 *  @return This function always returns CPHVB_SUCCESS unless it raises an exception with assert.
 */
template <typename T0, typename T1, typename T2, typename Instr>
cphvb_error traverse_naive_aaa( cphvb_instruction *instr, cphvb_tstate_naive* state, cphvb_index nelements ) {

    Instr opcode_func;                          // Element-wise functor-pointer

    cphvb_array *a0 = instr->operand[0];        // Operand pointers
    cphvb_array *a1 = instr->operand[1];
    cphvb_array *a2 = instr->operand[2];
                                                // Pointers to start of data elements
    T0* d0 = (T0*) cphvb_base_array(instr->operand[0])->data;
    T1* d1 = (T1*) cphvb_base_array(instr->operand[1])->data;
    T2* d2 = (T2*) cphvb_base_array(instr->operand[2])->data;

    assert(d0 != NULL);                         // Ensure that data is allocated
    assert(d1 != NULL);
    assert(d2 != NULL);

    cphvb_index j,                              // Traversal variables
                last_dim    = a0->ndim-1,
                last_e      = (nelements>0) ? nelements-1 : cphvb_nelements( a0->ndim, a0->shape )-1;

    cphvb_index off0;                           // Stride-offset
    cphvb_index off1;
    cphvb_index off2;

    while( state->cur_e <= last_e )
    {
        off0 = a0->start;                           // Compute offset based on coord
        off1 = a1->start;
        off2 = a2->start;

        for( j=0; j<=last_dim; ++j)
        {
            off0 += state->coord[j] * a0->stride[j];
            off1 += state->coord[j] * a1->stride[j];
            off2 += state->coord[j] * a2->stride[j];
        }
                                                    // Iterate over "last" / "innermost" dimension
        for(; (state->coord[last_dim] < a0->shape[last_dim]) && (state->cur_e <= last_e); state->coord[last_dim]++, state->cur_e++ )    
        {
            opcode_func( (off0+d0), (off1+d1), (off2+d2) );

            off0 += a0->stride[last_dim];
            off1 += a1->stride[last_dim];
            off2 += a2->stride[last_dim];
        }

        if (state->coord[last_dim] >= a0->shape[last_dim])
        {
            state->coord[last_dim] = 0;
            for(j = last_dim-1; j >= 0; --j)            // Increment coordinates for the remaining dimensions
            {
                state->coord[j]++;
                if (state->coord[j] < a0->shape[j]) {   // Still within this dimension
                    break;
                } else {                                // Reached the end of this dimension
                    state->coord[j] = 0;                // Reset coordinate
                }                                       // Loop then continues to increment the next dimension
            }
        }

    }

    return CPHVB_SUCCESS;

}

/**
 *  A naive implementation of executing an instruction.
 *
 *  @param instr The instruction to execute
 *  @param state State of the iteration
 *  @param nelements the number of elements on which the instruction should be applied.
 *  @return This function always returns CPHVB_SUCCESS unless it raises an exception with assert.
 */
template <typename T0, typename T1, typename T2, typename Instr>
cphvb_error traverse_naive_aac( cphvb_instruction *instr, cphvb_tstate_naive* state, cphvb_index nelements ) {

    Instr opcode_func;                          // Element-wise functor-pointer

    cphvb_array *a0 = instr->operand[0];        // Operand pointers
    cphvb_array *a1 = instr->operand[1];
                                                // Pointers to start of data elements
    T0* d0 = (T0*) cphvb_base_array(instr->operand[0])->data;
    T1* d1 = (T1*) cphvb_base_array(instr->operand[1])->data;
    T2* d2 = (T2*) &(instr->constant.value);

    assert(d0 != NULL);                         // Ensure that data is allocated
    assert(d1 != NULL);

    cphvb_index j,                              // Traversal variables
                last_dim    = a0->ndim-1,
                last_e      = (nelements>0) ? nelements-1 : cphvb_nelements( a0->ndim, a0->shape )-1;

    cphvb_index off0;                           // Stride-offset
    cphvb_index off1;

    while( state->cur_e <= last_e )
    {
        off0 = a0->start;                           // Compute offset based on coord
        off1 = a1->start;

        for( j=0; j<=last_dim; ++j)
        {
            off0 += state->coord[j] * a0->stride[j];
            off1 += state->coord[j] * a1->stride[j];
        }
                                                    // Iterate over "last" / "innermost" dimension
        for(; (state->coord[last_dim] < a0->shape[last_dim]) && (state->cur_e <= last_e); state->coord[last_dim]++, state->cur_e++ )    
        {
            opcode_func( (off0+d0), (off1+d1), d2 );

            off0 += a0->stride[last_dim];
            off1 += a1->stride[last_dim];
        }

        if (state->coord[last_dim] >= a0->shape[last_dim])
        {
            state->coord[last_dim] = 0;
            for(j = last_dim-1; j >= 0; --j)            // Increment coordinates for the remaining dimensions
            {
                state->coord[j]++;
                if (state->coord[j] < a0->shape[j]) {   // Still within this dimension
                    break;
                } else {                                // Reached the end of this dimension
                    state->coord[j] = 0;                // Reset coordinate
                }                                       // Loop then continues to increment the next dimension
            }
        }

    }

    return CPHVB_SUCCESS;

}

/**
 *  A naive implementation of executing an instruction.
 *
 *  @param instr The instruction to execute
 *  @param state State of the iteration
 *  @param nelements the number of elements on which the instruction should be applied.
 *  @return This function always returns CPHVB_SUCCESS unless it raises an exception with assert.
 */
template <typename T0, typename T1, typename T2, typename Instr>
cphvb_error traverse_naive_aca( cphvb_instruction *instr, cphvb_tstate_naive* state, cphvb_index nelements ) {

    Instr opcode_func;                          // Element-wise functor-pointer

    cphvb_array *a0 = instr->operand[0];        // Operand pointers
    cphvb_array *a2 = instr->operand[2];
                                                // Pointers to start of data elements
    T0* d0 = (T0*) cphvb_base_array(instr->operand[0])->data;
    T1* d1 = (T1*) &(instr->constant.value);
    T2* d2 = (T2*) cphvb_base_array(instr->operand[2])->data;

    assert(d0 != NULL);                         // Ensure that data is allocated
    assert(d2 != NULL);

    cphvb_index j,                              // Traversal variables
                last_dim    = a0->ndim-1,
                last_e      = (nelements>0) ? nelements-1 : cphvb_nelements( a0->ndim, a0->shape )-1;

    cphvb_index off0;                           // Stride-offset
    cphvb_index off2;

    while( state->cur_e <= last_e )
    {
        off0 = a0->start;                           // Compute offset based on coord
        off2 = a2->start;

        for( j=0; j<=last_dim; ++j)
        {
            off0 += state->coord[j] * a0->stride[j];
            off2 += state->coord[j] * a2->stride[j];
        }
                                                    // Iterate over "last" / "innermost" dimension
        for(; (state->coord[last_dim] < a0->shape[last_dim]) && (state->cur_e <= last_e); state->coord[last_dim]++, state->cur_e++ )    
        {
            opcode_func( (off0+d0), d1, (off2+d2) );

            off0 += a0->stride[last_dim];
            off2 += a2->stride[last_dim];
        }

        if (state->coord[last_dim] >= a0->shape[last_dim])
        {
            state->coord[last_dim] = 0;
            for(j = last_dim-1; j >= 0; --j)            // Increment coordinates for the remaining dimensions
            {
                state->coord[j]++;
                if (state->coord[j] < a0->shape[j]) {   // Still within this dimension
                    break;
                } else {                                // Reached the end of this dimension
                    state->coord[j] = 0;                // Reset coordinate
                }                                       // Loop then continues to increment the next dimension
            }
        }

    }

    return CPHVB_SUCCESS;

}

/**
 *  A naive implementation of executing an instruction.
 *
 *  @param instr The instruction to execute
 *  @param state State of the iteration
 *  @param nelements the number of elements on which the instruction should be applied.
 *  @return This function always returns CPHVB_SUCCESS unless it raises an exception with assert.
 */
template <typename T0, typename T1, typename Instr>
cphvb_error traverse_naive_aa( cphvb_instruction *instr, cphvb_tstate_naive* state, cphvb_index nelements ) {

    Instr opcode_func;                          // Element-wise functor-pointer

    cphvb_array *a0 = instr->operand[0];        // Operand pointers
    cphvb_array *a1 = instr->operand[1];
                                                // Pointers to start of data elements
    T0* d0 = (T0*) cphvb_base_array(instr->operand[0])->data;
    T1* d1 = (T1*) cphvb_base_array(instr->operand[1])->data;

    assert(d0 != NULL);                         // Ensure that data is allocated
    assert(d1 != NULL);

    cphvb_index j,                              // Traversal variables
                last_dim    = a0->ndim-1,
                last_e      = (nelements>0) ? nelements-1 : cphvb_nelements( a0->ndim, a0->shape )-1;

    cphvb_index off0;                           // Stride-offset
    cphvb_index off1;

    while( state->cur_e <= last_e )
    {
        off0 = a0->start;                           // Compute offset based on coord
        off1 = a1->start;

        for( j=0; j<=last_dim; ++j)
        {
            off0 += state->coord[j] * a0->stride[j];
            off1 += state->coord[j] * a1->stride[j];
        }
                                                    // Iterate over "last" / "innermost" dimension
        for(; (state->coord[last_dim] < a0->shape[last_dim]) && (state->cur_e <= last_e); state->coord[last_dim]++, state->cur_e++ )    
        {
            opcode_func( (off0+d0), (off1+d1) );

            off0 += a0->stride[last_dim];
            off1 += a1->stride[last_dim];
        }

        if (state->coord[last_dim] >= a0->shape[last_dim])
        {
            state->coord[last_dim] = 0;
            for(j = last_dim-1; j >= 0; --j)            // Increment coordinates for the remaining dimensions
            {
                state->coord[j]++;
                if (state->coord[j] < a0->shape[j]) {   // Still within this dimension
                    break;
                } else {                                // Reached the end of this dimension
                    state->coord[j] = 0;                // Reset coordinate
                }                                       // Loop then continues to increment the next dimension
            }
        }

    }

    return CPHVB_SUCCESS;

}

/**
 *  A naive implementation of executing an instruction.
 *
 *  @param instr The instruction to execute
 *  @param state State of the iteration
 *  @param nelements the number of elements on which the instruction should be applied.
 *  @return This function always returns CPHVB_SUCCESS unless it raises an exception with assert.
 */
template <typename T0, typename T1, typename Instr>
cphvb_error traverse_naive_ac( cphvb_instruction *instr, cphvb_tstate_naive* state, cphvb_index nelements ) {

    Instr opcode_func;                          // Element-wise functor-pointer

    cphvb_array *a0 = instr->operand[0];        // Operand pointers
                                                // Pointers to start of data elements
    T0* d0 = (T0*) cphvb_base_array(instr->operand[0])->data;
    T1* d1 = (T1*) &(instr->constant.value);

    assert(d0 != NULL);                         // Ensure that data is allocated

    cphvb_index j,                              // Traversal variables
                last_dim    = a0->ndim-1,
                last_e      = (nelements>0) ? nelements-1 : cphvb_nelements( a0->ndim, a0->shape )-1;

    cphvb_index off0;                           // Stride-offset

    while( state->cur_e <= last_e )
    {
        off0 = a0->start;                           // Compute offset based on coord

        for( j=0; j<=last_dim; ++j)
        {
            off0 += state->coord[j] * a0->stride[j];
        }
                                                    // Iterate over "last" / "innermost" dimension
        for(; (state->coord[last_dim] < a0->shape[last_dim]) && (state->cur_e <= last_e); state->coord[last_dim]++, state->cur_e++ )    
        {
            opcode_func( (off0+d0), d1 );

            off0 += a0->stride[last_dim];
        }

        if (state->coord[last_dim] >= a0->shape[last_dim])
        {
            state->coord[last_dim] = 0;
            for(j = last_dim-1; j >= 0; --j)            // Increment coordinates for the remaining dimensions
            {
                state->coord[j]++;
                if (state->coord[j] < a0->shape[j]) {   // Still within this dimension
                    break;
                } else {                                // Reached the end of this dimension
                    state->coord[j] = 0;                // Reset coordinate
                }                                       // Loop then continues to increment the next dimension
            }
        }

    }

    return CPHVB_SUCCESS;

}



