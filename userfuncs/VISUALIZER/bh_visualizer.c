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
#include <bh.h>
#include <assert.h>
#include <iostream>

#include <iostream>
#include "visualizer.hpp"

/**
 *
 * Implementation of the user-defined funtion "nselect".
 * Note that we follow the function signature defined by bh_userfunc_impl.
 *
 */
bool bh_visualize_initialized = false;
bh_error bh_visualizer(bh_userfunc *arg, void* ve_arg)
{
    bh_visualize_type *m_arg = (bh_visualize_type *) arg;
    assert(m_arg->nout == 1);
    assert(m_arg->nin == 1);
    bh_view *A   = &m_arg->operand[0];

    bh_int32 cm = m_arg->cm;
    bh_float32 min = m_arg->min;
    bh_float32 max = m_arg->max;
    bh_bool flat = m_arg->flat;
    bh_bool cube = m_arg->cube;

    //Make sure that the arrays memory are allocated.
    if(bh_data_malloc(A->base) != BH_SUCCESS)
        return BH_OUT_OF_MEMORY;
    if (! bh_visualize_initialized)
    {
        if (A->ndim == 3)
            Visualizer::getInstance().setValues(A, A->shape[0], A->shape[1], A->shape[2], cm, flat, cube, min, max);
        else
            Visualizer::getInstance().setValues(A, A->shape[0], A->shape[1], 1, cm, flat, cube, min, max);
        bh_visualize_initialized = true;
    }

    Visualizer::getInstance().run(A);
    return BH_SUCCESS;
}
