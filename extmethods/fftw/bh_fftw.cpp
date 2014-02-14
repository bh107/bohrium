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
#include "bh_fftw.h"
#include <bh.h>
#include <complex.h>
#include <assert.h>
#include <fftw3.h>
#if defined(_OPENMP)
#include <omp.h>
#else
inline int omp_get_max_threads() { return 1; }
inline int omp_get_thread_num()  { return 0; }
inline int omp_get_num_threads() { return 1; }
#endif


/* Implements fftn */
bh_error bh_fftw(bh_instruction *instr, void* arg)
{
    bh_view *out  = &instr->operand[0];
    bh_view *in   = &instr->operand[1];
    bh_int32 *args = (bh_int32 *) instr->operand[2].base->data;
    assert(args != NULL);
    assert(instr->operand[2].base->nelem == 1);
    assert(in->ndim == out->ndim);
    assert(out->base->type == BH_COMPLEX128);
    assert(in->base->type == BH_COMPLEX128);

    int sign = args[0];

    //Make sure that the arrays memory are allocated.
    if(bh_data_malloc(out->base) != BH_SUCCESS)
        return BH_OUT_OF_MEMORY;
    if(bh_data_malloc(in->base) != BH_SUCCESS)
        return BH_OUT_OF_MEMORY;

    fftw_complex *i = (fftw_complex*) in->base->data;
    fftw_complex *o = (fftw_complex*) out->base->data;

    //Convert shape to int
    int shape[BH_MAXDIM];
    for(int i=0; i<in->ndim; ++i)
    {
        shape[i] = in->shape[i];
        assert(shape[i] == out->shape[i]);
    }

    fftw_init_threads();
    fftw_plan_with_nthreads(omp_get_max_threads());

    fftw_plan p = fftw_plan_dft(in->ndim, shape, i, o, sign, FFTW_ESTIMATE);
    if(p == NULL)
        return BH_ERROR;

    fftw_execute(p);
    fftw_destroy_plan(p);
    fftw_cleanup_threads();

    return BH_SUCCESS;
}

