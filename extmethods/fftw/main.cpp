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
#include <stdexcept>
#include <cassert>
#include <fftw3.h>
#if defined(_OPENMP)
#include <omp.h>
#else
static inline int omp_get_max_threads() { return 1; }
static inline int omp_get_thread_num()  { return 0; }
static inline int omp_get_num_threads() { return 1; }
#endif

#include <bh_extmethod.hpp>

using namespace bohrium;
using namespace extmethod;
using namespace std;

namespace {
class Impl : public ExtmethodImpl {
public:
    void execute(bh_instruction *instr, void* arg) {
        bh_view *out  = &instr->operand[0];
        bh_view *in   = &instr->operand[1];
        bh_int32 *args = (bh_int32 *) instr->operand[2].base->getDataPtr();
        assert(args != NULL);
        assert(instr->operand[2].base->nelem() == 1);
        assert(in->ndim == out->ndim);
        assert(out->base->dtype() == BH_COMPLEX128);
        assert(in->base->dtype() == BH_COMPLEX128);

        int sign = args[0];

        //Make sure that the arrays memory are allocated.
        bh_data_malloc(out->base);
        bh_data_malloc(in->base);

        fftw_complex *i = (fftw_complex*) in->base->getDataPtr();
        fftw_complex *o = (fftw_complex*) out->base->getDataPtr();

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
            throw runtime_error("fftw plan fail!");

        fftw_execute(p);
        fftw_destroy_plan(p);
        fftw_cleanup_threads();
    }
};
} // Unnamed namespace

extern "C" ExtmethodImpl* fftw_create() {
    return new Impl();
}
extern "C" void fftw_destroy(ExtmethodImpl* self) {
    delete self;
}
