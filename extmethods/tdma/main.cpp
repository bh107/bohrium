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

#include <bh_extmethod.hpp>
#include <bh_main_memory.hpp>

using namespace bohrium;
using namespace extmethod;
using namespace std;

namespace {
class TDMAImpl : public ExtmethodImpl {
private:
    template<typename T>
    void tdma(const T* a, const T* b, const T* c, const T* d, T* c_prime, T* d_prime, const int n) const
    {
      // See https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
      c_prime[0] = c[0] / b[0];
      d_prime[0] = d[0] / b[0];
      for(int i=1; i < n; ++i)
      {
          const T m = 1. / (b[i] - a[i] * c_prime[i-1]);
          c_prime[i] = c[i] * m;
          d_prime[i] = (d[i] - a[i] * d_prime[i-1]) * m;
      }
      for(int i=n-2; i > -1; --i)
      {
          d_prime[i] -= c_prime[i] * d_prime[i+1];
      }
    }

    template<typename T>
    void tdma_reduce(const bh_view* diagonals, const bh_view* rhs, bh_view* out) const
    {
      const int m = rhs->shape[0];
      const int n = rhs->shape[1];
      T* tmp = (T*) malloc(n * m * sizeof(T));
      T *a = (T*) diagonals->base->data + diagonals->start;
      T *b = (T*) diagonals->base->data + diagonals->start + diagonals->stride[0];
      T *c = (T*) diagonals->base->data + diagonals->start + 2*diagonals->stride[0];
      T *r = (T*) rhs->base->data + rhs->start;
      T *o = (T*) out->base->data + out->start;

      #pragma omp parallel for
      for(int i=0; i < m; ++i)
      {
          tdma(a + i * diagonals->stride[1],
               b + i * diagonals->stride[1],
               c + i * diagonals->stride[1],
               r + i * rhs->stride[0],
               tmp + i * n,
               o + i * out->stride[0],
               n);
      }
      free(tmp);
    }

public:
    void execute(bh_instruction *instr, void* arg) {
        bh_view *diagonals = &instr->operand[1];
        assert(diagonals->ndim == 3);
        assert(diagonals->shape[0] == 3);

        bh_view *out = &instr->operand[0];
        assert(out->ndim == 2);
        assert(out->shape[0] == diagonals->shape[1] && out->shape[1] == diagonals->shape[2]);
        assert(out->base->type == diagonals->base->type);

        bh_view *rhs = &instr->operand[2];
        assert(rhs->ndim == 2);
        assert(rhs->shape[0] == diagonals->shape[1] && rhs->shape[1] == diagonals->shape[2]);
        assert(rhs->base->type == diagonals->base->type);

        // cout << diagonals->pprint(false) << endl;
        // cout << rhs->pprint(false) << endl;
        // cout << out->pprint(false) << endl;

        //Make sure that the arrays memory are allocated.
        bh_data_malloc(diagonals->base);
        bh_data_malloc(out->base);
        bh_data_malloc(rhs->base);

        switch(diagonals->base->type) {
            case bh_type::FLOAT32:
                tdma_reduce<bh_float32>(diagonals, rhs, out);
                break;
            case bh_type::FLOAT64:
                tdma_reduce<bh_float64>(diagonals, rhs, out);
                break;
            default:
                throw runtime_error("DTYPE must be float32 or float64");
        }
    }
};
} // Unnamed namespace

extern "C" ExtmethodImpl* tdma_create() {
    return new TDMAImpl();
}
extern "C" void tdma_destroy(ExtmethodImpl* self) {
    delete self;
}
