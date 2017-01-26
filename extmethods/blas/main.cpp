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

#include <bh_extmethod.hpp>

#if defined(__APPLE__) || defined(__MACOSX)
    #include <Accelerate/Accelerate.h>
#else
    #include <cblas.h>
#endif

#include <stdio.h>
#include <map>

using namespace bohrium;
using namespace extmethod;
using namespace std;

namespace {
    /*
      Implements sgemm, dgemm, cgemm, and zgemm from BLAS
      ?gemm calculates:
          C <- αAB + βC
      that is, it take three matrices and two constants (here always 1 and 0)
      and calculates the matrix product between the two first, add the third and
      stores the result in the third.

      Bohrium takes the C matrix as result operand, thus it is the first, and
      then A and B follows.
    */
    struct GemmImpl : public ExtmethodImpl {
    public:
        void execute(bh_instruction *instr, void* arg) {
            // All matrices must be contigous
            assert(instr->is_contiguous());
            // C is a m*n matrix
            bh_view* C = &instr->operand[0];
            int m = C->shape[0];
            int n = C->shape[1];

            // We allocate the C data, if not already present
            if (bh_data_malloc(C->base) != BH_SUCCESS) {
                cerr << "Cannot allocate memory for C-matrix" << endl;
                return;
            }

            // A is a m*k matrix
            bh_view* A = &instr->operand[1];
            int k = A->shape[1];

            // B is a k*n matrix
            bh_view* B = &instr->operand[2];

            assert(A->base->type == B->base->type);
            assert(B->base->type == C->base->type);

            void *A_data, *B_data, *C_data;
            bh_data_get(A, (bh_data_ptr*) &A_data);
            bh_data_get(B, (bh_data_ptr*) &B_data);
            bh_data_get(C, (bh_data_ptr*) &C_data);

            switch (A->base->type) {
                case BH_FLOAT32:
                    gemm((bh_float32*) A_data, (bh_float32*) B_data, (bh_float32*) C_data, m, n, k);
                    break;
                case BH_FLOAT64:
                    gemm((bh_float64*) A_data, (bh_float64*) B_data, (bh_float64*) C_data, m, n, k);
                    break;
                case BH_COMPLEX64:
                    gemm((bh_complex64*) A_data, (bh_complex64*) B_data, (bh_complex64*) C_data, m, n, k);
                    break;
                case BH_COMPLEX128:
                    gemm((bh_complex128*) A_data, (bh_complex128*) B_data, (bh_complex128*) C_data, m, n, k);
                    break;
                default:
                    cerr << bh_type_text(A->base->type) << " not supported by BLAS.\n" << endl;
                    return;
            }
        }

    private:
        void gemm(bh_float32* A_data, bh_float32* B_data, bh_float32* C_data, int m, int n, int k) {
            // Single precision
            cblas_sgemm(
                CblasRowMajor, // Layout
                CblasNoTrans,  // Transpose A
                CblasNoTrans,  // Transpose B
                m,             // Number of rows in A and C
                n,             // Number of columns in B and C
                k,             // Number of columns in A
                1.0,           // Alpha
                A_data,
                k,             // Stride of A
                B_data,
                n,             // Stride of B
                0.0,           // Beta
                C_data,
                n              // Stride of C
            );
        }

        void gemm(bh_float64* A_data, bh_float64* B_data, bh_float64* C_data, int m, int n, int k) {
            // Double precision
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, A_data, k, B_data, n, 0.0, C_data, n);
        }

        void gemm(bh_complex64* A_data, bh_complex64* B_data, bh_complex64* C_data, int m, int n, int k) {
            // Complex single precision
            bh_complex64 alpha; alpha.real = 1.0; alpha.imag = 0.0;
            bh_complex64  beta;  beta.real = 0.0;  beta.imag = 0.0;
            cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, (void*) &alpha, A_data, k, B_data, n, (void*) &beta, C_data, n);
        }

        void gemm(bh_complex128* A_data, bh_complex128* B_data, bh_complex128* C_data, int m, int n, int k) {
            // Complex double precision
            bh_complex128 alpha; alpha.real = 1.0; alpha.imag = 0.0;
            bh_complex128  beta;  beta.real = 0.0;  beta.imag = 0.0;
            cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, (void*) &alpha, A_data, k, B_data, n, (void*) &beta, C_data, n);
        }
    };
}

extern "C" ExtmethodImpl* blas_gemm_create() {
    return new GemmImpl();
}
extern "C" void blas_gemm_destroy(ExtmethodImpl* self) {
    delete self;
}
