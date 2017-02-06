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
#include "../ve/opencl/engine_opencl.hpp"

#include <clBLAS.h>

#include <stdio.h>
#include <map>

using namespace bohrium;
using namespace extmethod;
using namespace std;

namespace {
    /*
      Implements sgemm, dgemm, cgemm, and zgemm from clBLAS
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
        GemmImpl();
        ~GemmImpl();
        void execute(bh_instruction *instr, void* arg);
    private:
        cl_event event = NULL;
    };

    GemmImpl::GemmImpl(void) {
        clblasSetup();
    }

    GemmImpl::~GemmImpl(void) {
        clWaitForEvents(1, &event);
        clReleaseEvent(event);
        clblasTeardown();
    }

    void GemmImpl::execute(bh_instruction *instr, void* arg) {
        EngineOpenCL* engine = (EngineOpenCL*) arg;
        cl_command_queue queue = engine->getCQueue();

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

        cl_mem bufA = engine->getCBuffer(A->base);
        cl_mem bufB = engine->getCBuffer(B->base);
        cl_mem bufC = engine->getCBuffer(C->base);

        // Make sure that everything is copied to device, before executing clBlas method
        clFinish(queue);

        switch (A->base->type) {
            case BH_FLOAT32:
            {
                // Single precision
                clblasSgemm(
                    clblasRowMajor, // Layout
                    clblasNoTrans,  // Transpose A
                    clblasNoTrans,  // Transpose B
                    m,              // Number of rows in A and C
                    n,              // Number of columns in B and C
                    k,              // Number of columns in A
                    1.0,            // Alpha
                    bufA,           // A buffer
                    0,              // Offset for first element of A
                    k,              // LDA
                    bufB,           // B buffer
                    0,              // Offset for first element of B
                    n,              // LDB
                    0.0,            // Beta
                    bufC,           // C buffer
                    0,              // Offset for first element of C
                    n,              // LDC
                    1,              // Number of command queues
                    &queue,         // Command queues
                    0,              // Number of events in the event wait list
                    NULL,           // Event wait list
                    &event          // Event that identify a kernel execution
                );
                break;
            }
            case BH_FLOAT64:
            {
                // Double precision
                clblasDgemm(clblasRowMajor, clblasNoTrans, clblasNoTrans, m, n, k, 1.0, bufA, 0, k, bufB, 0, n, 0.0, bufC, 0, n, 1, &queue, 0, NULL, &event);
                break;
            }
            case BH_COMPLEX64:
            {
                // Complex single precision
                cl_float2 alpha = {{1.0f, 0.0f}};
                cl_float2  beta = {{0.0f, 0.0f}};
                clblasCgemm(clblasRowMajor, clblasNoTrans, clblasNoTrans, m, n, k, alpha, bufA, 0, k, bufB, 0, n, beta, bufC, 0, n, 1, &queue, 0, NULL, &event);
                break;
            }
            case BH_COMPLEX128:
            {
                // Complex double precision
                cl_double2 alpha = {{1.0f, 0.0f}};
                cl_double2  beta = {{0.0f, 0.0f}};
                clblasZgemm(clblasRowMajor, clblasNoTrans, clblasNoTrans, m, n, k, alpha, bufA, 0, k, bufB, 0, n, beta, bufC, 0, n, 1, &queue, 0, NULL, &event);
                break;
            }
            default:
                cerr << bh_type_text(A->base->type) << " not supported by BLAS.\n" << endl;
                return;
        }
    }
}

extern "C" ExtmethodImpl* blas_gemm_create() {
    return new GemmImpl();
}
extern "C" void blas_gemm_destroy(ExtmethodImpl* self) {
    delete self;
}
