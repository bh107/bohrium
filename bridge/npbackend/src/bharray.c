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

#include "bharray.h"
#include "memory.h"
#include "util.h"

static BhView bhview_new(const PyArrayObject *npy_view, const PyArrayObject *npy_base) {
    BhView ret;
    memset(&ret, 0, sizeof(BhView));
    ret.initiated = 1;
    ret.type_enum = PyArray_TYPE(npy_view);
    ret.ndim = PyArray_NDIM(npy_view);
    {
        size_t v = (size_t) PyArray_DATA((PyArrayObject *) npy_view);
        size_t b = (size_t) PyArray_DATA((PyArrayObject *) npy_base);
        ret.start = (v - b) / PyArray_ITEMSIZE(npy_view);
        if ((v - b) % PyArray_ITEMSIZE(npy_view) != 0) {
            fprintf(stderr, "Fatal error: bhview_new() - the view offset must be element aligned\n");
            assert(1 == 2);
            exit(-1);
        }
    }
    if (ret.ndim <= 0) { // bhc does not support zero-dim arrays
        ret.ndim = 1;
        ret.shape[0] = 1;
        ret.stride[0] = 1;
    } else {
        for(int i=0; i < ret.ndim; ++i) {
            ret.shape[i] = PyArray_DIM(npy_view, i);
            ret.stride[i] = PyArray_STRIDE(npy_view, i) / PyArray_ITEMSIZE(npy_view);
            if(PyArray_STRIDE(npy_view, i) % PyArray_ITEMSIZE(npy_view) != 0) {
                fprintf(stderr, "Fatal error: bhview_new() - the view stride must be element aligned\n");
                assert(1 == 2);
                exit(-1);
            }
        }
    }
    return ret;
}

static int bhview_identical(BhView *v1, BhView *v2) {
    if (v1->type_enum != v2->type_enum) {
        return 0;
    }
    if (v1->start != v2->start) {
        return 0;
    }
    assert(v1->ndim > 0);
    assert(v2->ndim > 0);
    if(v1->ndim != v2->ndim) {
        return 0;
    }
    for(int i=0; i < v1->ndim; ++i) {
        if (v1->shape[i] != v2->shape[i]) {
            return 0;
        }
        if (v1->stride[i] != v2->stride[i]) {
            return 0;
        }
    }
    return 1;
}


void *bharray_bhc(BhArray *ary) {

    // bhc doesn't support empty arrays
    if (PyArray_SIZE((PyArrayObject*) ary) <= 0) {
        fprintf(stderr, "Fatal error: bharray_bhc() - cannot create empty arrays/views\n");
        assert(1 == 2);
        exit(-1);
    }

    // Get the base `ary` and make sure it has a bhc_array
    BhArray *base = get_base((PyObject*) ary);
    if (base == NULL) {
        fprintf(stderr, "Fatal exception in bharray_bhc()\n");
        PyErr_Print();
        assert(1 == 2);
        exit(-1);
    } else if (base != ary) {
        bharray_bhc(base);
    } else if (!base->data_in_bhc) {
        mem_np2bhc(base);
    }

    if(PyArray_TYPE((PyArrayObject *) ary) != PyArray_TYPE((PyArrayObject *) base)) {
        fprintf(stderr, "Fatal error: bharray_bhc() - view and base must have the same dtype\n");
        assert(1 == 2);
        exit(-1);
    }

    // Find the expected view based on the values of `ary`, which might have changed since last time accessed
    BhView ex_view = bhview_new((PyArrayObject *) ary, (PyArrayObject *) base);
    const bhc_dtype ex_dtype = dtype_np2bhc(ex_view.type_enum);

    // Make sure that `ary->bhc_array` and `ary->view` is as expected
    if (!ary->view.initiated) {
        assert(ary->bhc_array == NULL);
        if (base == ary) {
            void *new_base = BhAPI_new(ex_dtype, PyArray_SIZE((PyArrayObject*) ary));
            ary->bhc_array = BhAPI_view(ex_dtype, new_base, ex_view.ndim, ex_view.start, ex_view.shape, ex_view.stride);
            BhAPI_destroy(ex_dtype, new_base);
        } else {
            ary->bhc_array = BhAPI_view(ex_dtype, base->bhc_array, ex_view.ndim,
                                      ex_view.start, ex_view.shape, ex_view.stride);
        }
        ary->view = ex_view;
        PyObject_CallMethod(loop, "add_slide_info", "O", ary);
    } else if(!bhview_identical(&ary->view, &ex_view)) {
        assert(ary->bhc_array != NULL);
        void *new = BhAPI_view(ex_dtype, ary->bhc_array, ex_view.ndim, ex_view.start, ex_view.shape, ex_view.stride);
        BhAPI_destroy(dtype_np2bhc(ary->view.type_enum), ary->bhc_array);
        ary->bhc_array = new;
        ary->view = ex_view;
    }
    assert(ary->view.initiated && bhview_identical(&ary->view, &ex_view));
    return ary->bhc_array;
}
