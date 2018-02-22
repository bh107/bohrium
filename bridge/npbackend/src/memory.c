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

#include "memory.h"
#include "util.h"
#include "handle_special_op.h"
#include <bh_mem_signal.h>

// Help function for unprotect memory
static void _munprotect(void *data, npy_intp size) {
    if(mprotect(data, size, PROT_WRITE) != 0) {
        int errsv = errno; // mprotect() sets the errno.
        fprintf(stderr,
                "Fatal error: _munprotect() could not (un-)mprotect a data region. "
                "Returned error code by mremap(): %s.\n",
                strerror(errsv));
        assert(1 == 2);
        exit(-1);
    }
}

// Help function for memory re-map
static void _mremap_data(void *dst, void *src, npy_intp size) {
#if MREMAP_FIXED
    if(mremap(src, size, size, MREMAP_FIXED|MREMAP_MAYMOVE, dst) == MAP_FAILED) {
        int errsv = errno; // mremap() sets the errno.
        fprintf(stderr,
                "Fatal error: _mremap_data() could not mremap a data region (src: %p, dst: %p, size: %ld). "
                "Returned error code by mremap(): %s.\n",
                src, dst, size,
                strerror(errsv));
        assert(1 == 2);
        exit(-1);
    }
#else
    // Systems that doesn't support mremap will use memcpy, which introduces a
    // race-condition if another thread access the 'dst' memory before memcpy finishes.
    _munprotect(dst, size);
    memcpy(dst, src, size);
    mem_unmap(src, size);
#endif
}

void mem_access_callback(void *id, void *addr) {
    PyObject *ary = (PyObject *) id;

    PyGILState_STATE GIL = PyGILState_Ensure();
    int err = PyErr_WarnEx(
            NULL,
            "Encountering an operation not supported by Bohrium. It will be handled by the original NumPy.",
            1
    );

    if(err == -1) {
        PyErr_WarnEx(
                NULL,
                "Encountering an operation not supported by Bohrium. [Sorry, you cannot upgrade this warning to an exception]",
                1
        );
        PyErr_Print();
    }
    PyErr_Clear();

    mem_bhc2np((BhArray*)ary);

    PyGILState_Release(GIL);
}

// Help function for protecting the memory of the NumPy part of 'ary'
static void _mprotect_np_part(BhArray *ary) {
    assert(get_base((PyObject*) ary) == ary); // `ary` must be a base
    assert(ary->mmap_allocated);
    assert(PyArray_CHKFLAGS((PyArrayObject*) ary, NPY_ARRAY_OWNDATA));

    // Finally, we memory protect the NumPy data
    if(mprotect(ary->base.data, ary_nbytes(ary), PROT_NONE) == -1) {
        int errsv = errno; // mprotect() sets the errno.
        fprintf(stderr,
                "Fatal error: _mprotect_np_part() could not protect a data region. "
                "Returned error code by mprotect: %s.\n",
                strerror(errsv));
        assert(1 == 2);
        exit(-1);
    }
    bh_mem_signal_attach(ary, ary->base.data, ary_nbytes(ary), mem_access_callback);
}

void* mem_map(uint64_t nbytes) {
    assert(nbytes > 0);
    // Allocate page-size aligned memory.
    // The MAP_PRIVATE and MAP_ANONYMOUS flags is not 100% portable. See:
    // <http://stackoverflow.com/questions/4779188/how-to-use-mmap-to-allocate-a-memory-in-heap>
    void *addr = mmap(0, nbytes, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if(addr == MAP_FAILED) {
        int errsv = errno; // mmap() sets the errno.
        fprintf(stderr,
                "Fatal error: mem_map() could not mmap new memory of size %ld). "
                "Returned error code by mmap: %s.\n",
                nbytes,
                strerror(errsv));
        assert(1 == 2);
        exit(-1);
    }
    return addr;
}

void mem_unmap(void *addr, npy_intp size) {
    if(munmap(addr, size) == -1) {
        int errsv = errno; // munmmap() sets the errno.
        fprintf(stderr,
                "Fatal error: mem_unmap() could not mummap the data region: %p (size: %ld). "
                "Returned error code by mmap: %s.\n",
                addr,
                size,
                strerror(errsv));
        assert(1 == 2);
        exit(-1);
    }
}

void protected_malloc(BhArray *ary) {
    if(ary->mmap_allocated || !PyArray_CHKFLAGS((PyArrayObject*) ary, NPY_ARRAY_OWNDATA)) {
        return;
    }
    if (get_base((PyObject*) ary) != ary) {
        fprintf(stderr, "Fatal error: protected_malloc() was given a array that "
                        "owns its memory but isn't a base array!.\n");
        assert(1 == 2);
        exit(-1);
    }
    ary->mmap_allocated = 1;
    void *addr = mem_map(ary_nbytes(ary));
    // Let's save the pointer to the NumPy allocated memory and use the mprotect'ed memory instead
    ary->npy_data = ary->base.data;
    ary->base.data = addr;
    bh_mem_signal_attach(ary, ary->base.data, ary_nbytes(ary), mem_access_callback);
    ary->data_in_bhc = 1;
}

void mem_signal_attach(const void *idx, const void *addr, uint64_t nbytes) {
    bh_mem_signal_attach(idx, addr, nbytes, mem_access_callback);
}

void mem_bhc2np(BhArray *base_array) {
    assert(get_base((PyObject*) base_array) == base_array);

    if (!base_array->data_in_bhc) {
        return;
    }

    // Let's detach the signal
    bh_mem_signal_detach(PyArray_DATA((PyArrayObject*) base_array));

    if(base_array->bhc_array != NULL) {
        void *d = get_data_pointer((BhArray*) base_array, 1, 0, 1);
        if(d == NULL) {
            _munprotect(PyArray_DATA((PyArrayObject*) base_array), ary_nbytes((BhArray*) base_array));
        } else {
            _mremap_data(PyArray_DATA((PyArrayObject*) base_array), d, ary_nbytes((BhArray*) base_array));
        }
    } else {
        // Let's make sure that the NumPy data isn't protected
        _munprotect(PyArray_DATA((PyArrayObject*) base_array), ary_nbytes((BhArray*) base_array));
    }
    base_array->data_in_bhc = 0;
}

void mem_np2bhc(BhArray *base_array) {
    assert(get_base((PyObject*) base_array) == base_array);

    if (base_array->data_in_bhc) {
        return;
    }
    base_array->data_in_bhc = 1;

    // Let's detach the signal
    bh_mem_signal_detach(PyArray_DATA((PyArrayObject*) base_array));

    // Then we unprotect the NumPy memory part
    _munprotect(PyArray_DATA((PyArrayObject*) base_array), ary_nbytes((BhArray*) base_array));

    // Copy the data from the NumPy part to the bhc part
    void *data = get_data_pointer((BhArray*) base_array, 1, 1, 0);
    memmove(data, PyArray_DATA((PyArrayObject*) base_array), PyArray_NBYTES((PyArrayObject*) base_array));

    // Finally, we memory protect the NumPy part of 'base' again
    _mprotect_np_part((BhArray*) base_array);
}
