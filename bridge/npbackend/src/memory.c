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
#include <frameobject.h>

// In OSX `MAP_ANONYMOUS` is called `MAP_ANON`
#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif

// Help function for unprotect memory
static void _munprotect(void *data, npy_intp size) {
    if(mprotect(data, size, PROT_WRITE | PROT_READ) != 0) {
        int errsv = errno; // mprotect() sets the errno.
        fprintf(stderr,
                "Fatal error: _munprotect() could not (un-)mprotect a data region. "
                "Returned error code by mprotect(): %s.\n",
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

// Help function to remove whitespaces (from <https://stackoverflow.com/a/122721>)
// Note: This function returns a pointer to a substring of the original string.
// If the given string was allocated dynamically, the caller must not overwrite
// that pointer with the returned value, since the original pointer must be
// deallocated using the same allocator with which it was allocated.  The return
// value must NOT be deallocated using free() etc.
static char *_trimwhitespace(char *str)
{
    char *end;

    // Trim leading space
    while(isspace((unsigned char)*str)) str++;

    if(*str == 0)  // All spaces?
        return str;

    // Trim trailing space
    end = str + strlen(str) - 1;
    while(end > str && isspace((unsigned char)*end)) end--;

    // Write new null terminator
    *(end+1) = 0;

    return str;
}

// Help function that prints the content of a specific line in `filename`
static void _display_file_line(const char *filename, int lineno) {
	if (filename != NULL) {
        FILE *file = fopen(filename, "r");
        int count = 0;
        if (file != NULL) {
            char line[1024];
            while (fgets(line, 1024, file) != NULL) {
                if (count++ == lineno-1) {
                    printf("  %s(%d): %s\n", filename, lineno, _trimwhitespace(line));
                    fclose(file);
                    return;
                }
            }
            fclose(file);
        }
    }
	printf("<string>:%d\n", lineno);
}

// Help function to print the current Python code line.
// Since we use `PyGILState_GetThisThreadState()`, we require Python version 2.7 or 3.3+
#if PY_VERSION_HEX >= 0x03000000 && PY_VERSION_HEX < 0x03030000
    static void _display_backtrace(int stack_limit) {
        printf("<< traceback info require Python 2.7 or 3.3+ >>\n");
    }
#else
    static void _display_backtrace(int stack_limit) {
        // First we try to get the current Python thread state using `PyGILState_GetThisThreadState()`
        PyThreadState *tstate = PyGILState_GetThisThreadState();
        if (NULL == tstate || NULL == tstate->frame) {
            // If that fails, we use the previously saved one `py_thread_state`
            tstate = py_thread_state;
            if (NULL == tstate || NULL == tstate->frame) {
                // If that also fails, we give up
                printf("<< sorry traceback info not available %p >>\n", tstate);
                return;
            }
        }
        // Now that we have the current Python thread state, we can print the backtrace
        PyFrameObject *frame = tstate->frame;
        for(int i=0; i < stack_limit && frame != NULL; ++i) {
            int line = PyCode_Addr2Line(frame->f_code, frame->f_lasti);
        #if defined(NPY_PY3K)
            Py_ssize_t filename_size;
            const char *filename = PyUnicode_AsUTF8AndSize(frame->f_code->co_filename, &filename_size);
        #else
            const char *filename = PyString_AsString(frame->f_code->co_filename);
        #endif
            _display_file_line(filename, line);
            frame = frame->f_back;
        }
    }
#endif

// The function that will be called when encountering an unsupported NumPy operation
int mem_access_callback(void *addr, void *id) {
    PyObject *ary = (PyObject *) id;
    if (bh_unsupported_warn) {
        printf("Encountering an operation not supported by Bohrium. It will be handled by the original NumPy:\n");
        _display_backtrace(4);
    }
    // If `addr` is protected, the data of `ary` must be in bhc
    assert(((BhArray*) ary)->data_in_bhc);
    // Let's copy the memory from bhc to the numpy address space
    mem_bhc2np((BhArray*)ary);
    return 1;
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
    assert(((BhArray*) ary)->data_in_bhc);
    BhAPI_mem_signal_attach(ary, ary->base.data, ary_nbytes(ary), mem_access_callback);
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
                "Fatal error: mem_map() could not mmap new memory of size %lu). "
                "Returned error code by mmap: %s.\n",
                (long unsigned int) nbytes,
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
    assert(((BhArray*) ary)->data_in_bhc);
    BhAPI_mem_signal_attach(ary, ary->base.data, ary_nbytes(ary), mem_access_callback);
    ary->data_in_bhc = 1;
}

void mem_signal_attach(void *idx, void *addr, uint64_t nbytes) {
    BhAPI_mem_signal_attach(idx, addr, nbytes, mem_access_callback);
}

void mem_bhc2np(BhArray *base_array) {
    assert(get_base((PyObject*) base_array) == base_array);

    if (!base_array->data_in_bhc) {
        return;
    }

    // Let's detach the signal
    BhAPI_mem_signal_detach(PyArray_DATA((PyArrayObject*) base_array));

    if(base_array->bhc_array != NULL) {
        void *d = get_data_pointer((BhArray*) base_array, 1, 0, 1);
        if(d == NULL) {
            _munprotect(PyArray_DATA((PyArrayObject*) base_array), ary_nbytes((BhArray*) base_array));
        } else {
            assert(!BhAPI_mem_signal_exist(d)); // `d` shouldn't be memory protected!
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
    BhAPI_mem_signal_detach(PyArray_DATA((PyArrayObject*) base_array));

    // Then we unprotect the NumPy memory part
    _munprotect(PyArray_DATA((PyArrayObject*) base_array), ary_nbytes((BhArray*) base_array));

    // Copy the data from the NumPy part to the bhc part
    void *data = get_data_pointer((BhArray*) base_array, 1, 1, 0);
    memmove(data, PyArray_DATA((PyArrayObject*) base_array), PyArray_NBYTES((PyArrayObject*) base_array));

    // Finally, we memory protect the NumPy part of 'base' again
    _mprotect_np_part((BhArray*) base_array);
}
