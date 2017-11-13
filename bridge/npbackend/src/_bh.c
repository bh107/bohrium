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

#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif

#include <Python.h>
#include <structmember.h>
#include <dlfcn.h>
#include <bh_mem_signal.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

// The NumPy API changed in version 1.7
#if(NPY_API_VERSION >= 0x00000007)
    #define BH_PyArrayObject PyArrayObject_fields
#else
    #define BH_PyArrayObject PyArrayObject
    #define NPY_ARRAY_OWNDATA NPY_OWNDATA
#endif

#if PY_MAJOR_VERSION >= 3
    #define NPY_PY3K
#endif

// Forward declaration
static PyObject* BhArray_data_bhc2np(PyObject *self, PyObject *args);
static PyTypeObject BhArrayType;

#define BhArray_CheckExact(op) (((PyObject*) (op))->ob_type == &BhArrayType)
PyObject *bhary          = NULL; // The bhary Python module
PyObject *ufuncs         = NULL; // The ufuncs Python module
PyObject *bohrium        = NULL; // The Bohrium Python module
PyObject *array_create   = NULL; // The array_create Python module
PyObject *reorganization = NULL; // The reorganization Python module
PyObject *masking        = NULL; // The masking Python module
int bh_sync_warn         = 0;    // Boolean: should we warn when copying from Bohrium to NumPy
int bh_mem_warn          = 0;    // Boolean: should we warn when about memory problems

#define bhc_exist(x) (((BhArray*) x)->bhc_ary != Py_None)

typedef struct {
    BH_PyArrayObject base;
    PyObject *bhc_ary;
    PyObject *bhc_ary_version;
    PyObject *bhc_view;
    PyObject *bhc_view_version;
    int mmap_allocated;
    void *npy_data; // NumPy allocated array data
} BhArray;

// Help function that returns number of bytes in 'ary'
// BUT minimum 'itemsize', which mimic the behavior of NumPy
static int64_t ary_nbytes(const BhArray *ary) {
    int64_t size = PyArray_NBYTES((PyArrayObject*) ary);
    if(size == 0) {
        return PyArray_ITEMSIZE((PyArrayObject*) ary);
    } else {
        return size;
    }
}

// Help function to retrieve the Bohrium-C data pointer
// Return -1 on error
static int get_bhc_data_pointer(PyObject *ary, int copy2host, int force_allocation, int nullify, void **out_data) {
    if(((BhArray*) ary)->mmap_allocated == 0) {
        PyErr_SetString(
            PyExc_TypeError,
            "The array data wasn't allocated through mmap(). Typically, this is because the base array was created from a template, which is not supported by Bohrium."
        );
        return -1;
    }

    PyObject *data = PyObject_CallMethod(bhary, "get_bhc_data_pointer", "Oiii", ary, copy2host, force_allocation, nullify);
    if(data == NULL) {
        return -1;
    }

#if defined(NPY_PY3K)
    if(!PyLong_Check(data)) {
#else
    if(!PyInt_Check(data)) {
#endif
        PyErr_SetString(
            PyExc_TypeError,
            "get_bhc_data_pointer(ary) should return a Python integer that represents a memory address."
        );
        Py_DECREF(data);
        return -1;
    }

    void *d = PyLong_AsVoidPtr(data);
    Py_DECREF(data);
    if(force_allocation && d == NULL) {
        PyErr_SetString(
            PyExc_TypeError,
            "get_bhc_data_pointer(ary, allocate=True) shouldn't return a NULL pointer"
        );
        return -1;
    }

    *out_data = d;
    return 0;
}

// Help function to set the Bohrium-C data from a numpy array
// Return -1 on error
static int set_bhc_data_from_ary(PyObject *self, PyObject *ary) {
    if(((BhArray*) self)->mmap_allocated == 0) {
        PyErr_SetString(
            PyExc_TypeError,
            "The array data wasn't allocated through mmap(). Typically, this is because the base array was created from a template, which is not supported by Bohrium."
        );
        return -1;
    }

    PyObject *ret = PyObject_CallMethod(bhary, "set_bhc_data_from_ary", "OO", self, ary);
    Py_XDECREF(ret);
    if(ret == NULL) {
        return -1;
    }

    return 0;
}

// Help function for unprotect memory
// Return -1 on error
static int _munprotect(void *data, npy_intp size) {
    if(mprotect(data, size, PROT_WRITE) != 0) {
        int errsv = errno; // mprotect() sets the errno.
        PyErr_Format(
            PyExc_RuntimeError,
            "Error - could not (un-)mprotect a data region. Returned error code by mprotect(): %s.\n",
            strerror(errsv)
        );
        return -1;
    }
    return 0;
}

// Help function for memory un-map
// Return -1 on error
static int _munmap(void *addr, npy_intp size) {
    if(munmap(addr, size) == -1) {
        int errsv = errno; // munmmap() sets the errno.
        PyErr_Format(
            PyExc_RuntimeError,
            "The Array Data Protection could not mummap the data region: %p (size: %ld). Returned error code by mmap: %s.\n",
            addr,
            size,
            strerror(errsv)
        );
        return -1;
    }

    return 0;
}

// Help function for memory re-map
// Return -1 on error
static int _mremap_data(void *dst, void *src, npy_intp size) {
#if MREMAP_FIXED
    if(mremap(src, size, size, MREMAP_FIXED|MREMAP_MAYMOVE, dst) == MAP_FAILED) {
        int errsv = errno; // mremap() sets the errno.
        PyErr_Format(
            PyExc_RuntimeError,
            "Error - could not mremap a data region (src: %p, dst: %p, size: %ld). Returned error code by mremap(): %s.\n",
            src, dst, size,
            strerror(errsv)
        );
        return -1;
    }

    return 0;
#else
    // Systems that doesn't support mremap will use memcpy, which introduces a
    // race-condition if another thread access the 'dst' memory before memcpy finishes.
    if(_munprotect(dst, size) != 0) {
        return -1;
    }

    memcpy(dst, src, size);
    return _munmap(src, size);
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

    if(bh_mem_warn && !bhc_exist(ary)) {
        printf("MEM_WARN: mem_access_callback() - base %p has no bhc object!\n", ary);
    }

    if(BhArray_data_bhc2np(ary, NULL) == NULL) {
        PyErr_Print();
    }

    PyGILState_Release(GIL);
}

// Help function for protecting the memory of the NumPy part of 'ary'
// Return -1 on error
static int _mprotect_np_part(BhArray *ary) {
    assert(((BhArray*) ary)->mmap_allocated);
    assert(PyArray_CHKFLAGS((PyArrayObject*) ary, NPY_ARRAY_OWNDATA));

    // Finally, we memory protect the NumPy data
    if(mprotect(ary->base.data, ary_nbytes(ary), PROT_NONE) == -1) {
        int errsv = errno; // mprotect() sets the errno.
        PyErr_Format(
            PyExc_RuntimeError,
            "Error - could not protect a data region. Returned error code by mprotect: %s.\n",
            strerror(errsv)
        );
        return -1;
    }

    bh_mem_signal_attach(ary, ary->base.data, ary_nbytes(ary), mem_access_callback);
    return 0;
}

// Help function for allocate protected memory through mmep
// Returns a pointer to the new memory or NULL on error
static void* _mmap_mem(uint64_t nbytes) {
    // Allocate page-size aligned memory.
    // The MAP_PRIVATE and MAP_ANONYMOUS flags is not 100% portable. See:
    // <http://stackoverflow.com/questions/4779188/how-to-use-mmap-to-allocate-a-memory-in-heap>
    void *addr = mmap(0, nbytes, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);

    if(addr == MAP_FAILED) {
        int errsv = errno; // mmap() sets the errno.
        PyErr_Format(
            PyExc_RuntimeError,
            "The Array Data Protection could not mmap a data region. Returned error code by mmap: %s.",
            strerror(errsv)
        );
        return NULL;
    }

    return addr;
}

// Help function for allocate protected memory for the NumPy part of 'ary'
// This function only allocates if the 'ary' is a new base array and avoids multiple
// allocations by checking and setting the ary->mmap_allocated
// Return -1 on error
static int _protected_malloc(BhArray *ary) {
    if(ary->mmap_allocated || !PyArray_CHKFLAGS((PyArrayObject*) ary, NPY_ARRAY_OWNDATA)) {
        return 0;
    }

    ary->mmap_allocated = 1;

    void *addr = _mmap_mem(ary_nbytes(ary));
    if(addr == NULL) {
        return -1;
    }

    // Let's save the pointer to the NumPy allocated memory and use the mprotect'ed memory instead
    ary->npy_data = ary->base.data;
    ary->base.data = addr;

    bh_mem_signal_attach(ary, ary->base.data, ary_nbytes(ary), mem_access_callback);
    return 0;
}

// Called when module exits
static void module_exit(void) {
    bh_mem_signal_shutdown();
}

// Help function that creates a simple new array.
// We parse to PyArray_NewFromDescr(), a new protected memory allocation
// Return the new Python object, or NULL on error
static PyObject* _simply_new_array(PyTypeObject *type, PyArray_Descr *descr, uint64_t nbytes, PyArray_Dims shape) {
    // Let's create a new NumPy array using our memory allocation
    void *addr = _mmap_mem(nbytes);

    if(addr == NULL) {
        return NULL;
    }

    PyObject *ret = PyArray_NewFromDescr(type, descr, shape.len, shape.ptr, NULL, addr, 0, NULL);
    if(ret == NULL) {
        return NULL;
    }

    ((BhArray*) ret)->base.flags |= NPY_ARRAY_OWNDATA;
    ((BhArray*) ret)->base.flags |= NPY_ARRAY_CARRAY;
    ((BhArray*) ret)->mmap_allocated = 1;

    PyArray_UpdateFlags((PyArrayObject*) ret, NPY_ARRAY_UPDATE_ALL);
    bh_mem_signal_attach(ret, ((BhArray*) ret)->base.data, nbytes, mem_access_callback);

    return ret;
}

static PyObject* BhArray_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    PyObject *ret;
    // If this is a "simple" new array, we can use our own memory allocation
    {
        static char *kwlist[] = {"shape", "dtype", NULL}; // We only support simple arrays
        PyArray_Descr *descr = NULL;
        PyArray_Dims shape = {NULL, 0};
        if(PyArg_ParseTupleAndKeywords(args, kwds, "O&|O&", kwlist,
                                       PyArray_IntpConverter, &shape,
                                       PyArray_DescrConverter, &descr))
        {
            int i; uint64_t nelem = 1;
            for(i = 0; i < shape.len; ++i) {
                nelem *= shape.ptr[i];
            }

            if(nelem > 0) {
                if(descr == NULL) { // Get default dtype
                    descr = PyArray_DescrFromType(NPY_DEFAULT_TYPE);
                }

                ret = _simply_new_array(type, descr, nelem * descr->elsize, shape);
                if (ret == NULL) {
                    return NULL;
                }

                if(shape.len > 0)
                {
                    assert(shape.ptr != NULL);
                    PyDimMem_FREE(shape.ptr);
                }

                return ret;
            }
        }
    }

    // If it is not a simple new array, we let NumPy create it
    ret = PyArray_Type.tp_new(type, args, kwds);
    if(ret == NULL) {
        return NULL;
    }

    // And then protect the memory afterwards
    if(_protected_malloc((BhArray *) ret) != 0) {
        return NULL;
    }

    return ret;
}

static PyObject* BhArray_finalize(PyObject *self, PyObject *args) {
    int e = PyObject_IsInstance(self, (PyObject*) &BhArrayType);
    if(e == -1) {
        return NULL;
    } else if (e == 0) {
        Py_RETURN_NONE;
    }

    ((BhArray*) self)->bhc_ary = Py_None;
    Py_INCREF(Py_None);

    ((BhArray*) self)->bhc_ary_version = PyLong_FromLong(0);

    ((BhArray*) self)->bhc_view = Py_None;
    Py_INCREF(Py_None);

    ((BhArray*) self)->bhc_view_version = Py_None;
    Py_INCREF(Py_None);

    if(_protected_malloc((BhArray *) self) != 0) {
        return NULL;
    }

    if(PyDataType_FLAGCHK(PyArray_DESCR((PyArrayObject*) self), NPY_ITEM_REFCOUNT)) {
        PyErr_Format(PyExc_RuntimeError, "Array of objects not supported by Bohrium.");
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject* BhArray_alloc(PyTypeObject *type, Py_ssize_t nitems) {
    PyObject *obj;
    obj = (PyObject *) PyObject_Malloc(type->tp_basicsize);
    PyObject_Init(obj, type);

    // Flag the array as uninitialized
    ((BhArray*) obj)->bhc_ary          = NULL;
    ((BhArray*) obj)->bhc_ary_version  = NULL;
    ((BhArray*) obj)->bhc_view         = NULL;
    ((BhArray*) obj)->bhc_view_version = NULL;
    ((BhArray*) obj)->npy_data         = NULL;
    ((BhArray*) obj)->mmap_allocated   = 0;

    return obj;
}

static void BhArray_dealloc(BhArray* self) {
    assert(BhArray_CheckExact(self));

    Py_XDECREF(self->bhc_view);
    Py_XDECREF(self->bhc_view_version);
    Py_XDECREF(self->bhc_ary_version);
    Py_XDECREF(self->bhc_ary);

    if (!PyArray_CHKFLAGS((PyArrayObject*) self, NPY_ARRAY_OWNDATA)) {
        BhArrayType.tp_base->tp_dealloc((PyObject*) self);
        return; // The array doesn't own the array data
    }

    assert(!PyDataType_FLAGCHK(PyArray_DESCR((PyArrayObject*) self), NPY_ITEM_REFCOUNT));

    if (self->mmap_allocated) {
        if(_munmap(PyArray_DATA((PyArrayObject*) self), ary_nbytes(self)) == -1) {
            PyErr_Print();
        }

        bh_mem_signal_detach(PyArray_DATA((PyArrayObject*) self));
        self->base.data = NULL;
    }

    if (self->npy_data != NULL) {
        self->base.data = self->npy_data;
    }

    // Notice, we have to call the 'tp_dealloc' of the base class (<http://legacy.python.org/dev/peps/pep-0253/>).
    // In our case, the base class is ndarray, which will call 'tp_free', which in turn calls 'BhArray_free()'
    BhArrayType.tp_base->tp_dealloc((PyObject*) self);
}

static void BhArray_free(PyObject * v) {
    PyObject_Free(v);
}

// Making the Bohrium memory available for NumPy.
// NB: this function should not fail before unprotecting the NumPy data
static PyObject* BhArray_data_bhc2np(PyObject *self, PyObject *args) {
    assert(args == NULL);
    assert(BhArray_CheckExact(self));

    // We move the whole array (i.e. the base array) from Bohrium to NumPy
    PyObject *base = PyObject_CallMethod(bhary, "get_base", "O", self);
    if(base == NULL) {
        base = self; // We have to keep going!
        Py_INCREF(base); // We call Py_DECREF(base) later
    }
    assert(BhArray_CheckExact(base));

    if(!PyArray_CHKFLAGS((PyArrayObject*) base, NPY_ARRAY_OWNDATA)) {
        PyErr_Format(PyExc_ValueError, "The base array doesn't own its data");
    }

    // Let's detach the signal
    bh_mem_signal_detach(PyArray_DATA((PyArrayObject*) base));

    if(bhc_exist(base)) {
        // Calling get_bhc_data_pointer(base, allocate=False, nullify=True)
        void *d = NULL;
        get_bhc_data_pointer(base, 1, 0, 1, &d);

        if(d == NULL) {
            _munprotect(PyArray_DATA((PyArrayObject*) base), ary_nbytes((BhArray*) base));
        } else {
            _mremap_data(PyArray_DATA((PyArrayObject*) base), d, ary_nbytes((BhArray*) base));
        }
        Py_DECREF(base);

        // Let's delete the current bhc_ary
        PyObject_CallMethod(bhary, "del_bhc", "O", self);
    } else {
        // Let's make sure that the NumPy data isn't protected
        _munprotect(PyArray_DATA((PyArrayObject*) base), ary_nbytes((BhArray*) base));
    }

    // Finally, we can return NULL on error (but not before!)
    if (PyErr_Occurred() != NULL) {
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject* BhArray_data_np2bhc(PyObject *self, PyObject *args) {
    assert(args == NULL);
    assert(BhArray_CheckExact(self));

    // We move the whole array (i.e. the base array) from Bohrium to NumPy
    PyObject *base = PyObject_CallMethod(bhary, "get_base", "O", self);
    if(base == NULL) {
        return NULL;
    }

    assert(BhArray_CheckExact(base));

    if(!PyArray_CHKFLAGS((PyArrayObject*) base, NPY_ARRAY_OWNDATA)) {
        PyErr_Format(PyExc_ValueError, "The base array doesn't own its data");
        return NULL;
    }

    // Make sure that bhc_ary exist
    if(!bhc_exist(base)) {
        PyObject *err = PyObject_CallMethod(bhary, "new_bhc_base", "O", base);
        if(err == NULL) {
            return NULL;
        }
        Py_DECREF(err);
    }

    // Then we unprotect the NumPy memory part
    bh_mem_signal_detach(PyArray_DATA((PyArrayObject*) base));
    if(_munprotect(PyArray_DATA((PyArrayObject*) base), ary_nbytes((BhArray*) base)) != 0) {
        return NULL;
    }

    // And sets the bhc data from the NumPy part of 'base'
    if(set_bhc_data_from_ary(base, base) == -1) {
        return NULL;
    }

    // Finally, we memory protect the NumPy part of 'base' again
    if(_mprotect_np_part((BhArray*) base) != 0) {
        return NULL;
    }

    Py_DECREF(base);
    Py_RETURN_NONE;
}

static PyObject* BhArray_data_fill(PyObject *self, PyObject *args) {
    PyObject *np_ary;
    if(!PyArg_ParseTuple(args, "O:ndarray", &np_ary)) {
        return NULL;
    }

    if(!PyArray_Check(np_ary)) {
        PyErr_SetString(PyExc_TypeError, "must be a NumPy array");
        return NULL;
    }

    if(!PyArray_ISCARRAY_RO((PyArrayObject*) np_ary)) {
        PyErr_SetString(PyExc_TypeError, "must be a C-style contiguous array");
        return NULL;
    }

    // Sets the bhc data from the NumPy array 'np_ary'
    if(set_bhc_data_from_ary(self, np_ary) == -1) {
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject* BhArray_copy2numpy(PyObject *self, PyObject *args) {
    assert(args == NULL);

    PyObject *ret = PyArray_NewLikeArray((PyArrayObject*) self, NPY_ANYORDER, NULL, 0);
    if(ret == NULL) {
        return NULL;
    }

    PyObject *err = PyObject_CallMethod(ufuncs, "assign", "OO", self, ret);
    if(err == NULL) {
        Py_DECREF(ret);
        return NULL;
    }

    Py_DECREF(err);
    return ret;
}

static PyObject* BhArray_numpy_wrapper(PyObject *self, PyObject *args) {
    assert(args == NULL);
    assert(BhArray_CheckExact(self));
    PyArrayObject *s = (PyArrayObject*) self;
    if(!PyArray_IS_C_CONTIGUOUS((PyArrayObject*) s)) {
        PyErr_Format(PyExc_RuntimeError, "Array must be C-style contiguous.");
        return NULL;
    }
    void *data;
    if (get_bhc_data_pointer(self, 1, 1, 0, &data) != 0) {
        return NULL;
    }
    return PyArray_SimpleNewFromData(PyArray_NDIM(s), PyArray_DIMS(s), PyArray_TYPE(s), data);
}

static PyObject* BhArray_resize(PyObject *self, PyObject *args) {
    PyErr_SetString(PyExc_NotImplementedError, "Bohrium arrays doesn't support resize");
    return NULL;
}


// Help function to make methods calling a Python function
static PyObject* method2function(char *name, PyObject *self, PyObject *args, PyObject *kwds) {
    // We parse the 'args' to bohrium.'name' with 'self' as the first argument
    Py_ssize_t i, size = PyTuple_Size(args);

    PyObject *func_args = PyTuple_New(size+1);
    if(func_args == NULL) {
        return NULL;
    }

    Py_INCREF(self);
    PyTuple_SET_ITEM(func_args, 0, self);

    for(i = 0; i < size; ++i) {
        PyObject *t = PyTuple_GET_ITEM(args, i);
        Py_INCREF(t);
        PyTuple_SET_ITEM(func_args, i+1, t);
    }

    PyObject *ret = PyObject_Call(PyObject_GetAttrString(bohrium, name), func_args, kwds);
    Py_DECREF(func_args);

    return ret;
}

static PyObject* BhArray_array_ufunc(PyObject *self, PyObject *args, PyObject *kwds) {
    return method2function("_handle__array_ufunc__", self, args, kwds);
}

static PyObject* BhArray_copy(PyObject *self, PyObject *args, PyObject *kwds) {
    return method2function("copy", self, args, kwds);
}

static PyObject* BhArray_reshape(PyObject *self, PyObject *args, PyObject *kwds) {
    return method2function("reshape", self, args, kwds);
}

static PyObject* BhArray_flatten(PyObject *self, PyObject *args, PyObject *kwds) {
    return method2function("flatten", self, args, kwds);
}

static PyObject* BhArray_sum(PyObject *self, PyObject *args, PyObject *kwds) {
    return method2function("sum", self, args, kwds);
}

static PyObject* BhArray_prod(PyObject *self, PyObject *args, PyObject *kwds) {
    return method2function("prod", self, args, kwds);
}

static PyObject* BhArray_cumsum(PyObject *self, PyObject *args, PyObject *kwds) {
    return method2function("cumsum", self, args, kwds);
}

static PyObject* BhArray_cumprod(PyObject *self, PyObject *args, PyObject *kwds) {
    return method2function("cumprod", self, args, kwds);
}

static PyObject* BhArray_any(PyObject *self, PyObject *args, PyObject *kwds) {
    return method2function("any", self, args, kwds);
}

static PyObject* BhArray_all(PyObject *self, PyObject *args, PyObject *kwds) {
    return method2function("all", self, args, kwds);
}

static PyObject* BhArray_conj(PyObject *self, PyObject *args, PyObject *kwds) {
    return method2function("conj", self, args, kwds);
}

static PyObject* BhArray_min(PyObject *self, PyObject *args, PyObject *kwds) {
    return method2function("min", self, args, kwds);
}

static PyObject* BhArray_max(PyObject *self, PyObject *args, PyObject *kwds) {
    return method2function("max", self, args, kwds);
}

static PyObject* BhArray_argmin(PyObject *self, PyObject *args, PyObject *kwds) {
    return method2function("argmin", self, args, kwds);
}

static PyObject* BhArray_argmax(PyObject *self, PyObject *args, PyObject *kwds) {
    return method2function("argmax", self, args, kwds);
}

static PyObject* BhArray_astype(PyObject *self, PyObject *args, PyObject *kwds) {
    return method2function("array", self, args, kwds);
}

static PyObject* BhArray_fill(PyObject *self, PyObject *args, PyObject *kwds) {
    return method2function("fill", self, args, kwds);
}

static PyObject* BhArray_trace(PyObject *self, PyObject *args, PyObject *kwds) {
    return method2function("trace", self, args, kwds);
}

static PyObject* BhArray_print_to_file(PyObject *self, PyObject *args, PyObject *kwds) {
    return method2function("print_to_file", self, args, kwds);
}

static PyObject* BhArray_take(PyObject *self, PyObject *args, PyObject *kwds) {
    return method2function("take", self, args, kwds);
}

static PyObject* BhArray_put(PyObject *self, PyObject *args, PyObject *kwds) {
    return method2function("put", self, args, kwds);
}

static PyMethodDef BhArrayMethods[] = {
    {"__array_finalize__", BhArray_finalize,                    METH_VARARGS,                 NULL},
    {"__array_ufunc__",    (PyCFunction) BhArray_array_ufunc,   METH_VARARGS | METH_KEYWORDS, "Handle ufunc"},
    {"_data_bhc2np",       BhArray_data_bhc2np,                 METH_NOARGS,                  "Copy the Bohrium-C data to NumPy data"},
    {"_data_np2bhc",       BhArray_data_np2bhc,                 METH_NOARGS,                  "Copy the NumPy data to Bohrium-C data"},
    {"_data_fill",         BhArray_data_fill,                   METH_VARARGS,                 "Fill the Bohrium-C data from a numpy NumPy"},
    {"copy2numpy",         BhArray_copy2numpy,                  METH_NOARGS,                  "Copy the array in C-style memory layout to a regular NumPy array"},
    {"_numpy_wrapper",     BhArray_numpy_wrapper,               METH_NOARGS,                  "Returns a NumPy array that wraps the data of this array. NB: no flush or data management!"},
    {"resize",             BhArray_resize,                      METH_VARARGS,                 "Change shape and size of array in-place"},
    {"copy",               (PyCFunction) BhArray_copy,          METH_VARARGS | METH_KEYWORDS, "a.copy(order='C')\n\nReturn a copy of the array."},
    {"reshape",            (PyCFunction) BhArray_reshape,       METH_VARARGS | METH_KEYWORDS, "a.reshape(shape)\n\nReturns an array containing the same data with a new shape.\n\nRefer to `bohrium.reshape` for full documentation."},
    {"flatten",            (PyCFunction) BhArray_flatten,       METH_VARARGS | METH_KEYWORDS, "a.flatten()\n\nReturn a copy of the array collapsed into one dimension."},
    {"ravel",              (PyCFunction) BhArray_flatten,       METH_VARARGS | METH_KEYWORDS, "a.ravel()\n\nReturn a copy of the array collapsed into one dimension."},
    {"sum",                (PyCFunction) BhArray_sum,           METH_VARARGS | METH_KEYWORDS, "a.sum(axis=None, dtype=None, out=None)\n\nReturn the sum of the array elements over the given axis.\n\nRefer to `bohrium.sum` for full documentation."},
    {"prod",               (PyCFunction) BhArray_prod,          METH_VARARGS | METH_KEYWORDS, "a.prod(axis=None, dtype=None, out=None)\n\nReturn the product of the array elements over the given axis\n\nRefer to `numpy.prod` for full documentation."},
    {"cumsum",             (PyCFunction) BhArray_cumsum,        METH_VARARGS | METH_KEYWORDS, "a.cumsum(axis=None, dtype=None, out=None)\n\nReturn the cumulative sum of the array elements over the given axis.\n\nRefer to `bohrium.cumsum` for full documentation."},
    {"cumprod",            (PyCFunction) BhArray_cumprod,       METH_VARARGS | METH_KEYWORDS, "a.cumprod(axis=None, dtype=None, out=None)\n\nReturn the cumulative product of the array elements over the given axis\n\nRefer to `numpy.cumprod` for full documentation."},
    {"any",                (PyCFunction) BhArray_any,           METH_VARARGS | METH_KEYWORDS, "a.any(axis=None, out=None)\n\nTest whether any array element along a given axis evaluates to True.\n\nRefer to `numpy.any` for full documentation."},
    {"all",                (PyCFunction) BhArray_all,           METH_VARARGS | METH_KEYWORDS, "a.all(axis=None, out=None)\n\nTest whether all array elements along a given axis evaluate to True.\n\nRefer to `numpy.all` for full documentation."},
    {"conj",               (PyCFunction) BhArray_conj,          METH_VARARGS | METH_KEYWORDS, "a.conj(x[, out])\n\nReturn the complex conjugate, element-wise.\n\nRefer to `numpy.conj` for full documentation."},
    {"conjugate",          (PyCFunction) BhArray_conj,          METH_VARARGS | METH_KEYWORDS, "a.conjugate(x[, out])\n\nReturn the complex conjugate, element-wise.\n\nRefer to `numpy.conj` for full documentation."},
    {"min",                (PyCFunction) BhArray_min,           METH_VARARGS | METH_KEYWORDS, "a.min(axis=None, out=None)\n\nReturn the minimum along a given axis.\n\nRefer to numpy.amin for full documentation."},
    {"max",                (PyCFunction) BhArray_max,           METH_VARARGS | METH_KEYWORDS, "a.max(axis=None, out=None)\n\nReturn the maximum along a given axis.\n\nRefer to numpy.amax for full documentation."},
    {"argmin",             (PyCFunction) BhArray_argmin,        METH_VARARGS | METH_KEYWORDS, "a.argmin(axis=None, out=None)\n\nReturns the indices of the minimum values along an axis.\n\nRefer to numpy.argmin for full documentation."},
    {"argmax",             (PyCFunction) BhArray_argmax,        METH_VARARGS | METH_KEYWORDS, "a.argmax(axis=None, out=None)\n\nReturns the indices of the maximum values along an axis.\n\nRefer to numpy.argmax for full documentation."},
    {"astype",             (PyCFunction) BhArray_astype,        METH_VARARGS | METH_KEYWORDS, "a.astype(dtype, order='C', subok=True, copy=True)\n\nCopy of the array, cast to a specified type."},
    {"fill",               (PyCFunction) BhArray_fill,          METH_VARARGS | METH_KEYWORDS, "a.fill(value)\n\nFill the array with a scalar value."},
    {"trace",              (PyCFunction) BhArray_trace,         METH_VARARGS | METH_KEYWORDS, "a.trace(offset=0, axis1=0, axis2=1, dtype=None, out=None)\n\nReturn the sum along diagonals of the array."},
    {"tofile",             (PyCFunction) BhArray_print_to_file, METH_VARARGS | METH_KEYWORDS, "a.tofile(fid, sep=\"\", format=\"%s\")\n\nWrite array to a file as text or binary (default)."},
    {"take",               (PyCFunction) BhArray_take,          METH_VARARGS | METH_KEYWORDS, "a.take(indices, axis=None, out=None, mode='raise')."},
    {"put",                (PyCFunction) BhArray_put,           METH_VARARGS | METH_KEYWORDS, "a.put(indices, values, mode='raise')\n\nSet a.flat[n] = values[n] for all n in indices."},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static PyMemberDef BhArrayMembers[] = {
    {"bhc_ary",          T_OBJECT_EX, offsetof(BhArray, bhc_ary),          0, "The Bohrium backend base-array"},
    {"bhc_ary_version",  T_OBJECT_EX, offsetof(BhArray, bhc_ary_version),  0, "The version of the Bohrium backend base-array"},
    {"bhc_view",         T_OBJECT_EX, offsetof(BhArray, bhc_view),         0, "The Bohrium backend view-array"},
    {"bhc_view_version", T_OBJECT_EX, offsetof(BhArray, bhc_view_version), 0, "The version of the Bohrium backend view-array"},
    {"bhc_mmap_allocated", T_BOOL, offsetof(BhArray, mmap_allocated), 0, "Is the base data allocated with mmap?"},
    {NULL}  /* Sentinel */
};

static int BhArray_SetSlice(PyObject *o, Py_ssize_t ilow, Py_ssize_t ihigh, PyObject *v) {
    if(v == NULL) {
        PyErr_SetString(PyExc_ValueError, "cannot delete array elements");
        return -1;
    }

    if(!PyArray_ISWRITEABLE((PyArrayObject *) o)) {
        PyErr_SetString(PyExc_ValueError, "assignment destination is read-only");
        return -1;
    }

#if defined(NPY_PY3K)
    PyObject *low = PyLong_FromSsize_t(ilow);
    PyObject *high = PyLong_FromSsize_t(ihigh);
#else
    PyObject *low = PyInt_FromSsize_t(ilow);
    PyObject *high = PyInt_FromSsize_t(ihigh);
#endif
    PyObject *slice = PySlice_New(low, high, NULL);
    PyObject *ret = PyObject_CallMethod(ufuncs, "setitem", "OOO", o, slice, v);

    Py_XDECREF(low);
    Py_XDECREF(high);
    Py_XDECREF(slice);
    if(ret == NULL) {
        return -1;
    }

    Py_XDECREF(ret);
    return 0;
}

// Help function that returns True when 'o' contains a list or array
static int obj_contains_a_list_or_ary(PyObject *o) {
    Py_ssize_t i;
    assert(o != NULL);

    if(PyArray_Check(o) || PyList_Check(o)) {
        return 1;
    }

    if(PyTuple_Check(o)) {
        for(i = 0; i < PyTuple_GET_SIZE(o); ++i) {
            PyObject *a = PyTuple_GET_ITEM(o, i);
            if(PyArray_Check(a) || PyList_Check(a)) {
                return 1;
            }
        }
    }

    return 0;
}

// Help function that returns True when 'k' is a bool mask with the same shape as 'o'
static int obj_is_a_bool_mask(PyObject *o, PyObject *k) {
    Py_ssize_t i;
    assert(o != NULL);
    assert(k != NULL);
    assert(PyArray_Check(o));

    if (!(PyArray_Check(k) && PyArray_TYPE((PyArrayObject*) k) == NPY_BOOL)) {
        return 0;
    }

    if (PyArray_NDIM((PyArrayObject*) o) != PyArray_NDIM((PyArrayObject*) k)) {
        return 0;
    }

    for(i = 0; i < PyArray_NDIM((PyArrayObject*) o); ++i) {
        if (PyArray_DIM((PyArrayObject*) o, i) != PyArray_DIM((PyArrayObject*) k, i)) {
            return 0;
        }
    }

    return 1;
}

static int BhArray_SetItem(PyObject *o, PyObject *k, PyObject *v) {
    Py_ssize_t i;
    assert(k != NULL);

    if(v == NULL) {
        PyErr_SetString(PyExc_ValueError, "cannot delete array elements");
        return -1;
    }

    if(!PyArray_ISWRITEABLE((PyArrayObject *) o)) {
        PyErr_SetString(PyExc_ValueError, "assignment destination is read-only");
        return -1;
    }

    // Let's handle assignments to a boolean masked array
    if (obj_is_a_bool_mask(o, k)) {
        PyObject *err = PyObject_CallMethod(masking, "masked_set", "OOO", o, k, v);
        if(err == NULL) {
            return -1;
        }

        Py_XDECREF(err);
        return 0;
    }

    // Generally, we do not support indexing with arrays
    if(obj_contains_a_list_or_ary(k) == 1) {
        // But when indexing array with an index array for each dimension in the array,
        // it corresponds to put_using_index_tuple()
        if (PySequence_Check(k) && PySequence_Size(k) == PyArray_NDIM((PyArrayObject*) o)) {
            PyObject *err = PyObject_CallMethod(reorganization, "put_using_index_tuple", "OOO", o, k, v);
            if(err == NULL) {
                return -1;
            }

            Py_XDECREF(err);
            return 0;
        }

        // And when indexing a vector, it corresponds to np.put()
        if (PyArray_NDIM((PyArrayObject*) o) == 1) {
            PyObject *err = PyObject_CallMethod(reorganization, "put", "OOO", o, k, v);
            if(err == NULL) {
                return -1;
            }

            Py_XDECREF(err);
            return 0;
        }

        // Else we let's NumPy handle it
        PyErr_WarnEx(
            NULL,
            "Bohrium does not support indexing with arrays. It will be handled by the original NumPy.",
            1
        );

        // Let's make sure that 'k' is a NumPy array
        if(BhArray_CheckExact(k)) {
            k = BhArray_copy2numpy(k, NULL);
            if(k == NULL) {
                return -1;
            }
        }

        if(PyTuple_Check(k)) {
            for(i = 0; i < PyTuple_GET_SIZE(k); ++i) {
                PyObject *a = PyTuple_GET_ITEM(k, i);
                if(BhArray_CheckExact(a)) {
                    // Let's replace the item with a NumPy copy.
                    PyObject *t = BhArray_copy2numpy(a, NULL);
                    if(t == NULL) {
                        return -1;
                    }

                    Py_DECREF(a);
                    PyTuple_SET_ITEM(k, i, t);
                }
            }
        }

        // Let's make sure that 'v' is a NumPy array
        if(BhArray_CheckExact(v)) {
            v = BhArray_copy2numpy(v, NULL);
            if(v == NULL) {
                return -1;
            }
        }

        // Finally, let's do the SetItem in NumPy
        if(BhArray_data_bhc2np(o, NULL) == NULL) {
            return -1;
        }

        return PyArray_Type.tp_as_mapping->mp_ass_subscript(o, k, v);
    }

    // It is a regular SetItem call, let's do it in Python
    PyObject *ret = PyObject_CallMethod(ufuncs, "setitem", "OOO", o, k, v);
    if(ret == NULL) {
        return -1;
    }

    Py_XDECREF(ret);
    return 0;
}

// Help function that returns true when 'k' is a scalar object
static int is_scalar_key(PyObject *k) {
#if defined(NPY_PY3K)
    return (PyLong_Check(k) || PyArray_IsScalar(k, Integer) || (PyIndex_Check(k) && !PySequence_Check(k)));
#else
    return (PyArray_IsIntegerScalar(k) || (PyIndex_Check(k) && !PySequence_Check(k)));
#endif
}


static PyObject* BhArray_GetItem(PyObject *o, PyObject *k) {
    Py_ssize_t i;
    assert(k != NULL);
    assert(BhArray_CheckExact(o));

    if (obj_is_a_bool_mask(o, k)) {
        return PyObject_CallMethod(masking, "masked_get", "OO", o, k);
    }

    // Generally, we do not support indexing with arrays
    if(obj_contains_a_list_or_ary(k)) {
        // But when indexing array with an index array for each dimension in the array,
        // it corresponds to take_using_index_tuple()
        if (PySequence_Check(k) && PySequence_Size(k) == PyArray_NDIM((PyArrayObject*) o)) {
            return PyObject_CallMethod(reorganization, "take_using_index_tuple", "OO", o, k);
        }

        // And when indexing a vector, it corresponds to np.take()
        if (PyArray_NDIM((PyArrayObject*) o) == 1) {
            return PyObject_CallMethod(reorganization, "take", "OO", o, k);
        }

        // Else we let's NumPy handle it
        PyErr_WarnEx(
            NULL,
            "Bohrium does not support indexing with arrays. Bohrium will return a NumPy copy of the indexed array.",
            1
        );

        o = BhArray_copy2numpy(o, NULL);
        if(o == NULL) {
            return NULL;
        }

        if(BhArray_CheckExact(k)) {
            k = BhArray_copy2numpy(k, NULL);
            if(k == NULL) {
                return NULL;
            }
        }

        if(PyTuple_Check(k)) {
            for(i = 0; i < PyTuple_GET_SIZE(k); ++i) {
                PyObject *a = PyTuple_GET_ITEM(k, i);
                if(BhArray_CheckExact(a)) {
                    // Let's replace the item with a NumPy copy.
                    PyObject *t = BhArray_copy2numpy(a, NULL);
                    if(t == NULL) {
                        return NULL;
                    }

                    Py_DECREF(a);
                    PyTuple_SET_ITEM(k, i, t);
                }
            }
        }

        return PyArray_Type.tp_as_mapping->mp_subscript(o, k);
    }

    // When returning a scalar value, we copy the scalar back to NumPy without any warning
    int scalar_output = 0;
    if (PyArray_NDIM((PyArrayObject*) o) <= 1 && is_scalar_key(k)) {
        // A scalar index into a vector returns a scalar
        scalar_output = 1;
    } else if(PyTuple_Check(k) && (PyTuple_GET_SIZE(k) == PyArray_NDIM((PyArrayObject*) o))) {
        scalar_output = 1;
        for(i = 0; i < PyTuple_GET_SIZE(k); ++i) {
            PyObject *a = PyTuple_GET_ITEM(k, i);
            if(!is_scalar_key(a)) {
                // A slice never results in a scalar output
                scalar_output = 0;
                break;
            }
        }
    }

    if (scalar_output) {
        if (bh_sync_warn) {
            int err = PyErr_WarnEx(NULL, "BH_SYNC_WARN: Copying the scalar output to NumPy", 1);
            if (err) {
                return NULL;
            }
        }

        if (BhArray_data_bhc2np(o, NULL) == NULL) {
            return NULL;
        }
    }

    return PyArray_Type.tp_as_mapping->mp_subscript(o, k);
}

static PyObject* BhArray_GetSeqItem(PyObject *o, Py_ssize_t i) {
    // If we wrap the index 'i' into a Python Object we can simply use BhArray_GetItem
#if defined(NPY_PY3K)
    PyObject *index = PyLong_FromSsize_t(i);
#else
    PyObject *index = PyInt_FromSsize_t(i);
#endif
    if(index == NULL) {
        return NULL;
    }

    PyObject *ret = BhArray_GetItem(o, index);
    Py_DECREF(index);

    return ret;
}

static PyMappingMethods array_as_mapping = {
    (lenfunc) 0,                     // mp_length
    (binaryfunc) BhArray_GetItem,    // mp_subscript
    (objobjargproc) BhArray_SetItem, // mp_ass_subscript
};

static PySequenceMethods array_as_sequence = {
    (lenfunc) 0,                              // sq_length
    (binaryfunc) NULL,                        // sq_concat is handled by nb_add
    (ssizeargfunc) NULL,                      // sq_repeat
    (ssizeargfunc) BhArray_GetSeqItem,        // sq_item
    (ssizessizeargfunc) 0,                    // sq_slice (Not in the Python doc)
    (ssizeobjargproc) BhArray_SetItem,        // sq_ass_item
    (ssizessizeobjargproc) BhArray_SetSlice,  // sq_ass_slice (Not in the Python doc)
    (objobjproc) 0,                           // sq_contains
    (binaryfunc) NULL,                        // sg_inplace_concat
    (ssizeargfunc) NULL,                      // sg_inplace_repeat
};

static PyObject* BhArray_Repr(PyObject *self) {
    assert(BhArray_CheckExact(self));

    PyObject *t = BhArray_copy2numpy(self, NULL);
    if(t == NULL) {
        return NULL;
    }

    PyObject *str = NULL;

    BH_PyArrayObject *base = &((BhArray*) self)->base;
    if (base->nd == 0) {
        // 0-rank array -> single value
        void *data = PyArray_GetPtr((PyArrayObject*) self, 0);
        char c[32];

        switch (base->descr->type) {
            case 'i': // int32
                snprintf(c, sizeof(c), "%d", *((npy_int*) data));
                break;
            case 'l': // int64
                snprintf(c, sizeof(c), "%ld", *((npy_long*) data));
                break;
            case 'I': // uint32
                 snprintf(c, sizeof(c), "%u", *((npy_uint*) data));
                break;
            case 'L': // uint64
                 snprintf(c, sizeof(c), "%lu", *((npy_ulong*) data));
                break;
            case 'f': // float
                snprintf(c, sizeof(c), "%.6g", *((npy_float*) data));
                break;
            case 'd': // double
                snprintf(c, sizeof(c), "%.6g", *((npy_double*) data));
                break;
        }

        if (c[0] == 0) {
            str = PyArray_Type.tp_repr(t);
        } else {
#if defined(NPY_PY3K)
            str = PyUnicode_FromString(c);
#else
            str = PyString_FromString(c);
#endif
        }
    } else {
        str = PyArray_Type.tp_repr(t);
    }

    Py_DECREF(t);

    return str;
}
static PyObject* BhArray_Str(PyObject *self) {
    assert(BhArray_CheckExact(self));

    PyObject *t = BhArray_copy2numpy(self, NULL);
    if(t == NULL) {
        return NULL;
    }

    PyObject *str = PyArray_Type.tp_str(t);
    Py_DECREF(t);

    return str;
}

// Importing the array_as_number struct
#include "operator_overload.c"

static PyTypeObject BhArrayType = {
#if defined(NPY_PY3K)
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                              // ob_size
#endif
    "bohrium.ndarray",              // tp_name
    sizeof(BhArray),                // tp_basicsize
    0,                              // tp_itemsize
    (destructor) BhArray_dealloc,   // tp_dealloc
    0,                              // tp_print
    0,                              // tp_getattr
    0,                              // tp_setattr
#if defined(NPY_PY3K)
    0,                              // tp_reserved
#else
    0,                              // tp_compare
#endif
    &BhArray_Repr,                  // tp_repr
    &array_as_number,               // tp_as_number
    &array_as_sequence,             // tp_as_sequence
    &array_as_mapping,              // tp_as_mapping
    0,                              // tp_hash
    0,                              // tp_call
    &BhArray_Str,                   // tp_str
    0,                              // tp_getattro
    0,                              // tp_setattro
    0,                              // tp_as_buffer
    Py_TPFLAGS_DEFAULT
#if !defined(NPY_PY3K)
    | Py_TPFLAGS_CHECKTYPES
#endif
    | Py_TPFLAGS_BASETYPE,          // tp_flags
    0,                              // tp_doc
    0,                              // tp_traverse
    0,                              // tp_clear
    (richcmpfunc)array_richcompare, // tp_richcompare
    0,                              // tp_weaklistoffset
    0,                              // tp_iter
    0,                              // tp_iternext
    BhArrayMethods,                 // tp_methods
    BhArrayMembers,                 // tp_members
    0,                              // tp_getset
    0,                              // tp_base
    0,                              // tp_dict
    0,                              // tp_descr_get
    0,                              // tp_descr_set
    0,                              // tp_dictoffset
    0,                              // tp_init
    BhArray_alloc,                  // tp_alloc
    BhArray_new,                    // tp_new
    (freefunc)BhArray_free,         // tp_free
    0,                              // tp_is_gc
    0,                              // tp_bases
    0,                              // tp_mro
    0,                              // tp_cache
    0,                              // tp_subclasses
    0,                              // tp_weaklist
    0,                              // tp_del
    0,                              // tp_version_tag
};

#if defined(NPY_PY3K)
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_bh",
        NULL,
        -1,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL
};
#endif

#if defined(NPY_PY3K)
#define RETVAL m
PyMODINIT_FUNC PyInit__bh(void)
#else
#define RETVAL
PyMODINIT_FUNC init_bh(void)
#endif
{
    PyObject *m;

#if defined(NPY_PY3K)
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule("_bh", NULL);
#endif
    if (m == NULL) {
        return RETVAL;
    }

    // Import NumPy
    import_array();

    BhArrayType.tp_base = &PyArray_Type;
    if (PyType_Ready(&BhArrayType) < 0) {
        return RETVAL;
    }

    // HACK: In order to force NumPy scalars on the left hand side of an operand to use Bohrium
    // we add all scalar types to the Method Resolution Order tuple.
    // This hack has undesirable consequences: <https://github.com/bh107/bohrium/issues/22>
    // Until NumPy introduces the "__numpy_ufunc__" method, we will accept that NumPy scalars
    // on the left hand side raises mem_access_callback()

    /*
    {
        Py_ssize_t i;
        PyObject *_info = PyImport_ImportModule("bohrium._info");
        if(_info == NULL) {
            return RETVAL;
        }

        PyObject *dtypes = PyObject_GetAttrString(_info, "numpy_types");
        if(dtypes == NULL) {
            return RETVAL;
        }

        Py_ssize_t ndtypes = PyList_GET_SIZE(dtypes);
        Py_ssize_t old_size = PyTuple_GET_SIZE(BhArrayType.tp_mro);
        Py_ssize_t new_size = old_size + ndtypes;
        if(_PyTuple_Resize(&BhArrayType.tp_mro, new_size) != 0) {
            return RETVAL;
        }

        for(i = 0; i < ndtypes; ++i) {
            PyObject *t = PyObject_GetAttrString(PyList_GET_ITEM(dtypes, i), "type");
            if(t == NULL) {
                return RETVAL;
            }

            PyTuple_SET_ITEM(BhArrayType.tp_mro, i+old_size, t);
        }

        Py_DECREF(_info);
    }
    */

    PyModule_AddObject(m, "ndarray", (PyObject*) &BhArrayType);

    bohrium        = PyImport_ImportModule("bohrium");
    bhary          = PyImport_ImportModule("bohrium.bhary");
    ufuncs         = PyImport_ImportModule("bohrium.ufuncs");
    array_create   = PyImport_ImportModule("bohrium.array_create");
    reorganization = PyImport_ImportModule("bohrium.reorganization");
    masking        = PyImport_ImportModule("bohrium.masking");

    if(bhary          == NULL ||
       ufuncs         == NULL ||
       bohrium        == NULL ||
       array_create   == NULL ||
       reorganization == NULL ||
       masking        == NULL) {
        return RETVAL;
    }

    // Check the 'BH_SYNC_WARN' flag
    char *value = getenv("BH_SYNC_WARN");
    if (value != NULL) {
        bh_sync_warn = 1;
    }

    // Check the 'BH_MEM_WARN' flag
    value = getenv("BH_MEM_WARN");
    if (value != NULL) {
        bh_mem_warn = 1;
    }

    // Initialize the signal handler
    bh_mem_signal_init();

    // Register an module exit function
    Py_AtExit(module_exit);
    return RETVAL;
}
