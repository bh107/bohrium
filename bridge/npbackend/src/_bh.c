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

#include "_bh.h"
#include <dlfcn.h>
#include "handle_array_op.h"
#include "handle_special_op.h"
#include "memory.h"

// Forward declaration
static PyObject* BhArray_data_bhc2np(PyObject *self);

PyObject *bh_api         = NULL; // The Bohrium API Python module
PyObject *ufuncs         = NULL; // The ufuncs Python module
PyObject *bohrium        = NULL; // The Bohrium Python module
PyObject *array_create   = NULL; // The array_create Python module
PyObject *reorganization = NULL; // The reorganization Python module
PyObject *masking        = NULL; // The masking Python module
PyObject *loop           = NULL; // The loop Python module
int bh_sync_warn         = 0;    // Boolean: should we warn when copying from Bohrium to NumPy
int bh_mem_warn          = 0;    // Boolean: should we warn when about memory problems
int bh_unsupported_warn  = 0;    // Boolean flag: should we warn when encountering a unsupported operation

// The current Python thread state
PyThreadState *py_thread_state = NULL;

// Called when module exits
static void module_exit(void) {
    PyFlush(NULL, NULL);
    BhAPI_mem_signal_shutdown();
}

// Help function that creates a simple new array.
// We parse to PyArray_NewFromDescr(), a new protected memory allocation
// Return the new Python object, or NULL on error
PyObject* simply_new_array(PyTypeObject *type, PyArray_Descr *descr, uint64_t nbytes, int ndim, npy_intp shape[]) {
    // Let's create a new NumPy array using our memory allocation
    if (nbytes == 0) {
        nbytes = descr->elsize;
    }
    void *addr = mem_map(nbytes);

    PyObject *ret = PyArray_NewFromDescr(type, descr, ndim, shape, NULL, addr, 0, NULL);
    if(ret == NULL) {
        return NULL;
    }

    ((BhArray*) ret)->base.flags |= NPY_ARRAY_OWNDATA;
    ((BhArray*) ret)->base.flags |= NPY_ARRAY_CARRAY;
    ((BhArray*) ret)->mmap_allocated = 1;
    ((BhArray*) ret)->data_in_bhc = 1;
    ((BhArray*) ret)->dynamic_view_info = NULL;

    PyArray_UpdateFlags((PyArrayObject*) ret, NPY_ARRAY_UPDATE_ALL);
    mem_signal_attach(ret, ((BhArray*) ret)->base.data, nbytes);
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

                ret = simply_new_array(type, descr, nelem * descr->elsize, shape.len, shape.ptr);
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
    protected_malloc((BhArray *) ret);
    return ret;
}

static PyObject* BhArray_finalize(PyObject *self, PyObject *args) {
    int e = PyObject_IsInstance(self, (PyObject*) &BhArrayType);
    if(e == -1) {
        return NULL;
    } else if (e == 0) {
        Py_RETURN_NONE;
    }

    ((BhArray*) self)->bhc_array = NULL;
    ((BhArray*) self)->view.initiated = 0;
    ((BhArray*) self)->data_in_bhc = 1;

    protected_malloc((BhArray *) self);

    Py_RETURN_NONE;
}

static PyObject* BhArray_alloc(PyTypeObject *type, Py_ssize_t nitems) {
    PyObject *obj;
    obj = (PyObject *) PyObject_Malloc(type->tp_basicsize);
    PyObject_Init(obj, type);

    // Flag the array as uninitialized
    ((BhArray*) obj)->npy_data         = NULL;
    ((BhArray*) obj)->mmap_allocated   = 0;

    ((BhArray*) obj)->bhc_array = NULL;
    ((BhArray*) obj)->view.initiated = 0;
    ((BhArray*) obj)->data_in_bhc = 0;
    ((BhArray*) obj)->dynamic_view_info = NULL;

    return obj;
}

static void BhArray_dealloc(BhArray* self) {
    assert(BhArray_CheckExact(self));

    if(self->bhc_array != NULL) {
        assert(self->view.initiated);
        BhAPI_destroy(dtype_np2bhc(self->view.type_enum), self->bhc_array);
    }

    if (!PyArray_CHKFLAGS((PyArrayObject*) self, NPY_ARRAY_OWNDATA)) {
        BhArrayType.tp_base->tp_dealloc((PyObject*) self);
        return; // The array doesn't own the array data
    }

    if (self->mmap_allocated) {
        mem_unmap(PyArray_DATA((PyArrayObject*) self), ary_nbytes(self));
        BhAPI_mem_signal_detach(PyArray_DATA((PyArrayObject*) self));
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
static PyObject* BhArray_data_bhc2np(PyObject *self) {
    assert(BhArray_CheckExact(self));

    // We move the whole array (i.e. the base array) from Bohrium to NumPy
    BhArray *base = get_base(self);
    if (base == NULL) {
        return NULL;
    }

    if(!PyArray_CHKFLAGS((PyArrayObject*) base, NPY_ARRAY_OWNDATA)) {
        PyErr_Format(PyExc_ValueError, "The base array doesn't own its data");
    }

    // Let's move data from the bhc domain to the NumPy domain
    mem_bhc2np(base);

    // Finally, we can return NULL on error (but not before!)
    if (PyErr_Occurred() != NULL) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* BhArray_data_fill(PyObject *self, PyObject *args) {
    assert(BhArray_CheckExact(self));
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

    // Copy the data from the NumPy array 'np_ary' to the bhc part of `self`
    void *data = get_data_pointer((BhArray*) self, 1, 1, 0);
    memmove(data, PyArray_DATA((PyArrayObject*) np_ary), PyArray_NBYTES((PyArrayObject*) np_ary));
    Py_RETURN_NONE;
}

static PyObject* BhArray_copy2numpy(PyObject *self, PyObject *args) {
    assert(args == NULL);
    PyObject *ret = PyArray_NewLikeArray((PyArrayObject*) self, NPY_ANYORDER, NULL, 0);
    if(ret == NULL) {
        return NULL;
    }
    PyObject *base = (PyObject*) get_base(self);
    if (base == NULL) {
        Py_DECREF(ret);
        return NULL;
    } else if(BhArray_data_bhc2np(base) == NULL) {
        Py_DECREF(ret);
        return NULL;
    }
    if(PyArray_CopyInto((PyArrayObject*) ret, (PyArrayObject*) self) == -1) {
        Py_DECREF(ret);
        return NULL;
    }
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
    void *data = get_data_pointer((BhArray*) self, 1, 1, 0);
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

    PyObject *py_name = PyObject_GetAttrString(bohrium, name);
    if (py_name == NULL) {
        Py_DECREF(func_args);
        return NULL;
    }

    PyObject *ret = PyObject_Call(PyObject_GetAttrString(bohrium, name), func_args, kwds);
    Py_DECREF(py_name);
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

static PyObject* BhArray_mean(PyObject *self, PyObject *args, PyObject *kwds) {
    return method2function("mean", self, args, kwds);
}

static PyObject* BhArray_dot(PyObject *self, PyObject *args, PyObject *kwds) {
    return method2function("dot", self, args, kwds);
}

static PyObject* BhArray_format(PyObject *self, PyObject *args, PyObject *kwds) {
    assert(BhArray_CheckExact(self));

    PyObject *npy_ary = BhArray_copy2numpy(self, NULL);
    if(npy_ary == NULL) {
        return NULL;
    }

    PyObject *__format__ = PyObject_GetAttrString(npy_ary, "__format__");
    if (__format__ == NULL) {
        Py_DECREF(npy_ary);
        return NULL;
    }
    PyObject *ret = PyObject_Call(__format__, args, kwds);
    Py_DECREF(npy_ary);
    Py_DECREF(__format__);
    return ret;
}

static PyMethodDef BhArrayMethods[] = {
    {"__array_finalize__", BhArray_finalize,                    METH_VARARGS,                 NULL},
    {"__array_ufunc__",    (PyCFunction) BhArray_array_ufunc,   METH_VARARGS | METH_KEYWORDS, "Handle ufunc"},
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
    {"mean",               (PyCFunction) BhArray_mean,          METH_VARARGS | METH_KEYWORDS, "a.mean(axis=None, dtype=None, out=None)\n\n Compute the arithmetic mean along the specified axis."},
    {"dot",                (PyCFunction) BhArray_dot,           METH_VARARGS | METH_KEYWORDS, "a.dot(b, out=None)\n\n Compute the dot product."},
    {"__format__",         (PyCFunction) BhArray_format,        METH_VARARGS | METH_KEYWORDS, "a.__format_()\n\n Implement the new string formatting in Python 3."},
    {NULL, NULL, 0, NULL} /* Sentinel */
};


static PyMemberDef BhArrayMembers[] = {
    {"bhc_mmap_allocated", T_BOOL, offsetof(BhArray, mmap_allocated), 0, "Is the base data allocated with mmap?"},
    {"bhc_dynamic_view_info", T_OBJECT, offsetof(BhArray, dynamic_view_info), 0, "The information regarding dynamic changes to a view within a do_while loop"},
    {NULL}  /* Sentinel */
};

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
        if(BhArray_data_bhc2np(o) == NULL) {
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

    PyObject* loop_check = PyObject_CallMethod(loop, "has_iterator", "O", k);
    if (loop_check == Py_True) {
        return PyObject_CallMethod(loop, "slide_from_view", "OO", o, k);
    }

    if(((BhArray*) o)->dynamic_view_info && ((BhArray*) o)->dynamic_view_info != Py_None) {
        return PyObject_CallMethod(loop, "inherit_dynamic_changes", "OO", o, k);
    }

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

        if (BhArray_data_bhc2np(o) == NULL) {
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
    (ssizessizeobjargproc) NULL,              // sq_ass_slice Uses setitem instead
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

PyTypeObject BhArrayType = {
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

// The methods (functions) of this module
static PyMethodDef _bhMethods[] = {
    {"ufunc", (PyCFunction) PyArrayOp, METH_VARARGS | METH_KEYWORDS,
              "Handle regular array operations, which is the operations " \
              "implemented in `bhc_array_operations_enum_typed.cpp`."},
    {"extmethod", (PyCFunction) PyExtMethod, METH_VARARGS | METH_KEYWORDS,
              "Handle extension methods."},
    {"flush", PyFlush,  METH_NOARGS,
              "Evaluate all delayed array operations"},
    {"flush_count", PyFlushCount,  METH_NOARGS,
            "Get the number of times flush has been called"},
    {"flush_and_repeat", (PyCFunction) PyFlushCountAndRepeat, METH_VARARGS | METH_KEYWORDS,
            "Flush and repeat the lazy evaluated operations while `condition` " \
            "is true and `nrepeats` hasn't been reach"},
    {"sync", (PyCFunction) PySync, METH_VARARGS | METH_KEYWORDS,
            "Sync `ary` to host memory."},
    {"slide_view", (PyCFunction) PySlideView, METH_VARARGS | METH_KEYWORDS,
            "Increase `ary`s offset by one."},
    {"add_reset", (PyCFunction) PyAddReset, METH_VARARGS | METH_KEYWORDS,
            "Add a reset for a given dimension."},
    {"random123", (PyCFunction) PyRandom123, METH_VARARGS | METH_KEYWORDS,
            "Create a new random array using the random123 algorithm.\n" \
            "The dtype is uint64 always."},
    {"get_data_pointer", (PyCFunction) PyGetDataPointer, METH_VARARGS | METH_KEYWORDS,
            "Return a pointer to the bhc data of `ary`\n"},
    {"set_data_pointer", (PyCFunction) PySetDataPointer, METH_VARARGS | METH_KEYWORDS,
            "Set the data pointer of `ary`\n"},
    {"mem_copy", (PyCFunction) PyMemCopy, METH_VARARGS | METH_KEYWORDS,
            "Copy the memory of `src` to `dst`\n"},
    {"get_device_context", PyGetDeviceContext,  METH_NOARGS,
            "Get the device context, such as OpenCL's cl_context, of the first VE in the runtime stack"},
    {"message", (PyCFunction) PyMessage, METH_VARARGS | METH_KEYWORDS,
            "Send and receive a message through the Bohrium stack\n"},
    {"same_view", (PyCFunction) PySameView, METH_VARARGS | METH_KEYWORDS,
            "Return true when `v1` and `v2` is exactly the same (incl. pointing to the same base)\n"},
    {"user_kernel", (PyCFunction) PyUserKernel, METH_VARARGS | METH_KEYWORDS,
            "Run a user kernel\n"},
    {"is_array_behaving", (PyCFunction) PyIsBehaving, METH_VARARGS | METH_KEYWORDS,
            "Check if a bohrium array is behaving\n"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

#if defined(NPY_PY3K)
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_bh",/* name of module */
    NULL, /* module documentation, may be NULL */
    -1,   /* size of per-interpreter state of the module or -1 if the module keeps state in global variables. */
    _bhMethods /* the methods of this module */
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
    m = Py_InitModule("_bh", _bhMethods);
#endif
    if (m == NULL) {
        return RETVAL;
    }

    bh_api = PyImport_ImportModule("bohrium_api");
    if (bh_api == NULL) {
        return RETVAL;
    }

    // Import Bohrium API
    if (import_bh_api() < 0)
        return RETVAL;

    // Import NumPy
    import_array();

    BhArrayType.tp_base = &PyArray_Type;
    if (PyType_Ready(&BhArrayType) < 0) {
        return RETVAL;
    }

    PyModule_AddObject(m, "ndarray", (PyObject*) &BhArrayType);

    bohrium = PyImport_ImportModule("bohrium");
    if(bohrium == NULL) {
        return RETVAL;
    }
    ufuncs = PyImport_ImportModule("bohrium.ufuncs");
    if(ufuncs == NULL) {
        return RETVAL;
    }
    array_create = PyImport_ImportModule("bohrium.array_create");
    if(array_create == NULL) {
        return RETVAL;
    }
    reorganization = PyImport_ImportModule("bohrium.reorganization");
    if(reorganization == NULL) {
        return RETVAL;
    }
    masking = PyImport_ImportModule("bohrium.masking");
    if(masking == NULL) {
        return RETVAL;
    }
    loop = PyImport_ImportModule("bohrium.loop");
    if(loop == NULL) {
        return RETVAL;
    }

    // Check environment variables
    bh_sync_warn = get_bool_env("BH_SYNC_WARN", 0);
    bh_mem_warn = get_bool_env("BH_MEM_WARN", 0);
    bh_unsupported_warn = get_bool_env("BH_UNSUP_WARN", 1);

    // Let's save the current Python thread state
    PyGILState_STATE gil = PyGILState_Ensure();
    py_thread_state = PyGILState_GetThisThreadState();
    PyGILState_Release(gil);

    // Initialize the signal handler
    BhAPI_mem_signal_init();

    // Register an module exit function
    Py_AtExit(module_exit);
    return RETVAL;
}
