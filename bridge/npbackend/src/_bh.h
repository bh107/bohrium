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

#pragma once

#include <Python.h>
#include <structmember.h>
#include <bohrium_api.h>


#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL bh_ARRAY_API
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

typedef struct {
    int initiated;
    int type_enum;
    int ndim;
    int64_t start;
    int64_t shape[NPY_MAXDIMS];
    int64_t stride[NPY_MAXDIMS];
} BhView;

// The declaration of the Bohrium ndarray
typedef struct {
    BH_PyArrayObject base;
    int mmap_allocated; // Is the memory allocated by us?
    void *npy_data; // NumPy allocated array data
    BhView view; // View information, which might be obsolete
    void *bhc_array; // bhc handle to the array
    int data_in_bhc; // Is the data in bhc?
    PyObject* dynamic_view_info; // Information regarding dynamic changes
                                 // to the view within a do_while loop
} BhArray;

// Exposing some global variables implemented in `_bh.c`
extern PyTypeObject BhArrayType; // Implemented in `_bh.c`
extern PyObject *bh_api;         // The Bohrium API Python module
extern PyObject *ufuncs;         // The ufuncs Python module
extern PyObject *bohrium;        // The Bohrium Python module
extern PyObject *array_create;   // The array_create Python module
extern PyObject *reorganization; // The reorganization Python module
extern PyObject *masking;        // The masking Python module
extern PyObject *loop;           // The loop Python module
extern int bh_sync_warn;         // Boolean flag: should we warn when copying from Bohrium to NumPy
extern int bh_mem_warn;          // Boolean flag: should we warn when about memory problems
extern int bh_unsupported_warn;  // Boolean flag: should we warn when encountering an unsupported operation
extern PyThreadState *py_thread_state; // The current Python thread state

// Help function that creates a simple new array.
// We parse to PyArray_NewFromDescr(), a new protected memory allocation
// Return the new Python object, or NULL on error
PyObject* simply_new_array(PyTypeObject *type, PyArray_Descr *descr, uint64_t nbytes, int ndim, npy_intp shape[]);
