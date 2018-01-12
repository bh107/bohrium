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

#include <bh_osx.h>
#include <Python.h>
#include <structmember.h>

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

typedef struct {
    BH_PyArrayObject base;
    PyObject *bhc_ary;
    PyObject *bhc_ary_version;
    PyObject *bhc_view;
    PyObject *bhc_view_version;
    int mmap_allocated;
    void *npy_data; // NumPy allocated array data
} BhArray;

// Forward declaration
static PyTypeObject BhArrayType;

#define BhArray_CheckExact(op) (((PyObject*) (op))->ob_type == &BhArrayType)
#define bhc_exist(x) (((BhArray*) x)->bhc_ary != Py_None)