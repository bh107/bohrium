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

#include "util.h"
#include "bharray.h"

PyObject *
PyExtMethod(PyObject *self, PyObject *args, PyObject *kwds) {
    char *name;
    PyObject *operand_fast_seq;
    {
        PyObject *operand_list;
        static char *kwlist[] = {"name", "operand_list:list", NULL};
        if (!PyArg_ParseTupleAndKeywords(args, kwds, "sO", kwlist, &name, &operand_list)) {
            return NULL;
        }
        operand_fast_seq = PySequence_Fast(operand_list, "`operand_list` should be a sequence.");
        if (operand_fast_seq == NULL) {
            return NULL;
        }
    }
    const Py_ssize_t nop = PySequence_Fast_GET_SIZE(operand_fast_seq);
    if (nop != 3) {
        PyErr_Format(PyExc_TypeError, "Expects three operands.");
        Py_DECREF(operand_fast_seq);
        return NULL;
    }

    // Read and normalize all operands
    bhc_dtype types[nop];
    bhc_bool constant;
    void *operands[nop];
    normalize_cleanup_handle cleanup;
    cleanup.objs2free_count = 0;
    for (int i = 0; i < nop; ++i) {
        PyObject *op = PySequence_Fast_GET_ITEM(operand_fast_seq, i); // Borrowed reference and will not fail
        int err = normalize_operand(op, &types[i], &constant, &operands[i], &cleanup);
        if (err != -1) {
            if (constant) {
                PyErr_Format(PyExc_TypeError, "Scalars isn't supported.");
                err = -1;
            } else if (types[0] != types[i]) {
                PyErr_Format(PyExc_TypeError, "The dtype of all operands must be the same.");
                err = -1;
            }
        }
        if (err == -1) {
            normalize_operand_cleanup(&cleanup);
            Py_DECREF(operand_fast_seq);
            if (PyErr_Occurred() != NULL) {
                return NULL;
            } else {
                Py_RETURN_NONE;
            }
        }

    }

    int err = BhAPI_extmethod(types[0], name, operands[0], operands[1], operands[2]);
    if (err) {
        PyErr_Format(PyExc_TypeError, "The current runtime system does not support the extension method '%s'", name);
    }

    // Clean up
    normalize_operand_cleanup(&cleanup);
    Py_DECREF(operand_fast_seq);

    if (PyErr_Occurred() != NULL) {
        return NULL;
    } else {
        Py_RETURN_NONE;
    }
}

PyObject* PyFlush(PyObject *self, PyObject *args) {
    BhAPI_flush();
    Py_RETURN_NONE;
}

PyObject* PyFlushCount(PyObject *self, PyObject *args) {
    return PyLong_FromLong(BhAPI_flush_count());
}

PyObject* PyFlushCountAndRepeat(PyObject *self, PyObject *args, PyObject *kwds) {
    unsigned long nrepeats;
    PyObject *condition = NULL;
    static char *kwlist[] = {"nrepeats", "condition:bharray", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "k|O", kwlist, &nrepeats, &condition)) {
        return NULL;
    }
    if (condition == NULL || condition == Py_None) {
        BhAPI_flush_and_repeat(nrepeats);
    } else if(!BhArray_CheckExact(condition)) {
        PyErr_Format(PyExc_TypeError, "The condition must a bharray array or None");
        return NULL;
    } else {
        bhc_dtype type;
        bhc_bool constant;
        void *operand;
        normalize_cleanup_handle cleanup;
        cleanup.objs2free_count = 0;
        int err = normalize_operand(condition, &type, &constant, &operand, &cleanup);
        if (err == -1) {
            normalize_operand_cleanup(&cleanup);
            if (PyErr_Occurred() != NULL) {
                return NULL;
            } else {
                Py_RETURN_NONE;
            }
        }
        BhAPI_flush_and_repeat_condition(nrepeats, operand);
        normalize_operand_cleanup(&cleanup);
    }
    Py_RETURN_NONE;
}

PyObject* PySync(PyObject *self, PyObject *args, PyObject *kwds) {
    PyObject *ary;
    static char *kwlist[] = {"ary", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &ary)) {
        return NULL;
    }

    bhc_dtype type;
    bhc_bool constant;
    void *operand;
    normalize_cleanup_handle cleanup;
    cleanup.objs2free_count = 0;
    int err = normalize_operand(ary, &type, &constant, &operand, &cleanup);
    if (err == -1) {
        normalize_operand_cleanup(&cleanup);
        if (PyErr_Occurred() != NULL) {
            return NULL;
        } else {
            Py_RETURN_NONE;
        }
    }
    BhAPI_sync(type, operand);
    normalize_operand_cleanup(&cleanup);
    Py_RETURN_NONE;
}

PyObject* PySlideView(PyObject *self, PyObject *args, PyObject *kwds) {
    PyObject *ary1;
    long long dim;
    long long slide;
    long long view_shape;
    long long array_shape;
    long long array_stride;
    long long step_delay;
    static char *kwlist[] = {"ary1", "dim", "slide", "view_shape", "array_shape",
                             "array_stride", "step_delay", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OLLLLLL", kwlist, &ary1, &dim, &slide,
                                     &view_shape, &array_shape, &array_stride, &step_delay)) {
        return NULL;
    }

    bhc_dtype type1;
    bhc_bool constant1;
    void *operand1;
    normalize_cleanup_handle cleanup1;
    int err1 = normalize_operand(ary1, &type1, &constant1, &operand1, &cleanup1);
    cleanup1.objs2free_count = 0;
    if (err1 == -1) {
        normalize_operand_cleanup(&cleanup1);
        if (PyErr_Occurred() != NULL) {
            return NULL;
        } else {
            Py_RETURN_NONE;
        }
    }

    BhAPI_slide_view(type1, operand1, dim, slide, view_shape, array_shape, array_stride, step_delay);
    normalize_operand_cleanup(&cleanup1);

    Py_RETURN_NONE;
}

PyObject* PyAddReset(PyObject *self, PyObject *args, PyObject *kwds) {
    PyObject *ary1;
    long long dim;
    long long reset_max;
    static char *kwlist[] = {"ary1", "dim", "reset_max", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OLL", kwlist, &ary1, &dim, &reset_max)) {
        return NULL;
    }

    bhc_dtype type1;
    bhc_bool constant1;
    void *operand1;
    normalize_cleanup_handle cleanup1;
    int err1 = normalize_operand(ary1, &type1, &constant1, &operand1, &cleanup1);
    cleanup1.objs2free_count = 0;
    if (err1 == -1) {
        normalize_operand_cleanup(&cleanup1);
        if (PyErr_Occurred() != NULL) {
            return NULL;
        } else {
            Py_RETURN_NONE;
        }
    }

    BhAPI_add_reset(type1, operand1, dim, reset_max);
    normalize_operand_cleanup(&cleanup1);

    Py_RETURN_NONE;
}

PyObject* PyRandom123(PyObject *self, PyObject *args, PyObject *kwds) {
    unsigned long long size;
    unsigned long long seed;
    unsigned long long key;
    static char *kwlist[] = {"size:int", "seed:int", "key:int", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "KKK", kwlist, &size, &seed, &key)) {
        return NULL;
    }

    npy_intp shape[1] = {size};
    PyObject *ret = simply_new_array(&BhArrayType, PyArray_DescrFromType(NPY_UINT64), size*8, 1, shape);
    if (ret == NULL) {
        return NULL;
    }
    if (size > 0) {
        bhc_dtype type;
        bhc_bool constant;
        void *operand;
        normalize_cleanup_handle cleanup;
        cleanup.objs2free_count = 0;
        int err = normalize_operand(ret, &type, &constant, &operand, &cleanup);
        if (err == -1) {
            normalize_operand_cleanup(&cleanup);
            if (PyErr_Occurred() != NULL) {
                return NULL;
            } else {
                return ret;
            }
        }
        BhAPI_random123(operand, seed, key);
        normalize_operand_cleanup(&cleanup);
    }
    return ret;
}

void *get_data_pointer(BhArray *ary, bhc_bool copy2host, bhc_bool force_alloc, bhc_bool nullify) {
    if (PyArray_SIZE((PyArrayObject*) ary) <= 0) {
        return NULL;
    }
    void *ary_ptr = bharray_bhc(ary);
    bhc_dtype dtype = dtype_np2bhc(PyArray_DESCR((PyArrayObject*) ary)->type_num);
    if (copy2host) {
        BhAPI_sync(dtype, ary_ptr);
    }
    BhAPI_flush();
    void *ret = BhAPI_data_get(dtype, ary_ptr, copy2host, force_alloc, nullify);
    return ret;
}

PyObject* PyGetDataPointer(PyObject *self, PyObject *args, PyObject *kwds) {
    PyObject *ary;
    npy_bool copy2host = 1;
    npy_bool allocate = 0;
    npy_bool nullify = 0;
    static char *kwlist[] = {"ary:bharray", "copy2host", "allocate", "nullify", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O&O&O&", kwlist, &ary,
                                     PyArray_BoolConverter, &copy2host,
                                     PyArray_BoolConverter, &allocate,
                                     PyArray_BoolConverter, &nullify)) {
        return NULL;
    }
    if (!BhArray_CheckExact(ary)) {
        PyErr_Format(PyExc_TypeError, "The `ary` must be a bharray.");
        return NULL;
    }
    if (PyArray_NBYTES((PyArrayObject*) ary) > 0) {
        void *data = get_data_pointer((BhArray *) ary, copy2host, allocate, nullify);
        if (data != NULL) {
            return PyLong_FromVoidPtr(data);
        }
    }
    return PyLong_FromLong(0);
}

PyObject* PySetDataPointer(PyObject *self, PyObject *args, PyObject *kwds) {
    PyObject *ary;
    PyObject *py_mem_ptr;
    npy_bool host_ptr = 1;
    static char *kwlist[] = {"ary:bharray", "mem_ptr", "host_ptr", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O&O&", kwlist, &ary, &py_mem_ptr,
                                     PyArray_BoolConverter, &host_ptr)) {
        return NULL;
    }
    if (!BhArray_CheckExact(ary)) {
        PyErr_Format(PyExc_TypeError, "The `ary` must be a bharray.");
        return NULL;
    }
    if (PyArray_NBYTES((PyArrayObject*) ary) == 0) {
        Py_RETURN_NONE;
    }
    void *mem_ptr = PyLong_AsVoidPtr(py_mem_ptr);
    if (PyErr_Occurred() != NULL) {
        return NULL;
    }

    void *ary_ptr = bharray_bhc((BhArray *) ary);
    bhc_dtype dtype = dtype_np2bhc(PyArray_DESCR((PyArrayObject*) ary)->type_num);
    BhAPI_sync(dtype, ary_ptr);
    BhAPI_flush();
    BhAPI_data_set(dtype, ary_ptr, host_ptr, mem_ptr);
    Py_RETURN_NONE;
}

PyObject* PyMemCopy(PyObject *self, PyObject *args, PyObject *kwds) {
    PyObject *src;
    const char *param = "";
    static char *kwlist[] = {"src:bharray", "param", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|s", kwlist, &src, &param)) {
        return NULL;
    }
    if (!BhArray_CheckExact((PyArrayObject*) src)) {
        PyErr_Format(PyExc_TypeError, "The `src` must be a bharray.");
        return NULL;
    }

    PyObject *dst = PyObject_CallMethod(array_create, "empty_like", "O", src);
    if(dst == NULL) {
        return NULL;
    }

    src = (PyObject *) PyArray_GETCONTIGUOUS((PyArrayObject*) src);
    if(src == NULL) {
        return NULL;
    }
    bhc_dtype dtype = dtype_np2bhc(PyArray_DESCR((PyArrayObject*) src)->type_num);

    if (PyArray_NBYTES((PyArrayObject*) src) > 0) {
        void *src_ptr = bharray_bhc((BhArray *) src);
        void *dst_ptr = bharray_bhc((BhArray *) dst);
        BhAPI_flush();
        BhAPI_data_copy(dtype, src_ptr, dst_ptr, param);
    }
    Py_DECREF(src);
    return dst;
}

PyObject* PyGetDeviceContext(PyObject *self, PyObject *args) {
    assert(args == NULL);
    return PyLong_FromVoidPtr(BhAPI_getDeviceContext());
}

PyObject* PyMessage(PyObject *self, PyObject *args, PyObject *kwds) {
    char *msg;
    static char *kwlist[] = {"msg:str", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", kwlist, &msg)) {
        return NULL;
    }
#if defined(NPY_PY3K)
    return PyUnicode_FromString(BhAPI_message(msg));
#else
    return PyString_FromString(BhAPI_message(msg));
#endif
}

PyObject* PyUserKernel(PyObject *self, PyObject *args, PyObject *kwds) {
    char *kernel, *compile_cmd, *tag, *param;
    PyObject *operand_fast_seq;
    {
        PyObject *operand_list;
        static char *kwlist[] = {"kernel:str", "operand_list:list", "compiler_cmd:str", "tag:str", "param:str", NULL};
        if (!PyArg_ParseTupleAndKeywords(args, kwds, "sOsss", kwlist, &kernel, &operand_list, &compile_cmd, &tag,
                                         &param)) {
            return NULL;
        }
        operand_fast_seq = PySequence_Fast(operand_list, "`operand_list` should be a sequence.");
        if (operand_fast_seq == NULL) {
            return NULL;
        }
    }
    const Py_ssize_t nop = PySequence_Fast_GET_SIZE(operand_fast_seq);

    // Read and normalize all operands
    bhc_dtype types[nop];
    bhc_bool constant;
    void *operands[nop];
    normalize_cleanup_handle cleanup;
    cleanup.objs2free_count = 0;
    for (int i = 0; i < nop; ++i) {
        PyObject *op = PySequence_Fast_GET_ITEM(operand_fast_seq, i); // Borrowed reference and will not fail
        int err = normalize_operand(op, &types[i], &constant, &operands[i], &cleanup);
        if (err != -1) {
            if (constant) {
                PyErr_Format(PyExc_TypeError, "Scalars isn't supported.");
                err = -1;
            }
        }
        if (err == -1) {
            normalize_operand_cleanup(&cleanup);
            Py_DECREF(operand_fast_seq);
            if (PyErr_Occurred() != NULL) {
                return NULL;
            } else {
                Py_RETURN_NONE;
            }
        }
    }

    const char *ret = BhAPI_user_kernel(kernel, nop, operands, compile_cmd, tag, param);
    normalize_operand_cleanup(&cleanup);
    Py_DECREF(operand_fast_seq);
#if defined(NPY_PY3K)
    return PyUnicode_FromString(ret);
#else
    return PyString_FromString(ret);
#endif
}
