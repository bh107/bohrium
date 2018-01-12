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

//NB: 'bohrium' is declared in _bhmodule.c

static PyObject *
array_add(PyObject *m1, PyObject *m2) {
    return PyObject_CallMethod(bohrium, "add", "OO", m1, m2);
}

static PyObject *
array_subtract(PyObject *m1, PyObject *m2) {
    return PyObject_CallMethod(bohrium, "subtract", "OO", m1, m2);
}

static PyObject *
array_multiply(PyObject *m1, PyObject *m2) {
    return PyObject_CallMethod(bohrium, "multiply", "OO", m1, m2);
}


#if !defined(NPY_PY3K)
static PyObject *
array_divide(PyObject *m1, PyObject *m2) {
    return PyObject_CallMethod(bohrium, "divide", "OO", m1, m2);
}
#endif

static PyObject *
array_remainder(PyObject *m1, PyObject *m2) {
    return PyObject_CallMethod(bohrium, "mod", "OO", m1, m2);
}

static PyObject *
array_power(PyObject *m1, PyObject *m2, PyObject *modulo) {
    return PyObject_CallMethod(bohrium, "power", "OO", m1, m2);
}

static PyObject *
array_negative(PyObject *m1) {
    return PyObject_CallMethod(bohrium, "negative", "O", m1);
}

static PyObject *
array_absolute(PyObject *m1) {
    return PyObject_CallMethod(bohrium, "absolute", "O", m1);
}

static PyObject *
array_invert(PyObject *m1) {
    return PyObject_CallMethod(bohrium, "invert", "O", m1);
}

static PyObject *
array_left_shift(PyObject *m1, PyObject *m2) {
    return PyObject_CallMethod(bohrium, "left_shift", "OO", m1, m2);
}

static PyObject *
array_right_shift(PyObject *m1, PyObject *m2) {
    return PyObject_CallMethod(bohrium, "right_shift", "OO", m1, m2);
}

static PyObject *
array_bitwise_and(PyObject *m1, PyObject *m2) {
    return PyObject_CallMethod(bohrium, "bitwise_and", "OO", m1, m2);
}

static PyObject *
array_bitwise_or(PyObject *m1, PyObject *m2) {
    return PyObject_CallMethod(bohrium, "bitwise_or", "OO", m1, m2);
}

static PyObject *
array_bitwise_xor(PyObject *m1, PyObject *m2) {
    return PyObject_CallMethod(bohrium, "bitwise_xor", "OO", m1, m2);
}

static PyObject *
array_inplace_add(PyObject *m1, PyObject *m2) {
    return PyObject_CallMethod(bohrium, "add", "OOO", m1, m2, m1);
}

static PyObject *
array_inplace_subtract(PyObject *m1, PyObject *m2) {
    return PyObject_CallMethod(bohrium, "subtract", "OOO", m1, m2, m1);
}

static PyObject *
array_inplace_multiply(PyObject *m1, PyObject *m2) {
    return PyObject_CallMethod(bohrium, "multiply", "OOO", m1, m2, m1);
}

#if !defined(NPY_PY3K)
static PyObject *
array_inplace_divide(PyObject *m1, PyObject *m2) {
    return PyObject_CallMethod(bohrium, "divide", "OOO", m1, m2, m1);
}
#endif

static PyObject *
array_inplace_remainder(PyObject *m1, PyObject *m2) {
    return PyObject_CallMethod(bohrium, "remainder", "OOO", m1, m2, m1);
}

static PyObject *
array_inplace_left_shift(PyObject *m1, PyObject *m2) {
    return PyObject_CallMethod(bohrium, "left_shift", "OOO", m1, m2, m1);
}

static PyObject *
array_inplace_right_shift(PyObject *m1, PyObject *m2) {
    return PyObject_CallMethod(bohrium, "right_shift", "OOO", m1, m2, m1);
}

static PyObject *
array_inplace_bitwise_and(PyObject *m1, PyObject *m2) {
    return PyObject_CallMethod(bohrium, "bitwise_and", "OOO", m1, m2, m1);
}

static PyObject *
array_inplace_bitwise_or(PyObject *m1, PyObject *m2) {
    return PyObject_CallMethod(bohrium, "bitwise_or", "OOO", m1, m2, m1);
}

static PyObject *
array_inplace_bitwise_xor(PyObject *m1, PyObject *m2) {
    return PyObject_CallMethod(bohrium, "bitwise_xor", "OOO", m1, m2, m1);
}

static PyObject *
array_floor_divide(PyObject *m1, PyObject *m2) {
    return PyObject_CallMethod(bohrium, "floor_divide", "OO", m1, m2);
}

static PyObject *
array_true_divide(PyObject *m1, PyObject *m2) {
    return PyObject_CallMethod(bohrium, "true_divide", "OO", m1, m2);
}

static PyObject *
array_inplace_floor_divide(PyObject *m1, PyObject *m2) {
    return PyObject_CallMethod(bohrium, "floor_divide", "OOO", m1, m2, m1);
}

static PyObject *
array_inplace_power(PyObject *m1, PyObject *m2, PyObject *modulo) {
    return PyObject_CallMethod(bohrium, "power", "OOO", m1, m2, m1);
}

static PyObject *
array_inplace_true_divide(PyObject *m1, PyObject *m2) {
    return PyObject_CallMethod(bohrium, "true_divide", "OOO", m1, m2, m1);
}

static PyObject *
array_divmod(PyObject *op1, PyObject *op2) {
    PyObject *divp, *modp, *result;

    divp = array_floor_divide(op1, op2);
    if (divp == NULL) {
        return NULL;
    }
    else if(divp == Py_NotImplemented) {
        return divp;
    }
    modp = array_remainder(op1, op2);
    if (modp == NULL) {
        Py_DECREF(divp);
        return NULL;
    }
    else if(modp == Py_NotImplemented) {
        Py_DECREF(divp);
        return modp;
    }
    result = Py_BuildValue("OO", divp, modp);
    Py_DECREF(divp);
    Py_DECREF(modp);
    return result;
}

static int
array_nonzero(PyArrayObject *mp) {
    npy_intp n;
    int ret;

    n = PyArray_SIZE(mp);
    if (n == 1) {
        PyArrayObject *np_ary = (PyArrayObject*) BhArray_copy2numpy((PyObject*)mp, NULL);
        if (np_ary == NULL) {
            return -1;
        }
        ret = PyArray_DESCR(np_ary)->f->nonzero(PyArray_DATA(np_ary), np_ary);
        Py_DECREF(np_ary);
        return ret;
    }
    else if (n == 0) {
        return 0;
    }
    else {
        PyErr_SetString(PyExc_ValueError,
                        "The truth value of an array " \
                        "with more than one element is ambiguous. " \
                        "Use a.any() or a.all()");
        return -1;
    }
}

static PyObject *
array_float(PyArrayObject *v) {
    if (PyArray_SIZE(v) != 1) {
        PyErr_SetString(PyExc_TypeError, "only length-1 arrays can be converted to Python scalars");
        return NULL;
    }

    PyObject *pv, *pv2;
    PyArrayObject *np_ary = (PyArrayObject*) BhArray_copy2numpy((PyObject*)v, NULL);
    if (np_ary == NULL) {
        return NULL;
    }
    pv = PyArray_DESCR(np_ary)->f->getitem(PyArray_DATA(np_ary), np_ary);

    if (pv == NULL) {
        return NULL;
    }

    if (Py_TYPE(pv)->tp_as_number == 0) {
        PyErr_SetString(PyExc_TypeError, "cannot convert to a float; scalar object is not a number");
        Py_DECREF(pv);
        return NULL;
    }

    if (Py_TYPE(pv)->tp_as_number->nb_float == 0) {
        PyErr_SetString(PyExc_TypeError, "don't know how to convert scalar number to float");
        Py_DECREF(pv);
        return NULL;
    }

    /*
     * If we still got an array which can hold references, stop
     * because it could point back at 'v'.
     */
    if (PyArray_Check(pv) && PyDataType_REFCHK(PyArray_DESCR((PyArrayObject *)pv))) {
        PyErr_SetString(PyExc_TypeError, "object array may be self-referencing");
        Py_DECREF(pv);
        return NULL;
    }

    pv2 = Py_TYPE(pv)->tp_as_number->nb_float(pv);
    Py_DECREF(pv);
    return pv2;
}

NPY_NO_EXPORT PyObject *
array_int(PyArrayObject *v) {
    if (PyArray_SIZE(v) != 1) {
        PyErr_SetString(PyExc_TypeError, "only length-1 arrays can be converted to Python scalars");
        return NULL;
    }

    PyObject *pv, *pv2;
    PyArrayObject *np_ary = (PyArrayObject*) BhArray_copy2numpy((PyObject*)v, NULL);
    if (np_ary == NULL) {
        return NULL;
    }
    pv = PyArray_DESCR(np_ary)->f->getitem(PyArray_DATA(np_ary), np_ary);

    if (pv == NULL) {
        return NULL;
    }

    if (Py_TYPE(pv)->tp_as_number == 0) {
        PyErr_SetString(PyExc_TypeError, "cannot convert to an int; scalar object is not a number");
        Py_DECREF(pv);
        return NULL;
    }

    if (Py_TYPE(pv)->tp_as_number->nb_int == 0) {
        PyErr_SetString(PyExc_TypeError, "don't know how to convert scalar number to int");
        Py_DECREF(pv);
        return NULL;
    }

    /*
     * If we still got an array which can hold references, stop
     * because it could point back at 'v'.
     */
    if (PyArray_Check(pv) && PyDataType_REFCHK(PyArray_DESCR((PyArrayObject *)pv))) {
        PyErr_SetString(PyExc_TypeError, "object array may be self-referencing");
        Py_DECREF(pv);
        return NULL;
    }

    pv2 = Py_TYPE(pv)->tp_as_number->nb_int(pv);
    Py_DECREF(pv);
    return pv2;
}

#if !defined(NPY_PY3K)
static PyObject *
array_long(PyArrayObject *v) {
    if (PyArray_SIZE(v) != 1) {
        PyErr_SetString(PyExc_TypeError, "only length-1 arrays can be converted to Python scalars");
        return NULL;
    }

    PyObject *pv, *pv2;
    PyArrayObject *np_ary = (PyArrayObject*) BhArray_copy2numpy((PyObject*)v, NULL);
    if (np_ary == NULL) {
        return NULL;
    }
    pv = PyArray_DESCR(np_ary)->f->getitem(PyArray_DATA(np_ary), np_ary);

    if (pv == NULL) {
        return NULL;
    }

    if (Py_TYPE(pv)->tp_as_number == 0) {
        PyErr_SetString(PyExc_TypeError, "cannot convert to a long; scalar object is not a number");
        Py_DECREF(pv);
        return NULL;
    }

    if (Py_TYPE(pv)->tp_as_number->nb_long == 0) {
        PyErr_SetString(PyExc_TypeError, "don't know how to convert scalar number to long");
        Py_DECREF(pv);
        return NULL;
    }

    /*
     * If we still got an array which can hold references, stop
     * because it could point back at 'v'.
     */
    if (PyArray_Check(pv) && PyDataType_REFCHK(PyArray_DESCR((PyArrayObject *)pv))) {
        PyErr_SetString(PyExc_TypeError, "object array may be self-referencing");
        Py_DECREF(pv);
        return NULL;
    }

    pv2 = Py_TYPE(pv)->tp_as_number->nb_long(pv);
    Py_DECREF(pv);
    return pv2;
}

static PyObject *
array_oct(PyArrayObject *v) {
    if (PyArray_SIZE(v) != 1) {
        PyErr_SetString(PyExc_TypeError, "only length-1 arrays can be converted to Python scalars");
        return NULL;
    }

    PyObject *pv, *pv2;
    PyArrayObject *np_ary = (PyArrayObject*) BhArray_copy2numpy((PyObject*)v, NULL);
    if (np_ary == NULL) {
        return NULL;
    }
    pv = PyArray_DESCR(np_ary)->f->getitem(PyArray_DATA(np_ary), np_ary);

    if (pv == NULL) {
        return NULL;
    }

    if (Py_TYPE(pv)->tp_as_number == 0) {
        PyErr_SetString(PyExc_TypeError, "cannot convert to oct; scalar object is not a number");
        Py_DECREF(pv);
        return NULL;
    }

    if (Py_TYPE(pv)->tp_as_number->nb_oct == 0) {
        PyErr_SetString(PyExc_TypeError, "don't know how to convert scalar number to oct");
        Py_DECREF(pv);
        return NULL;
    }

    /*
     * If we still got an array which can hold references, stop
     * because it could point back at 'v'.
     */
    if (PyArray_Check(pv) && PyDataType_REFCHK(PyArray_DESCR((PyArrayObject *)pv))) {
        PyErr_SetString(PyExc_TypeError, "object array may be self-referencing");
        Py_DECREF(pv);
        return NULL;
    }

    pv2 = Py_TYPE(pv)->tp_as_number->nb_oct(pv);
    Py_DECREF(pv);
    return pv2;
}

static PyObject *
array_hex(PyArrayObject *v) {
    if (PyArray_SIZE(v) != 1) {
        PyErr_SetString(PyExc_TypeError, "only length-1 arrays can be converted to Python scalars");
        return NULL;
    }

    PyObject *pv, *pv2;
    PyArrayObject *np_ary = (PyArrayObject*) BhArray_copy2numpy((PyObject*)v, NULL);
    if (np_ary == NULL) {
        return NULL;
    }
    pv = PyArray_DESCR(np_ary)->f->getitem(PyArray_DATA(np_ary), np_ary);

    if (pv == NULL) {
        return NULL;
    }

    if (Py_TYPE(pv)->tp_as_number == 0) {
        PyErr_SetString(PyExc_TypeError, "cannot convert to hex; scalar object is not a number");
        Py_DECREF(pv);
        return NULL;
    }

    if (Py_TYPE(pv)->tp_as_number->nb_hex == 0) {
        PyErr_SetString(PyExc_TypeError, "don't know how to convert scalar number to hex");
        Py_DECREF(pv);
        return NULL;
    }

    /*
     * If we still got an array which can hold references, stop
     * because it could point back at 'v'.
     */
    if (PyArray_Check(pv) && PyDataType_REFCHK(PyArray_DESCR((PyArrayObject *)pv))) {
        PyErr_SetString(PyExc_TypeError, "object array may be self-referencing");
        Py_DECREF(pv);
        return NULL;
    }

    pv2 = Py_TYPE(pv)->tp_as_number->nb_hex(pv);
    Py_DECREF(pv);
    return pv2;
}
#endif

static PyObject *
array_positive(PyArrayObject *v) {
    PyErr_SetString(PyExc_TypeError, "to positive is not implemented");
    return NULL;
}

PyNumberMethods array_as_number = {
    (binaryfunc)array_add,                      /*nb_add*/
    (binaryfunc)array_subtract,                 /*nb_subtract*/
    (binaryfunc)array_multiply,                 /*nb_multiply*/
#if !defined(NPY_PY3K)
    (binaryfunc)array_divide,                   /*nb_divide*/
#endif
    (binaryfunc)array_remainder,                /*nb_remainder*/
    (binaryfunc)array_divmod,                   /*nb_divmod*/
    (ternaryfunc)array_power,                   /*nb_power*/
    (unaryfunc)array_negative,                  /*nb_neg*/
    (unaryfunc)array_positive,                  /*nb_pos*/
    (unaryfunc)array_absolute,                  /*array_abs,*/
    (inquiry)array_nonzero,                     /*nb_nonzero*/
    (unaryfunc)array_invert,                    /*nb_invert*/
    (binaryfunc)array_left_shift,               /*nb_lshift*/
    (binaryfunc)array_right_shift,              /*nb_rshift*/
    (binaryfunc)array_bitwise_and,              /*nb_and*/
    (binaryfunc)array_bitwise_xor,              /*nb_xor*/
    (binaryfunc)array_bitwise_or,               /*nb_or*/
#if !defined(NPY_PY3K)
    0,                                          /*nb_coerce*/
#endif
    (unaryfunc)array_int,                       /*nb_int*/
#if defined(NPY_PY3K)
    0,                                          /*nb_reserved*/
#else
    (unaryfunc)array_long,                      /*nb_long*/
#endif
    (unaryfunc)array_float,                     /*nb_float*/
#if !defined(NPY_PY3K)
    (unaryfunc)array_oct,                       /*nb_oct*/
    (unaryfunc)array_hex,                       /*nb_hex*/
#endif
    (binaryfunc)array_inplace_add,              /*inplace_add*/
    (binaryfunc)array_inplace_subtract,         /*inplace_subtract*/
    (binaryfunc)array_inplace_multiply,         /*inplace_multiply*/
#if !defined(NPY_PY3K)
    (binaryfunc)array_inplace_divide,           /*inplace_divide*/
#endif
    (binaryfunc)array_inplace_remainder,        /*inplace_remainder*/
    (ternaryfunc)array_inplace_power,           /*inplace_power*/
    (binaryfunc)array_inplace_left_shift,       /*inplace_lshift*/
    (binaryfunc)array_inplace_right_shift,      /*inplace_rshift*/
    (binaryfunc)array_inplace_bitwise_and,      /*inplace_and*/
    (binaryfunc)array_inplace_bitwise_xor,      /*inplace_xor*/
    (binaryfunc)array_inplace_bitwise_or,       /*inplace_or*/

    (binaryfunc)array_floor_divide,             /*nb_floor_divide*/
    (binaryfunc)array_true_divide,              /*nb_true_divide*/
    (binaryfunc)array_inplace_floor_divide,     /*nb_inplace_floor_divide*/
    (binaryfunc)array_inplace_true_divide,      /*nb_inplace_true_divide*/
    (unaryfunc)0,                               /*nb_index */
};

static PyObject *
array_richcompare(PyArrayObject *self, PyObject *other, int cmp_op) {
    switch (cmp_op) {
    case Py_LT:
        return PyObject_CallMethod(bohrium, "less", "OO", self, other);
    case Py_LE:
        return PyObject_CallMethod(bohrium, "less_equal", "OO", self, other);
    case Py_EQ:
        return PyObject_CallMethod(bohrium, "equal", "OO", self, other);
    case Py_NE:
        return PyObject_CallMethod(bohrium, "not_equal", "OO", self, other);
    case Py_GT:
        return PyObject_CallMethod(bohrium, "greater", "OO", self, other);
    case Py_GE:
        return PyObject_CallMethod(bohrium, "greater_equal", "OO", self, other);
    default:
        return NULL;
    }
}
