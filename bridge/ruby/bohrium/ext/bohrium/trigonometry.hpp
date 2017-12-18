#pragma once

using namespace std;


/**
    Cos on a Bohrium array.

    @param result The resulting array.
    @param selfObj The calling object.
    @param otherObj The other object.
*/
template <typename T>
inline void _cos(bhDataObj* result, bhDataObj* selfObj) {
    bhxx::BhArray<T> selfArray = unwrap<T>(selfObj);

    bhxx::BhArray<T>* bhary = new bhxx::BhArray<T>(selfArray.shape);
    result->ary = ((void*) bhary);

    bhxx::cos(*bhary, selfArray);
}

template <typename T>
inline void _cos_bang(bhDataObj* selfObj) {
    bhxx::BhArray<T> selfArray = unwrap<T>(selfObj);

    bhxx::cos(selfArray, selfArray);
}

/**
    Cos on a Bohrium array.

    @param self The calling object.
    @param other The other object.
    @returns The resulting array.
*/
VALUE bh_array_m_cos(VALUE self) {
    bhDataObj *selfObj;
    Data_Get_Struct(self, bhDataObj, selfObj);

    VALUE res = bh_array_alloc(RBASIC_CLASS(self));
    bhDataObj *result;
    Data_Get_Struct(res, bhDataObj, result);
    result->type = selfObj->type;

    switch (selfObj->type) {
        
        
        case T_FLOAT:
        
            _cos<float>(result, selfObj);
            break;
        
        default:
            rb_raise(rb_eRuntimeError, "Wrong type.");
    }

    return res;
}

VALUE bh_array_m_cos_bang(VALUE self, VALUE other) {
    bhDataObj *selfObj;
    Data_Get_Struct(self, bhDataObj, selfObj);

    switch (selfObj->type) {
        
        
        case T_FLOAT:
        
            _cos<float>(selfObj, selfObj);
            break;
        
        default:
            rb_raise(rb_eRuntimeError, "Wrong type.");
    }

    return self;
}

/**
    Sin on a Bohrium array.

    @param result The resulting array.
    @param selfObj The calling object.
    @param otherObj The other object.
*/
template <typename T>
inline void _sin(bhDataObj* result, bhDataObj* selfObj) {
    bhxx::BhArray<T> selfArray = unwrap<T>(selfObj);

    bhxx::BhArray<T>* bhary = new bhxx::BhArray<T>(selfArray.shape);
    result->ary = ((void*) bhary);

    bhxx::sin(*bhary, selfArray);
}

template <typename T>
inline void _sin_bang(bhDataObj* selfObj) {
    bhxx::BhArray<T> selfArray = unwrap<T>(selfObj);

    bhxx::sin(selfArray, selfArray);
}

/**
    Sin on a Bohrium array.

    @param self The calling object.
    @param other The other object.
    @returns The resulting array.
*/
VALUE bh_array_m_sin(VALUE self) {
    bhDataObj *selfObj;
    Data_Get_Struct(self, bhDataObj, selfObj);

    VALUE res = bh_array_alloc(RBASIC_CLASS(self));
    bhDataObj *result;
    Data_Get_Struct(res, bhDataObj, result);
    result->type = selfObj->type;

    switch (selfObj->type) {
        
        
        case T_FLOAT:
        
            _sin<float>(result, selfObj);
            break;
        
        default:
            rb_raise(rb_eRuntimeError, "Wrong type.");
    }

    return res;
}

VALUE bh_array_m_sin_bang(VALUE self, VALUE other) {
    bhDataObj *selfObj;
    Data_Get_Struct(self, bhDataObj, selfObj);

    switch (selfObj->type) {
        
        
        case T_FLOAT:
        
            _sin<float>(selfObj, selfObj);
            break;
        
        default:
            rb_raise(rb_eRuntimeError, "Wrong type.");
    }

    return self;
}

/**
    Tan on a Bohrium array.

    @param result The resulting array.
    @param selfObj The calling object.
    @param otherObj The other object.
*/
template <typename T>
inline void _tan(bhDataObj* result, bhDataObj* selfObj) {
    bhxx::BhArray<T> selfArray = unwrap<T>(selfObj);

    bhxx::BhArray<T>* bhary = new bhxx::BhArray<T>(selfArray.shape);
    result->ary = ((void*) bhary);

    bhxx::tan(*bhary, selfArray);
}

template <typename T>
inline void _tan_bang(bhDataObj* selfObj) {
    bhxx::BhArray<T> selfArray = unwrap<T>(selfObj);

    bhxx::tan(selfArray, selfArray);
}

/**
    Tan on a Bohrium array.

    @param self The calling object.
    @param other The other object.
    @returns The resulting array.
*/
VALUE bh_array_m_tan(VALUE self) {
    bhDataObj *selfObj;
    Data_Get_Struct(self, bhDataObj, selfObj);

    VALUE res = bh_array_alloc(RBASIC_CLASS(self));
    bhDataObj *result;
    Data_Get_Struct(res, bhDataObj, result);
    result->type = selfObj->type;

    switch (selfObj->type) {
        
        
        case T_FLOAT:
        
            _tan<float>(result, selfObj);
            break;
        
        default:
            rb_raise(rb_eRuntimeError, "Wrong type.");
    }

    return res;
}

VALUE bh_array_m_tan_bang(VALUE self, VALUE other) {
    bhDataObj *selfObj;
    Data_Get_Struct(self, bhDataObj, selfObj);

    switch (selfObj->type) {
        
        
        case T_FLOAT:
        
            _tan<float>(selfObj, selfObj);
            break;
        
        default:
            rb_raise(rb_eRuntimeError, "Wrong type.");
    }

    return self;
}

/**
    Cosh on a Bohrium array.

    @param result The resulting array.
    @param selfObj The calling object.
    @param otherObj The other object.
*/
template <typename T>
inline void _cosh(bhDataObj* result, bhDataObj* selfObj) {
    bhxx::BhArray<T> selfArray = unwrap<T>(selfObj);

    bhxx::BhArray<T>* bhary = new bhxx::BhArray<T>(selfArray.shape);
    result->ary = ((void*) bhary);

    bhxx::cosh(*bhary, selfArray);
}

template <typename T>
inline void _cosh_bang(bhDataObj* selfObj) {
    bhxx::BhArray<T> selfArray = unwrap<T>(selfObj);

    bhxx::cosh(selfArray, selfArray);
}

/**
    Cosh on a Bohrium array.

    @param self The calling object.
    @param other The other object.
    @returns The resulting array.
*/
VALUE bh_array_m_cosh(VALUE self) {
    bhDataObj *selfObj;
    Data_Get_Struct(self, bhDataObj, selfObj);

    VALUE res = bh_array_alloc(RBASIC_CLASS(self));
    bhDataObj *result;
    Data_Get_Struct(res, bhDataObj, result);
    result->type = selfObj->type;

    switch (selfObj->type) {
        
        
        case T_FLOAT:
        
            _cosh<float>(result, selfObj);
            break;
        
        default:
            rb_raise(rb_eRuntimeError, "Wrong type.");
    }

    return res;
}

VALUE bh_array_m_cosh_bang(VALUE self, VALUE other) {
    bhDataObj *selfObj;
    Data_Get_Struct(self, bhDataObj, selfObj);

    switch (selfObj->type) {
        
        
        case T_FLOAT:
        
            _cosh<float>(selfObj, selfObj);
            break;
        
        default:
            rb_raise(rb_eRuntimeError, "Wrong type.");
    }

    return self;
}

/**
    Sinh on a Bohrium array.

    @param result The resulting array.
    @param selfObj The calling object.
    @param otherObj The other object.
*/
template <typename T>
inline void _sinh(bhDataObj* result, bhDataObj* selfObj) {
    bhxx::BhArray<T> selfArray = unwrap<T>(selfObj);

    bhxx::BhArray<T>* bhary = new bhxx::BhArray<T>(selfArray.shape);
    result->ary = ((void*) bhary);

    bhxx::sinh(*bhary, selfArray);
}

template <typename T>
inline void _sinh_bang(bhDataObj* selfObj) {
    bhxx::BhArray<T> selfArray = unwrap<T>(selfObj);

    bhxx::sinh(selfArray, selfArray);
}

/**
    Sinh on a Bohrium array.

    @param self The calling object.
    @param other The other object.
    @returns The resulting array.
*/
VALUE bh_array_m_sinh(VALUE self) {
    bhDataObj *selfObj;
    Data_Get_Struct(self, bhDataObj, selfObj);

    VALUE res = bh_array_alloc(RBASIC_CLASS(self));
    bhDataObj *result;
    Data_Get_Struct(res, bhDataObj, result);
    result->type = selfObj->type;

    switch (selfObj->type) {
        
        
        case T_FLOAT:
        
            _sinh<float>(result, selfObj);
            break;
        
        default:
            rb_raise(rb_eRuntimeError, "Wrong type.");
    }

    return res;
}

VALUE bh_array_m_sinh_bang(VALUE self, VALUE other) {
    bhDataObj *selfObj;
    Data_Get_Struct(self, bhDataObj, selfObj);

    switch (selfObj->type) {
        
        
        case T_FLOAT:
        
            _sinh<float>(selfObj, selfObj);
            break;
        
        default:
            rb_raise(rb_eRuntimeError, "Wrong type.");
    }

    return self;
}

/**
    Tanh on a Bohrium array.

    @param result The resulting array.
    @param selfObj The calling object.
    @param otherObj The other object.
*/
template <typename T>
inline void _tanh(bhDataObj* result, bhDataObj* selfObj) {
    bhxx::BhArray<T> selfArray = unwrap<T>(selfObj);

    bhxx::BhArray<T>* bhary = new bhxx::BhArray<T>(selfArray.shape);
    result->ary = ((void*) bhary);

    bhxx::tanh(*bhary, selfArray);
}

template <typename T>
inline void _tanh_bang(bhDataObj* selfObj) {
    bhxx::BhArray<T> selfArray = unwrap<T>(selfObj);

    bhxx::tanh(selfArray, selfArray);
}

/**
    Tanh on a Bohrium array.

    @param self The calling object.
    @param other The other object.
    @returns The resulting array.
*/
VALUE bh_array_m_tanh(VALUE self) {
    bhDataObj *selfObj;
    Data_Get_Struct(self, bhDataObj, selfObj);

    VALUE res = bh_array_alloc(RBASIC_CLASS(self));
    bhDataObj *result;
    Data_Get_Struct(res, bhDataObj, result);
    result->type = selfObj->type;

    switch (selfObj->type) {
        
        
        case T_FLOAT:
        
            _tanh<float>(result, selfObj);
            break;
        
        default:
            rb_raise(rb_eRuntimeError, "Wrong type.");
    }

    return res;
}

VALUE bh_array_m_tanh_bang(VALUE self, VALUE other) {
    bhDataObj *selfObj;
    Data_Get_Struct(self, bhDataObj, selfObj);

    switch (selfObj->type) {
        
        
        case T_FLOAT:
        
            _tanh<float>(selfObj, selfObj);
            break;
        
        default:
            rb_raise(rb_eRuntimeError, "Wrong type.");
    }

    return self;
}

/**
    Arccos on a Bohrium array.

    @param result The resulting array.
    @param selfObj The calling object.
    @param otherObj The other object.
*/
template <typename T>
inline void _arccos(bhDataObj* result, bhDataObj* selfObj) {
    bhxx::BhArray<T> selfArray = unwrap<T>(selfObj);

    bhxx::BhArray<T>* bhary = new bhxx::BhArray<T>(selfArray.shape);
    result->ary = ((void*) bhary);

    bhxx::arccos(*bhary, selfArray);
}

template <typename T>
inline void _arccos_bang(bhDataObj* selfObj) {
    bhxx::BhArray<T> selfArray = unwrap<T>(selfObj);

    bhxx::arccos(selfArray, selfArray);
}

/**
    Arccos on a Bohrium array.

    @param self The calling object.
    @param other The other object.
    @returns The resulting array.
*/
VALUE bh_array_m_arccos(VALUE self) {
    bhDataObj *selfObj;
    Data_Get_Struct(self, bhDataObj, selfObj);

    VALUE res = bh_array_alloc(RBASIC_CLASS(self));
    bhDataObj *result;
    Data_Get_Struct(res, bhDataObj, result);
    result->type = selfObj->type;

    switch (selfObj->type) {
        
        
        case T_FLOAT:
        
            _arccos<float>(result, selfObj);
            break;
        
        default:
            rb_raise(rb_eRuntimeError, "Wrong type.");
    }

    return res;
}

VALUE bh_array_m_arccos_bang(VALUE self, VALUE other) {
    bhDataObj *selfObj;
    Data_Get_Struct(self, bhDataObj, selfObj);

    switch (selfObj->type) {
        
        
        case T_FLOAT:
        
            _arccos<float>(selfObj, selfObj);
            break;
        
        default:
            rb_raise(rb_eRuntimeError, "Wrong type.");
    }

    return self;
}

/**
    Arcsin on a Bohrium array.

    @param result The resulting array.
    @param selfObj The calling object.
    @param otherObj The other object.
*/
template <typename T>
inline void _arcsin(bhDataObj* result, bhDataObj* selfObj) {
    bhxx::BhArray<T> selfArray = unwrap<T>(selfObj);

    bhxx::BhArray<T>* bhary = new bhxx::BhArray<T>(selfArray.shape);
    result->ary = ((void*) bhary);

    bhxx::arcsin(*bhary, selfArray);
}

template <typename T>
inline void _arcsin_bang(bhDataObj* selfObj) {
    bhxx::BhArray<T> selfArray = unwrap<T>(selfObj);

    bhxx::arcsin(selfArray, selfArray);
}

/**
    Arcsin on a Bohrium array.

    @param self The calling object.
    @param other The other object.
    @returns The resulting array.
*/
VALUE bh_array_m_arcsin(VALUE self) {
    bhDataObj *selfObj;
    Data_Get_Struct(self, bhDataObj, selfObj);

    VALUE res = bh_array_alloc(RBASIC_CLASS(self));
    bhDataObj *result;
    Data_Get_Struct(res, bhDataObj, result);
    result->type = selfObj->type;

    switch (selfObj->type) {
        
        
        case T_FLOAT:
        
            _arcsin<float>(result, selfObj);
            break;
        
        default:
            rb_raise(rb_eRuntimeError, "Wrong type.");
    }

    return res;
}

VALUE bh_array_m_arcsin_bang(VALUE self, VALUE other) {
    bhDataObj *selfObj;
    Data_Get_Struct(self, bhDataObj, selfObj);

    switch (selfObj->type) {
        
        
        case T_FLOAT:
        
            _arcsin<float>(selfObj, selfObj);
            break;
        
        default:
            rb_raise(rb_eRuntimeError, "Wrong type.");
    }

    return self;
}

/**
    Arctan on a Bohrium array.

    @param result The resulting array.
    @param selfObj The calling object.
    @param otherObj The other object.
*/
template <typename T>
inline void _arctan(bhDataObj* result, bhDataObj* selfObj) {
    bhxx::BhArray<T> selfArray = unwrap<T>(selfObj);

    bhxx::BhArray<T>* bhary = new bhxx::BhArray<T>(selfArray.shape);
    result->ary = ((void*) bhary);

    bhxx::arctan(*bhary, selfArray);
}

template <typename T>
inline void _arctan_bang(bhDataObj* selfObj) {
    bhxx::BhArray<T> selfArray = unwrap<T>(selfObj);

    bhxx::arctan(selfArray, selfArray);
}

/**
    Arctan on a Bohrium array.

    @param self The calling object.
    @param other The other object.
    @returns The resulting array.
*/
VALUE bh_array_m_arctan(VALUE self) {
    bhDataObj *selfObj;
    Data_Get_Struct(self, bhDataObj, selfObj);

    VALUE res = bh_array_alloc(RBASIC_CLASS(self));
    bhDataObj *result;
    Data_Get_Struct(res, bhDataObj, result);
    result->type = selfObj->type;

    switch (selfObj->type) {
        
        
        case T_FLOAT:
        
            _arctan<float>(result, selfObj);
            break;
        
        default:
            rb_raise(rb_eRuntimeError, "Wrong type.");
    }

    return res;
}

VALUE bh_array_m_arctan_bang(VALUE self, VALUE other) {
    bhDataObj *selfObj;
    Data_Get_Struct(self, bhDataObj, selfObj);

    switch (selfObj->type) {
        
        
        case T_FLOAT:
        
            _arctan<float>(selfObj, selfObj);
            break;
        
        default:
            rb_raise(rb_eRuntimeError, "Wrong type.");
    }

    return self;
}

/**
    Arccosh on a Bohrium array.

    @param result The resulting array.
    @param selfObj The calling object.
    @param otherObj The other object.
*/
template <typename T>
inline void _arccosh(bhDataObj* result, bhDataObj* selfObj) {
    bhxx::BhArray<T> selfArray = unwrap<T>(selfObj);

    bhxx::BhArray<T>* bhary = new bhxx::BhArray<T>(selfArray.shape);
    result->ary = ((void*) bhary);

    bhxx::arccosh(*bhary, selfArray);
}

template <typename T>
inline void _arccosh_bang(bhDataObj* selfObj) {
    bhxx::BhArray<T> selfArray = unwrap<T>(selfObj);

    bhxx::arccosh(selfArray, selfArray);
}

/**
    Arccosh on a Bohrium array.

    @param self The calling object.
    @param other The other object.
    @returns The resulting array.
*/
VALUE bh_array_m_arccosh(VALUE self) {
    bhDataObj *selfObj;
    Data_Get_Struct(self, bhDataObj, selfObj);

    VALUE res = bh_array_alloc(RBASIC_CLASS(self));
    bhDataObj *result;
    Data_Get_Struct(res, bhDataObj, result);
    result->type = selfObj->type;

    switch (selfObj->type) {
        
        
        case T_FLOAT:
        
            _arccosh<float>(result, selfObj);
            break;
        
        default:
            rb_raise(rb_eRuntimeError, "Wrong type.");
    }

    return res;
}

VALUE bh_array_m_arccosh_bang(VALUE self, VALUE other) {
    bhDataObj *selfObj;
    Data_Get_Struct(self, bhDataObj, selfObj);

    switch (selfObj->type) {
        
        
        case T_FLOAT:
        
            _arccosh<float>(selfObj, selfObj);
            break;
        
        default:
            rb_raise(rb_eRuntimeError, "Wrong type.");
    }

    return self;
}

/**
    Arcsinh on a Bohrium array.

    @param result The resulting array.
    @param selfObj The calling object.
    @param otherObj The other object.
*/
template <typename T>
inline void _arcsinh(bhDataObj* result, bhDataObj* selfObj) {
    bhxx::BhArray<T> selfArray = unwrap<T>(selfObj);

    bhxx::BhArray<T>* bhary = new bhxx::BhArray<T>(selfArray.shape);
    result->ary = ((void*) bhary);

    bhxx::arcsinh(*bhary, selfArray);
}

template <typename T>
inline void _arcsinh_bang(bhDataObj* selfObj) {
    bhxx::BhArray<T> selfArray = unwrap<T>(selfObj);

    bhxx::arcsinh(selfArray, selfArray);
}

/**
    Arcsinh on a Bohrium array.

    @param self The calling object.
    @param other The other object.
    @returns The resulting array.
*/
VALUE bh_array_m_arcsinh(VALUE self) {
    bhDataObj *selfObj;
    Data_Get_Struct(self, bhDataObj, selfObj);

    VALUE res = bh_array_alloc(RBASIC_CLASS(self));
    bhDataObj *result;
    Data_Get_Struct(res, bhDataObj, result);
    result->type = selfObj->type;

    switch (selfObj->type) {
        
        
        case T_FLOAT:
        
            _arcsinh<float>(result, selfObj);
            break;
        
        default:
            rb_raise(rb_eRuntimeError, "Wrong type.");
    }

    return res;
}

VALUE bh_array_m_arcsinh_bang(VALUE self, VALUE other) {
    bhDataObj *selfObj;
    Data_Get_Struct(self, bhDataObj, selfObj);

    switch (selfObj->type) {
        
        
        case T_FLOAT:
        
            _arcsinh<float>(selfObj, selfObj);
            break;
        
        default:
            rb_raise(rb_eRuntimeError, "Wrong type.");
    }

    return self;
}

/**
    Arctanh on a Bohrium array.

    @param result The resulting array.
    @param selfObj The calling object.
    @param otherObj The other object.
*/
template <typename T>
inline void _arctanh(bhDataObj* result, bhDataObj* selfObj) {
    bhxx::BhArray<T> selfArray = unwrap<T>(selfObj);

    bhxx::BhArray<T>* bhary = new bhxx::BhArray<T>(selfArray.shape);
    result->ary = ((void*) bhary);

    bhxx::arctanh(*bhary, selfArray);
}

template <typename T>
inline void _arctanh_bang(bhDataObj* selfObj) {
    bhxx::BhArray<T> selfArray = unwrap<T>(selfObj);

    bhxx::arctanh(selfArray, selfArray);
}

/**
    Arctanh on a Bohrium array.

    @param self The calling object.
    @param other The other object.
    @returns The resulting array.
*/
VALUE bh_array_m_arctanh(VALUE self) {
    bhDataObj *selfObj;
    Data_Get_Struct(self, bhDataObj, selfObj);

    VALUE res = bh_array_alloc(RBASIC_CLASS(self));
    bhDataObj *result;
    Data_Get_Struct(res, bhDataObj, result);
    result->type = selfObj->type;

    switch (selfObj->type) {
        
        
        case T_FLOAT:
        
            _arctanh<float>(result, selfObj);
            break;
        
        default:
            rb_raise(rb_eRuntimeError, "Wrong type.");
    }

    return res;
}

VALUE bh_array_m_arctanh_bang(VALUE self, VALUE other) {
    bhDataObj *selfObj;
    Data_Get_Struct(self, bhDataObj, selfObj);

    switch (selfObj->type) {
        
        
        case T_FLOAT:
        
            _arctanh<float>(selfObj, selfObj);
            break;
        
        default:
            rb_raise(rb_eRuntimeError, "Wrong type.");
    }

    return self;
}

/**
    Exp on a Bohrium array.

    @param result The resulting array.
    @param selfObj The calling object.
    @param otherObj The other object.
*/
template <typename T>
inline void _exp(bhDataObj* result, bhDataObj* selfObj) {
    bhxx::BhArray<T> selfArray = unwrap<T>(selfObj);

    bhxx::BhArray<T>* bhary = new bhxx::BhArray<T>(selfArray.shape);
    result->ary = ((void*) bhary);

    bhxx::exp(*bhary, selfArray);
}

template <typename T>
inline void _exp_bang(bhDataObj* selfObj) {
    bhxx::BhArray<T> selfArray = unwrap<T>(selfObj);

    bhxx::exp(selfArray, selfArray);
}

/**
    Exp on a Bohrium array.

    @param self The calling object.
    @param other The other object.
    @returns The resulting array.
*/
VALUE bh_array_m_exp(VALUE self) {
    bhDataObj *selfObj;
    Data_Get_Struct(self, bhDataObj, selfObj);

    VALUE res = bh_array_alloc(RBASIC_CLASS(self));
    bhDataObj *result;
    Data_Get_Struct(res, bhDataObj, result);
    result->type = selfObj->type;

    switch (selfObj->type) {
        
        
        case T_FLOAT:
        
            _exp<float>(result, selfObj);
            break;
        
        default:
            rb_raise(rb_eRuntimeError, "Wrong type.");
    }

    return res;
}

VALUE bh_array_m_exp_bang(VALUE self, VALUE other) {
    bhDataObj *selfObj;
    Data_Get_Struct(self, bhDataObj, selfObj);

    switch (selfObj->type) {
        
        
        case T_FLOAT:
        
            _exp<float>(selfObj, selfObj);
            break;
        
        default:
            rb_raise(rb_eRuntimeError, "Wrong type.");
    }

    return self;
}

/**
    Exp2 on a Bohrium array.

    @param result The resulting array.
    @param selfObj The calling object.
    @param otherObj The other object.
*/
template <typename T>
inline void _exp2(bhDataObj* result, bhDataObj* selfObj) {
    bhxx::BhArray<T> selfArray = unwrap<T>(selfObj);

    bhxx::BhArray<T>* bhary = new bhxx::BhArray<T>(selfArray.shape);
    result->ary = ((void*) bhary);

    bhxx::exp2(*bhary, selfArray);
}

template <typename T>
inline void _exp2_bang(bhDataObj* selfObj) {
    bhxx::BhArray<T> selfArray = unwrap<T>(selfObj);

    bhxx::exp2(selfArray, selfArray);
}

/**
    Exp2 on a Bohrium array.

    @param self The calling object.
    @param other The other object.
    @returns The resulting array.
*/
VALUE bh_array_m_exp2(VALUE self) {
    bhDataObj *selfObj;
    Data_Get_Struct(self, bhDataObj, selfObj);

    VALUE res = bh_array_alloc(RBASIC_CLASS(self));
    bhDataObj *result;
    Data_Get_Struct(res, bhDataObj, result);
    result->type = selfObj->type;

    switch (selfObj->type) {
        
        
        case T_FLOAT:
        
            _exp2<float>(result, selfObj);
            break;
        
        default:
            rb_raise(rb_eRuntimeError, "Wrong type.");
    }

    return res;
}

VALUE bh_array_m_exp2_bang(VALUE self, VALUE other) {
    bhDataObj *selfObj;
    Data_Get_Struct(self, bhDataObj, selfObj);

    switch (selfObj->type) {
        
        
        case T_FLOAT:
        
            _exp2<float>(selfObj, selfObj);
            break;
        
        default:
            rb_raise(rb_eRuntimeError, "Wrong type.");
    }

    return self;
}

