#pragma once

using namespace std;


/**
    Add two Bohrium arrays of type T.

    @param result The resulting array.
    @param selfObj The calling object.
    @param otherObj The other object.
*/
template <typename T>
inline void _add(bhDataObj* result, bhDataObj* selfObj, bhDataObj* otherObj) {
    bhxx::BhArray<T> selfArray = unwrap<T>(selfObj);
    bhxx::BhArray<T> otherArray = unwrap<T>(otherObj);

    bhxx::BhArray<T>* bhary = new bhxx::BhArray<T>(selfArray.shape);
    result->ary = ((void*) bhary);

    bhxx::add(*bhary, selfArray, otherArray);
}

template <typename T>
inline void _add_bang(bhDataObj* selfObj, bhDataObj* otherObj) {
    bhxx::BhArray<T> selfArray = unwrap<T>(selfObj);
    bhxx::BhArray<T> otherArray = unwrap<T>(otherObj);

    bhxx::add(selfArray, selfArray, otherArray);
}

/**
    Add two Bohrium arrays.

    @param self The calling object.
    @param other The other object.
    @returns The resulting array.
*/
VALUE bh_array_m_add(VALUE self, VALUE other) {
    bhDataObj *selfObj, *otherObj;
    Data_Get_Struct(self, bhDataObj, selfObj);
    Data_Get_Struct(other, bhDataObj, otherObj);

    assert(selfObj->type == otherObj->type);

    VALUE res = bh_array_alloc(RBASIC_CLASS(self));
    bhDataObj *result;
    Data_Get_Struct(res, bhDataObj, result);
    result->type = selfObj->type;

    switch (selfObj->type) {
        
        
        case T_BIGNUM:
        
        case T_FIXNUM:
        
            _add<int64_t>(result, selfObj, otherObj);
            break;
        
        
        case T_FLOAT:
        
            _add<float>(result, selfObj, otherObj);
            break;
        
        
        case T_TRUE:
        
        case T_FALSE:
        
            _add<bool>(result, selfObj, otherObj);
            break;
        
        default:
            rb_raise(rb_eRuntimeError, "Wrong type.");
    }

    return res;
}

VALUE bh_array_m_add_bang(VALUE self, VALUE other) {
    bhDataObj *selfObj, *otherObj;
    Data_Get_Struct(self, bhDataObj, selfObj);
    Data_Get_Struct(other, bhDataObj, otherObj);

    assert(selfObj->type == otherObj->type);

    switch (selfObj->type) {
        
        
        case T_BIGNUM:
        
        case T_FIXNUM:
        
            _add<int64_t>(selfObj, selfObj, otherObj);
            break;
        
        
        case T_FLOAT:
        
            _add<float>(selfObj, selfObj, otherObj);
            break;
        
        
        case T_TRUE:
        
        case T_FALSE:
        
            _add<bool>(selfObj, selfObj, otherObj);
            break;
        
        default:
            rb_raise(rb_eRuntimeError, "Wrong type.");
    }

    return self;
}

/**
    Subtract two Bohrium arrays of type T.

    @param result The resulting array.
    @param selfObj The calling object.
    @param otherObj The other object.
*/
template <typename T>
inline void _subtract(bhDataObj* result, bhDataObj* selfObj, bhDataObj* otherObj) {
    bhxx::BhArray<T> selfArray = unwrap<T>(selfObj);
    bhxx::BhArray<T> otherArray = unwrap<T>(otherObj);

    bhxx::BhArray<T>* bhary = new bhxx::BhArray<T>(selfArray.shape);
    result->ary = ((void*) bhary);

    bhxx::subtract(*bhary, selfArray, otherArray);
}

template <typename T>
inline void _subtract_bang(bhDataObj* selfObj, bhDataObj* otherObj) {
    bhxx::BhArray<T> selfArray = unwrap<T>(selfObj);
    bhxx::BhArray<T> otherArray = unwrap<T>(otherObj);

    bhxx::subtract(selfArray, selfArray, otherArray);
}

/**
    Subtract two Bohrium arrays.

    @param self The calling object.
    @param other The other object.
    @returns The resulting array.
*/
VALUE bh_array_m_subtract(VALUE self, VALUE other) {
    bhDataObj *selfObj, *otherObj;
    Data_Get_Struct(self, bhDataObj, selfObj);
    Data_Get_Struct(other, bhDataObj, otherObj);

    assert(selfObj->type == otherObj->type);

    VALUE res = bh_array_alloc(RBASIC_CLASS(self));
    bhDataObj *result;
    Data_Get_Struct(res, bhDataObj, result);
    result->type = selfObj->type;

    switch (selfObj->type) {
        
        
        case T_BIGNUM:
        
        case T_FIXNUM:
        
            _subtract<int64_t>(result, selfObj, otherObj);
            break;
        
        
        case T_FLOAT:
        
            _subtract<float>(result, selfObj, otherObj);
            break;
        
        default:
            rb_raise(rb_eRuntimeError, "Wrong type.");
    }

    return res;
}

VALUE bh_array_m_subtract_bang(VALUE self, VALUE other) {
    bhDataObj *selfObj, *otherObj;
    Data_Get_Struct(self, bhDataObj, selfObj);
    Data_Get_Struct(other, bhDataObj, otherObj);

    assert(selfObj->type == otherObj->type);

    switch (selfObj->type) {
        
        
        case T_BIGNUM:
        
        case T_FIXNUM:
        
            _subtract<int64_t>(selfObj, selfObj, otherObj);
            break;
        
        
        case T_FLOAT:
        
            _subtract<float>(selfObj, selfObj, otherObj);
            break;
        
        default:
            rb_raise(rb_eRuntimeError, "Wrong type.");
    }

    return self;
}

/**
    Multiply two Bohrium arrays of type T.

    @param result The resulting array.
    @param selfObj The calling object.
    @param otherObj The other object.
*/
template <typename T>
inline void _multiply(bhDataObj* result, bhDataObj* selfObj, bhDataObj* otherObj) {
    bhxx::BhArray<T> selfArray = unwrap<T>(selfObj);
    bhxx::BhArray<T> otherArray = unwrap<T>(otherObj);

    bhxx::BhArray<T>* bhary = new bhxx::BhArray<T>(selfArray.shape);
    result->ary = ((void*) bhary);

    bhxx::multiply(*bhary, selfArray, otherArray);
}

template <typename T>
inline void _multiply_bang(bhDataObj* selfObj, bhDataObj* otherObj) {
    bhxx::BhArray<T> selfArray = unwrap<T>(selfObj);
    bhxx::BhArray<T> otherArray = unwrap<T>(otherObj);

    bhxx::multiply(selfArray, selfArray, otherArray);
}

/**
    Multiply two Bohrium arrays.

    @param self The calling object.
    @param other The other object.
    @returns The resulting array.
*/
VALUE bh_array_m_multiply(VALUE self, VALUE other) {
    bhDataObj *selfObj, *otherObj;
    Data_Get_Struct(self, bhDataObj, selfObj);
    Data_Get_Struct(other, bhDataObj, otherObj);

    assert(selfObj->type == otherObj->type);

    VALUE res = bh_array_alloc(RBASIC_CLASS(self));
    bhDataObj *result;
    Data_Get_Struct(res, bhDataObj, result);
    result->type = selfObj->type;

    switch (selfObj->type) {
        
        
        case T_BIGNUM:
        
        case T_FIXNUM:
        
            _multiply<int64_t>(result, selfObj, otherObj);
            break;
        
        
        case T_FLOAT:
        
            _multiply<float>(result, selfObj, otherObj);
            break;
        
        
        case T_TRUE:
        
        case T_FALSE:
        
            _multiply<bool>(result, selfObj, otherObj);
            break;
        
        default:
            rb_raise(rb_eRuntimeError, "Wrong type.");
    }

    return res;
}

VALUE bh_array_m_multiply_bang(VALUE self, VALUE other) {
    bhDataObj *selfObj, *otherObj;
    Data_Get_Struct(self, bhDataObj, selfObj);
    Data_Get_Struct(other, bhDataObj, otherObj);

    assert(selfObj->type == otherObj->type);

    switch (selfObj->type) {
        
        
        case T_BIGNUM:
        
        case T_FIXNUM:
        
            _multiply<int64_t>(selfObj, selfObj, otherObj);
            break;
        
        
        case T_FLOAT:
        
            _multiply<float>(selfObj, selfObj, otherObj);
            break;
        
        
        case T_TRUE:
        
        case T_FALSE:
        
            _multiply<bool>(selfObj, selfObj, otherObj);
            break;
        
        default:
            rb_raise(rb_eRuntimeError, "Wrong type.");
    }

    return self;
}

/**
    Divide two Bohrium arrays of type T.

    @param result The resulting array.
    @param selfObj The calling object.
    @param otherObj The other object.
*/
template <typename T>
inline void _divide(bhDataObj* result, bhDataObj* selfObj, bhDataObj* otherObj) {
    bhxx::BhArray<T> selfArray = unwrap<T>(selfObj);
    bhxx::BhArray<T> otherArray = unwrap<T>(otherObj);

    bhxx::BhArray<T>* bhary = new bhxx::BhArray<T>(selfArray.shape);
    result->ary = ((void*) bhary);

    bhxx::divide(*bhary, selfArray, otherArray);
}

template <typename T>
inline void _divide_bang(bhDataObj* selfObj, bhDataObj* otherObj) {
    bhxx::BhArray<T> selfArray = unwrap<T>(selfObj);
    bhxx::BhArray<T> otherArray = unwrap<T>(otherObj);

    bhxx::divide(selfArray, selfArray, otherArray);
}

/**
    Divide two Bohrium arrays.

    @param self The calling object.
    @param other The other object.
    @returns The resulting array.
*/
VALUE bh_array_m_divide(VALUE self, VALUE other) {
    bhDataObj *selfObj, *otherObj;
    Data_Get_Struct(self, bhDataObj, selfObj);
    Data_Get_Struct(other, bhDataObj, otherObj);

    assert(selfObj->type == otherObj->type);

    VALUE res = bh_array_alloc(RBASIC_CLASS(self));
    bhDataObj *result;
    Data_Get_Struct(res, bhDataObj, result);
    result->type = selfObj->type;

    switch (selfObj->type) {
        
        
        case T_BIGNUM:
        
        case T_FIXNUM:
        
            _divide<int64_t>(result, selfObj, otherObj);
            break;
        
        
        case T_FLOAT:
        
            _divide<float>(result, selfObj, otherObj);
            break;
        
        default:
            rb_raise(rb_eRuntimeError, "Wrong type.");
    }

    return res;
}

VALUE bh_array_m_divide_bang(VALUE self, VALUE other) {
    bhDataObj *selfObj, *otherObj;
    Data_Get_Struct(self, bhDataObj, selfObj);
    Data_Get_Struct(other, bhDataObj, otherObj);

    assert(selfObj->type == otherObj->type);

    switch (selfObj->type) {
        
        
        case T_BIGNUM:
        
        case T_FIXNUM:
        
            _divide<int64_t>(selfObj, selfObj, otherObj);
            break;
        
        
        case T_FLOAT:
        
            _divide<float>(selfObj, selfObj, otherObj);
            break;
        
        default:
            rb_raise(rb_eRuntimeError, "Wrong type.");
    }

    return self;
}

/**
    Power two Bohrium arrays of type T.

    @param result The resulting array.
    @param selfObj The calling object.
    @param otherObj The other object.
*/
template <typename T>
inline void _power(bhDataObj* result, bhDataObj* selfObj, bhDataObj* otherObj) {
    bhxx::BhArray<T> selfArray = unwrap<T>(selfObj);
    bhxx::BhArray<T> otherArray = unwrap<T>(otherObj);

    bhxx::BhArray<T>* bhary = new bhxx::BhArray<T>(selfArray.shape);
    result->ary = ((void*) bhary);

    bhxx::power(*bhary, selfArray, otherArray);
}

template <typename T>
inline void _power_bang(bhDataObj* selfObj, bhDataObj* otherObj) {
    bhxx::BhArray<T> selfArray = unwrap<T>(selfObj);
    bhxx::BhArray<T> otherArray = unwrap<T>(otherObj);

    bhxx::power(selfArray, selfArray, otherArray);
}

/**
    Power two Bohrium arrays.

    @param self The calling object.
    @param other The other object.
    @returns The resulting array.
*/
VALUE bh_array_m_power(VALUE self, VALUE other) {
    bhDataObj *selfObj, *otherObj;
    Data_Get_Struct(self, bhDataObj, selfObj);
    Data_Get_Struct(other, bhDataObj, otherObj);

    assert(selfObj->type == otherObj->type);

    VALUE res = bh_array_alloc(RBASIC_CLASS(self));
    bhDataObj *result;
    Data_Get_Struct(res, bhDataObj, result);
    result->type = selfObj->type;

    switch (selfObj->type) {
        
        
        case T_BIGNUM:
        
        case T_FIXNUM:
        
            _power<int64_t>(result, selfObj, otherObj);
            break;
        
        
        case T_FLOAT:
        
            _power<float>(result, selfObj, otherObj);
            break;
        
        default:
            rb_raise(rb_eRuntimeError, "Wrong type.");
    }

    return res;
}

VALUE bh_array_m_power_bang(VALUE self, VALUE other) {
    bhDataObj *selfObj, *otherObj;
    Data_Get_Struct(self, bhDataObj, selfObj);
    Data_Get_Struct(other, bhDataObj, otherObj);

    assert(selfObj->type == otherObj->type);

    switch (selfObj->type) {
        
        
        case T_BIGNUM:
        
        case T_FIXNUM:
        
            _power<int64_t>(selfObj, selfObj, otherObj);
            break;
        
        
        case T_FLOAT:
        
            _power<float>(selfObj, selfObj, otherObj);
            break;
        
        default:
            rb_raise(rb_eRuntimeError, "Wrong type.");
    }

    return self;
}

/**
    Maximum two Bohrium arrays of type T.

    @param result The resulting array.
    @param selfObj The calling object.
    @param otherObj The other object.
*/
template <typename T>
inline void _maximum(bhDataObj* result, bhDataObj* selfObj, bhDataObj* otherObj) {
    bhxx::BhArray<T> selfArray = unwrap<T>(selfObj);
    bhxx::BhArray<T> otherArray = unwrap<T>(otherObj);

    bhxx::BhArray<T>* bhary = new bhxx::BhArray<T>(selfArray.shape);
    result->ary = ((void*) bhary);

    bhxx::maximum(*bhary, selfArray, otherArray);
}

template <typename T>
inline void _maximum_bang(bhDataObj* selfObj, bhDataObj* otherObj) {
    bhxx::BhArray<T> selfArray = unwrap<T>(selfObj);
    bhxx::BhArray<T> otherArray = unwrap<T>(otherObj);

    bhxx::maximum(selfArray, selfArray, otherArray);
}

/**
    Maximum two Bohrium arrays.

    @param self The calling object.
    @param other The other object.
    @returns The resulting array.
*/
VALUE bh_array_m_maximum(VALUE self, VALUE other) {
    bhDataObj *selfObj, *otherObj;
    Data_Get_Struct(self, bhDataObj, selfObj);
    Data_Get_Struct(other, bhDataObj, otherObj);

    assert(selfObj->type == otherObj->type);

    VALUE res = bh_array_alloc(RBASIC_CLASS(self));
    bhDataObj *result;
    Data_Get_Struct(res, bhDataObj, result);
    result->type = selfObj->type;

    switch (selfObj->type) {
        
        
        case T_BIGNUM:
        
        case T_FIXNUM:
        
            _maximum<int64_t>(result, selfObj, otherObj);
            break;
        
        
        case T_FLOAT:
        
            _maximum<float>(result, selfObj, otherObj);
            break;
        
        default:
            rb_raise(rb_eRuntimeError, "Wrong type.");
    }

    return res;
}

VALUE bh_array_m_maximum_bang(VALUE self, VALUE other) {
    bhDataObj *selfObj, *otherObj;
    Data_Get_Struct(self, bhDataObj, selfObj);
    Data_Get_Struct(other, bhDataObj, otherObj);

    assert(selfObj->type == otherObj->type);

    switch (selfObj->type) {
        
        
        case T_BIGNUM:
        
        case T_FIXNUM:
        
            _maximum<int64_t>(selfObj, selfObj, otherObj);
            break;
        
        
        case T_FLOAT:
        
            _maximum<float>(selfObj, selfObj, otherObj);
            break;
        
        default:
            rb_raise(rb_eRuntimeError, "Wrong type.");
    }

    return self;
}

/**
    Minimum two Bohrium arrays of type T.

    @param result The resulting array.
    @param selfObj The calling object.
    @param otherObj The other object.
*/
template <typename T>
inline void _minimum(bhDataObj* result, bhDataObj* selfObj, bhDataObj* otherObj) {
    bhxx::BhArray<T> selfArray = unwrap<T>(selfObj);
    bhxx::BhArray<T> otherArray = unwrap<T>(otherObj);

    bhxx::BhArray<T>* bhary = new bhxx::BhArray<T>(selfArray.shape);
    result->ary = ((void*) bhary);

    bhxx::minimum(*bhary, selfArray, otherArray);
}

template <typename T>
inline void _minimum_bang(bhDataObj* selfObj, bhDataObj* otherObj) {
    bhxx::BhArray<T> selfArray = unwrap<T>(selfObj);
    bhxx::BhArray<T> otherArray = unwrap<T>(otherObj);

    bhxx::minimum(selfArray, selfArray, otherArray);
}

/**
    Minimum two Bohrium arrays.

    @param self The calling object.
    @param other The other object.
    @returns The resulting array.
*/
VALUE bh_array_m_minimum(VALUE self, VALUE other) {
    bhDataObj *selfObj, *otherObj;
    Data_Get_Struct(self, bhDataObj, selfObj);
    Data_Get_Struct(other, bhDataObj, otherObj);

    assert(selfObj->type == otherObj->type);

    VALUE res = bh_array_alloc(RBASIC_CLASS(self));
    bhDataObj *result;
    Data_Get_Struct(res, bhDataObj, result);
    result->type = selfObj->type;

    switch (selfObj->type) {
        
        
        case T_BIGNUM:
        
        case T_FIXNUM:
        
            _minimum<int64_t>(result, selfObj, otherObj);
            break;
        
        
        case T_FLOAT:
        
            _minimum<float>(result, selfObj, otherObj);
            break;
        
        default:
            rb_raise(rb_eRuntimeError, "Wrong type.");
    }

    return res;
}

VALUE bh_array_m_minimum_bang(VALUE self, VALUE other) {
    bhDataObj *selfObj, *otherObj;
    Data_Get_Struct(self, bhDataObj, selfObj);
    Data_Get_Struct(other, bhDataObj, otherObj);

    assert(selfObj->type == otherObj->type);

    switch (selfObj->type) {
        
        
        case T_BIGNUM:
        
        case T_FIXNUM:
        
            _minimum<int64_t>(selfObj, selfObj, otherObj);
            break;
        
        
        case T_FLOAT:
        
            _minimum<float>(selfObj, selfObj, otherObj);
            break;
        
        default:
            rb_raise(rb_eRuntimeError, "Wrong type.");
    }

    return self;
}

