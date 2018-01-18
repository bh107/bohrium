#pragma once

using namespace std;



/**
    Add_reduce a Bohrium array of type T.

    @param result The resulting array.
    @param selfObj The calling object.
    @param arg The constant argument.
*/
template <typename T>
inline void _add_reduce(bhDataObj<T>* result, bhDataObj<T>* selfObj, size_t arg) {
    bhxx::BhArray<T> selfArray = selfObj->bhary;
    // Cannot choose a dimension larger than the array shape size.
    assert(selfArray.shape.size() > arg);

    // FIXME: We assume we only ever have two dimensions
    bhxx::Shape _shape;
    size_t self_size = selfArray.shape.size();
    if (self_size == 1) {
        _shape = {1};
    } else {
        if (selfArray.shape[self_size - arg - 1] == 1) {
            _shape = {selfArray.shape[arg]};
        } else {
            _shape = {selfArray.shape[self_size - arg - 1]};
        }
    }

    bhxx::BhArray<T> bhary = *(new bhxx::BhArray<T>(_shape));

    bhxx::add_reduce(bhary, selfArray, arg);

    result->bhary = bhary;
    result->type = selfObj->type;
}

/**
    Add_reduce a Bohrium array.

    @param self The calling object.
    @param arg The constant argument.
    @returns The resulting array.
*/
VALUE bh_array_m_add_reduce(VALUE self, VALUE arg) {
    UNPACK(int64_t, tmpObj);

    VALUE res = bh_array_alloc(RBASIC_CLASS(self));

    switch (tmpObj->type) {
        
            
            case T_FIXNUM: {
                UNPACK(int64_t, selfObj);
                UNPACK_(int64_t, result, res);
                _add_reduce<int64_t>(result, selfObj, NUM2INT(arg));
                break;
            }
            
            case T_BIGNUM: {
                UNPACK(int64_t, selfObj);
                UNPACK_(int64_t, result, res);
                _add_reduce<int64_t>(result, selfObj, NUM2INT(arg));
                break;
            }
            
        
        default:
            rb_raise(rb_eRuntimeError, "Wrong type for 'add_reduce'.");
    }

    return res;
}


/**
    Multiply_reduce a Bohrium array of type T.

    @param result The resulting array.
    @param selfObj The calling object.
    @param arg The constant argument.
*/
template <typename T>
inline void _multiply_reduce(bhDataObj<T>* result, bhDataObj<T>* selfObj, size_t arg) {
    bhxx::BhArray<T> selfArray = selfObj->bhary;
    // Cannot choose a dimension larger than the array shape size.
    assert(selfArray.shape.size() > arg);

    // FIXME: We assume we only ever have two dimensions
    bhxx::Shape _shape;
    size_t self_size = selfArray.shape.size();
    if (self_size == 1) {
        _shape = {1};
    } else {
        if (selfArray.shape[self_size - arg - 1] == 1) {
            _shape = {selfArray.shape[arg]};
        } else {
            _shape = {selfArray.shape[self_size - arg - 1]};
        }
    }

    bhxx::BhArray<T> bhary = *(new bhxx::BhArray<T>(_shape));

    bhxx::multiply_reduce(bhary, selfArray, arg);

    result->bhary = bhary;
    result->type = selfObj->type;
}

/**
    Multiply_reduce a Bohrium array.

    @param self The calling object.
    @param arg The constant argument.
    @returns The resulting array.
*/
VALUE bh_array_m_multiply_reduce(VALUE self, VALUE arg) {
    UNPACK(int64_t, tmpObj);

    VALUE res = bh_array_alloc(RBASIC_CLASS(self));

    switch (tmpObj->type) {
        
            
            case T_FIXNUM: {
                UNPACK(int64_t, selfObj);
                UNPACK_(int64_t, result, res);
                _multiply_reduce<int64_t>(result, selfObj, NUM2INT(arg));
                break;
            }
            
            case T_BIGNUM: {
                UNPACK(int64_t, selfObj);
                UNPACK_(int64_t, result, res);
                _multiply_reduce<int64_t>(result, selfObj, NUM2INT(arg));
                break;
            }
            
        
        default:
            rb_raise(rb_eRuntimeError, "Wrong type for 'multiply_reduce'.");
    }

    return res;
}


/**
    Minimum_reduce a Bohrium array of type T.

    @param result The resulting array.
    @param selfObj The calling object.
    @param arg The constant argument.
*/
template <typename T>
inline void _minimum_reduce(bhDataObj<T>* result, bhDataObj<T>* selfObj, size_t arg) {
    bhxx::BhArray<T> selfArray = selfObj->bhary;
    // Cannot choose a dimension larger than the array shape size.
    assert(selfArray.shape.size() > arg);

    // FIXME: We assume we only ever have two dimensions
    bhxx::Shape _shape;
    size_t self_size = selfArray.shape.size();
    if (self_size == 1) {
        _shape = {1};
    } else {
        if (selfArray.shape[self_size - arg - 1] == 1) {
            _shape = {selfArray.shape[arg]};
        } else {
            _shape = {selfArray.shape[self_size - arg - 1]};
        }
    }

    bhxx::BhArray<T> bhary = *(new bhxx::BhArray<T>(_shape));

    bhxx::minimum_reduce(bhary, selfArray, arg);

    result->bhary = bhary;
    result->type = selfObj->type;
}

/**
    Minimum_reduce a Bohrium array.

    @param self The calling object.
    @param arg The constant argument.
    @returns The resulting array.
*/
VALUE bh_array_m_minimum_reduce(VALUE self, VALUE arg) {
    UNPACK(int64_t, tmpObj);

    VALUE res = bh_array_alloc(RBASIC_CLASS(self));

    switch (tmpObj->type) {
        
            
            case T_FIXNUM: {
                UNPACK(int64_t, selfObj);
                UNPACK_(int64_t, result, res);
                _minimum_reduce<int64_t>(result, selfObj, NUM2INT(arg));
                break;
            }
            
            case T_BIGNUM: {
                UNPACK(int64_t, selfObj);
                UNPACK_(int64_t, result, res);
                _minimum_reduce<int64_t>(result, selfObj, NUM2INT(arg));
                break;
            }
            
        
        default:
            rb_raise(rb_eRuntimeError, "Wrong type for 'minimum_reduce'.");
    }

    return res;
}


/**
    Maximum_reduce a Bohrium array of type T.

    @param result The resulting array.
    @param selfObj The calling object.
    @param arg The constant argument.
*/
template <typename T>
inline void _maximum_reduce(bhDataObj<T>* result, bhDataObj<T>* selfObj, size_t arg) {
    bhxx::BhArray<T> selfArray = selfObj->bhary;
    // Cannot choose a dimension larger than the array shape size.
    assert(selfArray.shape.size() > arg);

    // FIXME: We assume we only ever have two dimensions
    bhxx::Shape _shape;
    size_t self_size = selfArray.shape.size();
    if (self_size == 1) {
        _shape = {1};
    } else {
        if (selfArray.shape[self_size - arg - 1] == 1) {
            _shape = {selfArray.shape[arg]};
        } else {
            _shape = {selfArray.shape[self_size - arg - 1]};
        }
    }

    bhxx::BhArray<T> bhary = *(new bhxx::BhArray<T>(_shape));

    bhxx::maximum_reduce(bhary, selfArray, arg);

    result->bhary = bhary;
    result->type = selfObj->type;
}

/**
    Maximum_reduce a Bohrium array.

    @param self The calling object.
    @param arg The constant argument.
    @returns The resulting array.
*/
VALUE bh_array_m_maximum_reduce(VALUE self, VALUE arg) {
    UNPACK(int64_t, tmpObj);

    VALUE res = bh_array_alloc(RBASIC_CLASS(self));

    switch (tmpObj->type) {
        
            
            case T_FIXNUM: {
                UNPACK(int64_t, selfObj);
                UNPACK_(int64_t, result, res);
                _maximum_reduce<int64_t>(result, selfObj, NUM2INT(arg));
                break;
            }
            
            case T_BIGNUM: {
                UNPACK(int64_t, selfObj);
                UNPACK_(int64_t, result, res);
                _maximum_reduce<int64_t>(result, selfObj, NUM2INT(arg));
                break;
            }
            
        
        default:
            rb_raise(rb_eRuntimeError, "Wrong type for 'maximum_reduce'.");
    }

    return res;
}


/**
    Bitwise_and_reduce a Bohrium array of type T.

    @param result The resulting array.
    @param selfObj The calling object.
    @param arg The constant argument.
*/
template <typename T>
inline void _bitwise_and_reduce(bhDataObj<T>* result, bhDataObj<T>* selfObj, size_t arg) {
    bhxx::BhArray<T> selfArray = selfObj->bhary;
    // Cannot choose a dimension larger than the array shape size.
    assert(selfArray.shape.size() > arg);

    // FIXME: We assume we only ever have two dimensions
    bhxx::Shape _shape;
    size_t self_size = selfArray.shape.size();
    if (self_size == 1) {
        _shape = {1};
    } else {
        if (selfArray.shape[self_size - arg - 1] == 1) {
            _shape = {selfArray.shape[arg]};
        } else {
            _shape = {selfArray.shape[self_size - arg - 1]};
        }
    }

    bhxx::BhArray<T> bhary = *(new bhxx::BhArray<T>(_shape));

    bhxx::bitwise_and_reduce(bhary, selfArray, arg);

    result->bhary = bhary;
    result->type = selfObj->type;
}

/**
    Bitwise_and_reduce a Bohrium array.

    @param self The calling object.
    @param arg The constant argument.
    @returns The resulting array.
*/
VALUE bh_array_m_bitwise_and_reduce(VALUE self, VALUE arg) {
    UNPACK(int64_t, tmpObj);

    VALUE res = bh_array_alloc(RBASIC_CLASS(self));

    switch (tmpObj->type) {
        
            
            case T_FIXNUM: {
                UNPACK(int64_t, selfObj);
                UNPACK_(int64_t, result, res);
                _bitwise_and_reduce<int64_t>(result, selfObj, NUM2INT(arg));
                break;
            }
            
            case T_BIGNUM: {
                UNPACK(int64_t, selfObj);
                UNPACK_(int64_t, result, res);
                _bitwise_and_reduce<int64_t>(result, selfObj, NUM2INT(arg));
                break;
            }
            
        
        default:
            rb_raise(rb_eRuntimeError, "Wrong type for 'bitwise_and_reduce'.");
    }

    return res;
}


/**
    Bitwise_or_reduce a Bohrium array of type T.

    @param result The resulting array.
    @param selfObj The calling object.
    @param arg The constant argument.
*/
template <typename T>
inline void _bitwise_or_reduce(bhDataObj<T>* result, bhDataObj<T>* selfObj, size_t arg) {
    bhxx::BhArray<T> selfArray = selfObj->bhary;
    // Cannot choose a dimension larger than the array shape size.
    assert(selfArray.shape.size() > arg);

    // FIXME: We assume we only ever have two dimensions
    bhxx::Shape _shape;
    size_t self_size = selfArray.shape.size();
    if (self_size == 1) {
        _shape = {1};
    } else {
        if (selfArray.shape[self_size - arg - 1] == 1) {
            _shape = {selfArray.shape[arg]};
        } else {
            _shape = {selfArray.shape[self_size - arg - 1]};
        }
    }

    bhxx::BhArray<T> bhary = *(new bhxx::BhArray<T>(_shape));

    bhxx::bitwise_or_reduce(bhary, selfArray, arg);

    result->bhary = bhary;
    result->type = selfObj->type;
}

/**
    Bitwise_or_reduce a Bohrium array.

    @param self The calling object.
    @param arg The constant argument.
    @returns The resulting array.
*/
VALUE bh_array_m_bitwise_or_reduce(VALUE self, VALUE arg) {
    UNPACK(int64_t, tmpObj);

    VALUE res = bh_array_alloc(RBASIC_CLASS(self));

    switch (tmpObj->type) {
        
            
            case T_FIXNUM: {
                UNPACK(int64_t, selfObj);
                UNPACK_(int64_t, result, res);
                _bitwise_or_reduce<int64_t>(result, selfObj, NUM2INT(arg));
                break;
            }
            
            case T_BIGNUM: {
                UNPACK(int64_t, selfObj);
                UNPACK_(int64_t, result, res);
                _bitwise_or_reduce<int64_t>(result, selfObj, NUM2INT(arg));
                break;
            }
            
        
        default:
            rb_raise(rb_eRuntimeError, "Wrong type for 'bitwise_or_reduce'.");
    }

    return res;
}


/**
    Bitwise_xor_reduce a Bohrium array of type T.

    @param result The resulting array.
    @param selfObj The calling object.
    @param arg The constant argument.
*/
template <typename T>
inline void _bitwise_xor_reduce(bhDataObj<T>* result, bhDataObj<T>* selfObj, size_t arg) {
    bhxx::BhArray<T> selfArray = selfObj->bhary;
    // Cannot choose a dimension larger than the array shape size.
    assert(selfArray.shape.size() > arg);

    // FIXME: We assume we only ever have two dimensions
    bhxx::Shape _shape;
    size_t self_size = selfArray.shape.size();
    if (self_size == 1) {
        _shape = {1};
    } else {
        if (selfArray.shape[self_size - arg - 1] == 1) {
            _shape = {selfArray.shape[arg]};
        } else {
            _shape = {selfArray.shape[self_size - arg - 1]};
        }
    }

    bhxx::BhArray<T> bhary = *(new bhxx::BhArray<T>(_shape));

    bhxx::bitwise_xor_reduce(bhary, selfArray, arg);

    result->bhary = bhary;
    result->type = selfObj->type;
}

/**
    Bitwise_xor_reduce a Bohrium array.

    @param self The calling object.
    @param arg The constant argument.
    @returns The resulting array.
*/
VALUE bh_array_m_bitwise_xor_reduce(VALUE self, VALUE arg) {
    UNPACK(int64_t, tmpObj);

    VALUE res = bh_array_alloc(RBASIC_CLASS(self));

    switch (tmpObj->type) {
        
            
            case T_FIXNUM: {
                UNPACK(int64_t, selfObj);
                UNPACK_(int64_t, result, res);
                _bitwise_xor_reduce<int64_t>(result, selfObj, NUM2INT(arg));
                break;
            }
            
            case T_BIGNUM: {
                UNPACK(int64_t, selfObj);
                UNPACK_(int64_t, result, res);
                _bitwise_xor_reduce<int64_t>(result, selfObj, NUM2INT(arg));
                break;
            }
            
        
        default:
            rb_raise(rb_eRuntimeError, "Wrong type for 'bitwise_xor_reduce'.");
    }

    return res;
}


/**
    Add_accumulate a Bohrium array of type T.

    @param result The resulting array.
    @param selfObj The calling object.
    @param arg The constant argument.
*/
template <typename T>
inline void _add_accumulate(bhDataObj<T>* result, bhDataObj<T>* selfObj, size_t arg) {
    bhxx::BhArray<T> selfArray = selfObj->bhary;
    // Cannot choose a dimension larger than the array shape size.
    assert(selfArray.shape.size() > arg);

    // FIXME: We assume we only ever have two dimensions
    bhxx::Shape _shape;
    size_t self_size = selfArray.shape.size();
    if (self_size == 1) {
        _shape = {1};
    } else {
        if (selfArray.shape[self_size - arg - 1] == 1) {
            _shape = {selfArray.shape[arg]};
        } else {
            _shape = {selfArray.shape[self_size - arg - 1]};
        }
    }

    bhxx::BhArray<T> bhary = *(new bhxx::BhArray<T>(_shape));

    bhxx::add_accumulate(bhary, selfArray, arg);

    result->bhary = bhary;
    result->type = selfObj->type;
}

/**
    Add_accumulate a Bohrium array.

    @param self The calling object.
    @param arg The constant argument.
    @returns The resulting array.
*/
VALUE bh_array_m_add_accumulate(VALUE self, VALUE arg) {
    UNPACK(int64_t, tmpObj);

    VALUE res = bh_array_alloc(RBASIC_CLASS(self));

    switch (tmpObj->type) {
        
            
            case T_FIXNUM: {
                UNPACK(int64_t, selfObj);
                UNPACK_(int64_t, result, res);
                _add_accumulate<int64_t>(result, selfObj, NUM2INT(arg));
                break;
            }
            
            case T_BIGNUM: {
                UNPACK(int64_t, selfObj);
                UNPACK_(int64_t, result, res);
                _add_accumulate<int64_t>(result, selfObj, NUM2INT(arg));
                break;
            }
            
        
        default:
            rb_raise(rb_eRuntimeError, "Wrong type for 'add_accumulate'.");
    }

    return res;
}


/**
    Multiply_accumulate a Bohrium array of type T.

    @param result The resulting array.
    @param selfObj The calling object.
    @param arg The constant argument.
*/
template <typename T>
inline void _multiply_accumulate(bhDataObj<T>* result, bhDataObj<T>* selfObj, size_t arg) {
    bhxx::BhArray<T> selfArray = selfObj->bhary;
    // Cannot choose a dimension larger than the array shape size.
    assert(selfArray.shape.size() > arg);

    // FIXME: We assume we only ever have two dimensions
    bhxx::Shape _shape;
    size_t self_size = selfArray.shape.size();
    if (self_size == 1) {
        _shape = {1};
    } else {
        if (selfArray.shape[self_size - arg - 1] == 1) {
            _shape = {selfArray.shape[arg]};
        } else {
            _shape = {selfArray.shape[self_size - arg - 1]};
        }
    }

    bhxx::BhArray<T> bhary = *(new bhxx::BhArray<T>(_shape));

    bhxx::multiply_accumulate(bhary, selfArray, arg);

    result->bhary = bhary;
    result->type = selfObj->type;
}

/**
    Multiply_accumulate a Bohrium array.

    @param self The calling object.
    @param arg The constant argument.
    @returns The resulting array.
*/
VALUE bh_array_m_multiply_accumulate(VALUE self, VALUE arg) {
    UNPACK(int64_t, tmpObj);

    VALUE res = bh_array_alloc(RBASIC_CLASS(self));

    switch (tmpObj->type) {
        
            
            case T_FIXNUM: {
                UNPACK(int64_t, selfObj);
                UNPACK_(int64_t, result, res);
                _multiply_accumulate<int64_t>(result, selfObj, NUM2INT(arg));
                break;
            }
            
            case T_BIGNUM: {
                UNPACK(int64_t, selfObj);
                UNPACK_(int64_t, result, res);
                _multiply_accumulate<int64_t>(result, selfObj, NUM2INT(arg));
                break;
            }
            
        
        default:
            rb_raise(rb_eRuntimeError, "Wrong type for 'multiply_accumulate'.");
    }

    return res;
}

