#pragma once

using namespace std;

/**
    Returns the element at index `i` of `ary` with type T.

    @param ary The array to index.
    @param i The index.
    @return An integer or float depending on the array.
*/
template <typename T>
T _get(VALUE ary, unsigned long i) {
    VALUE val = rb_ary_entry(ary, i);
    switch (TYPE(val)) {
        case T_BIGNUM:
        case T_FIXNUM:
            return NUM2INT(val);
        case T_FLOAT:
            return NUM2DBL(val);
        case T_TRUE:
        case T_FALSE:
            if (val == Qtrue) {
                return true;
            } else if (val == Qfalse) {
                return false;
            }
        default:
            rb_raise(rb_eRuntimeError, "Wrong type for array given.");
    }
}

/**
    Allocate memory for the data object on the Ruby object.

    @param self The calling Ruby object.
    @return The newly allocated Ruby object.
*/
VALUE bh_array_alloc(VALUE self) {
    void* ptr;
    return Data_Make_Struct(self, bhDataObj<int64_t>, NULL, RUBY_DEFAULT_FREE, ptr);
}

/**
    Convert the Bohrium array to a string.

    @param self The calling object.
    @return A Ruby string representing the Bohrium array.
*/
VALUE bh_array_m_to_s(VALUE self) {
    stringstream ss;

    bhDataObj<int64_t> *tmpObj;
    Data_Get_Struct(self, bhDataObj<int64_t>, tmpObj);

    switch (tmpObj->type) {
        case T_FIXNUM: {
            bhDataObj<int64_t> *dataObj;
            Data_Get_Struct(self, bhDataObj<int64_t>, dataObj);
            dataObj->bhary.pprint(ss);
            break;
        }
        case T_FLOAT: {
            bhDataObj<float> *dataObj;
            Data_Get_Struct(self, bhDataObj<float>, dataObj);
            dataObj->bhary.pprint(ss);
            break;
        }
        case T_TRUE:
        case T_FALSE: {
            bhDataObj<bool> *dataObj;
            Data_Get_Struct(self, bhDataObj<bool>, dataObj);
            dataObj->bhary.pprint(ss);
            break;
        }
        default:
            rb_raise(rb_eRuntimeError, "#to_s: Type not supported.");
    }
    VALUE result = rb_str_new2(ss.str().c_str());

    return result;
}

/**
    Print the Bohrium array to STDOUT.

    @param self The calling object.
    @return self
*/
VALUE bh_array_m_print(VALUE self) {
    bhDataObj<int64_t> *tmpObj;
    Data_Get_Struct(self, bhDataObj<int64_t>, tmpObj);

    switch (tmpObj->type) {
        case T_FIXNUM: {
            bhDataObj<int64_t> *dataObj;
            Data_Get_Struct(self, bhDataObj<int64_t>, dataObj);
            dataObj->bhary.pprint(cout);
            break;
        }
        case T_FLOAT: {
            bhDataObj<float> *dataObj;
            Data_Get_Struct(self, bhDataObj<float>, dataObj);
            dataObj->bhary.pprint(cout);
            break;
        }
        case T_TRUE:
        case T_FALSE: {
            bhDataObj<bool> *dataObj;
            Data_Get_Struct(self, bhDataObj<bool>, dataObj);
            dataObj->bhary.pprint(cout);
            break;
        }
        default:
            rb_raise(rb_eRuntimeError, "#print: Type not supported.");
    }

    return Qnil;
}

/**
    Helper function for converting Bohrium into Ruby arrays.

    @param self The calling object.
    @param rb_ary The return array.
*/
template <typename T>
inline void _to_ary(VALUE self, VALUE rb_ary) {
    bhDataObj<T> *dataObj;
    Data_Get_Struct(self, bhDataObj<T>, dataObj);

    bhxx::BhArray<T> bh_ary = dataObj->bhary;

    bhxx::Runtime::instance().sync(bh_ary.base);
    bhxx::Runtime::instance().flush();

    for(size_t i = 0; i < static_cast<size_t>(bh_ary.base->nelem); ++i) {
        if (std::is_same<T, int64_t>::value) {
            rb_ary_push(rb_ary, INT2NUM(bh_ary.data()[i]));
        } else if (std::is_same<T, float>::value) {
            rb_ary_push(rb_ary, DBL2NUM(bh_ary.data()[i]));
        } else if (std::is_same<T, bool>::value) {
            rb_ary_push(rb_ary, bh_ary.data()[i] ? Qtrue : Qfalse);
        } else {
            rb_raise(rb_eRuntimeError, "Invalid type.");
        }

    }
}

/**
    Convert a Bohrium array into a Ruby array.

    @param self The calling object.
    @return A Ruby array with the data.
*/
VALUE bh_array_m_to_ary(VALUE self) {
    VALUE rb_ary = rb_ary_new();

    bhDataObj<int64_t> *tmpObj;
    Data_Get_Struct(self, bhDataObj<int64_t>, tmpObj);

    switch (tmpObj->type) {
        case T_FIXNUM:
            _to_ary<int64_t>(self, rb_ary);
            break;
        case T_FLOAT:
            _to_ary<float>(self, rb_ary);
            break;
        case T_TRUE:
        case T_FALSE:
            _to_ary<bool>(self, rb_ary);
            break;
        default:
            rb_raise(rb_eRuntimeError, "#to_ary: Type not supported.");
    }

    return rb_ary;
}

/**
    Returns the number of elements in the array.

    @param self The calling object.
    @return Number of elements.
*/
VALUE bh_array_m_size(VALUE self) {
    bhDataObj<int64_t> *dataObj;
    Data_Get_Struct(self, bhDataObj<int64_t>, dataObj);
    return INT2NUM(dataObj->bhary.base->nelem);
}

/**
    Returns the shape of the array.

    @param self The calling object.
    @return Shape of the array as a Ruby array.
*/
VALUE bh_array_m_shape(VALUE self) {
    bhDataObj<int64_t> *dataObj;
    Data_Get_Struct(self, bhDataObj<int64_t>, dataObj);
    VALUE rb_ary = rb_ary_new();

    bhxx::Shape shape(dataObj->bhary.shape);
    for (auto it : shape) {
        rb_ary_push(rb_ary, INT2NUM(it));
    }
    return rb_ary;
}

/**
    Returns a reshaped array.

    @param self The calling object.
    @param new_shape The new shape as a Ruby array.
    @return Reshaped array.
*/
VALUE bh_array_m_reshape(VALUE self, VALUE new_shape) {
    if (TYPE(new_shape) != T_ARRAY) {
        rb_raise(rb_eRuntimeError, "New shape has to be an array.");
    }

    // We don't need to worry about the actual type of the dataObj
    // as we are only changing the shape of the  data.
    bhDataObj<int64_t> *dataObj;
    Data_Get_Struct(self, bhDataObj<int64_t>, dataObj);

    vector<size_t> vec;
    unsigned long size = rb_array_len(new_shape);
    vec.reserve(size);
    for(unsigned long i = 0; i < size; ++i) {
        vec.push_back(_get<int64_t>(new_shape, i));
    }
    bhxx::Shape shape(vec);

    try {
        dataObj->bhary = bhxx::reshape(dataObj->bhary, shape);
    } catch(const std::runtime_error& e) {
        // Convert potential C++ error to Ruby exception.
        rb_raise(rb_eRuntimeError, "%s", e.what());
    }

    return self;
}

