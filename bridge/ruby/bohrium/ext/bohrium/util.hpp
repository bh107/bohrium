#pragma once

using namespace std;

/**
    Unwrap a Bohrium array from a data object.

    @param data The data object to unwrap the array from
    @return The unwrapped array.
*/
template <typename T>
inline bhxx::BhArray<T>& unwrap(const bhDataObj *data) {
    return *((bhxx::BhArray<T>*) data->ary);
}

/**
    Allocate memory for the data object on the Ruby object.

    @param self The calling Ruby object.
    @return The newly allocated Ruby object.
*/
VALUE bh_array_alloc(VALUE self) {
    void* ptr;
    return Data_Make_Struct(self, bhDataObj, NULL, RUBY_DEFAULT_FREE, ptr);
}

/**
    Convert the Bohrium array to a string.

    @param self The calling object.
    @return A Ruby string representing the Bohrium array.
*/
VALUE bh_array_m_to_s(VALUE self) {
    bhDataObj *dataObj;
    Data_Get_Struct(self, bhDataObj, dataObj);

    stringstream ss;
    switch (dataObj->type) {
        case T_FIXNUM:
            unwrap<int64_t>(dataObj).pprint(ss);
            break;
        case T_FLOAT:
            unwrap<float>(dataObj).pprint(ss);
            break;
        case T_TRUE:
        case T_FALSE:
            unwrap<bool>(dataObj).pprint(ss);
            break;
        default:
            rb_raise(rb_eRuntimeError, "Type not supported.");
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
    bhDataObj *dataObj;
    Data_Get_Struct(self, bhDataObj, dataObj);

    switch (dataObj->type) {
        case T_FIXNUM:
            unwrap<int64_t>(dataObj).pprint(cout);
            break;
        case T_FLOAT:
            unwrap<float>(dataObj).pprint(cout);
            break;
        case T_TRUE:
        case T_FALSE:
            unwrap<bool>(dataObj).pprint(cout);
            break;
        default:
            rb_raise(rb_eRuntimeError, "Type not supported.");
    }

    return self;
}

/**
    Convert a Bohrium array into a Ruby array.

    @param self The calling object.
    @return A Ruby array with the data.
*/
VALUE bh_array_m_to_ary(VALUE self) {
    bhDataObj *dataObj;
    Data_Get_Struct(self, bhDataObj, dataObj);
    VALUE rb_ary = rb_ary_new();

    switch (dataObj->type) {
        case T_FIXNUM: {
            bhxx::BhArray<int64_t> bh_ary = unwrap<int64_t>(dataObj);

            bhxx::Runtime::instance().sync(bh_ary.base);
            bhxx::Runtime::instance().flush();

            for(size_t i = 0; i < static_cast<size_t>(bh_ary.base->nelem); ++i) {
                rb_ary_push(rb_ary, INT2NUM(bh_ary.data()[i]));
            }
            break;
        }
        case T_FLOAT: {
            bhxx::BhArray<float> bh_ary = unwrap<float>(dataObj);

            bhxx::Runtime::instance().sync(bh_ary.base);
            bhxx::Runtime::instance().flush();

            for(size_t i = 0; i < static_cast<size_t>(bh_ary.base->nelem); ++i) {
                rb_ary_push(rb_ary, DBL2NUM(bh_ary.data()[i]));
            }
            break;
        }
        case T_TRUE:
        case T_FALSE: {
            bhxx::BhArray<bool> bh_ary = unwrap<bool>(dataObj);

            bhxx::Runtime::instance().sync(bh_ary.base);
            bhxx::Runtime::instance().flush();

            for(size_t i = 0; i < static_cast<size_t>(bh_ary.base->nelem); ++i) {
                rb_ary_push(rb_ary, bh_ary.data()[i] ? Qtrue : Qfalse);
            }
            break;
        }
        default:
            rb_raise(rb_eRuntimeError, "Type not supported.");
    }

    return rb_ary;
}
