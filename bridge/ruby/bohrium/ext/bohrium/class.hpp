#pragma once

using namespace std;

/**
    Creates an array of the given size and value using Bohriums identity method.

    @param result The data object to add the result to.
    @param width The width of the array.
    @param height The height of the array.
    @param value The value each element should have.
    @param type The Ruby type of this value.
*/
template <typename T>
void _identity(bhDataObj *result, unsigned long width, unsigned long height, T value, ruby_value_type type) {
    bhxx::BhArray<T>* bhary = new bhxx::BhArray<T>({width, height});
    bhxx::identity(*bhary, value);
    result->type = type;
    result->ary = ((void*) bhary);
}

/**
    Create an array and fill it with ones (or value given).

    @param argc Number of elements in argv.
    @param argv The arguments given the method from Ruby.
    @param klass The class we are defining on.
    @return The created array.
*/
VALUE bh_array_m_ones(int argc, VALUE *argv, VALUE klass) {
    unsigned long x = 1, y = 1;

    VALUE res = bh_array_alloc(klass);
    bhDataObj *result;
    Data_Get_Struct(res, bhDataObj, result);

    if(argc >= 1) { x = NUM2INT(argv[0]); }
    if(argc >= 2) { y = NUM2INT(argv[1]); }
    if(argc >= 3) {
        switch (TYPE(argv[2])) {
            case T_FIXNUM:
                _identity(result, x, y, (int64_t) NUM2INT(argv[2]), T_FIXNUM);
                break;
            case T_FLOAT:
                _identity(result, x, y, (float) NUM2DBL(argv[2]), T_FLOAT);
                break;
            // TODO: Add T_TRUE and T_FALSE (Booleans)
            default:
                rb_raise(rb_eRuntimeError, "Wrong type for array given.");
        }
    } else {
        _identity(result, x, y, (int64_t) 1, T_FIXNUM);
    }

    return res;
}

/**
    Create an array and fill it with zeros.

    @param argc Number of elements in argv.
    @param argv The arguments given the method from Ruby.
    @param klass The class we are defining on.
    @return The created array.
*/
VALUE bh_array_m_zeros(int argc, VALUE *argv, VALUE klass) {
                     argv[2] = INT2NUM(0);   // Force 'value' to 0
    if (argc <= 1) { argv[1] = INT2NUM(1); } // Force 'y' to 1, if not given
    if (argc <= 0) { argv[0] = INT2NUM(1); } // Force 'x' to 1, if not given
    return bh_array_m_ones(3, argv, klass);
}
