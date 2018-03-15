#pragma once

using namespace std;

template <typename T>
void del_bhc(bhDataObj<T>* dataObj);

inline bool is_constant(VALUE val) {
    switch (TYPE(val)) {
        case T_DATA:
            return false;
            break;
        case T_FIXNUM:
        case T_BIGNUM:
        case T_FLOAT:
        case T_TRUE:
        case T_FALSE:
            return true;
            break;
        default:
            rb_raise(rb_eRuntimeError, "Invalid type.");
    }
}

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

VALUE bh_flush(VALUE self) {
    bhxx::Runtime::instance().flush();
    return Qnil;
}

void bh_array_free(bhDataObj<int64_t>* dataObj) {
    del_bhc(dataObj);
    delete dataObj;
}

void bh_array_mark(bhDataObj<int64_t>* dataObj) {
    // No-op, as we have no Ruby values to mark for garbage collection.
}

/**
    Allocate memory for the data object on the Ruby object.

    @param klass The calling class.
    @return The newly allocated Ruby object.
*/
VALUE bh_array_alloc(VALUE klass) {
    bhDataObj<int64_t>* dataObj;
    return Data_Make_Struct(klass, bhDataObj<int64_t>, bh_array_mark, bh_array_free, dataObj);
}

template <typename T>
bhDataObj<T>* get_parent(bhDataObj<T>* dataObj) {
    if (dataObj->parent == nullptr) {
        return dataObj;
    }

    bhDataObj<T>* parent = dataObj->parent;
    while (parent != nullptr) {
        parent = parent->parent;
    }

    return parent;
}

template <typename T, typename S>
bool identical_views(bhxx::BhArray<T>* bhary, bhxx::BhArray<S>* otherary) {
    if (!std::is_same<T, S>::value) {
        return false;
    } else if (bhary->shape != otherary->shape) {
        return false;
    } else if (bhary->stride != otherary->stride) {
        return false;
    }

    size_t v1_offset = bhary->offset;
    size_t v2_offset = otherary->offset;

    return v1_offset == v2_offset;
}

template <typename T>
void del_bhc(bhDataObj<T>* dataObj) {
    delete dataObj->bhary;
    delete dataObj->view;

    bhDataObj<T>* parent = get_parent(dataObj);
    if (parent == dataObj) {
        // dataObj has no parent
        dataObj->bhary_version += 1;
    } else {
        del_bhc(parent);
    }
}

template <typename T>
bhxx::BhArray<T>* get_bhary(bhDataObj<T>* dataObj) {
    bhDataObj<T>* parent = get_parent<T>(dataObj);

    // There exists a view
    if (dataObj->view != nullptr) {
        // This view has the same version as the parents array version
        if (parent->bhary_version == dataObj->view_version) {
            // The views are identical
            if (identical_views(dataObj->bhary, dataObj->view)) {
                return dataObj->view;
            } else {
                dataObj->view = nullptr;
            }
        }
    }

    bhxx::BhArray<T>* ret = new bhxx::BhArray<T>(
        parent->bhary->base,
        dataObj->bhary->shape,
        dataObj->bhary->stride,
        dataObj->bhary->offset
    );

    dataObj->view = ret;
    dataObj->view_version = parent->bhary_version;

    return ret;
}

/**
    Prints the Bohrium array to a stream. This stream can be STDOUT.

    @param self The calling object.
    @param ss The stream to print to
*/
inline void _print_to(VALUE self, ostream &ss) {
    UNPACK(int64_t, tmpObj);

    switch (tmpObj->type) {
        case T_FIXNUM: {
            UNPACK(int64_t, dataObj);
            get_bhary(dataObj)->pprint(ss);
            break;
        }
        case T_FLOAT: {
            UNPACK(float, dataObj);
            get_bhary(dataObj)->pprint(ss);
            break;
        }
        case T_TRUE:
        case T_FALSE: {
            UNPACK(bool, dataObj);
            get_bhary(dataObj)->pprint(ss);
            break;
        }
        default:
            rb_raise(rb_eRuntimeError, "Type not supported.");
    }
}

/**
    Convert the Bohrium array to a string.

    @param self The calling object.
    @return A Ruby string representing the Bohrium array.
*/
VALUE bh_array_m_to_s(VALUE self) {
    stringstream ss;
    _print_to(self, ss);
    return rb_str_new2(ss.str().c_str());
}

/**
    Print the Bohrium array to STDOUT.

    @param self The calling object.
    @return nil
*/
VALUE bh_array_m_print(VALUE self) {
    _print_to(self, cout);
    return Qnil;
}

/**
    Helper function for returning typed data.

    @param bh_view The Bohrium view to fetch from.
    @param index The index into the view.
    @return The typed data.
*/
template <typename T>
inline VALUE _typed_data(bhxx::BhArray<T> bh_view, size_t index) {
    if (std::is_same<T, int64_t>::value) {
        return INT2NUM(bh_view.data()[index]);
    } else if (std::is_same<T, float>::value) {
        return DBL2NUM(bh_view.data()[index]);
    } else if (std::is_same<T, bool>::value) {
        return bh_view.data()[index] ? Qtrue : Qfalse;
    } else {
        rb_raise(rb_eRuntimeError, "Invalid type.");
    }
}

/**
    Helper function for converting Bohrium into Ruby arrays.
    Will return two-dimensional

    @param self The calling object.
    @param rb_ary The return array.
*/
template <typename T>
inline void _to_ary(VALUE self, VALUE rb_ary) {
    UNPACK(T, dataObj);

    bhxx::BhArray<T> bh_view = bhxx::as_contiguous(*(get_bhary(dataObj)));
    bhxx::Runtime::instance().sync(bh_view.base);
    bhxx::Runtime::instance().flush();

    if (bh_view.shape.size() > 2) {
        // FIXME: If dimensions are above 2, we return one dimensional data.
        for(size_t i = 0; i < bh_view.numberOfElements(); ++i) {
            rb_ary_push(rb_ary, _typed_data<T>(bh_view, i));
        }
    } else {
        // For 1 or 2 dimensions, we return a possibly nested array.
        if (bh_view.shape.size() == 1) {
            // One-dimensional array
            for(size_t i = 0; i < bh_view.shape[0]; ++i) {
                rb_ary_push(rb_ary, _typed_data<T>(bh_view, i));
            }
        } else {
            // Two-dimensional array
            VALUE inner_ary;

            for (size_t row = 0; row < bh_view.shape[0]; ++row) {
                inner_ary = rb_ary_new();
                for (size_t col = 0; col < bh_view.shape[1]; ++col) {
                    rb_ary_push(inner_ary, _typed_data<T>(bh_view, row * bh_view.shape[1] + col));
                }
                rb_ary_push(rb_ary, inner_ary);
            }
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

    UNPACK(int64_t, tmpObj);

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
    UNPACK(int64_t, dataObj);
    return INT2NUM(get_bhary(dataObj)->numberOfElements());
}

/**
    Returns the shape of the array.

    @param self The calling object.
    @return Shape of the array as a Ruby array.
*/
VALUE bh_array_m_shape(VALUE self) {
    UNPACK(int64_t, dataObj);
    VALUE rb_ary = rb_ary_new();

    bhxx::Shape shape(get_bhary(dataObj)->shape);
    for (auto it : shape) {
        rb_ary_push(rb_ary, INT2NUM(it));
    }
    return rb_ary;
}

/**
    Returns the class of the array, that is the type.

    @param self The calling object.
    @return Type of the array as a Ruby class.
*/
VALUE bh_array_m_type(VALUE self) {
    // Doesn't matter what the actual type of the data is,
    // since we ask it about it's type afterwards.
    UNPACK(int64_t, dataObj);
    switch(dataObj->type) {
        case T_FIXNUM:
            return rb_cInteger;
        case T_FLOAT:
            return rb_cFloat;
        case T_TRUE:
            return rb_cTrueClass;
        case T_FALSE:
            return rb_cFalseClass;
        default:
            rb_raise(rb_eRuntimeError, "Cannot determine type of array.");
    }
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
    UNPACK(int64_t, dataObj);
    VALUE returnObj = bh_array_alloc(cBhArray);

    vector<size_t> vec;
    unsigned long size = rb_array_len(new_shape);
    vec.reserve(size);
    for(unsigned long i = 0; i < size; ++i) {
        vec.push_back(_get<int64_t>(new_shape, i));
    }
    bhxx::Shape shape(vec);

    try {
        UNPACK_(int64_t, newObj, returnObj);
        bhxx::BhArray<int64_t>* bhary = new bhxx::BhArray<int64_t>(get_bhary(dataObj)->shape);
        bhxx::identity(*bhary, *(get_bhary(dataObj)));
        bhary->shape  = shape;
        bhary->stride = contiguous_stride(shape);
        newObj->bhary = bhary;
        newObj->type  = dataObj->type;
    } catch(const std::runtime_error& e) {
        // Convert potential C++ error to Ruby exception.
        rb_raise(rb_eRuntimeError, "%s", e.what());
    }

    return returnObj;
}

/**
    Get a shape from a set of ranges.

    @param num_ranges The number of ranges.
    @param end An array of arrays of two elements. This is the start and end points of the ranges.
    @param stride_size The stride size.
    @return The shape.
*/
inline bhxx::Shape* _shape_from_range(int num_ranges, VALUE *end, size_t stride_size) {
    std::vector<size_t> shape_vector;
    int64_t a = 0, b = 0;

    for(int i = 0; i < num_ranges; ++i) {
        a = _get<int64_t>(end[i], 0);
        b = _get<int64_t>(end[i], 1);
        shape_vector.push_back(b-a+1);
    }

    return new bhxx::Shape(shape_vector);
}

/**
    Helper function to get a view from an array of ranges.

    @param dataObj The data object, which contains the data.
    @param num_ranges The number of ranges to use to find the view.
    @param end An array of arrays of two elements. This is the start and end points of the ranges.
    @return The array which has the shape of the view on the data.
*/
template <typename T>
inline bhxx::BhArray<T>* _view_from_ranges(bhDataObj<T>* dataObj, int num_ranges, VALUE *end) {
    bhxx::Shape _shape = *_shape_from_range(num_ranges, end, get_bhary(dataObj)->stride.size());
    // Only allow same stride for now.
    bhxx::Stride _stride(get_bhary(dataObj)->stride);

    size_t start = _get<int64_t>(end[0], 0);
    if (num_ranges == 2) {
        start *= get_bhary(dataObj)->shape[1];
        start += _get<int64_t>(end[1], 0);
    }

    // Create new array, that share the same base.
    return new bhxx::BhArray<T>(
        get_bhary(dataObj)->base,
        _shape,
        _stride,
        start
    );
}

/**
    Returns a view of the array.

    @param argc Amount of arguments given.
    @param argv Array of arguments in C-style.
    @param self The calling object.
    @return An array that is the view from the arguments.
*/
VALUE bh_array_m_view_from_ranges(int argc, VALUE *argv, VALUE self) {
    UNPACK(int64_t, tmpObj);

    VALUE returnObj = bh_array_alloc(cBhArray);

    // argv is an array of arrays of two elements
    switch(tmpObj->type) {
        case T_FIXNUM: {
            UNPACK(int64_t, dataObj);
            UNPACK_(int64_t, newObj, returnObj);
            newObj->bhary = _view_from_ranges<int64_t>(dataObj, argc, argv);
            newObj->type  = dataObj->type;
            break;
        }
        case T_FLOAT: {
            UNPACK(float, dataObj);
            UNPACK_(float, newObj, returnObj);
            newObj->bhary = _view_from_ranges<float>(dataObj, argc, argv);
            newObj->type  = dataObj->type;
            break;
        }
        case T_TRUE:
        case T_FALSE: {
            UNPACK(bool, dataObj);
            UNPACK_(bool, newObj, returnObj);
            newObj->bhary = _view_from_ranges<bool>(dataObj, argc, argv);
            newObj->type  = dataObj->type;
            break;
        }
        default:
            rb_raise(rb_eRuntimeError, "Cannot determine type of array.");
    }

    return returnObj;
}

/**
    Helper function to set view from ranges
*/
template <typename T>
inline void _set_from_ranges(bhDataObj<T>* dataObj, int num_ranges, VALUE *end, VALUE val) {
    // Get the view from the ranges
    bhxx::BhArray<T>* view = _view_from_ranges<T>(dataObj, num_ranges, end);

    // If the value type is a single value, we broadcast it to the entire view.
    switch(TYPE(val)) {
        case T_FIXNUM:
            bhxx::identity(*view, NUM2INT(val));
            break;
        case T_FLOAT:
            bhxx::identity(*view, NUM2DBL(val));
            break;
        case T_TRUE:
            bhxx::identity(*view, true);
            break;
        case T_FALSE:
            bhxx::identity(*view, false);
            break;
        case T_ARRAY:
            // This case should be handled from Ruby.
            // The array is converted into a BhArray before being sent here.
            rb_raise(rb_eRuntimeError, "Got an array while trying to set view. Shouldn't be possible.");
            break;
        default:
            if (RBASIC_CLASS(val) == cBhArray) {
                // If the value type is another array, we set the current view equal to that view.
                UNPACK_(int64_t, dataObj, val);
                bhxx::identity(*view, *(get_bhary(dataObj)));
            } else {
                rb_raise(rb_eRuntimeError, "Got invalid type while setting values in array.");
            }
    }
}

VALUE bh_array_m_set_from_ranges(int argc, VALUE *argv, VALUE self) {
    UNPACK(int64_t, tmpObj);

    switch(tmpObj->type) {
        case T_FIXNUM: {
            UNPACK(int64_t, dataObj);

            // Last argument is the value we want to set.
            // This can be a fixed value or another view of the same size.
            _set_from_ranges<int64_t>(dataObj, argc-1, argv, argv[argc-1]);
            break;
        }
        case T_FLOAT: {
            UNPACK(float, dataObj);
            _set_from_ranges<float>(dataObj, argc-1, argv, argv[argc-1]);
            break;
        }
        case T_TRUE:
        case T_FALSE: {
            UNPACK(bool, dataObj);
            _set_from_ranges<bool>(dataObj, argc-1, argv, argv[argc-1]);
            break;
        }
        default:
            rb_raise(rb_eRuntimeError, "Cannot determine type of array.");
    }

    return self;
}
