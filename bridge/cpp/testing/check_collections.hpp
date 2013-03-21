#include <iostream>

template<typename LeftIter, typename RightIter>
::testing::AssertionResult CheckEqualCollections(LeftIter left_begin,
                                                 LeftIter left_end,
                                                 RightIter right_begin)
{
    std::stringstream message;
    std::size_t index(0);
    bool equal(true);

    for(;left_begin != left_end; left_begin++, right_begin++) {
        if (*left_begin != *right_begin) {
            equal = false;
            message << "\n  Mismatch in position " << index << ": ";
            message << *left_begin << " != " <<  *right_begin;
        }
        ++index;
    }
    if (message.str().size()) {
        message << "\n";
    }
    return equal ? ::testing::AssertionSuccess() :
                   ::testing::AssertionFailure() << message.str();
}

::testing::AssertionResult VerifySlicing(bh_array* view, int* shape, int* stride, int ndim, int start)
{
    std::stringstream message;
    bool equal(true), equal_shape(true), equal_stride(true);

    if (ndim != (view->ndim)) {
        equal = false;
        message << "\n    view->ndim != expected {";
        message << "\n        Actual: " << view->ndim;
        message << "\n      Expected: " << ndim;
        message << "\n    }";
    }

    if (start != (view->start)) {
        equal = false;
        message << "\n    view->start != expected {";
        message << "\n        Actual: " << view->start;
        message << "\n      Expected: " << start;
        message << "\n    }";
    }
    
    for(int i=0; equal && (i < view->ndim); i++) {      // Determine error
        if (shape[i] != view->shape[i]) {
            equal_shape = false;
        } 
        if (stride[i] != view->stride[i]) {
            equal_stride = false;
        } 
    }
    if (!equal_shape) {                                 // Pretty-print the shape
        message << "\n    view->shape != expected {";

        message << "\n     view->shape = " << view->shape[0];
        for(int i=1; i < view->ndim; i++) {
            message << "," << view->shape[i];
        } 
        message << "\n        expected = " << shape[0];
        for(int i=1; i < ndim; i++) {
            message << "," << shape[i];
        }
        message << "\n    }";
    }

    if (!equal_stride) {
        message << "\n    view->stride != expected {";

        message << "\n     view->stride = " << view->stride[0];
        for(int i=1; i < view->ndim; i++) {
            message << "," << view->stride[i];
        } 
        message << "\n         expected = " << stride[0];
        for(int i=1; i < ndim; i++) {
            message << "," << stride[i];
        }
        message << "\n    }";
    }

    if (message.str().size()) {
        message << "\n";
    }
    return ((equal && equal_shape) && equal_stride) ? ::testing::AssertionSuccess() :
                   ::testing::AssertionFailure() << message.str();

}

