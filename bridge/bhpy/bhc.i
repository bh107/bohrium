%module bhc
%{
#include<bh_c.h>
%}

%typemap(in) uint64_t {
    $1 = PyInt_AsUnsignedLongMask($input);
}

%typemap(in) int64_t const * {
    Py_ssize_t i = PySequence_Size($input);
    int64_t l[10];
    for(i=0; i<PySequence_Size($input); ++i)
    {
        l[i] = PyInt_AsUnsignedLongMask(PySequence_GetItem($input, i));
    }
    $1 = l;
}

%typemap(out) bh_float32 {
    $result = PyFloat_FromDouble($1);
}

#ifndef __BH_C_DATA_TYPES_H
#define __BH_C_DATA_TYPES_H

#include <stdint.h>
#include <bh_type.h>

#ifdef _WIN32
#define DLLEXPORT __declspec( dllexport )
#else
#define DLLEXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Common runtime methods
DLLEXPORT void bh_runtime_flush();

// Common slice range
struct bh_slice_range;
typedef struct bh_slice_range* bh_slice_range_p;

// Common type forward definition
#ifndef __BH_ARRAY_H
struct bh_base;
struct bh_view;
typedef struct bh_base* bh_base_p;
typedef struct bh_view* bh_view_p;
#else
typedef bh_base* bh_base_p;
typedef bh_view* bh_view_p;
#endif



// Forward definitions
struct bh_multi_array_bool8;
struct bh_slice_bool8;

// Shorthand pointer defs
typedef struct bh_multi_array_bool8* bh_multi_array_bool8_p;
typedef struct bh_slice_range_bool8* bh_slice_range_bool8_p;

// Sync the current base
void bh_multi_array_bool8_sync(const bh_multi_array_bool8_p self);

// Sets the temp status of an array
DLLEXPORT void bh_multi_array_bool8_set_temp(const bh_multi_array_bool8_p self, bh_bool temp);

// Gets the temp status of an array
DLLEXPORT bh_bool bh_multi_array_bool8_get_temp(const bh_multi_array_bool8_p self);

// Create a base pointer from existing data
DLLEXPORT bh_base_p bh_multi_array_bool8_create_base(bh_bool* data, int64_t nelem);

// Destroy a base pointer
DLLEXPORT void bh_multi_array_bool8_destroy_base(bh_base_p base);

// Gets the data pointer from a base
DLLEXPORT bh_bool* bh_multi_array_bool8_get_base_data(bh_base_p base);

// Gets the number of elements in a base
DLLEXPORT int64_t bh_multi_array_bool8_get_base_nelem(bh_base_p base);

// Sets the data pointer for a base
DLLEXPORT void bh_multi_array_bool8_set_base_data(bh_base_p base, bh_bool* data);

// Get the base from an existing array
DLLEXPORT bh_base_p bh_multi_array_bool8_get_base(const bh_multi_array_bool8_p self);

// Construct a new array from bh_base_p and view setup
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_new_from_base(const bh_base_p base);

// Construct a new array from bh_base_p and view setup
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_new_from_view(const bh_base_p base, uint64_t rank, const int64_t start, const int64_t* shape, const int64_t* stride);

// Construct a new empty array
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_new_empty(uint64_t rank, const int64_t* shape);


// Construct a new random-filled array
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_new_random(const int64_t length);

// Construct a new array, filled with the specified value
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_new_value(const bh_bool value, uint64_t rank, const int64_t* shape);

// Construct a copy of the array
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_new_copy(bh_multi_array_bool8_p other);

// Destroy the pointer and release resources
DLLEXPORT void bh_multi_array_bool8_destroy(bh_multi_array_bool8_p self);

// Gets the number of elements in the array
DLLEXPORT uint64_t bh_multi_array_bool8_get_length(bh_multi_array_bool8_p self);

// Gets the number of dimensions in the array
DLLEXPORT uint64_t bh_multi_array_bool8_get_rank(bh_multi_array_bool8_p self);

// Gets the number of elements in the dimension
DLLEXPORT uint64_t bh_multi_array_bool8_get_dimension_size(bh_multi_array_bool8_p self, const int64_t dimension);

// Update with a scalar
DLLEXPORT void bh_multi_array_bool8_assign_scalar(bh_multi_array_bool8_p self, const bh_bool value);

// Update with an array
DLLEXPORT void bh_multi_array_bool8_assign_array(bh_multi_array_bool8_p self, bh_multi_array_bool8_p other);

// Flatten view
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_flatten(bh_multi_array_bool8_p self);

// Transpose view
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_transpose(bh_multi_array_bool8_p self);


// All
DLLEXPORT bh_bool bh_multi_array_bool8_all(bh_multi_array_bool8_p self);

// Any
DLLEXPORT bh_bool bh_multi_array_bool8_any(bh_multi_array_bool8_p self);

// Max
DLLEXPORT bh_bool bh_multi_array_bool8_max(bh_multi_array_bool8_p self);

// Min
DLLEXPORT bh_bool bh_multi_array_bool8_min(bh_multi_array_bool8_p self);





// Forward definitions
struct bh_multi_array_int8;
struct bh_slice_int8;

// Shorthand pointer defs
typedef struct bh_multi_array_int8* bh_multi_array_int8_p;
typedef struct bh_slice_range_int8* bh_slice_range_int8_p;

// Sync the current base
void bh_multi_array_int8_sync(const bh_multi_array_int8_p self);

// Sets the temp status of an array
DLLEXPORT void bh_multi_array_int8_set_temp(const bh_multi_array_int8_p self, bh_bool temp);

// Gets the temp status of an array
DLLEXPORT bh_bool bh_multi_array_int8_get_temp(const bh_multi_array_int8_p self);

// Create a base pointer from existing data
DLLEXPORT bh_base_p bh_multi_array_int8_create_base(bh_int8* data, int64_t nelem);

// Destroy a base pointer
DLLEXPORT void bh_multi_array_int8_destroy_base(bh_base_p base);

// Gets the data pointer from a base
DLLEXPORT bh_int8* bh_multi_array_int8_get_base_data(bh_base_p base);

// Gets the number of elements in a base
DLLEXPORT int64_t bh_multi_array_int8_get_base_nelem(bh_base_p base);

// Sets the data pointer for a base
DLLEXPORT void bh_multi_array_int8_set_base_data(bh_base_p base, bh_int8* data);

// Get the base from an existing array
DLLEXPORT bh_base_p bh_multi_array_int8_get_base(const bh_multi_array_int8_p self);

// Construct a new array from bh_base_p and view setup
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_new_from_base(const bh_base_p base);

// Construct a new array from bh_base_p and view setup
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_new_from_view(const bh_base_p base, uint64_t rank, const int64_t start, const int64_t* shape, const int64_t* stride);

// Construct a new empty array
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_new_empty(uint64_t rank, const int64_t* shape);

// Construct a new zero-filled array
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_new_zeroes(uint64_t rank, const int64_t* shape);

// Construct a new one-filled array
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_new_ones(uint64_t rank, const int64_t* shape);

// Construct a new array with sequential numbers
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_new_range(const int64_t start, const int64_t end, const int64_t skip);

// Construct a new random-filled array
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_new_random(const int64_t length);

// Construct a new array, filled with the specified value
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_new_value(const bh_int8 value, uint64_t rank, const int64_t* shape);

// Construct a copy of the array
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_new_copy(bh_multi_array_int8_p other);

// Destroy the pointer and release resources
DLLEXPORT void bh_multi_array_int8_destroy(bh_multi_array_int8_p self);

// Gets the number of elements in the array
DLLEXPORT uint64_t bh_multi_array_int8_get_length(bh_multi_array_int8_p self);

// Gets the number of dimensions in the array
DLLEXPORT uint64_t bh_multi_array_int8_get_rank(bh_multi_array_int8_p self);

// Gets the number of elements in the dimension
DLLEXPORT uint64_t bh_multi_array_int8_get_dimension_size(bh_multi_array_int8_p self, const int64_t dimension);

// Update with a scalar
DLLEXPORT void bh_multi_array_int8_assign_scalar(bh_multi_array_int8_p self, const bh_int8 value);

// Update with an array
DLLEXPORT void bh_multi_array_int8_assign_array(bh_multi_array_int8_p self, bh_multi_array_int8_p other);

// Flatten view
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_flatten(bh_multi_array_int8_p self);

// Transpose view
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_transpose(bh_multi_array_int8_p self);

// Partial add reduction
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_partial_reduce_add(bh_multi_array_int8_p self, const int64_t axis);

// Partial multiply reduction
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_partial_reduce_multiply(bh_multi_array_int8_p self, const int64_t axis);

// Partial min reduction
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_partial_reduce_min(bh_multi_array_int8_p self, const int64_t axis);

// Partial max reduction
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_partial_reduce_max(bh_multi_array_int8_p self, const int64_t axis);

// Partial logical_and reduction
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_partial_reduce_logical_and(bh_multi_array_int8_p self, const int64_t axis);

// Partial logical_or reduction
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_partial_reduce_logical_or(bh_multi_array_int8_p self, const int64_t axis);

// Partial logical_xor reduction
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_partial_reduce_logical_xor(bh_multi_array_int8_p self, const int64_t axis);

// Partial bitwise_and reduction
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_partial_reduce_bitwise_and(bh_multi_array_int8_p self, const int64_t axis);

// Partial bitwise_or reduction
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_partial_reduce_bitwise_or(bh_multi_array_int8_p self, const int64_t axis);

// Partial bitwise_xor reduction
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_partial_reduce_bitwise_xor(bh_multi_array_int8_p self, const int64_t axis);


// Sum
DLLEXPORT bh_int8 bh_multi_array_int8_sum(bh_multi_array_int8_p self);

// Product
DLLEXPORT bh_int8 bh_multi_array_int8_product(bh_multi_array_int8_p self);


// Max
DLLEXPORT bh_int8 bh_multi_array_int8_max(bh_multi_array_int8_p self);

// Min
DLLEXPORT bh_int8 bh_multi_array_int8_min(bh_multi_array_int8_p self);





// Forward definitions
struct bh_multi_array_int16;
struct bh_slice_int16;

// Shorthand pointer defs
typedef struct bh_multi_array_int16* bh_multi_array_int16_p;
typedef struct bh_slice_range_int16* bh_slice_range_int16_p;

// Sync the current base
void bh_multi_array_int16_sync(const bh_multi_array_int16_p self);

// Sets the temp status of an array
DLLEXPORT void bh_multi_array_int16_set_temp(const bh_multi_array_int16_p self, bh_bool temp);

// Gets the temp status of an array
DLLEXPORT bh_bool bh_multi_array_int16_get_temp(const bh_multi_array_int16_p self);

// Create a base pointer from existing data
DLLEXPORT bh_base_p bh_multi_array_int16_create_base(bh_int16* data, int64_t nelem);

// Destroy a base pointer
DLLEXPORT void bh_multi_array_int16_destroy_base(bh_base_p base);

// Gets the data pointer from a base
DLLEXPORT bh_int16* bh_multi_array_int16_get_base_data(bh_base_p base);

// Gets the number of elements in a base
DLLEXPORT int64_t bh_multi_array_int16_get_base_nelem(bh_base_p base);

// Sets the data pointer for a base
DLLEXPORT void bh_multi_array_int16_set_base_data(bh_base_p base, bh_int16* data);

// Get the base from an existing array
DLLEXPORT bh_base_p bh_multi_array_int16_get_base(const bh_multi_array_int16_p self);

// Construct a new array from bh_base_p and view setup
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_new_from_base(const bh_base_p base);

// Construct a new array from bh_base_p and view setup
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_new_from_view(const bh_base_p base, uint64_t rank, const int64_t start, const int64_t* shape, const int64_t* stride);

// Construct a new empty array
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_new_empty(uint64_t rank, const int64_t* shape);

// Construct a new zero-filled array
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_new_zeroes(uint64_t rank, const int64_t* shape);

// Construct a new one-filled array
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_new_ones(uint64_t rank, const int64_t* shape);

// Construct a new array with sequential numbers
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_new_range(const int64_t start, const int64_t end, const int64_t skip);

// Construct a new random-filled array
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_new_random(const int64_t length);

// Construct a new array, filled with the specified value
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_new_value(const bh_int16 value, uint64_t rank, const int64_t* shape);

// Construct a copy of the array
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_new_copy(bh_multi_array_int16_p other);

// Destroy the pointer and release resources
DLLEXPORT void bh_multi_array_int16_destroy(bh_multi_array_int16_p self);

// Gets the number of elements in the array
DLLEXPORT uint64_t bh_multi_array_int16_get_length(bh_multi_array_int16_p self);

// Gets the number of dimensions in the array
DLLEXPORT uint64_t bh_multi_array_int16_get_rank(bh_multi_array_int16_p self);

// Gets the number of elements in the dimension
DLLEXPORT uint64_t bh_multi_array_int16_get_dimension_size(bh_multi_array_int16_p self, const int64_t dimension);

// Update with a scalar
DLLEXPORT void bh_multi_array_int16_assign_scalar(bh_multi_array_int16_p self, const bh_int16 value);

// Update with an array
DLLEXPORT void bh_multi_array_int16_assign_array(bh_multi_array_int16_p self, bh_multi_array_int16_p other);

// Flatten view
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_flatten(bh_multi_array_int16_p self);

// Transpose view
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_transpose(bh_multi_array_int16_p self);

// Partial add reduction
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_partial_reduce_add(bh_multi_array_int16_p self, const int64_t axis);

// Partial multiply reduction
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_partial_reduce_multiply(bh_multi_array_int16_p self, const int64_t axis);

// Partial min reduction
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_partial_reduce_min(bh_multi_array_int16_p self, const int64_t axis);

// Partial max reduction
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_partial_reduce_max(bh_multi_array_int16_p self, const int64_t axis);

// Partial logical_and reduction
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_partial_reduce_logical_and(bh_multi_array_int16_p self, const int64_t axis);

// Partial logical_or reduction
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_partial_reduce_logical_or(bh_multi_array_int16_p self, const int64_t axis);

// Partial logical_xor reduction
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_partial_reduce_logical_xor(bh_multi_array_int16_p self, const int64_t axis);

// Partial bitwise_and reduction
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_partial_reduce_bitwise_and(bh_multi_array_int16_p self, const int64_t axis);

// Partial bitwise_or reduction
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_partial_reduce_bitwise_or(bh_multi_array_int16_p self, const int64_t axis);

// Partial bitwise_xor reduction
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_partial_reduce_bitwise_xor(bh_multi_array_int16_p self, const int64_t axis);


// Sum
DLLEXPORT bh_int16 bh_multi_array_int16_sum(bh_multi_array_int16_p self);

// Product
DLLEXPORT bh_int16 bh_multi_array_int16_product(bh_multi_array_int16_p self);


// Max
DLLEXPORT bh_int16 bh_multi_array_int16_max(bh_multi_array_int16_p self);

// Min
DLLEXPORT bh_int16 bh_multi_array_int16_min(bh_multi_array_int16_p self);





// Forward definitions
struct bh_multi_array_int32;
struct bh_slice_int32;

// Shorthand pointer defs
typedef struct bh_multi_array_int32* bh_multi_array_int32_p;
typedef struct bh_slice_range_int32* bh_slice_range_int32_p;

// Sync the current base
void bh_multi_array_int32_sync(const bh_multi_array_int32_p self);

// Sets the temp status of an array
DLLEXPORT void bh_multi_array_int32_set_temp(const bh_multi_array_int32_p self, bh_bool temp);

// Gets the temp status of an array
DLLEXPORT bh_bool bh_multi_array_int32_get_temp(const bh_multi_array_int32_p self);

// Create a base pointer from existing data
DLLEXPORT bh_base_p bh_multi_array_int32_create_base(bh_int32* data, int64_t nelem);

// Destroy a base pointer
DLLEXPORT void bh_multi_array_int32_destroy_base(bh_base_p base);

// Gets the data pointer from a base
DLLEXPORT bh_int32* bh_multi_array_int32_get_base_data(bh_base_p base);

// Gets the number of elements in a base
DLLEXPORT int64_t bh_multi_array_int32_get_base_nelem(bh_base_p base);

// Sets the data pointer for a base
DLLEXPORT void bh_multi_array_int32_set_base_data(bh_base_p base, bh_int32* data);

// Get the base from an existing array
DLLEXPORT bh_base_p bh_multi_array_int32_get_base(const bh_multi_array_int32_p self);

// Construct a new array from bh_base_p and view setup
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_new_from_base(const bh_base_p base);

// Construct a new array from bh_base_p and view setup
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_new_from_view(const bh_base_p base, uint64_t rank, const int64_t start, const int64_t* shape, const int64_t* stride);

// Construct a new empty array
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_new_empty(uint64_t rank, const int64_t* shape);

// Construct a new zero-filled array
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_new_zeroes(uint64_t rank, const int64_t* shape);

// Construct a new one-filled array
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_new_ones(uint64_t rank, const int64_t* shape);

// Construct a new array with sequential numbers
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_new_range(const int64_t start, const int64_t end, const int64_t skip);

// Construct a new random-filled array
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_new_random(const int64_t length);

// Construct a new array, filled with the specified value
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_new_value(const bh_int32 value, uint64_t rank, const int64_t* shape);

// Construct a copy of the array
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_new_copy(bh_multi_array_int32_p other);

// Destroy the pointer and release resources
DLLEXPORT void bh_multi_array_int32_destroy(bh_multi_array_int32_p self);

// Gets the number of elements in the array
DLLEXPORT uint64_t bh_multi_array_int32_get_length(bh_multi_array_int32_p self);

// Gets the number of dimensions in the array
DLLEXPORT uint64_t bh_multi_array_int32_get_rank(bh_multi_array_int32_p self);

// Gets the number of elements in the dimension
DLLEXPORT uint64_t bh_multi_array_int32_get_dimension_size(bh_multi_array_int32_p self, const int64_t dimension);

// Update with a scalar
DLLEXPORT void bh_multi_array_int32_assign_scalar(bh_multi_array_int32_p self, const bh_int32 value);

// Update with an array
DLLEXPORT void bh_multi_array_int32_assign_array(bh_multi_array_int32_p self, bh_multi_array_int32_p other);

// Flatten view
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_flatten(bh_multi_array_int32_p self);

// Transpose view
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_transpose(bh_multi_array_int32_p self);

// Partial add reduction
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_partial_reduce_add(bh_multi_array_int32_p self, const int64_t axis);

// Partial multiply reduction
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_partial_reduce_multiply(bh_multi_array_int32_p self, const int64_t axis);

// Partial min reduction
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_partial_reduce_min(bh_multi_array_int32_p self, const int64_t axis);

// Partial max reduction
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_partial_reduce_max(bh_multi_array_int32_p self, const int64_t axis);

// Partial logical_and reduction
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_partial_reduce_logical_and(bh_multi_array_int32_p self, const int64_t axis);

// Partial logical_or reduction
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_partial_reduce_logical_or(bh_multi_array_int32_p self, const int64_t axis);

// Partial logical_xor reduction
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_partial_reduce_logical_xor(bh_multi_array_int32_p self, const int64_t axis);

// Partial bitwise_and reduction
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_partial_reduce_bitwise_and(bh_multi_array_int32_p self, const int64_t axis);

// Partial bitwise_or reduction
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_partial_reduce_bitwise_or(bh_multi_array_int32_p self, const int64_t axis);

// Partial bitwise_xor reduction
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_partial_reduce_bitwise_xor(bh_multi_array_int32_p self, const int64_t axis);


// Sum
DLLEXPORT bh_int32 bh_multi_array_int32_sum(bh_multi_array_int32_p self);

// Product
DLLEXPORT bh_int32 bh_multi_array_int32_product(bh_multi_array_int32_p self);


// Max
DLLEXPORT bh_int32 bh_multi_array_int32_max(bh_multi_array_int32_p self);

// Min
DLLEXPORT bh_int32 bh_multi_array_int32_min(bh_multi_array_int32_p self);





// Forward definitions
struct bh_multi_array_int64;
struct bh_slice_int64;

// Shorthand pointer defs
typedef struct bh_multi_array_int64* bh_multi_array_int64_p;
typedef struct bh_slice_range_int64* bh_slice_range_int64_p;

// Sync the current base
void bh_multi_array_int64_sync(const bh_multi_array_int64_p self);

// Sets the temp status of an array
DLLEXPORT void bh_multi_array_int64_set_temp(const bh_multi_array_int64_p self, bh_bool temp);

// Gets the temp status of an array
DLLEXPORT bh_bool bh_multi_array_int64_get_temp(const bh_multi_array_int64_p self);

// Create a base pointer from existing data
DLLEXPORT bh_base_p bh_multi_array_int64_create_base(bh_int64* data, int64_t nelem);

// Destroy a base pointer
DLLEXPORT void bh_multi_array_int64_destroy_base(bh_base_p base);

// Gets the data pointer from a base
DLLEXPORT bh_int64* bh_multi_array_int64_get_base_data(bh_base_p base);

// Gets the number of elements in a base
DLLEXPORT int64_t bh_multi_array_int64_get_base_nelem(bh_base_p base);

// Sets the data pointer for a base
DLLEXPORT void bh_multi_array_int64_set_base_data(bh_base_p base, bh_int64* data);

// Get the base from an existing array
DLLEXPORT bh_base_p bh_multi_array_int64_get_base(const bh_multi_array_int64_p self);

// Construct a new array from bh_base_p and view setup
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_new_from_base(const bh_base_p base);

// Construct a new array from bh_base_p and view setup
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_new_from_view(const bh_base_p base, uint64_t rank, const int64_t start, const int64_t* shape, const int64_t* stride);

// Construct a new empty array
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_new_empty(uint64_t rank, const int64_t* shape);

// Construct a new zero-filled array
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_new_zeroes(uint64_t rank, const int64_t* shape);

// Construct a new one-filled array
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_new_ones(uint64_t rank, const int64_t* shape);

// Construct a new array with sequential numbers
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_new_range(const int64_t start, const int64_t end, const int64_t skip);

// Construct a new random-filled array
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_new_random(const int64_t length);

// Construct a new array, filled with the specified value
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_new_value(const bh_int64 value, uint64_t rank, const int64_t* shape);

// Construct a copy of the array
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_new_copy(bh_multi_array_int64_p other);

// Destroy the pointer and release resources
DLLEXPORT void bh_multi_array_int64_destroy(bh_multi_array_int64_p self);

// Gets the number of elements in the array
DLLEXPORT uint64_t bh_multi_array_int64_get_length(bh_multi_array_int64_p self);

// Gets the number of dimensions in the array
DLLEXPORT uint64_t bh_multi_array_int64_get_rank(bh_multi_array_int64_p self);

// Gets the number of elements in the dimension
DLLEXPORT uint64_t bh_multi_array_int64_get_dimension_size(bh_multi_array_int64_p self, const int64_t dimension);

// Update with a scalar
DLLEXPORT void bh_multi_array_int64_assign_scalar(bh_multi_array_int64_p self, const bh_int64 value);

// Update with an array
DLLEXPORT void bh_multi_array_int64_assign_array(bh_multi_array_int64_p self, bh_multi_array_int64_p other);

// Flatten view
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_flatten(bh_multi_array_int64_p self);

// Transpose view
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_transpose(bh_multi_array_int64_p self);

// Partial add reduction
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_partial_reduce_add(bh_multi_array_int64_p self, const int64_t axis);

// Partial multiply reduction
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_partial_reduce_multiply(bh_multi_array_int64_p self, const int64_t axis);

// Partial min reduction
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_partial_reduce_min(bh_multi_array_int64_p self, const int64_t axis);

// Partial max reduction
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_partial_reduce_max(bh_multi_array_int64_p self, const int64_t axis);

// Partial logical_and reduction
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_partial_reduce_logical_and(bh_multi_array_int64_p self, const int64_t axis);

// Partial logical_or reduction
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_partial_reduce_logical_or(bh_multi_array_int64_p self, const int64_t axis);

// Partial logical_xor reduction
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_partial_reduce_logical_xor(bh_multi_array_int64_p self, const int64_t axis);

// Partial bitwise_and reduction
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_partial_reduce_bitwise_and(bh_multi_array_int64_p self, const int64_t axis);

// Partial bitwise_or reduction
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_partial_reduce_bitwise_or(bh_multi_array_int64_p self, const int64_t axis);

// Partial bitwise_xor reduction
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_partial_reduce_bitwise_xor(bh_multi_array_int64_p self, const int64_t axis);


// Sum
DLLEXPORT bh_int64 bh_multi_array_int64_sum(bh_multi_array_int64_p self);

// Product
DLLEXPORT bh_int64 bh_multi_array_int64_product(bh_multi_array_int64_p self);


// Max
DLLEXPORT bh_int64 bh_multi_array_int64_max(bh_multi_array_int64_p self);

// Min
DLLEXPORT bh_int64 bh_multi_array_int64_min(bh_multi_array_int64_p self);





// Forward definitions
struct bh_multi_array_uint8;
struct bh_slice_uint8;

// Shorthand pointer defs
typedef struct bh_multi_array_uint8* bh_multi_array_uint8_p;
typedef struct bh_slice_range_uint8* bh_slice_range_uint8_p;

// Sync the current base
void bh_multi_array_uint8_sync(const bh_multi_array_uint8_p self);

// Sets the temp status of an array
DLLEXPORT void bh_multi_array_uint8_set_temp(const bh_multi_array_uint8_p self, bh_bool temp);

// Gets the temp status of an array
DLLEXPORT bh_bool bh_multi_array_uint8_get_temp(const bh_multi_array_uint8_p self);

// Create a base pointer from existing data
DLLEXPORT bh_base_p bh_multi_array_uint8_create_base(bh_uint8* data, int64_t nelem);

// Destroy a base pointer
DLLEXPORT void bh_multi_array_uint8_destroy_base(bh_base_p base);

// Gets the data pointer from a base
DLLEXPORT bh_uint8* bh_multi_array_uint8_get_base_data(bh_base_p base);

// Gets the number of elements in a base
DLLEXPORT int64_t bh_multi_array_uint8_get_base_nelem(bh_base_p base);

// Sets the data pointer for a base
DLLEXPORT void bh_multi_array_uint8_set_base_data(bh_base_p base, bh_uint8* data);

// Get the base from an existing array
DLLEXPORT bh_base_p bh_multi_array_uint8_get_base(const bh_multi_array_uint8_p self);

// Construct a new array from bh_base_p and view setup
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_new_from_base(const bh_base_p base);

// Construct a new array from bh_base_p and view setup
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_new_from_view(const bh_base_p base, uint64_t rank, const int64_t start, const int64_t* shape, const int64_t* stride);

// Construct a new empty array
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_new_empty(uint64_t rank, const int64_t* shape);

// Construct a new zero-filled array
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_new_zeroes(uint64_t rank, const int64_t* shape);

// Construct a new one-filled array
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_new_ones(uint64_t rank, const int64_t* shape);

// Construct a new array with sequential numbers
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_new_range(const int64_t start, const int64_t end, const int64_t skip);

// Construct a new random-filled array
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_new_random(const int64_t length);

// Construct a new array, filled with the specified value
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_new_value(const bh_uint8 value, uint64_t rank, const int64_t* shape);

// Construct a copy of the array
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_new_copy(bh_multi_array_uint8_p other);

// Destroy the pointer and release resources
DLLEXPORT void bh_multi_array_uint8_destroy(bh_multi_array_uint8_p self);

// Gets the number of elements in the array
DLLEXPORT uint64_t bh_multi_array_uint8_get_length(bh_multi_array_uint8_p self);

// Gets the number of dimensions in the array
DLLEXPORT uint64_t bh_multi_array_uint8_get_rank(bh_multi_array_uint8_p self);

// Gets the number of elements in the dimension
DLLEXPORT uint64_t bh_multi_array_uint8_get_dimension_size(bh_multi_array_uint8_p self, const int64_t dimension);

// Update with a scalar
DLLEXPORT void bh_multi_array_uint8_assign_scalar(bh_multi_array_uint8_p self, const bh_uint8 value);

// Update with an array
DLLEXPORT void bh_multi_array_uint8_assign_array(bh_multi_array_uint8_p self, bh_multi_array_uint8_p other);

// Flatten view
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_flatten(bh_multi_array_uint8_p self);

// Transpose view
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_transpose(bh_multi_array_uint8_p self);

// Partial add reduction
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_partial_reduce_add(bh_multi_array_uint8_p self, const int64_t axis);

// Partial multiply reduction
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_partial_reduce_multiply(bh_multi_array_uint8_p self, const int64_t axis);

// Partial min reduction
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_partial_reduce_min(bh_multi_array_uint8_p self, const int64_t axis);

// Partial max reduction
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_partial_reduce_max(bh_multi_array_uint8_p self, const int64_t axis);

// Partial logical_and reduction
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_partial_reduce_logical_and(bh_multi_array_uint8_p self, const int64_t axis);

// Partial logical_or reduction
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_partial_reduce_logical_or(bh_multi_array_uint8_p self, const int64_t axis);

// Partial logical_xor reduction
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_partial_reduce_logical_xor(bh_multi_array_uint8_p self, const int64_t axis);

// Partial bitwise_and reduction
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_partial_reduce_bitwise_and(bh_multi_array_uint8_p self, const int64_t axis);

// Partial bitwise_or reduction
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_partial_reduce_bitwise_or(bh_multi_array_uint8_p self, const int64_t axis);

// Partial bitwise_xor reduction
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_partial_reduce_bitwise_xor(bh_multi_array_uint8_p self, const int64_t axis);


// Sum
DLLEXPORT bh_uint8 bh_multi_array_uint8_sum(bh_multi_array_uint8_p self);

// Product
DLLEXPORT bh_uint8 bh_multi_array_uint8_product(bh_multi_array_uint8_p self);


// Max
DLLEXPORT bh_uint8 bh_multi_array_uint8_max(bh_multi_array_uint8_p self);

// Min
DLLEXPORT bh_uint8 bh_multi_array_uint8_min(bh_multi_array_uint8_p self);





// Forward definitions
struct bh_multi_array_uint16;
struct bh_slice_uint16;

// Shorthand pointer defs
typedef struct bh_multi_array_uint16* bh_multi_array_uint16_p;
typedef struct bh_slice_range_uint16* bh_slice_range_uint16_p;

// Sync the current base
void bh_multi_array_uint16_sync(const bh_multi_array_uint16_p self);

// Sets the temp status of an array
DLLEXPORT void bh_multi_array_uint16_set_temp(const bh_multi_array_uint16_p self, bh_bool temp);

// Gets the temp status of an array
DLLEXPORT bh_bool bh_multi_array_uint16_get_temp(const bh_multi_array_uint16_p self);

// Create a base pointer from existing data
DLLEXPORT bh_base_p bh_multi_array_uint16_create_base(bh_uint16* data, int64_t nelem);

// Destroy a base pointer
DLLEXPORT void bh_multi_array_uint16_destroy_base(bh_base_p base);

// Gets the data pointer from a base
DLLEXPORT bh_uint16* bh_multi_array_uint16_get_base_data(bh_base_p base);

// Gets the number of elements in a base
DLLEXPORT int64_t bh_multi_array_uint16_get_base_nelem(bh_base_p base);

// Sets the data pointer for a base
DLLEXPORT void bh_multi_array_uint16_set_base_data(bh_base_p base, bh_uint16* data);

// Get the base from an existing array
DLLEXPORT bh_base_p bh_multi_array_uint16_get_base(const bh_multi_array_uint16_p self);

// Construct a new array from bh_base_p and view setup
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_new_from_base(const bh_base_p base);

// Construct a new array from bh_base_p and view setup
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_new_from_view(const bh_base_p base, uint64_t rank, const int64_t start, const int64_t* shape, const int64_t* stride);

// Construct a new empty array
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_new_empty(uint64_t rank, const int64_t* shape);

// Construct a new zero-filled array
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_new_zeroes(uint64_t rank, const int64_t* shape);

// Construct a new one-filled array
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_new_ones(uint64_t rank, const int64_t* shape);

// Construct a new array with sequential numbers
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_new_range(const int64_t start, const int64_t end, const int64_t skip);

// Construct a new random-filled array
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_new_random(const int64_t length);

// Construct a new array, filled with the specified value
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_new_value(const bh_uint16 value, uint64_t rank, const int64_t* shape);

// Construct a copy of the array
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_new_copy(bh_multi_array_uint16_p other);

// Destroy the pointer and release resources
DLLEXPORT void bh_multi_array_uint16_destroy(bh_multi_array_uint16_p self);

// Gets the number of elements in the array
DLLEXPORT uint64_t bh_multi_array_uint16_get_length(bh_multi_array_uint16_p self);

// Gets the number of dimensions in the array
DLLEXPORT uint64_t bh_multi_array_uint16_get_rank(bh_multi_array_uint16_p self);

// Gets the number of elements in the dimension
DLLEXPORT uint64_t bh_multi_array_uint16_get_dimension_size(bh_multi_array_uint16_p self, const int64_t dimension);

// Update with a scalar
DLLEXPORT void bh_multi_array_uint16_assign_scalar(bh_multi_array_uint16_p self, const bh_uint16 value);

// Update with an array
DLLEXPORT void bh_multi_array_uint16_assign_array(bh_multi_array_uint16_p self, bh_multi_array_uint16_p other);

// Flatten view
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_flatten(bh_multi_array_uint16_p self);

// Transpose view
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_transpose(bh_multi_array_uint16_p self);

// Partial add reduction
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_partial_reduce_add(bh_multi_array_uint16_p self, const int64_t axis);

// Partial multiply reduction
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_partial_reduce_multiply(bh_multi_array_uint16_p self, const int64_t axis);

// Partial min reduction
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_partial_reduce_min(bh_multi_array_uint16_p self, const int64_t axis);

// Partial max reduction
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_partial_reduce_max(bh_multi_array_uint16_p self, const int64_t axis);

// Partial logical_and reduction
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_partial_reduce_logical_and(bh_multi_array_uint16_p self, const int64_t axis);

// Partial logical_or reduction
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_partial_reduce_logical_or(bh_multi_array_uint16_p self, const int64_t axis);

// Partial logical_xor reduction
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_partial_reduce_logical_xor(bh_multi_array_uint16_p self, const int64_t axis);

// Partial bitwise_and reduction
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_partial_reduce_bitwise_and(bh_multi_array_uint16_p self, const int64_t axis);

// Partial bitwise_or reduction
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_partial_reduce_bitwise_or(bh_multi_array_uint16_p self, const int64_t axis);

// Partial bitwise_xor reduction
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_partial_reduce_bitwise_xor(bh_multi_array_uint16_p self, const int64_t axis);


// Sum
DLLEXPORT bh_uint16 bh_multi_array_uint16_sum(bh_multi_array_uint16_p self);

// Product
DLLEXPORT bh_uint16 bh_multi_array_uint16_product(bh_multi_array_uint16_p self);


// Max
DLLEXPORT bh_uint16 bh_multi_array_uint16_max(bh_multi_array_uint16_p self);

// Min
DLLEXPORT bh_uint16 bh_multi_array_uint16_min(bh_multi_array_uint16_p self);





// Forward definitions
struct bh_multi_array_uint32;
struct bh_slice_uint32;

// Shorthand pointer defs
typedef struct bh_multi_array_uint32* bh_multi_array_uint32_p;
typedef struct bh_slice_range_uint32* bh_slice_range_uint32_p;

// Sync the current base
void bh_multi_array_uint32_sync(const bh_multi_array_uint32_p self);

// Sets the temp status of an array
DLLEXPORT void bh_multi_array_uint32_set_temp(const bh_multi_array_uint32_p self, bh_bool temp);

// Gets the temp status of an array
DLLEXPORT bh_bool bh_multi_array_uint32_get_temp(const bh_multi_array_uint32_p self);

// Create a base pointer from existing data
DLLEXPORT bh_base_p bh_multi_array_uint32_create_base(bh_uint32* data, int64_t nelem);

// Destroy a base pointer
DLLEXPORT void bh_multi_array_uint32_destroy_base(bh_base_p base);

// Gets the data pointer from a base
DLLEXPORT bh_uint32* bh_multi_array_uint32_get_base_data(bh_base_p base);

// Gets the number of elements in a base
DLLEXPORT int64_t bh_multi_array_uint32_get_base_nelem(bh_base_p base);

// Sets the data pointer for a base
DLLEXPORT void bh_multi_array_uint32_set_base_data(bh_base_p base, bh_uint32* data);

// Get the base from an existing array
DLLEXPORT bh_base_p bh_multi_array_uint32_get_base(const bh_multi_array_uint32_p self);

// Construct a new array from bh_base_p and view setup
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_new_from_base(const bh_base_p base);

// Construct a new array from bh_base_p and view setup
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_new_from_view(const bh_base_p base, uint64_t rank, const int64_t start, const int64_t* shape, const int64_t* stride);

// Construct a new empty array
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_new_empty(uint64_t rank, const int64_t* shape);

// Construct a new zero-filled array
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_new_zeroes(uint64_t rank, const int64_t* shape);

// Construct a new one-filled array
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_new_ones(uint64_t rank, const int64_t* shape);

// Construct a new array with sequential numbers
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_new_range(const int64_t start, const int64_t end, const int64_t skip);

// Construct a new random-filled array
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_new_random(const int64_t length);

// Construct a new array, filled with the specified value
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_new_value(const bh_uint32 value, uint64_t rank, const int64_t* shape);

// Construct a copy of the array
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_new_copy(bh_multi_array_uint32_p other);

// Destroy the pointer and release resources
DLLEXPORT void bh_multi_array_uint32_destroy(bh_multi_array_uint32_p self);

// Gets the number of elements in the array
DLLEXPORT uint64_t bh_multi_array_uint32_get_length(bh_multi_array_uint32_p self);

// Gets the number of dimensions in the array
DLLEXPORT uint64_t bh_multi_array_uint32_get_rank(bh_multi_array_uint32_p self);

// Gets the number of elements in the dimension
DLLEXPORT uint64_t bh_multi_array_uint32_get_dimension_size(bh_multi_array_uint32_p self, const int64_t dimension);

// Update with a scalar
DLLEXPORT void bh_multi_array_uint32_assign_scalar(bh_multi_array_uint32_p self, const bh_uint32 value);

// Update with an array
DLLEXPORT void bh_multi_array_uint32_assign_array(bh_multi_array_uint32_p self, bh_multi_array_uint32_p other);

// Flatten view
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_flatten(bh_multi_array_uint32_p self);

// Transpose view
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_transpose(bh_multi_array_uint32_p self);

// Partial add reduction
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_partial_reduce_add(bh_multi_array_uint32_p self, const int64_t axis);

// Partial multiply reduction
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_partial_reduce_multiply(bh_multi_array_uint32_p self, const int64_t axis);

// Partial min reduction
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_partial_reduce_min(bh_multi_array_uint32_p self, const int64_t axis);

// Partial max reduction
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_partial_reduce_max(bh_multi_array_uint32_p self, const int64_t axis);

// Partial logical_and reduction
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_partial_reduce_logical_and(bh_multi_array_uint32_p self, const int64_t axis);

// Partial logical_or reduction
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_partial_reduce_logical_or(bh_multi_array_uint32_p self, const int64_t axis);

// Partial logical_xor reduction
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_partial_reduce_logical_xor(bh_multi_array_uint32_p self, const int64_t axis);

// Partial bitwise_and reduction
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_partial_reduce_bitwise_and(bh_multi_array_uint32_p self, const int64_t axis);

// Partial bitwise_or reduction
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_partial_reduce_bitwise_or(bh_multi_array_uint32_p self, const int64_t axis);

// Partial bitwise_xor reduction
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_partial_reduce_bitwise_xor(bh_multi_array_uint32_p self, const int64_t axis);


// Sum
DLLEXPORT bh_uint32 bh_multi_array_uint32_sum(bh_multi_array_uint32_p self);

// Product
DLLEXPORT bh_uint32 bh_multi_array_uint32_product(bh_multi_array_uint32_p self);


// Max
DLLEXPORT bh_uint32 bh_multi_array_uint32_max(bh_multi_array_uint32_p self);

// Min
DLLEXPORT bh_uint32 bh_multi_array_uint32_min(bh_multi_array_uint32_p self);





// Forward definitions
struct bh_multi_array_uint64;
struct bh_slice_uint64;

// Shorthand pointer defs
typedef struct bh_multi_array_uint64* bh_multi_array_uint64_p;
typedef struct bh_slice_range_uint64* bh_slice_range_uint64_p;

// Sync the current base
void bh_multi_array_uint64_sync(const bh_multi_array_uint64_p self);

// Sets the temp status of an array
DLLEXPORT void bh_multi_array_uint64_set_temp(const bh_multi_array_uint64_p self, bh_bool temp);

// Gets the temp status of an array
DLLEXPORT bh_bool bh_multi_array_uint64_get_temp(const bh_multi_array_uint64_p self);

// Create a base pointer from existing data
DLLEXPORT bh_base_p bh_multi_array_uint64_create_base(bh_uint64* data, int64_t nelem);

// Destroy a base pointer
DLLEXPORT void bh_multi_array_uint64_destroy_base(bh_base_p base);

// Gets the data pointer from a base
DLLEXPORT bh_uint64* bh_multi_array_uint64_get_base_data(bh_base_p base);

// Gets the number of elements in a base
DLLEXPORT int64_t bh_multi_array_uint64_get_base_nelem(bh_base_p base);

// Sets the data pointer for a base
DLLEXPORT void bh_multi_array_uint64_set_base_data(bh_base_p base, bh_uint64* data);

// Get the base from an existing array
DLLEXPORT bh_base_p bh_multi_array_uint64_get_base(const bh_multi_array_uint64_p self);

// Construct a new array from bh_base_p and view setup
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_new_from_base(const bh_base_p base);

// Construct a new array from bh_base_p and view setup
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_new_from_view(const bh_base_p base, uint64_t rank, const int64_t start, const int64_t* shape, const int64_t* stride);

// Construct a new empty array
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_new_empty(uint64_t rank, const int64_t* shape);

// Construct a new zero-filled array
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_new_zeroes(uint64_t rank, const int64_t* shape);

// Construct a new one-filled array
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_new_ones(uint64_t rank, const int64_t* shape);

// Construct a new array with sequential numbers
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_new_range(const int64_t start, const int64_t end, const int64_t skip);

// Construct a new random-filled array
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_new_random(const int64_t length);

// Construct a new array, filled with the specified value
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_new_value(const bh_uint64 value, uint64_t rank, const int64_t* shape);

// Construct a copy of the array
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_new_copy(bh_multi_array_uint64_p other);

// Destroy the pointer and release resources
DLLEXPORT void bh_multi_array_uint64_destroy(bh_multi_array_uint64_p self);

// Gets the number of elements in the array
DLLEXPORT uint64_t bh_multi_array_uint64_get_length(bh_multi_array_uint64_p self);

// Gets the number of dimensions in the array
DLLEXPORT uint64_t bh_multi_array_uint64_get_rank(bh_multi_array_uint64_p self);

// Gets the number of elements in the dimension
DLLEXPORT uint64_t bh_multi_array_uint64_get_dimension_size(bh_multi_array_uint64_p self, const int64_t dimension);

// Update with a scalar
DLLEXPORT void bh_multi_array_uint64_assign_scalar(bh_multi_array_uint64_p self, const bh_uint64 value);

// Update with an array
DLLEXPORT void bh_multi_array_uint64_assign_array(bh_multi_array_uint64_p self, bh_multi_array_uint64_p other);

// Flatten view
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_flatten(bh_multi_array_uint64_p self);

// Transpose view
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_transpose(bh_multi_array_uint64_p self);

// Partial add reduction
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_partial_reduce_add(bh_multi_array_uint64_p self, const int64_t axis);

// Partial multiply reduction
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_partial_reduce_multiply(bh_multi_array_uint64_p self, const int64_t axis);

// Partial min reduction
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_partial_reduce_min(bh_multi_array_uint64_p self, const int64_t axis);

// Partial max reduction
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_partial_reduce_max(bh_multi_array_uint64_p self, const int64_t axis);

// Partial logical_and reduction
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_partial_reduce_logical_and(bh_multi_array_uint64_p self, const int64_t axis);

// Partial logical_or reduction
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_partial_reduce_logical_or(bh_multi_array_uint64_p self, const int64_t axis);

// Partial logical_xor reduction
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_partial_reduce_logical_xor(bh_multi_array_uint64_p self, const int64_t axis);

// Partial bitwise_and reduction
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_partial_reduce_bitwise_and(bh_multi_array_uint64_p self, const int64_t axis);

// Partial bitwise_or reduction
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_partial_reduce_bitwise_or(bh_multi_array_uint64_p self, const int64_t axis);

// Partial bitwise_xor reduction
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_partial_reduce_bitwise_xor(bh_multi_array_uint64_p self, const int64_t axis);


// Sum
DLLEXPORT bh_uint64 bh_multi_array_uint64_sum(bh_multi_array_uint64_p self);

// Product
DLLEXPORT bh_uint64 bh_multi_array_uint64_product(bh_multi_array_uint64_p self);


// Max
DLLEXPORT bh_uint64 bh_multi_array_uint64_max(bh_multi_array_uint64_p self);

// Min
DLLEXPORT bh_uint64 bh_multi_array_uint64_min(bh_multi_array_uint64_p self);





// Forward definitions
struct bh_multi_array_float32;
struct bh_slice_float32;

// Shorthand pointer defs
typedef struct bh_multi_array_float32* bh_multi_array_float32_p;
typedef struct bh_slice_range_float32* bh_slice_range_float32_p;

// Sync the current base
void bh_multi_array_float32_sync(const bh_multi_array_float32_p self);

// Sets the temp status of an array
DLLEXPORT void bh_multi_array_float32_set_temp(const bh_multi_array_float32_p self, bh_bool temp);

// Gets the temp status of an array
DLLEXPORT bh_bool bh_multi_array_float32_get_temp(const bh_multi_array_float32_p self);

// Create a base pointer from existing data
DLLEXPORT bh_base_p bh_multi_array_float32_create_base(bh_float32* data, int64_t nelem);

// Destroy a base pointer
DLLEXPORT void bh_multi_array_float32_destroy_base(bh_base_p base);

// Gets the data pointer from a base
DLLEXPORT bh_float32* bh_multi_array_float32_get_base_data(bh_base_p base);

// Gets the number of elements in a base
DLLEXPORT int64_t bh_multi_array_float32_get_base_nelem(bh_base_p base);

// Sets the data pointer for a base
DLLEXPORT void bh_multi_array_float32_set_base_data(bh_base_p base, bh_float32* data);

// Get the base from an existing array
DLLEXPORT bh_base_p bh_multi_array_float32_get_base(const bh_multi_array_float32_p self);

// Construct a new array from bh_base_p and view setup
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_new_from_base(const bh_base_p base);

// Construct a new array from bh_base_p and view setup
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_new_from_view(const bh_base_p base, uint64_t rank, const int64_t start, const int64_t* shape, const int64_t* stride);

// Construct a new empty array
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_new_empty(uint64_t rank, const int64_t* shape);

// Construct a new zero-filled array
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_new_zeroes(uint64_t rank, const int64_t* shape);

// Construct a new one-filled array
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_new_ones(uint64_t rank, const int64_t* shape);

// Construct a new array with sequential numbers
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_new_range(const int64_t start, const int64_t end, const int64_t skip);

// Construct a new random-filled array
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_new_random(const int64_t length);

// Construct a new array, filled with the specified value
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_new_value(const bh_float32 value, uint64_t rank, const int64_t* shape);

// Construct a copy of the array
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_new_copy(bh_multi_array_float32_p other);

// Destroy the pointer and release resources
DLLEXPORT void bh_multi_array_float32_destroy(bh_multi_array_float32_p self);

// Gets the number of elements in the array
DLLEXPORT uint64_t bh_multi_array_float32_get_length(bh_multi_array_float32_p self);

// Gets the number of dimensions in the array
DLLEXPORT uint64_t bh_multi_array_float32_get_rank(bh_multi_array_float32_p self);

// Gets the number of elements in the dimension
DLLEXPORT uint64_t bh_multi_array_float32_get_dimension_size(bh_multi_array_float32_p self, const int64_t dimension);

// Update with a scalar
DLLEXPORT void bh_multi_array_float32_assign_scalar(bh_multi_array_float32_p self, const bh_float32 value);

// Update with an array
DLLEXPORT void bh_multi_array_float32_assign_array(bh_multi_array_float32_p self, bh_multi_array_float32_p other);

// Flatten view
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_flatten(bh_multi_array_float32_p self);

// Transpose view
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_transpose(bh_multi_array_float32_p self);

// Partial add reduction
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_partial_reduce_add(bh_multi_array_float32_p self, const int64_t axis);

// Partial multiply reduction
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_partial_reduce_multiply(bh_multi_array_float32_p self, const int64_t axis);

// Partial min reduction
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_partial_reduce_min(bh_multi_array_float32_p self, const int64_t axis);

// Partial max reduction
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_partial_reduce_max(bh_multi_array_float32_p self, const int64_t axis);

// Partial logical_and reduction
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_partial_reduce_logical_and(bh_multi_array_float32_p self, const int64_t axis);

// Partial logical_or reduction
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_partial_reduce_logical_or(bh_multi_array_float32_p self, const int64_t axis);

// Partial logical_xor reduction
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_partial_reduce_logical_xor(bh_multi_array_float32_p self, const int64_t axis);

// Partial bitwise_and reduction
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_partial_reduce_bitwise_and(bh_multi_array_float32_p self, const int64_t axis);

// Partial bitwise_or reduction
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_partial_reduce_bitwise_or(bh_multi_array_float32_p self, const int64_t axis);

// Partial bitwise_xor reduction
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_partial_reduce_bitwise_xor(bh_multi_array_float32_p self, const int64_t axis);


// Sum
DLLEXPORT bh_float32 bh_multi_array_float32_sum(bh_multi_array_float32_p self);

// Product
DLLEXPORT bh_float32 bh_multi_array_float32_product(bh_multi_array_float32_p self);


// Max
DLLEXPORT bh_float32 bh_multi_array_float32_max(bh_multi_array_float32_p self);

// Min
DLLEXPORT bh_float32 bh_multi_array_float32_min(bh_multi_array_float32_p self);





// Forward definitions
struct bh_multi_array_float64;
struct bh_slice_float64;

// Shorthand pointer defs
typedef struct bh_multi_array_float64* bh_multi_array_float64_p;
typedef struct bh_slice_range_float64* bh_slice_range_float64_p;

// Sync the current base
void bh_multi_array_float64_sync(const bh_multi_array_float64_p self);

// Sets the temp status of an array
DLLEXPORT void bh_multi_array_float64_set_temp(const bh_multi_array_float64_p self, bh_bool temp);

// Gets the temp status of an array
DLLEXPORT bh_bool bh_multi_array_float64_get_temp(const bh_multi_array_float64_p self);

// Create a base pointer from existing data
DLLEXPORT bh_base_p bh_multi_array_float64_create_base(bh_float64* data, int64_t nelem);

// Destroy a base pointer
DLLEXPORT void bh_multi_array_float64_destroy_base(bh_base_p base);

// Gets the data pointer from a base
DLLEXPORT bh_float64* bh_multi_array_float64_get_base_data(bh_base_p base);

// Gets the number of elements in a base
DLLEXPORT int64_t bh_multi_array_float64_get_base_nelem(bh_base_p base);

// Sets the data pointer for a base
DLLEXPORT void bh_multi_array_float64_set_base_data(bh_base_p base, bh_float64* data);

// Get the base from an existing array
DLLEXPORT bh_base_p bh_multi_array_float64_get_base(const bh_multi_array_float64_p self);

// Construct a new array from bh_base_p and view setup
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_new_from_base(const bh_base_p base);

// Construct a new array from bh_base_p and view setup
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_new_from_view(const bh_base_p base, uint64_t rank, const int64_t start, const int64_t* shape, const int64_t* stride);

// Construct a new empty array
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_new_empty(uint64_t rank, const int64_t* shape);

// Construct a new zero-filled array
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_new_zeroes(uint64_t rank, const int64_t* shape);

// Construct a new one-filled array
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_new_ones(uint64_t rank, const int64_t* shape);

// Construct a new array with sequential numbers
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_new_range(const int64_t start, const int64_t end, const int64_t skip);

// Construct a new random-filled array
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_new_random(const int64_t length);

// Construct a new array, filled with the specified value
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_new_value(const bh_float64 value, uint64_t rank, const int64_t* shape);

// Construct a copy of the array
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_new_copy(bh_multi_array_float64_p other);

// Destroy the pointer and release resources
DLLEXPORT void bh_multi_array_float64_destroy(bh_multi_array_float64_p self);

// Gets the number of elements in the array
DLLEXPORT uint64_t bh_multi_array_float64_get_length(bh_multi_array_float64_p self);

// Gets the number of dimensions in the array
DLLEXPORT uint64_t bh_multi_array_float64_get_rank(bh_multi_array_float64_p self);

// Gets the number of elements in the dimension
DLLEXPORT uint64_t bh_multi_array_float64_get_dimension_size(bh_multi_array_float64_p self, const int64_t dimension);

// Update with a scalar
DLLEXPORT void bh_multi_array_float64_assign_scalar(bh_multi_array_float64_p self, const bh_float64 value);

// Update with an array
DLLEXPORT void bh_multi_array_float64_assign_array(bh_multi_array_float64_p self, bh_multi_array_float64_p other);

// Flatten view
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_flatten(bh_multi_array_float64_p self);

// Transpose view
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_transpose(bh_multi_array_float64_p self);

// Partial add reduction
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_partial_reduce_add(bh_multi_array_float64_p self, const int64_t axis);

// Partial multiply reduction
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_partial_reduce_multiply(bh_multi_array_float64_p self, const int64_t axis);

// Partial min reduction
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_partial_reduce_min(bh_multi_array_float64_p self, const int64_t axis);

// Partial max reduction
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_partial_reduce_max(bh_multi_array_float64_p self, const int64_t axis);

// Partial logical_and reduction
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_partial_reduce_logical_and(bh_multi_array_float64_p self, const int64_t axis);

// Partial logical_or reduction
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_partial_reduce_logical_or(bh_multi_array_float64_p self, const int64_t axis);

// Partial logical_xor reduction
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_partial_reduce_logical_xor(bh_multi_array_float64_p self, const int64_t axis);

// Partial bitwise_and reduction
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_partial_reduce_bitwise_and(bh_multi_array_float64_p self, const int64_t axis);

// Partial bitwise_or reduction
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_partial_reduce_bitwise_or(bh_multi_array_float64_p self, const int64_t axis);

// Partial bitwise_xor reduction
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_partial_reduce_bitwise_xor(bh_multi_array_float64_p self, const int64_t axis);


// Sum
DLLEXPORT bh_float64 bh_multi_array_float64_sum(bh_multi_array_float64_p self);

// Product
DLLEXPORT bh_float64 bh_multi_array_float64_product(bh_multi_array_float64_p self);


// Max
DLLEXPORT bh_float64 bh_multi_array_float64_max(bh_multi_array_float64_p self);

// Min
DLLEXPORT bh_float64 bh_multi_array_float64_min(bh_multi_array_float64_p self);





// Forward definitions
struct bh_multi_array_complex64;
struct bh_slice_complex64;

// Shorthand pointer defs
typedef struct bh_multi_array_complex64* bh_multi_array_complex64_p;
typedef struct bh_slice_range_complex64* bh_slice_range_complex64_p;

// Sync the current base
void bh_multi_array_complex64_sync(const bh_multi_array_complex64_p self);

// Sets the temp status of an array
DLLEXPORT void bh_multi_array_complex64_set_temp(const bh_multi_array_complex64_p self, bh_bool temp);

// Gets the temp status of an array
DLLEXPORT bh_bool bh_multi_array_complex64_get_temp(const bh_multi_array_complex64_p self);

// Create a base pointer from existing data
DLLEXPORT bh_base_p bh_multi_array_complex64_create_base(bh_complex64* data, int64_t nelem);

// Destroy a base pointer
DLLEXPORT void bh_multi_array_complex64_destroy_base(bh_base_p base);

// Gets the data pointer from a base
DLLEXPORT bh_complex64* bh_multi_array_complex64_get_base_data(bh_base_p base);

// Gets the number of elements in a base
DLLEXPORT int64_t bh_multi_array_complex64_get_base_nelem(bh_base_p base);

// Sets the data pointer for a base
DLLEXPORT void bh_multi_array_complex64_set_base_data(bh_base_p base, bh_complex64* data);

// Get the base from an existing array
DLLEXPORT bh_base_p bh_multi_array_complex64_get_base(const bh_multi_array_complex64_p self);

// Construct a new array from bh_base_p and view setup
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_new_from_base(const bh_base_p base);

// Construct a new array from bh_base_p and view setup
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_new_from_view(const bh_base_p base, uint64_t rank, const int64_t start, const int64_t* shape, const int64_t* stride);

// Construct a new empty array
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_new_empty(uint64_t rank, const int64_t* shape);

// Construct a new zero-filled array
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_new_zeroes(uint64_t rank, const int64_t* shape);

// Construct a new one-filled array
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_new_ones(uint64_t rank, const int64_t* shape);

// Construct a new array with sequential numbers
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_new_range(const int64_t start, const int64_t end, const int64_t skip);

// Construct a new random-filled array
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_new_random(const int64_t length);

// Construct a new array, filled with the specified value
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_new_value(const bh_complex64 value, uint64_t rank, const int64_t* shape);

// Construct a copy of the array
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_new_copy(bh_multi_array_complex64_p other);

// Destroy the pointer and release resources
DLLEXPORT void bh_multi_array_complex64_destroy(bh_multi_array_complex64_p self);

// Gets the number of elements in the array
DLLEXPORT uint64_t bh_multi_array_complex64_get_length(bh_multi_array_complex64_p self);

// Gets the number of dimensions in the array
DLLEXPORT uint64_t bh_multi_array_complex64_get_rank(bh_multi_array_complex64_p self);

// Gets the number of elements in the dimension
DLLEXPORT uint64_t bh_multi_array_complex64_get_dimension_size(bh_multi_array_complex64_p self, const int64_t dimension);

// Update with a scalar
DLLEXPORT void bh_multi_array_complex64_assign_scalar(bh_multi_array_complex64_p self, const bh_complex64 value);

// Update with an array
DLLEXPORT void bh_multi_array_complex64_assign_array(bh_multi_array_complex64_p self, bh_multi_array_complex64_p other);

// Flatten view
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_flatten(bh_multi_array_complex64_p self);

// Transpose view
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_transpose(bh_multi_array_complex64_p self);

// Partial add reduction
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_partial_reduce_add(bh_multi_array_complex64_p self, const int64_t axis);

// Partial multiply reduction
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_partial_reduce_multiply(bh_multi_array_complex64_p self, const int64_t axis);

// Partial min reduction
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_partial_reduce_min(bh_multi_array_complex64_p self, const int64_t axis);

// Partial max reduction
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_partial_reduce_max(bh_multi_array_complex64_p self, const int64_t axis);

// Partial logical_and reduction
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_partial_reduce_logical_and(bh_multi_array_complex64_p self, const int64_t axis);

// Partial logical_or reduction
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_partial_reduce_logical_or(bh_multi_array_complex64_p self, const int64_t axis);

// Partial logical_xor reduction
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_partial_reduce_logical_xor(bh_multi_array_complex64_p self, const int64_t axis);

// Partial bitwise_and reduction
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_partial_reduce_bitwise_and(bh_multi_array_complex64_p self, const int64_t axis);

// Partial bitwise_or reduction
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_partial_reduce_bitwise_or(bh_multi_array_complex64_p self, const int64_t axis);

// Partial bitwise_xor reduction
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_partial_reduce_bitwise_xor(bh_multi_array_complex64_p self, const int64_t axis);


// Sum
DLLEXPORT bh_complex64 bh_multi_array_complex64_sum(bh_multi_array_complex64_p self);

// Product
DLLEXPORT bh_complex64 bh_multi_array_complex64_product(bh_multi_array_complex64_p self);




// Get the real component of a complex number
DLLEXPORT bh_multi_array_float32_p bh_multi_array_complex64_real(bh_multi_array_complex64_p self);

// Get the imaginary component of a complex number
DLLEXPORT bh_multi_array_float32_p bh_multi_array_complex64_imag(bh_multi_array_complex64_p self);




// Forward definitions
struct bh_multi_array_complex128;
struct bh_slice_complex128;

// Shorthand pointer defs
typedef struct bh_multi_array_complex128* bh_multi_array_complex128_p;
typedef struct bh_slice_range_complex128* bh_slice_range_complex128_p;

// Sync the current base
void bh_multi_array_complex128_sync(const bh_multi_array_complex128_p self);

// Sets the temp status of an array
DLLEXPORT void bh_multi_array_complex128_set_temp(const bh_multi_array_complex128_p self, bh_bool temp);

// Gets the temp status of an array
DLLEXPORT bh_bool bh_multi_array_complex128_get_temp(const bh_multi_array_complex128_p self);

// Create a base pointer from existing data
DLLEXPORT bh_base_p bh_multi_array_complex128_create_base(bh_complex128* data, int64_t nelem);

// Destroy a base pointer
DLLEXPORT void bh_multi_array_complex128_destroy_base(bh_base_p base);

// Gets the data pointer from a base
DLLEXPORT bh_complex128* bh_multi_array_complex128_get_base_data(bh_base_p base);

// Gets the number of elements in a base
DLLEXPORT int64_t bh_multi_array_complex128_get_base_nelem(bh_base_p base);

// Sets the data pointer for a base
DLLEXPORT void bh_multi_array_complex128_set_base_data(bh_base_p base, bh_complex128* data);

// Get the base from an existing array
DLLEXPORT bh_base_p bh_multi_array_complex128_get_base(const bh_multi_array_complex128_p self);

// Construct a new array from bh_base_p and view setup
DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_new_from_base(const bh_base_p base);

// Construct a new array from bh_base_p and view setup
DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_new_from_view(const bh_base_p base, uint64_t rank, const int64_t start, const int64_t* shape, const int64_t* stride);

// Construct a new empty array
DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_new_empty(uint64_t rank, const int64_t* shape);

// Construct a new zero-filled array
DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_new_zeroes(uint64_t rank, const int64_t* shape);

// Construct a new one-filled array
DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_new_ones(uint64_t rank, const int64_t* shape);

// Construct a new array with sequential numbers
DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_new_range(const int64_t start, const int64_t end, const int64_t skip);

// Construct a new random-filled array
DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_new_random(const int64_t length);

// Construct a new array, filled with the specified value
DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_new_value(const bh_complex128 value, uint64_t rank, const int64_t* shape);

// Construct a copy of the array
DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_new_copy(bh_multi_array_complex128_p other);

// Destroy the pointer and release resources
DLLEXPORT void bh_multi_array_complex128_destroy(bh_multi_array_complex128_p self);

// Gets the number of elements in the array
DLLEXPORT uint64_t bh_multi_array_complex128_get_length(bh_multi_array_complex128_p self);

// Gets the number of dimensions in the array
DLLEXPORT uint64_t bh_multi_array_complex128_get_rank(bh_multi_array_complex128_p self);

// Gets the number of elements in the dimension
DLLEXPORT uint64_t bh_multi_array_complex128_get_dimension_size(bh_multi_array_complex128_p self, const int64_t dimension);

// Update with a scalar
DLLEXPORT void bh_multi_array_complex128_assign_scalar(bh_multi_array_complex128_p self, const bh_complex128 value);

// Update with an array
DLLEXPORT void bh_multi_array_complex128_assign_array(bh_multi_array_complex128_p self, bh_multi_array_complex128_p other);

// Flatten view
DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_flatten(bh_multi_array_complex128_p self);

// Transpose view
DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_transpose(bh_multi_array_complex128_p self);

// Partial add reduction
DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_partial_reduce_add(bh_multi_array_complex128_p self, const int64_t axis);

// Partial multiply reduction
DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_partial_reduce_multiply(bh_multi_array_complex128_p self, const int64_t axis);

// Partial min reduction
DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_partial_reduce_min(bh_multi_array_complex128_p self, const int64_t axis);

// Partial max reduction
DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_partial_reduce_max(bh_multi_array_complex128_p self, const int64_t axis);

// Partial logical_and reduction
DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_partial_reduce_logical_and(bh_multi_array_complex128_p self, const int64_t axis);

// Partial logical_or reduction
DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_partial_reduce_logical_or(bh_multi_array_complex128_p self, const int64_t axis);

// Partial logical_xor reduction
DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_partial_reduce_logical_xor(bh_multi_array_complex128_p self, const int64_t axis);

// Partial bitwise_and reduction
DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_partial_reduce_bitwise_and(bh_multi_array_complex128_p self, const int64_t axis);

// Partial bitwise_or reduction
DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_partial_reduce_bitwise_or(bh_multi_array_complex128_p self, const int64_t axis);

// Partial bitwise_xor reduction
DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_partial_reduce_bitwise_xor(bh_multi_array_complex128_p self, const int64_t axis);


// Sum
DLLEXPORT bh_complex128 bh_multi_array_complex128_sum(bh_multi_array_complex128_p self);

// Product
DLLEXPORT bh_complex128 bh_multi_array_complex128_product(bh_multi_array_complex128_p self);





// Get the real component of a complex number
DLLEXPORT bh_multi_array_float64_p bh_multi_array_complex128_real(bh_multi_array_complex128_p self);

// Get the imaginary component of a complex number
DLLEXPORT bh_multi_array_float64_p bh_multi_array_complex128_imag(bh_multi_array_complex128_p self);



#ifdef __cplusplus
}
#endif

#endif

#ifndef __BH_C_INTERFACE_H
#define __BH_C_INTERFACE_H

#include <stdint.h>
#include <bh_type.h>

#ifdef _WIN32
#define DLLEXPORT __declspec( dllexport )
#else
#define DLLEXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Copy methods

DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_convert_bool8(bh_multi_array_bool8_p other);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_convert_float32(bh_multi_array_float32_p other);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_convert_float64(bh_multi_array_float64_p other);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_convert_int16(bh_multi_array_int16_p other);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_convert_int32(bh_multi_array_int32_p other);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_convert_int64(bh_multi_array_int64_p other);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_convert_int8(bh_multi_array_int8_p other);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_convert_uint16(bh_multi_array_uint16_p other);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_convert_uint32(bh_multi_array_uint32_p other);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_convert_uint64(bh_multi_array_uint64_p other);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_convert_uint8(bh_multi_array_uint8_p other);
DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_convert_bool8(bh_multi_array_bool8_p other);
DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_convert_complex128(bh_multi_array_complex128_p other);
DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_convert_complex64(bh_multi_array_complex64_p other);
DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_convert_float32(bh_multi_array_float32_p other);
DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_convert_float64(bh_multi_array_float64_p other);
DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_convert_int16(bh_multi_array_int16_p other);
DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_convert_int32(bh_multi_array_int32_p other);
DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_convert_int64(bh_multi_array_int64_p other);
DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_convert_int8(bh_multi_array_int8_p other);
DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_convert_uint16(bh_multi_array_uint16_p other);
DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_convert_uint32(bh_multi_array_uint32_p other);
DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_convert_uint64(bh_multi_array_uint64_p other);
DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_convert_uint8(bh_multi_array_uint8_p other);
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_convert_bool8(bh_multi_array_bool8_p other);
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_convert_complex128(bh_multi_array_complex128_p other);
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_convert_complex64(bh_multi_array_complex64_p other);
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_convert_float32(bh_multi_array_float32_p other);
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_convert_float64(bh_multi_array_float64_p other);
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_convert_int16(bh_multi_array_int16_p other);
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_convert_int32(bh_multi_array_int32_p other);
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_convert_int64(bh_multi_array_int64_p other);
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_convert_int8(bh_multi_array_int8_p other);
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_convert_uint16(bh_multi_array_uint16_p other);
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_convert_uint32(bh_multi_array_uint32_p other);
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_convert_uint64(bh_multi_array_uint64_p other);
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_convert_uint8(bh_multi_array_uint8_p other);
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_convert_bool8(bh_multi_array_bool8_p other);
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_convert_float32(bh_multi_array_float32_p other);
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_convert_float64(bh_multi_array_float64_p other);
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_convert_int16(bh_multi_array_int16_p other);
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_convert_int32(bh_multi_array_int32_p other);
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_convert_int64(bh_multi_array_int64_p other);
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_convert_int8(bh_multi_array_int8_p other);
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_convert_uint16(bh_multi_array_uint16_p other);
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_convert_uint32(bh_multi_array_uint32_p other);
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_convert_uint64(bh_multi_array_uint64_p other);
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_convert_uint8(bh_multi_array_uint8_p other);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_convert_bool8(bh_multi_array_bool8_p other);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_convert_float32(bh_multi_array_float32_p other);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_convert_float64(bh_multi_array_float64_p other);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_convert_int16(bh_multi_array_int16_p other);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_convert_int32(bh_multi_array_int32_p other);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_convert_int64(bh_multi_array_int64_p other);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_convert_int8(bh_multi_array_int8_p other);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_convert_uint16(bh_multi_array_uint16_p other);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_convert_uint32(bh_multi_array_uint32_p other);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_convert_uint64(bh_multi_array_uint64_p other);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_convert_uint8(bh_multi_array_uint8_p other);
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_convert_bool8(bh_multi_array_bool8_p other);
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_convert_float32(bh_multi_array_float32_p other);
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_convert_float64(bh_multi_array_float64_p other);
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_convert_int16(bh_multi_array_int16_p other);
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_convert_int32(bh_multi_array_int32_p other);
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_convert_int64(bh_multi_array_int64_p other);
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_convert_int8(bh_multi_array_int8_p other);
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_convert_uint16(bh_multi_array_uint16_p other);
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_convert_uint32(bh_multi_array_uint32_p other);
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_convert_uint64(bh_multi_array_uint64_p other);
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_convert_uint8(bh_multi_array_uint8_p other);
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_convert_bool8(bh_multi_array_bool8_p other);
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_convert_float32(bh_multi_array_float32_p other);
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_convert_float64(bh_multi_array_float64_p other);
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_convert_int16(bh_multi_array_int16_p other);
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_convert_int32(bh_multi_array_int32_p other);
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_convert_int64(bh_multi_array_int64_p other);
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_convert_int8(bh_multi_array_int8_p other);
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_convert_uint16(bh_multi_array_uint16_p other);
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_convert_uint32(bh_multi_array_uint32_p other);
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_convert_uint64(bh_multi_array_uint64_p other);
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_convert_uint8(bh_multi_array_uint8_p other);
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_convert_bool8(bh_multi_array_bool8_p other);
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_convert_float32(bh_multi_array_float32_p other);
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_convert_float64(bh_multi_array_float64_p other);
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_convert_int16(bh_multi_array_int16_p other);
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_convert_int32(bh_multi_array_int32_p other);
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_convert_int64(bh_multi_array_int64_p other);
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_convert_int8(bh_multi_array_int8_p other);
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_convert_uint16(bh_multi_array_uint16_p other);
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_convert_uint32(bh_multi_array_uint32_p other);
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_convert_uint64(bh_multi_array_uint64_p other);
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_convert_uint8(bh_multi_array_uint8_p other);
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_convert_bool8(bh_multi_array_bool8_p other);
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_convert_float32(bh_multi_array_float32_p other);
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_convert_float64(bh_multi_array_float64_p other);
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_convert_int16(bh_multi_array_int16_p other);
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_convert_int32(bh_multi_array_int32_p other);
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_convert_int64(bh_multi_array_int64_p other);
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_convert_int8(bh_multi_array_int8_p other);
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_convert_uint16(bh_multi_array_uint16_p other);
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_convert_uint32(bh_multi_array_uint32_p other);
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_convert_uint64(bh_multi_array_uint64_p other);
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_convert_uint8(bh_multi_array_uint8_p other);
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_convert_bool8(bh_multi_array_bool8_p other);
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_convert_float32(bh_multi_array_float32_p other);
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_convert_float64(bh_multi_array_float64_p other);
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_convert_int16(bh_multi_array_int16_p other);
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_convert_int32(bh_multi_array_int32_p other);
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_convert_int64(bh_multi_array_int64_p other);
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_convert_int8(bh_multi_array_int8_p other);
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_convert_uint16(bh_multi_array_uint16_p other);
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_convert_uint32(bh_multi_array_uint32_p other);
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_convert_uint64(bh_multi_array_uint64_p other);
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_convert_uint8(bh_multi_array_uint8_p other);
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_convert_bool8(bh_multi_array_bool8_p other);
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_convert_float32(bh_multi_array_float32_p other);
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_convert_float64(bh_multi_array_float64_p other);
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_convert_int16(bh_multi_array_int16_p other);
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_convert_int32(bh_multi_array_int32_p other);
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_convert_int64(bh_multi_array_int64_p other);
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_convert_int8(bh_multi_array_int8_p other);
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_convert_uint16(bh_multi_array_uint16_p other);
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_convert_uint32(bh_multi_array_uint32_p other);
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_convert_uint64(bh_multi_array_uint64_p other);
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_convert_uint8(bh_multi_array_uint8_p other);
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_convert_bool8(bh_multi_array_bool8_p other);
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_convert_float32(bh_multi_array_float32_p other);
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_convert_float64(bh_multi_array_float64_p other);
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_convert_int16(bh_multi_array_int16_p other);
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_convert_int32(bh_multi_array_int32_p other);
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_convert_int64(bh_multi_array_int64_p other);
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_convert_int8(bh_multi_array_int8_p other);
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_convert_uint16(bh_multi_array_uint16_p other);
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_convert_uint32(bh_multi_array_uint32_p other);
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_convert_uint64(bh_multi_array_uint64_p other);
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_convert_uint8(bh_multi_array_uint8_p other);
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_convert_bool8(bh_multi_array_bool8_p other);
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_convert_float32(bh_multi_array_float32_p other);
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_convert_float64(bh_multi_array_float64_p other);
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_convert_int16(bh_multi_array_int16_p other);
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_convert_int32(bh_multi_array_int32_p other);
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_convert_int64(bh_multi_array_int64_p other);
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_convert_int8(bh_multi_array_int8_p other);
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_convert_uint16(bh_multi_array_uint16_p other);
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_convert_uint32(bh_multi_array_uint32_p other);
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_convert_uint64(bh_multi_array_uint64_p other);
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_convert_uint8(bh_multi_array_uint8_p other);


// Binary functions

DLLEXPORT void bh_multi_array_bool8_add_in_place(bh_multi_array_bool8_p self, bh_multi_array_bool8_p rhs);
DLLEXPORT void bh_multi_array_bool8_add_in_place_scalar_rhs(bh_multi_array_bool8_p self, bh_bool rhs);
DLLEXPORT void bh_multi_array_complex128_add_in_place(bh_multi_array_complex128_p self, bh_multi_array_complex128_p rhs);
DLLEXPORT void bh_multi_array_complex128_add_in_place_scalar_rhs(bh_multi_array_complex128_p self, bh_complex128 rhs);
DLLEXPORT void bh_multi_array_complex64_add_in_place(bh_multi_array_complex64_p self, bh_multi_array_complex64_p rhs);
DLLEXPORT void bh_multi_array_complex64_add_in_place_scalar_rhs(bh_multi_array_complex64_p self, bh_complex64 rhs);
DLLEXPORT void bh_multi_array_float32_add_in_place(bh_multi_array_float32_p self, bh_multi_array_float32_p rhs);
DLLEXPORT void bh_multi_array_float32_add_in_place_scalar_rhs(bh_multi_array_float32_p self, bh_float32 rhs);
DLLEXPORT void bh_multi_array_float64_add_in_place(bh_multi_array_float64_p self, bh_multi_array_float64_p rhs);
DLLEXPORT void bh_multi_array_float64_add_in_place_scalar_rhs(bh_multi_array_float64_p self, bh_float64 rhs);
DLLEXPORT void bh_multi_array_int16_add_in_place(bh_multi_array_int16_p self, bh_multi_array_int16_p rhs);
DLLEXPORT void bh_multi_array_int16_add_in_place_scalar_rhs(bh_multi_array_int16_p self, bh_int16 rhs);
DLLEXPORT void bh_multi_array_int32_add_in_place(bh_multi_array_int32_p self, bh_multi_array_int32_p rhs);
DLLEXPORT void bh_multi_array_int32_add_in_place_scalar_rhs(bh_multi_array_int32_p self, bh_int32 rhs);
DLLEXPORT void bh_multi_array_int64_add_in_place(bh_multi_array_int64_p self, bh_multi_array_int64_p rhs);
DLLEXPORT void bh_multi_array_int64_add_in_place_scalar_rhs(bh_multi_array_int64_p self, bh_int64 rhs);
DLLEXPORT void bh_multi_array_int8_add_in_place(bh_multi_array_int8_p self, bh_multi_array_int8_p rhs);
DLLEXPORT void bh_multi_array_int8_add_in_place_scalar_rhs(bh_multi_array_int8_p self, bh_int8 rhs);
DLLEXPORT void bh_multi_array_uint16_add_in_place(bh_multi_array_uint16_p self, bh_multi_array_uint16_p rhs);
DLLEXPORT void bh_multi_array_uint16_add_in_place_scalar_rhs(bh_multi_array_uint16_p self, bh_uint16 rhs);
DLLEXPORT void bh_multi_array_uint32_add_in_place(bh_multi_array_uint32_p self, bh_multi_array_uint32_p rhs);
DLLEXPORT void bh_multi_array_uint32_add_in_place_scalar_rhs(bh_multi_array_uint32_p self, bh_uint32 rhs);
DLLEXPORT void bh_multi_array_uint64_add_in_place(bh_multi_array_uint64_p self, bh_multi_array_uint64_p rhs);
DLLEXPORT void bh_multi_array_uint64_add_in_place_scalar_rhs(bh_multi_array_uint64_p self, bh_uint64 rhs);
DLLEXPORT void bh_multi_array_uint8_add_in_place(bh_multi_array_uint8_p self, bh_multi_array_uint8_p rhs);
DLLEXPORT void bh_multi_array_uint8_add_in_place_scalar_rhs(bh_multi_array_uint8_p self, bh_uint8 rhs);

DLLEXPORT void bh_multi_array_bool8_subtract_in_place(bh_multi_array_bool8_p self, bh_multi_array_bool8_p rhs);
DLLEXPORT void bh_multi_array_bool8_subtract_in_place_scalar_rhs(bh_multi_array_bool8_p self, bh_bool rhs);
DLLEXPORT void bh_multi_array_complex128_subtract_in_place(bh_multi_array_complex128_p self, bh_multi_array_complex128_p rhs);
DLLEXPORT void bh_multi_array_complex128_subtract_in_place_scalar_rhs(bh_multi_array_complex128_p self, bh_complex128 rhs);
DLLEXPORT void bh_multi_array_complex64_subtract_in_place(bh_multi_array_complex64_p self, bh_multi_array_complex64_p rhs);
DLLEXPORT void bh_multi_array_complex64_subtract_in_place_scalar_rhs(bh_multi_array_complex64_p self, bh_complex64 rhs);
DLLEXPORT void bh_multi_array_float32_subtract_in_place(bh_multi_array_float32_p self, bh_multi_array_float32_p rhs);
DLLEXPORT void bh_multi_array_float32_subtract_in_place_scalar_rhs(bh_multi_array_float32_p self, bh_float32 rhs);
DLLEXPORT void bh_multi_array_float64_subtract_in_place(bh_multi_array_float64_p self, bh_multi_array_float64_p rhs);
DLLEXPORT void bh_multi_array_float64_subtract_in_place_scalar_rhs(bh_multi_array_float64_p self, bh_float64 rhs);
DLLEXPORT void bh_multi_array_int16_subtract_in_place(bh_multi_array_int16_p self, bh_multi_array_int16_p rhs);
DLLEXPORT void bh_multi_array_int16_subtract_in_place_scalar_rhs(bh_multi_array_int16_p self, bh_int16 rhs);
DLLEXPORT void bh_multi_array_int32_subtract_in_place(bh_multi_array_int32_p self, bh_multi_array_int32_p rhs);
DLLEXPORT void bh_multi_array_int32_subtract_in_place_scalar_rhs(bh_multi_array_int32_p self, bh_int32 rhs);
DLLEXPORT void bh_multi_array_int64_subtract_in_place(bh_multi_array_int64_p self, bh_multi_array_int64_p rhs);
DLLEXPORT void bh_multi_array_int64_subtract_in_place_scalar_rhs(bh_multi_array_int64_p self, bh_int64 rhs);
DLLEXPORT void bh_multi_array_int8_subtract_in_place(bh_multi_array_int8_p self, bh_multi_array_int8_p rhs);
DLLEXPORT void bh_multi_array_int8_subtract_in_place_scalar_rhs(bh_multi_array_int8_p self, bh_int8 rhs);
DLLEXPORT void bh_multi_array_uint16_subtract_in_place(bh_multi_array_uint16_p self, bh_multi_array_uint16_p rhs);
DLLEXPORT void bh_multi_array_uint16_subtract_in_place_scalar_rhs(bh_multi_array_uint16_p self, bh_uint16 rhs);
DLLEXPORT void bh_multi_array_uint32_subtract_in_place(bh_multi_array_uint32_p self, bh_multi_array_uint32_p rhs);
DLLEXPORT void bh_multi_array_uint32_subtract_in_place_scalar_rhs(bh_multi_array_uint32_p self, bh_uint32 rhs);
DLLEXPORT void bh_multi_array_uint64_subtract_in_place(bh_multi_array_uint64_p self, bh_multi_array_uint64_p rhs);
DLLEXPORT void bh_multi_array_uint64_subtract_in_place_scalar_rhs(bh_multi_array_uint64_p self, bh_uint64 rhs);
DLLEXPORT void bh_multi_array_uint8_subtract_in_place(bh_multi_array_uint8_p self, bh_multi_array_uint8_p rhs);
DLLEXPORT void bh_multi_array_uint8_subtract_in_place_scalar_rhs(bh_multi_array_uint8_p self, bh_uint8 rhs);

DLLEXPORT void bh_multi_array_bool8_multiply_in_place(bh_multi_array_bool8_p self, bh_multi_array_bool8_p rhs);
DLLEXPORT void bh_multi_array_bool8_multiply_in_place_scalar_rhs(bh_multi_array_bool8_p self, bh_bool rhs);
DLLEXPORT void bh_multi_array_complex128_multiply_in_place(bh_multi_array_complex128_p self, bh_multi_array_complex128_p rhs);
DLLEXPORT void bh_multi_array_complex128_multiply_in_place_scalar_rhs(bh_multi_array_complex128_p self, bh_complex128 rhs);
DLLEXPORT void bh_multi_array_complex64_multiply_in_place(bh_multi_array_complex64_p self, bh_multi_array_complex64_p rhs);
DLLEXPORT void bh_multi_array_complex64_multiply_in_place_scalar_rhs(bh_multi_array_complex64_p self, bh_complex64 rhs);
DLLEXPORT void bh_multi_array_float32_multiply_in_place(bh_multi_array_float32_p self, bh_multi_array_float32_p rhs);
DLLEXPORT void bh_multi_array_float32_multiply_in_place_scalar_rhs(bh_multi_array_float32_p self, bh_float32 rhs);
DLLEXPORT void bh_multi_array_float64_multiply_in_place(bh_multi_array_float64_p self, bh_multi_array_float64_p rhs);
DLLEXPORT void bh_multi_array_float64_multiply_in_place_scalar_rhs(bh_multi_array_float64_p self, bh_float64 rhs);
DLLEXPORT void bh_multi_array_int16_multiply_in_place(bh_multi_array_int16_p self, bh_multi_array_int16_p rhs);
DLLEXPORT void bh_multi_array_int16_multiply_in_place_scalar_rhs(bh_multi_array_int16_p self, bh_int16 rhs);
DLLEXPORT void bh_multi_array_int32_multiply_in_place(bh_multi_array_int32_p self, bh_multi_array_int32_p rhs);
DLLEXPORT void bh_multi_array_int32_multiply_in_place_scalar_rhs(bh_multi_array_int32_p self, bh_int32 rhs);
DLLEXPORT void bh_multi_array_int64_multiply_in_place(bh_multi_array_int64_p self, bh_multi_array_int64_p rhs);
DLLEXPORT void bh_multi_array_int64_multiply_in_place_scalar_rhs(bh_multi_array_int64_p self, bh_int64 rhs);
DLLEXPORT void bh_multi_array_int8_multiply_in_place(bh_multi_array_int8_p self, bh_multi_array_int8_p rhs);
DLLEXPORT void bh_multi_array_int8_multiply_in_place_scalar_rhs(bh_multi_array_int8_p self, bh_int8 rhs);
DLLEXPORT void bh_multi_array_uint16_multiply_in_place(bh_multi_array_uint16_p self, bh_multi_array_uint16_p rhs);
DLLEXPORT void bh_multi_array_uint16_multiply_in_place_scalar_rhs(bh_multi_array_uint16_p self, bh_uint16 rhs);
DLLEXPORT void bh_multi_array_uint32_multiply_in_place(bh_multi_array_uint32_p self, bh_multi_array_uint32_p rhs);
DLLEXPORT void bh_multi_array_uint32_multiply_in_place_scalar_rhs(bh_multi_array_uint32_p self, bh_uint32 rhs);
DLLEXPORT void bh_multi_array_uint64_multiply_in_place(bh_multi_array_uint64_p self, bh_multi_array_uint64_p rhs);
DLLEXPORT void bh_multi_array_uint64_multiply_in_place_scalar_rhs(bh_multi_array_uint64_p self, bh_uint64 rhs);
DLLEXPORT void bh_multi_array_uint8_multiply_in_place(bh_multi_array_uint8_p self, bh_multi_array_uint8_p rhs);
DLLEXPORT void bh_multi_array_uint8_multiply_in_place_scalar_rhs(bh_multi_array_uint8_p self, bh_uint8 rhs);

DLLEXPORT void bh_multi_array_complex128_divide_in_place(bh_multi_array_complex128_p self, bh_multi_array_complex128_p rhs);
DLLEXPORT void bh_multi_array_complex128_divide_in_place_scalar_rhs(bh_multi_array_complex128_p self, bh_complex128 rhs);
DLLEXPORT void bh_multi_array_complex64_divide_in_place(bh_multi_array_complex64_p self, bh_multi_array_complex64_p rhs);
DLLEXPORT void bh_multi_array_complex64_divide_in_place_scalar_rhs(bh_multi_array_complex64_p self, bh_complex64 rhs);
DLLEXPORT void bh_multi_array_float32_divide_in_place(bh_multi_array_float32_p self, bh_multi_array_float32_p rhs);
DLLEXPORT void bh_multi_array_float32_divide_in_place_scalar_rhs(bh_multi_array_float32_p self, bh_float32 rhs);
DLLEXPORT void bh_multi_array_float64_divide_in_place(bh_multi_array_float64_p self, bh_multi_array_float64_p rhs);
DLLEXPORT void bh_multi_array_float64_divide_in_place_scalar_rhs(bh_multi_array_float64_p self, bh_float64 rhs);
DLLEXPORT void bh_multi_array_int16_divide_in_place(bh_multi_array_int16_p self, bh_multi_array_int16_p rhs);
DLLEXPORT void bh_multi_array_int16_divide_in_place_scalar_rhs(bh_multi_array_int16_p self, bh_int16 rhs);
DLLEXPORT void bh_multi_array_int32_divide_in_place(bh_multi_array_int32_p self, bh_multi_array_int32_p rhs);
DLLEXPORT void bh_multi_array_int32_divide_in_place_scalar_rhs(bh_multi_array_int32_p self, bh_int32 rhs);
DLLEXPORT void bh_multi_array_int64_divide_in_place(bh_multi_array_int64_p self, bh_multi_array_int64_p rhs);
DLLEXPORT void bh_multi_array_int64_divide_in_place_scalar_rhs(bh_multi_array_int64_p self, bh_int64 rhs);
DLLEXPORT void bh_multi_array_int8_divide_in_place(bh_multi_array_int8_p self, bh_multi_array_int8_p rhs);
DLLEXPORT void bh_multi_array_int8_divide_in_place_scalar_rhs(bh_multi_array_int8_p self, bh_int8 rhs);
DLLEXPORT void bh_multi_array_uint16_divide_in_place(bh_multi_array_uint16_p self, bh_multi_array_uint16_p rhs);
DLLEXPORT void bh_multi_array_uint16_divide_in_place_scalar_rhs(bh_multi_array_uint16_p self, bh_uint16 rhs);
DLLEXPORT void bh_multi_array_uint32_divide_in_place(bh_multi_array_uint32_p self, bh_multi_array_uint32_p rhs);
DLLEXPORT void bh_multi_array_uint32_divide_in_place_scalar_rhs(bh_multi_array_uint32_p self, bh_uint32 rhs);
DLLEXPORT void bh_multi_array_uint64_divide_in_place(bh_multi_array_uint64_p self, bh_multi_array_uint64_p rhs);
DLLEXPORT void bh_multi_array_uint64_divide_in_place_scalar_rhs(bh_multi_array_uint64_p self, bh_uint64 rhs);
DLLEXPORT void bh_multi_array_uint8_divide_in_place(bh_multi_array_uint8_p self, bh_multi_array_uint8_p rhs);
DLLEXPORT void bh_multi_array_uint8_divide_in_place_scalar_rhs(bh_multi_array_uint8_p self, bh_uint8 rhs);

DLLEXPORT void bh_multi_array_float32_modulo_in_place(bh_multi_array_float32_p self, bh_multi_array_float32_p rhs);
DLLEXPORT void bh_multi_array_float32_modulo_in_place_scalar_rhs(bh_multi_array_float32_p self, bh_float32 rhs);
DLLEXPORT void bh_multi_array_float64_modulo_in_place(bh_multi_array_float64_p self, bh_multi_array_float64_p rhs);
DLLEXPORT void bh_multi_array_float64_modulo_in_place_scalar_rhs(bh_multi_array_float64_p self, bh_float64 rhs);
DLLEXPORT void bh_multi_array_int16_modulo_in_place(bh_multi_array_int16_p self, bh_multi_array_int16_p rhs);
DLLEXPORT void bh_multi_array_int16_modulo_in_place_scalar_rhs(bh_multi_array_int16_p self, bh_int16 rhs);
DLLEXPORT void bh_multi_array_int32_modulo_in_place(bh_multi_array_int32_p self, bh_multi_array_int32_p rhs);
DLLEXPORT void bh_multi_array_int32_modulo_in_place_scalar_rhs(bh_multi_array_int32_p self, bh_int32 rhs);
DLLEXPORT void bh_multi_array_int64_modulo_in_place(bh_multi_array_int64_p self, bh_multi_array_int64_p rhs);
DLLEXPORT void bh_multi_array_int64_modulo_in_place_scalar_rhs(bh_multi_array_int64_p self, bh_int64 rhs);
DLLEXPORT void bh_multi_array_int8_modulo_in_place(bh_multi_array_int8_p self, bh_multi_array_int8_p rhs);
DLLEXPORT void bh_multi_array_int8_modulo_in_place_scalar_rhs(bh_multi_array_int8_p self, bh_int8 rhs);
DLLEXPORT void bh_multi_array_uint16_modulo_in_place(bh_multi_array_uint16_p self, bh_multi_array_uint16_p rhs);
DLLEXPORT void bh_multi_array_uint16_modulo_in_place_scalar_rhs(bh_multi_array_uint16_p self, bh_uint16 rhs);
DLLEXPORT void bh_multi_array_uint32_modulo_in_place(bh_multi_array_uint32_p self, bh_multi_array_uint32_p rhs);
DLLEXPORT void bh_multi_array_uint32_modulo_in_place_scalar_rhs(bh_multi_array_uint32_p self, bh_uint32 rhs);
DLLEXPORT void bh_multi_array_uint64_modulo_in_place(bh_multi_array_uint64_p self, bh_multi_array_uint64_p rhs);
DLLEXPORT void bh_multi_array_uint64_modulo_in_place_scalar_rhs(bh_multi_array_uint64_p self, bh_uint64 rhs);
DLLEXPORT void bh_multi_array_uint8_modulo_in_place(bh_multi_array_uint8_p self, bh_multi_array_uint8_p rhs);
DLLEXPORT void bh_multi_array_uint8_modulo_in_place_scalar_rhs(bh_multi_array_uint8_p self, bh_uint8 rhs);

DLLEXPORT void bh_multi_array_bool8_bitwise_and_in_place(bh_multi_array_bool8_p self, bh_multi_array_bool8_p rhs);
DLLEXPORT void bh_multi_array_bool8_bitwise_and_in_place_scalar_rhs(bh_multi_array_bool8_p self, bh_bool rhs);
DLLEXPORT void bh_multi_array_int16_bitwise_and_in_place(bh_multi_array_int16_p self, bh_multi_array_int16_p rhs);
DLLEXPORT void bh_multi_array_int16_bitwise_and_in_place_scalar_rhs(bh_multi_array_int16_p self, bh_int16 rhs);
DLLEXPORT void bh_multi_array_int32_bitwise_and_in_place(bh_multi_array_int32_p self, bh_multi_array_int32_p rhs);
DLLEXPORT void bh_multi_array_int32_bitwise_and_in_place_scalar_rhs(bh_multi_array_int32_p self, bh_int32 rhs);
DLLEXPORT void bh_multi_array_int64_bitwise_and_in_place(bh_multi_array_int64_p self, bh_multi_array_int64_p rhs);
DLLEXPORT void bh_multi_array_int64_bitwise_and_in_place_scalar_rhs(bh_multi_array_int64_p self, bh_int64 rhs);
DLLEXPORT void bh_multi_array_int8_bitwise_and_in_place(bh_multi_array_int8_p self, bh_multi_array_int8_p rhs);
DLLEXPORT void bh_multi_array_int8_bitwise_and_in_place_scalar_rhs(bh_multi_array_int8_p self, bh_int8 rhs);
DLLEXPORT void bh_multi_array_uint16_bitwise_and_in_place(bh_multi_array_uint16_p self, bh_multi_array_uint16_p rhs);
DLLEXPORT void bh_multi_array_uint16_bitwise_and_in_place_scalar_rhs(bh_multi_array_uint16_p self, bh_uint16 rhs);
DLLEXPORT void bh_multi_array_uint32_bitwise_and_in_place(bh_multi_array_uint32_p self, bh_multi_array_uint32_p rhs);
DLLEXPORT void bh_multi_array_uint32_bitwise_and_in_place_scalar_rhs(bh_multi_array_uint32_p self, bh_uint32 rhs);
DLLEXPORT void bh_multi_array_uint64_bitwise_and_in_place(bh_multi_array_uint64_p self, bh_multi_array_uint64_p rhs);
DLLEXPORT void bh_multi_array_uint64_bitwise_and_in_place_scalar_rhs(bh_multi_array_uint64_p self, bh_uint64 rhs);
DLLEXPORT void bh_multi_array_uint8_bitwise_and_in_place(bh_multi_array_uint8_p self, bh_multi_array_uint8_p rhs);
DLLEXPORT void bh_multi_array_uint8_bitwise_and_in_place_scalar_rhs(bh_multi_array_uint8_p self, bh_uint8 rhs);

DLLEXPORT void bh_multi_array_bool8_bitwise_or_in_place(bh_multi_array_bool8_p self, bh_multi_array_bool8_p rhs);
DLLEXPORT void bh_multi_array_bool8_bitwise_or_in_place_scalar_rhs(bh_multi_array_bool8_p self, bh_bool rhs);
DLLEXPORT void bh_multi_array_int16_bitwise_or_in_place(bh_multi_array_int16_p self, bh_multi_array_int16_p rhs);
DLLEXPORT void bh_multi_array_int16_bitwise_or_in_place_scalar_rhs(bh_multi_array_int16_p self, bh_int16 rhs);
DLLEXPORT void bh_multi_array_int32_bitwise_or_in_place(bh_multi_array_int32_p self, bh_multi_array_int32_p rhs);
DLLEXPORT void bh_multi_array_int32_bitwise_or_in_place_scalar_rhs(bh_multi_array_int32_p self, bh_int32 rhs);
DLLEXPORT void bh_multi_array_int64_bitwise_or_in_place(bh_multi_array_int64_p self, bh_multi_array_int64_p rhs);
DLLEXPORT void bh_multi_array_int64_bitwise_or_in_place_scalar_rhs(bh_multi_array_int64_p self, bh_int64 rhs);
DLLEXPORT void bh_multi_array_int8_bitwise_or_in_place(bh_multi_array_int8_p self, bh_multi_array_int8_p rhs);
DLLEXPORT void bh_multi_array_int8_bitwise_or_in_place_scalar_rhs(bh_multi_array_int8_p self, bh_int8 rhs);
DLLEXPORT void bh_multi_array_uint16_bitwise_or_in_place(bh_multi_array_uint16_p self, bh_multi_array_uint16_p rhs);
DLLEXPORT void bh_multi_array_uint16_bitwise_or_in_place_scalar_rhs(bh_multi_array_uint16_p self, bh_uint16 rhs);
DLLEXPORT void bh_multi_array_uint32_bitwise_or_in_place(bh_multi_array_uint32_p self, bh_multi_array_uint32_p rhs);
DLLEXPORT void bh_multi_array_uint32_bitwise_or_in_place_scalar_rhs(bh_multi_array_uint32_p self, bh_uint32 rhs);
DLLEXPORT void bh_multi_array_uint64_bitwise_or_in_place(bh_multi_array_uint64_p self, bh_multi_array_uint64_p rhs);
DLLEXPORT void bh_multi_array_uint64_bitwise_or_in_place_scalar_rhs(bh_multi_array_uint64_p self, bh_uint64 rhs);
DLLEXPORT void bh_multi_array_uint8_bitwise_or_in_place(bh_multi_array_uint8_p self, bh_multi_array_uint8_p rhs);
DLLEXPORT void bh_multi_array_uint8_bitwise_or_in_place_scalar_rhs(bh_multi_array_uint8_p self, bh_uint8 rhs);

DLLEXPORT void bh_multi_array_bool8_bitwise_xor_in_place(bh_multi_array_bool8_p self, bh_multi_array_bool8_p rhs);
DLLEXPORT void bh_multi_array_bool8_bitwise_xor_in_place_scalar_rhs(bh_multi_array_bool8_p self, bh_bool rhs);
DLLEXPORT void bh_multi_array_int16_bitwise_xor_in_place(bh_multi_array_int16_p self, bh_multi_array_int16_p rhs);
DLLEXPORT void bh_multi_array_int16_bitwise_xor_in_place_scalar_rhs(bh_multi_array_int16_p self, bh_int16 rhs);
DLLEXPORT void bh_multi_array_int32_bitwise_xor_in_place(bh_multi_array_int32_p self, bh_multi_array_int32_p rhs);
DLLEXPORT void bh_multi_array_int32_bitwise_xor_in_place_scalar_rhs(bh_multi_array_int32_p self, bh_int32 rhs);
DLLEXPORT void bh_multi_array_int64_bitwise_xor_in_place(bh_multi_array_int64_p self, bh_multi_array_int64_p rhs);
DLLEXPORT void bh_multi_array_int64_bitwise_xor_in_place_scalar_rhs(bh_multi_array_int64_p self, bh_int64 rhs);
DLLEXPORT void bh_multi_array_int8_bitwise_xor_in_place(bh_multi_array_int8_p self, bh_multi_array_int8_p rhs);
DLLEXPORT void bh_multi_array_int8_bitwise_xor_in_place_scalar_rhs(bh_multi_array_int8_p self, bh_int8 rhs);
DLLEXPORT void bh_multi_array_uint16_bitwise_xor_in_place(bh_multi_array_uint16_p self, bh_multi_array_uint16_p rhs);
DLLEXPORT void bh_multi_array_uint16_bitwise_xor_in_place_scalar_rhs(bh_multi_array_uint16_p self, bh_uint16 rhs);
DLLEXPORT void bh_multi_array_uint32_bitwise_xor_in_place(bh_multi_array_uint32_p self, bh_multi_array_uint32_p rhs);
DLLEXPORT void bh_multi_array_uint32_bitwise_xor_in_place_scalar_rhs(bh_multi_array_uint32_p self, bh_uint32 rhs);
DLLEXPORT void bh_multi_array_uint64_bitwise_xor_in_place(bh_multi_array_uint64_p self, bh_multi_array_uint64_p rhs);
DLLEXPORT void bh_multi_array_uint64_bitwise_xor_in_place_scalar_rhs(bh_multi_array_uint64_p self, bh_uint64 rhs);
DLLEXPORT void bh_multi_array_uint8_bitwise_xor_in_place(bh_multi_array_uint8_p self, bh_multi_array_uint8_p rhs);
DLLEXPORT void bh_multi_array_uint8_bitwise_xor_in_place_scalar_rhs(bh_multi_array_uint8_p self, bh_uint8 rhs);

DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_add(bh_multi_array_bool8_p lhs, bh_multi_array_bool8_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_add_scalar_lhs(bh_bool lhs, bh_multi_array_bool8_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_add_scalar_rhs(bh_multi_array_bool8_p lhs, bh_bool rhs);
DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_add(bh_multi_array_complex128_p lhs, bh_multi_array_complex128_p rhs);
DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_add_scalar_lhs(bh_complex128 lhs, bh_multi_array_complex128_p rhs);
DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_add_scalar_rhs(bh_multi_array_complex128_p lhs, bh_complex128 rhs);
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_add(bh_multi_array_complex64_p lhs, bh_multi_array_complex64_p rhs);
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_add_scalar_lhs(bh_complex64 lhs, bh_multi_array_complex64_p rhs);
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_add_scalar_rhs(bh_multi_array_complex64_p lhs, bh_complex64 rhs);
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_add(bh_multi_array_float32_p lhs, bh_multi_array_float32_p rhs);
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_add_scalar_lhs(bh_float32 lhs, bh_multi_array_float32_p rhs);
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_add_scalar_rhs(bh_multi_array_float32_p lhs, bh_float32 rhs);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_add(bh_multi_array_float64_p lhs, bh_multi_array_float64_p rhs);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_add_scalar_lhs(bh_float64 lhs, bh_multi_array_float64_p rhs);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_add_scalar_rhs(bh_multi_array_float64_p lhs, bh_float64 rhs);
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_add(bh_multi_array_int16_p lhs, bh_multi_array_int16_p rhs);
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_add_scalar_lhs(bh_int16 lhs, bh_multi_array_int16_p rhs);
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_add_scalar_rhs(bh_multi_array_int16_p lhs, bh_int16 rhs);
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_add(bh_multi_array_int32_p lhs, bh_multi_array_int32_p rhs);
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_add_scalar_lhs(bh_int32 lhs, bh_multi_array_int32_p rhs);
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_add_scalar_rhs(bh_multi_array_int32_p lhs, bh_int32 rhs);
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_add(bh_multi_array_int64_p lhs, bh_multi_array_int64_p rhs);
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_add_scalar_lhs(bh_int64 lhs, bh_multi_array_int64_p rhs);
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_add_scalar_rhs(bh_multi_array_int64_p lhs, bh_int64 rhs);
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_add(bh_multi_array_int8_p lhs, bh_multi_array_int8_p rhs);
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_add_scalar_lhs(bh_int8 lhs, bh_multi_array_int8_p rhs);
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_add_scalar_rhs(bh_multi_array_int8_p lhs, bh_int8 rhs);
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_add(bh_multi_array_uint16_p lhs, bh_multi_array_uint16_p rhs);
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_add_scalar_lhs(bh_uint16 lhs, bh_multi_array_uint16_p rhs);
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_add_scalar_rhs(bh_multi_array_uint16_p lhs, bh_uint16 rhs);
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_add(bh_multi_array_uint32_p lhs, bh_multi_array_uint32_p rhs);
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_add_scalar_lhs(bh_uint32 lhs, bh_multi_array_uint32_p rhs);
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_add_scalar_rhs(bh_multi_array_uint32_p lhs, bh_uint32 rhs);
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_add(bh_multi_array_uint64_p lhs, bh_multi_array_uint64_p rhs);
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_add_scalar_lhs(bh_uint64 lhs, bh_multi_array_uint64_p rhs);
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_add_scalar_rhs(bh_multi_array_uint64_p lhs, bh_uint64 rhs);
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_add(bh_multi_array_uint8_p lhs, bh_multi_array_uint8_p rhs);
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_add_scalar_lhs(bh_uint8 lhs, bh_multi_array_uint8_p rhs);
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_add_scalar_rhs(bh_multi_array_uint8_p lhs, bh_uint8 rhs);

DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_subtract(bh_multi_array_bool8_p lhs, bh_multi_array_bool8_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_subtract_scalar_lhs(bh_bool lhs, bh_multi_array_bool8_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_subtract_scalar_rhs(bh_multi_array_bool8_p lhs, bh_bool rhs);
DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_subtract(bh_multi_array_complex128_p lhs, bh_multi_array_complex128_p rhs);
DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_subtract_scalar_lhs(bh_complex128 lhs, bh_multi_array_complex128_p rhs);
DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_subtract_scalar_rhs(bh_multi_array_complex128_p lhs, bh_complex128 rhs);
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_subtract(bh_multi_array_complex64_p lhs, bh_multi_array_complex64_p rhs);
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_subtract_scalar_lhs(bh_complex64 lhs, bh_multi_array_complex64_p rhs);
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_subtract_scalar_rhs(bh_multi_array_complex64_p lhs, bh_complex64 rhs);
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_subtract(bh_multi_array_float32_p lhs, bh_multi_array_float32_p rhs);
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_subtract_scalar_lhs(bh_float32 lhs, bh_multi_array_float32_p rhs);
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_subtract_scalar_rhs(bh_multi_array_float32_p lhs, bh_float32 rhs);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_subtract(bh_multi_array_float64_p lhs, bh_multi_array_float64_p rhs);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_subtract_scalar_lhs(bh_float64 lhs, bh_multi_array_float64_p rhs);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_subtract_scalar_rhs(bh_multi_array_float64_p lhs, bh_float64 rhs);
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_subtract(bh_multi_array_int16_p lhs, bh_multi_array_int16_p rhs);
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_subtract_scalar_lhs(bh_int16 lhs, bh_multi_array_int16_p rhs);
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_subtract_scalar_rhs(bh_multi_array_int16_p lhs, bh_int16 rhs);
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_subtract(bh_multi_array_int32_p lhs, bh_multi_array_int32_p rhs);
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_subtract_scalar_lhs(bh_int32 lhs, bh_multi_array_int32_p rhs);
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_subtract_scalar_rhs(bh_multi_array_int32_p lhs, bh_int32 rhs);
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_subtract(bh_multi_array_int64_p lhs, bh_multi_array_int64_p rhs);
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_subtract_scalar_lhs(bh_int64 lhs, bh_multi_array_int64_p rhs);
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_subtract_scalar_rhs(bh_multi_array_int64_p lhs, bh_int64 rhs);
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_subtract(bh_multi_array_int8_p lhs, bh_multi_array_int8_p rhs);
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_subtract_scalar_lhs(bh_int8 lhs, bh_multi_array_int8_p rhs);
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_subtract_scalar_rhs(bh_multi_array_int8_p lhs, bh_int8 rhs);
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_subtract(bh_multi_array_uint16_p lhs, bh_multi_array_uint16_p rhs);
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_subtract_scalar_lhs(bh_uint16 lhs, bh_multi_array_uint16_p rhs);
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_subtract_scalar_rhs(bh_multi_array_uint16_p lhs, bh_uint16 rhs);
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_subtract(bh_multi_array_uint32_p lhs, bh_multi_array_uint32_p rhs);
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_subtract_scalar_lhs(bh_uint32 lhs, bh_multi_array_uint32_p rhs);
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_subtract_scalar_rhs(bh_multi_array_uint32_p lhs, bh_uint32 rhs);
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_subtract(bh_multi_array_uint64_p lhs, bh_multi_array_uint64_p rhs);
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_subtract_scalar_lhs(bh_uint64 lhs, bh_multi_array_uint64_p rhs);
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_subtract_scalar_rhs(bh_multi_array_uint64_p lhs, bh_uint64 rhs);
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_subtract(bh_multi_array_uint8_p lhs, bh_multi_array_uint8_p rhs);
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_subtract_scalar_lhs(bh_uint8 lhs, bh_multi_array_uint8_p rhs);
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_subtract_scalar_rhs(bh_multi_array_uint8_p lhs, bh_uint8 rhs);

DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_multiply(bh_multi_array_bool8_p lhs, bh_multi_array_bool8_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_multiply_scalar_lhs(bh_bool lhs, bh_multi_array_bool8_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_multiply_scalar_rhs(bh_multi_array_bool8_p lhs, bh_bool rhs);
DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_multiply(bh_multi_array_complex128_p lhs, bh_multi_array_complex128_p rhs);
DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_multiply_scalar_lhs(bh_complex128 lhs, bh_multi_array_complex128_p rhs);
DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_multiply_scalar_rhs(bh_multi_array_complex128_p lhs, bh_complex128 rhs);
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_multiply(bh_multi_array_complex64_p lhs, bh_multi_array_complex64_p rhs);
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_multiply_scalar_lhs(bh_complex64 lhs, bh_multi_array_complex64_p rhs);
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_multiply_scalar_rhs(bh_multi_array_complex64_p lhs, bh_complex64 rhs);
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_multiply(bh_multi_array_float32_p lhs, bh_multi_array_float32_p rhs);
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_multiply_scalar_lhs(bh_float32 lhs, bh_multi_array_float32_p rhs);
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_multiply_scalar_rhs(bh_multi_array_float32_p lhs, bh_float32 rhs);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_multiply(bh_multi_array_float64_p lhs, bh_multi_array_float64_p rhs);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_multiply_scalar_lhs(bh_float64 lhs, bh_multi_array_float64_p rhs);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_multiply_scalar_rhs(bh_multi_array_float64_p lhs, bh_float64 rhs);
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_multiply(bh_multi_array_int16_p lhs, bh_multi_array_int16_p rhs);
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_multiply_scalar_lhs(bh_int16 lhs, bh_multi_array_int16_p rhs);
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_multiply_scalar_rhs(bh_multi_array_int16_p lhs, bh_int16 rhs);
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_multiply(bh_multi_array_int32_p lhs, bh_multi_array_int32_p rhs);
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_multiply_scalar_lhs(bh_int32 lhs, bh_multi_array_int32_p rhs);
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_multiply_scalar_rhs(bh_multi_array_int32_p lhs, bh_int32 rhs);
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_multiply(bh_multi_array_int64_p lhs, bh_multi_array_int64_p rhs);
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_multiply_scalar_lhs(bh_int64 lhs, bh_multi_array_int64_p rhs);
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_multiply_scalar_rhs(bh_multi_array_int64_p lhs, bh_int64 rhs);
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_multiply(bh_multi_array_int8_p lhs, bh_multi_array_int8_p rhs);
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_multiply_scalar_lhs(bh_int8 lhs, bh_multi_array_int8_p rhs);
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_multiply_scalar_rhs(bh_multi_array_int8_p lhs, bh_int8 rhs);
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_multiply(bh_multi_array_uint16_p lhs, bh_multi_array_uint16_p rhs);
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_multiply_scalar_lhs(bh_uint16 lhs, bh_multi_array_uint16_p rhs);
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_multiply_scalar_rhs(bh_multi_array_uint16_p lhs, bh_uint16 rhs);
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_multiply(bh_multi_array_uint32_p lhs, bh_multi_array_uint32_p rhs);
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_multiply_scalar_lhs(bh_uint32 lhs, bh_multi_array_uint32_p rhs);
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_multiply_scalar_rhs(bh_multi_array_uint32_p lhs, bh_uint32 rhs);
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_multiply(bh_multi_array_uint64_p lhs, bh_multi_array_uint64_p rhs);
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_multiply_scalar_lhs(bh_uint64 lhs, bh_multi_array_uint64_p rhs);
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_multiply_scalar_rhs(bh_multi_array_uint64_p lhs, bh_uint64 rhs);
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_multiply(bh_multi_array_uint8_p lhs, bh_multi_array_uint8_p rhs);
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_multiply_scalar_lhs(bh_uint8 lhs, bh_multi_array_uint8_p rhs);
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_multiply_scalar_rhs(bh_multi_array_uint8_p lhs, bh_uint8 rhs);

DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_divide(bh_multi_array_complex128_p lhs, bh_multi_array_complex128_p rhs);
DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_divide_scalar_lhs(bh_complex128 lhs, bh_multi_array_complex128_p rhs);
DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_divide_scalar_rhs(bh_multi_array_complex128_p lhs, bh_complex128 rhs);
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_divide(bh_multi_array_complex64_p lhs, bh_multi_array_complex64_p rhs);
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_divide_scalar_lhs(bh_complex64 lhs, bh_multi_array_complex64_p rhs);
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_divide_scalar_rhs(bh_multi_array_complex64_p lhs, bh_complex64 rhs);
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_divide(bh_multi_array_float32_p lhs, bh_multi_array_float32_p rhs);
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_divide_scalar_lhs(bh_float32 lhs, bh_multi_array_float32_p rhs);
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_divide_scalar_rhs(bh_multi_array_float32_p lhs, bh_float32 rhs);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_divide(bh_multi_array_float64_p lhs, bh_multi_array_float64_p rhs);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_divide_scalar_lhs(bh_float64 lhs, bh_multi_array_float64_p rhs);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_divide_scalar_rhs(bh_multi_array_float64_p lhs, bh_float64 rhs);
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_divide(bh_multi_array_int16_p lhs, bh_multi_array_int16_p rhs);
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_divide_scalar_lhs(bh_int16 lhs, bh_multi_array_int16_p rhs);
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_divide_scalar_rhs(bh_multi_array_int16_p lhs, bh_int16 rhs);
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_divide(bh_multi_array_int32_p lhs, bh_multi_array_int32_p rhs);
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_divide_scalar_lhs(bh_int32 lhs, bh_multi_array_int32_p rhs);
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_divide_scalar_rhs(bh_multi_array_int32_p lhs, bh_int32 rhs);
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_divide(bh_multi_array_int64_p lhs, bh_multi_array_int64_p rhs);
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_divide_scalar_lhs(bh_int64 lhs, bh_multi_array_int64_p rhs);
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_divide_scalar_rhs(bh_multi_array_int64_p lhs, bh_int64 rhs);
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_divide(bh_multi_array_int8_p lhs, bh_multi_array_int8_p rhs);
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_divide_scalar_lhs(bh_int8 lhs, bh_multi_array_int8_p rhs);
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_divide_scalar_rhs(bh_multi_array_int8_p lhs, bh_int8 rhs);
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_divide(bh_multi_array_uint16_p lhs, bh_multi_array_uint16_p rhs);
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_divide_scalar_lhs(bh_uint16 lhs, bh_multi_array_uint16_p rhs);
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_divide_scalar_rhs(bh_multi_array_uint16_p lhs, bh_uint16 rhs);
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_divide(bh_multi_array_uint32_p lhs, bh_multi_array_uint32_p rhs);
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_divide_scalar_lhs(bh_uint32 lhs, bh_multi_array_uint32_p rhs);
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_divide_scalar_rhs(bh_multi_array_uint32_p lhs, bh_uint32 rhs);
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_divide(bh_multi_array_uint64_p lhs, bh_multi_array_uint64_p rhs);
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_divide_scalar_lhs(bh_uint64 lhs, bh_multi_array_uint64_p rhs);
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_divide_scalar_rhs(bh_multi_array_uint64_p lhs, bh_uint64 rhs);
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_divide(bh_multi_array_uint8_p lhs, bh_multi_array_uint8_p rhs);
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_divide_scalar_lhs(bh_uint8 lhs, bh_multi_array_uint8_p rhs);
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_divide_scalar_rhs(bh_multi_array_uint8_p lhs, bh_uint8 rhs);

DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_modulo(bh_multi_array_float32_p lhs, bh_multi_array_float32_p rhs);
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_modulo_scalar_lhs(bh_float32 lhs, bh_multi_array_float32_p rhs);
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_modulo_scalar_rhs(bh_multi_array_float32_p lhs, bh_float32 rhs);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_modulo(bh_multi_array_float64_p lhs, bh_multi_array_float64_p rhs);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_modulo_scalar_lhs(bh_float64 lhs, bh_multi_array_float64_p rhs);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_modulo_scalar_rhs(bh_multi_array_float64_p lhs, bh_float64 rhs);
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_modulo(bh_multi_array_int16_p lhs, bh_multi_array_int16_p rhs);
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_modulo_scalar_lhs(bh_int16 lhs, bh_multi_array_int16_p rhs);
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_modulo_scalar_rhs(bh_multi_array_int16_p lhs, bh_int16 rhs);
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_modulo(bh_multi_array_int32_p lhs, bh_multi_array_int32_p rhs);
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_modulo_scalar_lhs(bh_int32 lhs, bh_multi_array_int32_p rhs);
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_modulo_scalar_rhs(bh_multi_array_int32_p lhs, bh_int32 rhs);
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_modulo(bh_multi_array_int64_p lhs, bh_multi_array_int64_p rhs);
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_modulo_scalar_lhs(bh_int64 lhs, bh_multi_array_int64_p rhs);
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_modulo_scalar_rhs(bh_multi_array_int64_p lhs, bh_int64 rhs);
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_modulo(bh_multi_array_int8_p lhs, bh_multi_array_int8_p rhs);
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_modulo_scalar_lhs(bh_int8 lhs, bh_multi_array_int8_p rhs);
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_modulo_scalar_rhs(bh_multi_array_int8_p lhs, bh_int8 rhs);
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_modulo(bh_multi_array_uint16_p lhs, bh_multi_array_uint16_p rhs);
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_modulo_scalar_lhs(bh_uint16 lhs, bh_multi_array_uint16_p rhs);
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_modulo_scalar_rhs(bh_multi_array_uint16_p lhs, bh_uint16 rhs);
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_modulo(bh_multi_array_uint32_p lhs, bh_multi_array_uint32_p rhs);
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_modulo_scalar_lhs(bh_uint32 lhs, bh_multi_array_uint32_p rhs);
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_modulo_scalar_rhs(bh_multi_array_uint32_p lhs, bh_uint32 rhs);
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_modulo(bh_multi_array_uint64_p lhs, bh_multi_array_uint64_p rhs);
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_modulo_scalar_lhs(bh_uint64 lhs, bh_multi_array_uint64_p rhs);
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_modulo_scalar_rhs(bh_multi_array_uint64_p lhs, bh_uint64 rhs);
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_modulo(bh_multi_array_uint8_p lhs, bh_multi_array_uint8_p rhs);
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_modulo_scalar_lhs(bh_uint8 lhs, bh_multi_array_uint8_p rhs);
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_modulo_scalar_rhs(bh_multi_array_uint8_p lhs, bh_uint8 rhs);

DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_equal_to(bh_multi_array_bool8_p lhs, bh_multi_array_bool8_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_equal_to_scalar_lhs(bh_bool lhs, bh_multi_array_bool8_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_equal_to_scalar_rhs(bh_multi_array_bool8_p lhs, bh_bool rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_complex128_equal_to(bh_multi_array_complex128_p lhs, bh_multi_array_complex128_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_complex128_equal_to_scalar_lhs(bh_complex128 lhs, bh_multi_array_complex128_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_complex128_equal_to_scalar_rhs(bh_multi_array_complex128_p lhs, bh_complex128 rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_complex64_equal_to(bh_multi_array_complex64_p lhs, bh_multi_array_complex64_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_complex64_equal_to_scalar_lhs(bh_complex64 lhs, bh_multi_array_complex64_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_complex64_equal_to_scalar_rhs(bh_multi_array_complex64_p lhs, bh_complex64 rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_float32_equal_to(bh_multi_array_float32_p lhs, bh_multi_array_float32_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_float32_equal_to_scalar_lhs(bh_float32 lhs, bh_multi_array_float32_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_float32_equal_to_scalar_rhs(bh_multi_array_float32_p lhs, bh_float32 rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_float64_equal_to(bh_multi_array_float64_p lhs, bh_multi_array_float64_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_float64_equal_to_scalar_lhs(bh_float64 lhs, bh_multi_array_float64_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_float64_equal_to_scalar_rhs(bh_multi_array_float64_p lhs, bh_float64 rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int16_equal_to(bh_multi_array_int16_p lhs, bh_multi_array_int16_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int16_equal_to_scalar_lhs(bh_int16 lhs, bh_multi_array_int16_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int16_equal_to_scalar_rhs(bh_multi_array_int16_p lhs, bh_int16 rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int32_equal_to(bh_multi_array_int32_p lhs, bh_multi_array_int32_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int32_equal_to_scalar_lhs(bh_int32 lhs, bh_multi_array_int32_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int32_equal_to_scalar_rhs(bh_multi_array_int32_p lhs, bh_int32 rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int64_equal_to(bh_multi_array_int64_p lhs, bh_multi_array_int64_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int64_equal_to_scalar_lhs(bh_int64 lhs, bh_multi_array_int64_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int64_equal_to_scalar_rhs(bh_multi_array_int64_p lhs, bh_int64 rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int8_equal_to(bh_multi_array_int8_p lhs, bh_multi_array_int8_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int8_equal_to_scalar_lhs(bh_int8 lhs, bh_multi_array_int8_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int8_equal_to_scalar_rhs(bh_multi_array_int8_p lhs, bh_int8 rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint16_equal_to(bh_multi_array_uint16_p lhs, bh_multi_array_uint16_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint16_equal_to_scalar_lhs(bh_uint16 lhs, bh_multi_array_uint16_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint16_equal_to_scalar_rhs(bh_multi_array_uint16_p lhs, bh_uint16 rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint32_equal_to(bh_multi_array_uint32_p lhs, bh_multi_array_uint32_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint32_equal_to_scalar_lhs(bh_uint32 lhs, bh_multi_array_uint32_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint32_equal_to_scalar_rhs(bh_multi_array_uint32_p lhs, bh_uint32 rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint64_equal_to(bh_multi_array_uint64_p lhs, bh_multi_array_uint64_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint64_equal_to_scalar_lhs(bh_uint64 lhs, bh_multi_array_uint64_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint64_equal_to_scalar_rhs(bh_multi_array_uint64_p lhs, bh_uint64 rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint8_equal_to(bh_multi_array_uint8_p lhs, bh_multi_array_uint8_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint8_equal_to_scalar_lhs(bh_uint8 lhs, bh_multi_array_uint8_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint8_equal_to_scalar_rhs(bh_multi_array_uint8_p lhs, bh_uint8 rhs);

DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_not_equal_to(bh_multi_array_bool8_p lhs, bh_multi_array_bool8_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_not_equal_to_scalar_lhs(bh_bool lhs, bh_multi_array_bool8_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_not_equal_to_scalar_rhs(bh_multi_array_bool8_p lhs, bh_bool rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_complex128_not_equal_to(bh_multi_array_complex128_p lhs, bh_multi_array_complex128_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_complex128_not_equal_to_scalar_lhs(bh_complex128 lhs, bh_multi_array_complex128_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_complex128_not_equal_to_scalar_rhs(bh_multi_array_complex128_p lhs, bh_complex128 rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_complex64_not_equal_to(bh_multi_array_complex64_p lhs, bh_multi_array_complex64_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_complex64_not_equal_to_scalar_lhs(bh_complex64 lhs, bh_multi_array_complex64_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_complex64_not_equal_to_scalar_rhs(bh_multi_array_complex64_p lhs, bh_complex64 rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_float32_not_equal_to(bh_multi_array_float32_p lhs, bh_multi_array_float32_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_float32_not_equal_to_scalar_lhs(bh_float32 lhs, bh_multi_array_float32_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_float32_not_equal_to_scalar_rhs(bh_multi_array_float32_p lhs, bh_float32 rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_float64_not_equal_to(bh_multi_array_float64_p lhs, bh_multi_array_float64_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_float64_not_equal_to_scalar_lhs(bh_float64 lhs, bh_multi_array_float64_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_float64_not_equal_to_scalar_rhs(bh_multi_array_float64_p lhs, bh_float64 rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int16_not_equal_to(bh_multi_array_int16_p lhs, bh_multi_array_int16_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int16_not_equal_to_scalar_lhs(bh_int16 lhs, bh_multi_array_int16_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int16_not_equal_to_scalar_rhs(bh_multi_array_int16_p lhs, bh_int16 rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int32_not_equal_to(bh_multi_array_int32_p lhs, bh_multi_array_int32_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int32_not_equal_to_scalar_lhs(bh_int32 lhs, bh_multi_array_int32_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int32_not_equal_to_scalar_rhs(bh_multi_array_int32_p lhs, bh_int32 rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int64_not_equal_to(bh_multi_array_int64_p lhs, bh_multi_array_int64_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int64_not_equal_to_scalar_lhs(bh_int64 lhs, bh_multi_array_int64_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int64_not_equal_to_scalar_rhs(bh_multi_array_int64_p lhs, bh_int64 rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int8_not_equal_to(bh_multi_array_int8_p lhs, bh_multi_array_int8_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int8_not_equal_to_scalar_lhs(bh_int8 lhs, bh_multi_array_int8_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int8_not_equal_to_scalar_rhs(bh_multi_array_int8_p lhs, bh_int8 rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint16_not_equal_to(bh_multi_array_uint16_p lhs, bh_multi_array_uint16_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint16_not_equal_to_scalar_lhs(bh_uint16 lhs, bh_multi_array_uint16_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint16_not_equal_to_scalar_rhs(bh_multi_array_uint16_p lhs, bh_uint16 rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint32_not_equal_to(bh_multi_array_uint32_p lhs, bh_multi_array_uint32_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint32_not_equal_to_scalar_lhs(bh_uint32 lhs, bh_multi_array_uint32_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint32_not_equal_to_scalar_rhs(bh_multi_array_uint32_p lhs, bh_uint32 rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint64_not_equal_to(bh_multi_array_uint64_p lhs, bh_multi_array_uint64_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint64_not_equal_to_scalar_lhs(bh_uint64 lhs, bh_multi_array_uint64_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint64_not_equal_to_scalar_rhs(bh_multi_array_uint64_p lhs, bh_uint64 rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint8_not_equal_to(bh_multi_array_uint8_p lhs, bh_multi_array_uint8_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint8_not_equal_to_scalar_lhs(bh_uint8 lhs, bh_multi_array_uint8_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint8_not_equal_to_scalar_rhs(bh_multi_array_uint8_p lhs, bh_uint8 rhs);

DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_greater_than(bh_multi_array_bool8_p lhs, bh_multi_array_bool8_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_greater_than_scalar_lhs(bh_bool lhs, bh_multi_array_bool8_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_greater_than_scalar_rhs(bh_multi_array_bool8_p lhs, bh_bool rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_float32_greater_than(bh_multi_array_float32_p lhs, bh_multi_array_float32_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_float32_greater_than_scalar_lhs(bh_float32 lhs, bh_multi_array_float32_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_float32_greater_than_scalar_rhs(bh_multi_array_float32_p lhs, bh_float32 rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_float64_greater_than(bh_multi_array_float64_p lhs, bh_multi_array_float64_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_float64_greater_than_scalar_lhs(bh_float64 lhs, bh_multi_array_float64_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_float64_greater_than_scalar_rhs(bh_multi_array_float64_p lhs, bh_float64 rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int16_greater_than(bh_multi_array_int16_p lhs, bh_multi_array_int16_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int16_greater_than_scalar_lhs(bh_int16 lhs, bh_multi_array_int16_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int16_greater_than_scalar_rhs(bh_multi_array_int16_p lhs, bh_int16 rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int32_greater_than(bh_multi_array_int32_p lhs, bh_multi_array_int32_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int32_greater_than_scalar_lhs(bh_int32 lhs, bh_multi_array_int32_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int32_greater_than_scalar_rhs(bh_multi_array_int32_p lhs, bh_int32 rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int64_greater_than(bh_multi_array_int64_p lhs, bh_multi_array_int64_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int64_greater_than_scalar_lhs(bh_int64 lhs, bh_multi_array_int64_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int64_greater_than_scalar_rhs(bh_multi_array_int64_p lhs, bh_int64 rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int8_greater_than(bh_multi_array_int8_p lhs, bh_multi_array_int8_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int8_greater_than_scalar_lhs(bh_int8 lhs, bh_multi_array_int8_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int8_greater_than_scalar_rhs(bh_multi_array_int8_p lhs, bh_int8 rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint16_greater_than(bh_multi_array_uint16_p lhs, bh_multi_array_uint16_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint16_greater_than_scalar_lhs(bh_uint16 lhs, bh_multi_array_uint16_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint16_greater_than_scalar_rhs(bh_multi_array_uint16_p lhs, bh_uint16 rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint32_greater_than(bh_multi_array_uint32_p lhs, bh_multi_array_uint32_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint32_greater_than_scalar_lhs(bh_uint32 lhs, bh_multi_array_uint32_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint32_greater_than_scalar_rhs(bh_multi_array_uint32_p lhs, bh_uint32 rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint64_greater_than(bh_multi_array_uint64_p lhs, bh_multi_array_uint64_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint64_greater_than_scalar_lhs(bh_uint64 lhs, bh_multi_array_uint64_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint64_greater_than_scalar_rhs(bh_multi_array_uint64_p lhs, bh_uint64 rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint8_greater_than(bh_multi_array_uint8_p lhs, bh_multi_array_uint8_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint8_greater_than_scalar_lhs(bh_uint8 lhs, bh_multi_array_uint8_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint8_greater_than_scalar_rhs(bh_multi_array_uint8_p lhs, bh_uint8 rhs);

DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_greater_than_or_equal_to(bh_multi_array_bool8_p lhs, bh_multi_array_bool8_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_greater_than_or_equal_to_scalar_lhs(bh_bool lhs, bh_multi_array_bool8_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_greater_than_or_equal_to_scalar_rhs(bh_multi_array_bool8_p lhs, bh_bool rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_float32_greater_than_or_equal_to(bh_multi_array_float32_p lhs, bh_multi_array_float32_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_float32_greater_than_or_equal_to_scalar_lhs(bh_float32 lhs, bh_multi_array_float32_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_float32_greater_than_or_equal_to_scalar_rhs(bh_multi_array_float32_p lhs, bh_float32 rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_float64_greater_than_or_equal_to(bh_multi_array_float64_p lhs, bh_multi_array_float64_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_float64_greater_than_or_equal_to_scalar_lhs(bh_float64 lhs, bh_multi_array_float64_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_float64_greater_than_or_equal_to_scalar_rhs(bh_multi_array_float64_p lhs, bh_float64 rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int16_greater_than_or_equal_to(bh_multi_array_int16_p lhs, bh_multi_array_int16_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int16_greater_than_or_equal_to_scalar_lhs(bh_int16 lhs, bh_multi_array_int16_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int16_greater_than_or_equal_to_scalar_rhs(bh_multi_array_int16_p lhs, bh_int16 rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int32_greater_than_or_equal_to(bh_multi_array_int32_p lhs, bh_multi_array_int32_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int32_greater_than_or_equal_to_scalar_lhs(bh_int32 lhs, bh_multi_array_int32_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int32_greater_than_or_equal_to_scalar_rhs(bh_multi_array_int32_p lhs, bh_int32 rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int64_greater_than_or_equal_to(bh_multi_array_int64_p lhs, bh_multi_array_int64_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int64_greater_than_or_equal_to_scalar_lhs(bh_int64 lhs, bh_multi_array_int64_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int64_greater_than_or_equal_to_scalar_rhs(bh_multi_array_int64_p lhs, bh_int64 rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int8_greater_than_or_equal_to(bh_multi_array_int8_p lhs, bh_multi_array_int8_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int8_greater_than_or_equal_to_scalar_lhs(bh_int8 lhs, bh_multi_array_int8_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int8_greater_than_or_equal_to_scalar_rhs(bh_multi_array_int8_p lhs, bh_int8 rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint16_greater_than_or_equal_to(bh_multi_array_uint16_p lhs, bh_multi_array_uint16_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint16_greater_than_or_equal_to_scalar_lhs(bh_uint16 lhs, bh_multi_array_uint16_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint16_greater_than_or_equal_to_scalar_rhs(bh_multi_array_uint16_p lhs, bh_uint16 rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint32_greater_than_or_equal_to(bh_multi_array_uint32_p lhs, bh_multi_array_uint32_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint32_greater_than_or_equal_to_scalar_lhs(bh_uint32 lhs, bh_multi_array_uint32_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint32_greater_than_or_equal_to_scalar_rhs(bh_multi_array_uint32_p lhs, bh_uint32 rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint64_greater_than_or_equal_to(bh_multi_array_uint64_p lhs, bh_multi_array_uint64_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint64_greater_than_or_equal_to_scalar_lhs(bh_uint64 lhs, bh_multi_array_uint64_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint64_greater_than_or_equal_to_scalar_rhs(bh_multi_array_uint64_p lhs, bh_uint64 rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint8_greater_than_or_equal_to(bh_multi_array_uint8_p lhs, bh_multi_array_uint8_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint8_greater_than_or_equal_to_scalar_lhs(bh_uint8 lhs, bh_multi_array_uint8_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint8_greater_than_or_equal_to_scalar_rhs(bh_multi_array_uint8_p lhs, bh_uint8 rhs);

DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_less_than(bh_multi_array_bool8_p lhs, bh_multi_array_bool8_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_less_than_scalar_lhs(bh_bool lhs, bh_multi_array_bool8_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_less_than_scalar_rhs(bh_multi_array_bool8_p lhs, bh_bool rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_float32_less_than(bh_multi_array_float32_p lhs, bh_multi_array_float32_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_float32_less_than_scalar_lhs(bh_float32 lhs, bh_multi_array_float32_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_float32_less_than_scalar_rhs(bh_multi_array_float32_p lhs, bh_float32 rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_float64_less_than(bh_multi_array_float64_p lhs, bh_multi_array_float64_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_float64_less_than_scalar_lhs(bh_float64 lhs, bh_multi_array_float64_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_float64_less_than_scalar_rhs(bh_multi_array_float64_p lhs, bh_float64 rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int16_less_than(bh_multi_array_int16_p lhs, bh_multi_array_int16_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int16_less_than_scalar_lhs(bh_int16 lhs, bh_multi_array_int16_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int16_less_than_scalar_rhs(bh_multi_array_int16_p lhs, bh_int16 rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int32_less_than(bh_multi_array_int32_p lhs, bh_multi_array_int32_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int32_less_than_scalar_lhs(bh_int32 lhs, bh_multi_array_int32_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int32_less_than_scalar_rhs(bh_multi_array_int32_p lhs, bh_int32 rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int64_less_than(bh_multi_array_int64_p lhs, bh_multi_array_int64_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int64_less_than_scalar_lhs(bh_int64 lhs, bh_multi_array_int64_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int64_less_than_scalar_rhs(bh_multi_array_int64_p lhs, bh_int64 rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int8_less_than(bh_multi_array_int8_p lhs, bh_multi_array_int8_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int8_less_than_scalar_lhs(bh_int8 lhs, bh_multi_array_int8_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int8_less_than_scalar_rhs(bh_multi_array_int8_p lhs, bh_int8 rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint16_less_than(bh_multi_array_uint16_p lhs, bh_multi_array_uint16_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint16_less_than_scalar_lhs(bh_uint16 lhs, bh_multi_array_uint16_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint16_less_than_scalar_rhs(bh_multi_array_uint16_p lhs, bh_uint16 rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint32_less_than(bh_multi_array_uint32_p lhs, bh_multi_array_uint32_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint32_less_than_scalar_lhs(bh_uint32 lhs, bh_multi_array_uint32_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint32_less_than_scalar_rhs(bh_multi_array_uint32_p lhs, bh_uint32 rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint64_less_than(bh_multi_array_uint64_p lhs, bh_multi_array_uint64_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint64_less_than_scalar_lhs(bh_uint64 lhs, bh_multi_array_uint64_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint64_less_than_scalar_rhs(bh_multi_array_uint64_p lhs, bh_uint64 rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint8_less_than(bh_multi_array_uint8_p lhs, bh_multi_array_uint8_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint8_less_than_scalar_lhs(bh_uint8 lhs, bh_multi_array_uint8_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint8_less_than_scalar_rhs(bh_multi_array_uint8_p lhs, bh_uint8 rhs);

DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_less_than_or_equal_to(bh_multi_array_bool8_p lhs, bh_multi_array_bool8_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_less_than_or_equal_to_scalar_lhs(bh_bool lhs, bh_multi_array_bool8_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_less_than_or_equal_to_scalar_rhs(bh_multi_array_bool8_p lhs, bh_bool rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_float32_less_than_or_equal_to(bh_multi_array_float32_p lhs, bh_multi_array_float32_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_float32_less_than_or_equal_to_scalar_lhs(bh_float32 lhs, bh_multi_array_float32_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_float32_less_than_or_equal_to_scalar_rhs(bh_multi_array_float32_p lhs, bh_float32 rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_float64_less_than_or_equal_to(bh_multi_array_float64_p lhs, bh_multi_array_float64_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_float64_less_than_or_equal_to_scalar_lhs(bh_float64 lhs, bh_multi_array_float64_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_float64_less_than_or_equal_to_scalar_rhs(bh_multi_array_float64_p lhs, bh_float64 rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int16_less_than_or_equal_to(bh_multi_array_int16_p lhs, bh_multi_array_int16_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int16_less_than_or_equal_to_scalar_lhs(bh_int16 lhs, bh_multi_array_int16_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int16_less_than_or_equal_to_scalar_rhs(bh_multi_array_int16_p lhs, bh_int16 rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int32_less_than_or_equal_to(bh_multi_array_int32_p lhs, bh_multi_array_int32_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int32_less_than_or_equal_to_scalar_lhs(bh_int32 lhs, bh_multi_array_int32_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int32_less_than_or_equal_to_scalar_rhs(bh_multi_array_int32_p lhs, bh_int32 rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int64_less_than_or_equal_to(bh_multi_array_int64_p lhs, bh_multi_array_int64_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int64_less_than_or_equal_to_scalar_lhs(bh_int64 lhs, bh_multi_array_int64_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int64_less_than_or_equal_to_scalar_rhs(bh_multi_array_int64_p lhs, bh_int64 rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int8_less_than_or_equal_to(bh_multi_array_int8_p lhs, bh_multi_array_int8_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int8_less_than_or_equal_to_scalar_lhs(bh_int8 lhs, bh_multi_array_int8_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_int8_less_than_or_equal_to_scalar_rhs(bh_multi_array_int8_p lhs, bh_int8 rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint16_less_than_or_equal_to(bh_multi_array_uint16_p lhs, bh_multi_array_uint16_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint16_less_than_or_equal_to_scalar_lhs(bh_uint16 lhs, bh_multi_array_uint16_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint16_less_than_or_equal_to_scalar_rhs(bh_multi_array_uint16_p lhs, bh_uint16 rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint32_less_than_or_equal_to(bh_multi_array_uint32_p lhs, bh_multi_array_uint32_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint32_less_than_or_equal_to_scalar_lhs(bh_uint32 lhs, bh_multi_array_uint32_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint32_less_than_or_equal_to_scalar_rhs(bh_multi_array_uint32_p lhs, bh_uint32 rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint64_less_than_or_equal_to(bh_multi_array_uint64_p lhs, bh_multi_array_uint64_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint64_less_than_or_equal_to_scalar_lhs(bh_uint64 lhs, bh_multi_array_uint64_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint64_less_than_or_equal_to_scalar_rhs(bh_multi_array_uint64_p lhs, bh_uint64 rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint8_less_than_or_equal_to(bh_multi_array_uint8_p lhs, bh_multi_array_uint8_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint8_less_than_or_equal_to_scalar_lhs(bh_uint8 lhs, bh_multi_array_uint8_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_uint8_less_than_or_equal_to_scalar_rhs(bh_multi_array_uint8_p lhs, bh_uint8 rhs);

DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_logical_and(bh_multi_array_bool8_p lhs, bh_multi_array_bool8_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_logical_and_scalar_lhs(bh_bool lhs, bh_multi_array_bool8_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_logical_and_scalar_rhs(bh_multi_array_bool8_p lhs, bh_bool rhs);

DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_logical_or(bh_multi_array_bool8_p lhs, bh_multi_array_bool8_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_logical_or_scalar_lhs(bh_bool lhs, bh_multi_array_bool8_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_logical_or_scalar_rhs(bh_multi_array_bool8_p lhs, bh_bool rhs);

DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_bitwise_and(bh_multi_array_bool8_p lhs, bh_multi_array_bool8_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_bitwise_and_scalar_lhs(bh_bool lhs, bh_multi_array_bool8_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_bitwise_and_scalar_rhs(bh_multi_array_bool8_p lhs, bh_bool rhs);
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_bitwise_and(bh_multi_array_int16_p lhs, bh_multi_array_int16_p rhs);
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_bitwise_and_scalar_lhs(bh_int16 lhs, bh_multi_array_int16_p rhs);
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_bitwise_and_scalar_rhs(bh_multi_array_int16_p lhs, bh_int16 rhs);
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_bitwise_and(bh_multi_array_int32_p lhs, bh_multi_array_int32_p rhs);
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_bitwise_and_scalar_lhs(bh_int32 lhs, bh_multi_array_int32_p rhs);
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_bitwise_and_scalar_rhs(bh_multi_array_int32_p lhs, bh_int32 rhs);
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_bitwise_and(bh_multi_array_int64_p lhs, bh_multi_array_int64_p rhs);
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_bitwise_and_scalar_lhs(bh_int64 lhs, bh_multi_array_int64_p rhs);
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_bitwise_and_scalar_rhs(bh_multi_array_int64_p lhs, bh_int64 rhs);
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_bitwise_and(bh_multi_array_int8_p lhs, bh_multi_array_int8_p rhs);
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_bitwise_and_scalar_lhs(bh_int8 lhs, bh_multi_array_int8_p rhs);
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_bitwise_and_scalar_rhs(bh_multi_array_int8_p lhs, bh_int8 rhs);
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_bitwise_and(bh_multi_array_uint16_p lhs, bh_multi_array_uint16_p rhs);
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_bitwise_and_scalar_lhs(bh_uint16 lhs, bh_multi_array_uint16_p rhs);
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_bitwise_and_scalar_rhs(bh_multi_array_uint16_p lhs, bh_uint16 rhs);
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_bitwise_and(bh_multi_array_uint32_p lhs, bh_multi_array_uint32_p rhs);
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_bitwise_and_scalar_lhs(bh_uint32 lhs, bh_multi_array_uint32_p rhs);
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_bitwise_and_scalar_rhs(bh_multi_array_uint32_p lhs, bh_uint32 rhs);
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_bitwise_and(bh_multi_array_uint64_p lhs, bh_multi_array_uint64_p rhs);
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_bitwise_and_scalar_lhs(bh_uint64 lhs, bh_multi_array_uint64_p rhs);
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_bitwise_and_scalar_rhs(bh_multi_array_uint64_p lhs, bh_uint64 rhs);
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_bitwise_and(bh_multi_array_uint8_p lhs, bh_multi_array_uint8_p rhs);
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_bitwise_and_scalar_lhs(bh_uint8 lhs, bh_multi_array_uint8_p rhs);
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_bitwise_and_scalar_rhs(bh_multi_array_uint8_p lhs, bh_uint8 rhs);

DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_bitwise_or(bh_multi_array_bool8_p lhs, bh_multi_array_bool8_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_bitwise_or_scalar_lhs(bh_bool lhs, bh_multi_array_bool8_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_bitwise_or_scalar_rhs(bh_multi_array_bool8_p lhs, bh_bool rhs);
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_bitwise_or(bh_multi_array_int16_p lhs, bh_multi_array_int16_p rhs);
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_bitwise_or_scalar_lhs(bh_int16 lhs, bh_multi_array_int16_p rhs);
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_bitwise_or_scalar_rhs(bh_multi_array_int16_p lhs, bh_int16 rhs);
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_bitwise_or(bh_multi_array_int32_p lhs, bh_multi_array_int32_p rhs);
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_bitwise_or_scalar_lhs(bh_int32 lhs, bh_multi_array_int32_p rhs);
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_bitwise_or_scalar_rhs(bh_multi_array_int32_p lhs, bh_int32 rhs);
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_bitwise_or(bh_multi_array_int64_p lhs, bh_multi_array_int64_p rhs);
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_bitwise_or_scalar_lhs(bh_int64 lhs, bh_multi_array_int64_p rhs);
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_bitwise_or_scalar_rhs(bh_multi_array_int64_p lhs, bh_int64 rhs);
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_bitwise_or(bh_multi_array_int8_p lhs, bh_multi_array_int8_p rhs);
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_bitwise_or_scalar_lhs(bh_int8 lhs, bh_multi_array_int8_p rhs);
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_bitwise_or_scalar_rhs(bh_multi_array_int8_p lhs, bh_int8 rhs);
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_bitwise_or(bh_multi_array_uint16_p lhs, bh_multi_array_uint16_p rhs);
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_bitwise_or_scalar_lhs(bh_uint16 lhs, bh_multi_array_uint16_p rhs);
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_bitwise_or_scalar_rhs(bh_multi_array_uint16_p lhs, bh_uint16 rhs);
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_bitwise_or(bh_multi_array_uint32_p lhs, bh_multi_array_uint32_p rhs);
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_bitwise_or_scalar_lhs(bh_uint32 lhs, bh_multi_array_uint32_p rhs);
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_bitwise_or_scalar_rhs(bh_multi_array_uint32_p lhs, bh_uint32 rhs);
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_bitwise_or(bh_multi_array_uint64_p lhs, bh_multi_array_uint64_p rhs);
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_bitwise_or_scalar_lhs(bh_uint64 lhs, bh_multi_array_uint64_p rhs);
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_bitwise_or_scalar_rhs(bh_multi_array_uint64_p lhs, bh_uint64 rhs);
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_bitwise_or(bh_multi_array_uint8_p lhs, bh_multi_array_uint8_p rhs);
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_bitwise_or_scalar_lhs(bh_uint8 lhs, bh_multi_array_uint8_p rhs);
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_bitwise_or_scalar_rhs(bh_multi_array_uint8_p lhs, bh_uint8 rhs);

DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_bitwise_xor(bh_multi_array_bool8_p lhs, bh_multi_array_bool8_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_bitwise_xor_scalar_lhs(bh_bool lhs, bh_multi_array_bool8_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_bitwise_xor_scalar_rhs(bh_multi_array_bool8_p lhs, bh_bool rhs);
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_bitwise_xor(bh_multi_array_int16_p lhs, bh_multi_array_int16_p rhs);
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_bitwise_xor_scalar_lhs(bh_int16 lhs, bh_multi_array_int16_p rhs);
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_bitwise_xor_scalar_rhs(bh_multi_array_int16_p lhs, bh_int16 rhs);
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_bitwise_xor(bh_multi_array_int32_p lhs, bh_multi_array_int32_p rhs);
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_bitwise_xor_scalar_lhs(bh_int32 lhs, bh_multi_array_int32_p rhs);
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_bitwise_xor_scalar_rhs(bh_multi_array_int32_p lhs, bh_int32 rhs);
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_bitwise_xor(bh_multi_array_int64_p lhs, bh_multi_array_int64_p rhs);
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_bitwise_xor_scalar_lhs(bh_int64 lhs, bh_multi_array_int64_p rhs);
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_bitwise_xor_scalar_rhs(bh_multi_array_int64_p lhs, bh_int64 rhs);
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_bitwise_xor(bh_multi_array_int8_p lhs, bh_multi_array_int8_p rhs);
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_bitwise_xor_scalar_lhs(bh_int8 lhs, bh_multi_array_int8_p rhs);
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_bitwise_xor_scalar_rhs(bh_multi_array_int8_p lhs, bh_int8 rhs);
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_bitwise_xor(bh_multi_array_uint16_p lhs, bh_multi_array_uint16_p rhs);
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_bitwise_xor_scalar_lhs(bh_uint16 lhs, bh_multi_array_uint16_p rhs);
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_bitwise_xor_scalar_rhs(bh_multi_array_uint16_p lhs, bh_uint16 rhs);
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_bitwise_xor(bh_multi_array_uint32_p lhs, bh_multi_array_uint32_p rhs);
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_bitwise_xor_scalar_lhs(bh_uint32 lhs, bh_multi_array_uint32_p rhs);
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_bitwise_xor_scalar_rhs(bh_multi_array_uint32_p lhs, bh_uint32 rhs);
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_bitwise_xor(bh_multi_array_uint64_p lhs, bh_multi_array_uint64_p rhs);
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_bitwise_xor_scalar_lhs(bh_uint64 lhs, bh_multi_array_uint64_p rhs);
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_bitwise_xor_scalar_rhs(bh_multi_array_uint64_p lhs, bh_uint64 rhs);
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_bitwise_xor(bh_multi_array_uint8_p lhs, bh_multi_array_uint8_p rhs);
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_bitwise_xor_scalar_lhs(bh_uint8 lhs, bh_multi_array_uint8_p rhs);
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_bitwise_xor_scalar_rhs(bh_multi_array_uint8_p lhs, bh_uint8 rhs);

DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_power(bh_multi_array_complex128_p lhs, bh_multi_array_complex128_p rhs);
DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_power_scalar_lhs(bh_complex128 lhs, bh_multi_array_complex128_p rhs);
DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_power_scalar_rhs(bh_multi_array_complex128_p lhs, bh_complex128 rhs);
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_power(bh_multi_array_complex64_p lhs, bh_multi_array_complex64_p rhs);
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_power_scalar_lhs(bh_complex64 lhs, bh_multi_array_complex64_p rhs);
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_power_scalar_rhs(bh_multi_array_complex64_p lhs, bh_complex64 rhs);
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_power(bh_multi_array_float32_p lhs, bh_multi_array_float32_p rhs);
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_power_scalar_lhs(bh_float32 lhs, bh_multi_array_float32_p rhs);
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_power_scalar_rhs(bh_multi_array_float32_p lhs, bh_float32 rhs);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_power(bh_multi_array_float64_p lhs, bh_multi_array_float64_p rhs);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_power_scalar_lhs(bh_float64 lhs, bh_multi_array_float64_p rhs);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_power_scalar_rhs(bh_multi_array_float64_p lhs, bh_float64 rhs);
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_power(bh_multi_array_int16_p lhs, bh_multi_array_int16_p rhs);
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_power_scalar_lhs(bh_int16 lhs, bh_multi_array_int16_p rhs);
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_power_scalar_rhs(bh_multi_array_int16_p lhs, bh_int16 rhs);
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_power(bh_multi_array_int32_p lhs, bh_multi_array_int32_p rhs);
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_power_scalar_lhs(bh_int32 lhs, bh_multi_array_int32_p rhs);
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_power_scalar_rhs(bh_multi_array_int32_p lhs, bh_int32 rhs);
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_power(bh_multi_array_int64_p lhs, bh_multi_array_int64_p rhs);
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_power_scalar_lhs(bh_int64 lhs, bh_multi_array_int64_p rhs);
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_power_scalar_rhs(bh_multi_array_int64_p lhs, bh_int64 rhs);
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_power(bh_multi_array_int8_p lhs, bh_multi_array_int8_p rhs);
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_power_scalar_lhs(bh_int8 lhs, bh_multi_array_int8_p rhs);
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_power_scalar_rhs(bh_multi_array_int8_p lhs, bh_int8 rhs);
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_power(bh_multi_array_uint16_p lhs, bh_multi_array_uint16_p rhs);
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_power_scalar_lhs(bh_uint16 lhs, bh_multi_array_uint16_p rhs);
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_power_scalar_rhs(bh_multi_array_uint16_p lhs, bh_uint16 rhs);
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_power(bh_multi_array_uint32_p lhs, bh_multi_array_uint32_p rhs);
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_power_scalar_lhs(bh_uint32 lhs, bh_multi_array_uint32_p rhs);
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_power_scalar_rhs(bh_multi_array_uint32_p lhs, bh_uint32 rhs);
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_power(bh_multi_array_uint64_p lhs, bh_multi_array_uint64_p rhs);
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_power_scalar_lhs(bh_uint64 lhs, bh_multi_array_uint64_p rhs);
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_power_scalar_rhs(bh_multi_array_uint64_p lhs, bh_uint64 rhs);
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_power(bh_multi_array_uint8_p lhs, bh_multi_array_uint8_p rhs);
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_power_scalar_lhs(bh_uint8 lhs, bh_multi_array_uint8_p rhs);
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_power_scalar_rhs(bh_multi_array_uint8_p lhs, bh_uint8 rhs);

DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_maximum(bh_multi_array_bool8_p lhs, bh_multi_array_bool8_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_maximum_scalar_lhs(bh_bool lhs, bh_multi_array_bool8_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_maximum_scalar_rhs(bh_multi_array_bool8_p lhs, bh_bool rhs);
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_maximum(bh_multi_array_float32_p lhs, bh_multi_array_float32_p rhs);
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_maximum_scalar_lhs(bh_float32 lhs, bh_multi_array_float32_p rhs);
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_maximum_scalar_rhs(bh_multi_array_float32_p lhs, bh_float32 rhs);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_maximum(bh_multi_array_float64_p lhs, bh_multi_array_float64_p rhs);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_maximum_scalar_lhs(bh_float64 lhs, bh_multi_array_float64_p rhs);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_maximum_scalar_rhs(bh_multi_array_float64_p lhs, bh_float64 rhs);
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_maximum(bh_multi_array_int16_p lhs, bh_multi_array_int16_p rhs);
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_maximum_scalar_lhs(bh_int16 lhs, bh_multi_array_int16_p rhs);
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_maximum_scalar_rhs(bh_multi_array_int16_p lhs, bh_int16 rhs);
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_maximum(bh_multi_array_int32_p lhs, bh_multi_array_int32_p rhs);
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_maximum_scalar_lhs(bh_int32 lhs, bh_multi_array_int32_p rhs);
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_maximum_scalar_rhs(bh_multi_array_int32_p lhs, bh_int32 rhs);
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_maximum(bh_multi_array_int64_p lhs, bh_multi_array_int64_p rhs);
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_maximum_scalar_lhs(bh_int64 lhs, bh_multi_array_int64_p rhs);
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_maximum_scalar_rhs(bh_multi_array_int64_p lhs, bh_int64 rhs);
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_maximum(bh_multi_array_int8_p lhs, bh_multi_array_int8_p rhs);
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_maximum_scalar_lhs(bh_int8 lhs, bh_multi_array_int8_p rhs);
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_maximum_scalar_rhs(bh_multi_array_int8_p lhs, bh_int8 rhs);
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_maximum(bh_multi_array_uint16_p lhs, bh_multi_array_uint16_p rhs);
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_maximum_scalar_lhs(bh_uint16 lhs, bh_multi_array_uint16_p rhs);
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_maximum_scalar_rhs(bh_multi_array_uint16_p lhs, bh_uint16 rhs);
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_maximum(bh_multi_array_uint32_p lhs, bh_multi_array_uint32_p rhs);
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_maximum_scalar_lhs(bh_uint32 lhs, bh_multi_array_uint32_p rhs);
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_maximum_scalar_rhs(bh_multi_array_uint32_p lhs, bh_uint32 rhs);
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_maximum(bh_multi_array_uint64_p lhs, bh_multi_array_uint64_p rhs);
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_maximum_scalar_lhs(bh_uint64 lhs, bh_multi_array_uint64_p rhs);
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_maximum_scalar_rhs(bh_multi_array_uint64_p lhs, bh_uint64 rhs);
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_maximum(bh_multi_array_uint8_p lhs, bh_multi_array_uint8_p rhs);
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_maximum_scalar_lhs(bh_uint8 lhs, bh_multi_array_uint8_p rhs);
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_maximum_scalar_rhs(bh_multi_array_uint8_p lhs, bh_uint8 rhs);

DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_minimin(bh_multi_array_bool8_p lhs, bh_multi_array_bool8_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_minimin_scalar_lhs(bh_bool lhs, bh_multi_array_bool8_p rhs);
DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_minimin_scalar_rhs(bh_multi_array_bool8_p lhs, bh_bool rhs);
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_minimin(bh_multi_array_float32_p lhs, bh_multi_array_float32_p rhs);
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_minimin_scalar_lhs(bh_float32 lhs, bh_multi_array_float32_p rhs);
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_minimin_scalar_rhs(bh_multi_array_float32_p lhs, bh_float32 rhs);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_minimin(bh_multi_array_float64_p lhs, bh_multi_array_float64_p rhs);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_minimin_scalar_lhs(bh_float64 lhs, bh_multi_array_float64_p rhs);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_minimin_scalar_rhs(bh_multi_array_float64_p lhs, bh_float64 rhs);
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_minimin(bh_multi_array_int16_p lhs, bh_multi_array_int16_p rhs);
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_minimin_scalar_lhs(bh_int16 lhs, bh_multi_array_int16_p rhs);
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_minimin_scalar_rhs(bh_multi_array_int16_p lhs, bh_int16 rhs);
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_minimin(bh_multi_array_int32_p lhs, bh_multi_array_int32_p rhs);
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_minimin_scalar_lhs(bh_int32 lhs, bh_multi_array_int32_p rhs);
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_minimin_scalar_rhs(bh_multi_array_int32_p lhs, bh_int32 rhs);
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_minimin(bh_multi_array_int64_p lhs, bh_multi_array_int64_p rhs);
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_minimin_scalar_lhs(bh_int64 lhs, bh_multi_array_int64_p rhs);
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_minimin_scalar_rhs(bh_multi_array_int64_p lhs, bh_int64 rhs);
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_minimin(bh_multi_array_int8_p lhs, bh_multi_array_int8_p rhs);
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_minimin_scalar_lhs(bh_int8 lhs, bh_multi_array_int8_p rhs);
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_minimin_scalar_rhs(bh_multi_array_int8_p lhs, bh_int8 rhs);
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_minimin(bh_multi_array_uint16_p lhs, bh_multi_array_uint16_p rhs);
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_minimin_scalar_lhs(bh_uint16 lhs, bh_multi_array_uint16_p rhs);
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_minimin_scalar_rhs(bh_multi_array_uint16_p lhs, bh_uint16 rhs);
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_minimin(bh_multi_array_uint32_p lhs, bh_multi_array_uint32_p rhs);
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_minimin_scalar_lhs(bh_uint32 lhs, bh_multi_array_uint32_p rhs);
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_minimin_scalar_rhs(bh_multi_array_uint32_p lhs, bh_uint32 rhs);
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_minimin(bh_multi_array_uint64_p lhs, bh_multi_array_uint64_p rhs);
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_minimin_scalar_lhs(bh_uint64 lhs, bh_multi_array_uint64_p rhs);
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_minimin_scalar_rhs(bh_multi_array_uint64_p lhs, bh_uint64 rhs);
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_minimin(bh_multi_array_uint8_p lhs, bh_multi_array_uint8_p rhs);
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_minimin_scalar_lhs(bh_uint8 lhs, bh_multi_array_uint8_p rhs);
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_minimin_scalar_rhs(bh_multi_array_uint8_p lhs, bh_uint8 rhs);


// Unary functions

DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_logical_not(bh_multi_array_bool8_p lhs);

DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_bitwise_invert(bh_multi_array_bool8_p lhs);
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_bitwise_invert(bh_multi_array_int16_p lhs);
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_bitwise_invert(bh_multi_array_int32_p lhs);
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_bitwise_invert(bh_multi_array_int64_p lhs);
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_bitwise_invert(bh_multi_array_int8_p lhs);
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_bitwise_invert(bh_multi_array_uint16_p lhs);
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_bitwise_invert(bh_multi_array_uint32_p lhs);
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_bitwise_invert(bh_multi_array_uint64_p lhs);
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_bitwise_invert(bh_multi_array_uint8_p lhs);

DLLEXPORT bh_multi_array_bool8_p bh_multi_array_bool8_absolute(bh_multi_array_bool8_p lhs);
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_absolute(bh_multi_array_float32_p lhs);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_absolute(bh_multi_array_float64_p lhs);
DLLEXPORT bh_multi_array_int16_p bh_multi_array_int16_absolute(bh_multi_array_int16_p lhs);
DLLEXPORT bh_multi_array_int32_p bh_multi_array_int32_absolute(bh_multi_array_int32_p lhs);
DLLEXPORT bh_multi_array_int64_p bh_multi_array_int64_absolute(bh_multi_array_int64_p lhs);
DLLEXPORT bh_multi_array_int8_p bh_multi_array_int8_absolute(bh_multi_array_int8_p lhs);
DLLEXPORT bh_multi_array_uint16_p bh_multi_array_uint16_absolute(bh_multi_array_uint16_p lhs);
DLLEXPORT bh_multi_array_uint32_p bh_multi_array_uint32_absolute(bh_multi_array_uint32_p lhs);
DLLEXPORT bh_multi_array_uint64_p bh_multi_array_uint64_absolute(bh_multi_array_uint64_p lhs);
DLLEXPORT bh_multi_array_uint8_p bh_multi_array_uint8_absolute(bh_multi_array_uint8_p lhs);

DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_sin(bh_multi_array_complex128_p lhs);
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_sin(bh_multi_array_complex64_p lhs);
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_sin(bh_multi_array_float32_p lhs);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_sin(bh_multi_array_float64_p lhs);

DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_cos(bh_multi_array_complex128_p lhs);
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_cos(bh_multi_array_complex64_p lhs);
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_cos(bh_multi_array_float32_p lhs);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_cos(bh_multi_array_float64_p lhs);

DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_tan(bh_multi_array_complex128_p lhs);
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_tan(bh_multi_array_complex64_p lhs);
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_tan(bh_multi_array_float32_p lhs);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_tan(bh_multi_array_float64_p lhs);

DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_sinh(bh_multi_array_complex128_p lhs);
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_sinh(bh_multi_array_complex64_p lhs);
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_sinh(bh_multi_array_float32_p lhs);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_sinh(bh_multi_array_float64_p lhs);

DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_cosh(bh_multi_array_complex128_p lhs);
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_cosh(bh_multi_array_complex64_p lhs);
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_cosh(bh_multi_array_float32_p lhs);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_cosh(bh_multi_array_float64_p lhs);

DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_tanh(bh_multi_array_complex128_p lhs);
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_tanh(bh_multi_array_complex64_p lhs);
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_tanh(bh_multi_array_float32_p lhs);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_tanh(bh_multi_array_float64_p lhs);

DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_asin(bh_multi_array_float32_p lhs);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_asin(bh_multi_array_float64_p lhs);

DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_acos(bh_multi_array_float32_p lhs);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_acos(bh_multi_array_float64_p lhs);

DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_atan(bh_multi_array_float32_p lhs);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_atan(bh_multi_array_float64_p lhs);

DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_asinh(bh_multi_array_float32_p lhs);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_asinh(bh_multi_array_float64_p lhs);

DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_acosh(bh_multi_array_float32_p lhs);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_acosh(bh_multi_array_float64_p lhs);

DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_atanh(bh_multi_array_float32_p lhs);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_atanh(bh_multi_array_float64_p lhs);

DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_exp(bh_multi_array_complex128_p lhs);
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_exp(bh_multi_array_complex64_p lhs);
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_exp(bh_multi_array_float32_p lhs);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_exp(bh_multi_array_float64_p lhs);

DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_exp2(bh_multi_array_float32_p lhs);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_exp2(bh_multi_array_float64_p lhs);

DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_expm1(bh_multi_array_float32_p lhs);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_expm1(bh_multi_array_float64_p lhs);

DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_log(bh_multi_array_complex128_p lhs);
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_log(bh_multi_array_complex64_p lhs);
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_log(bh_multi_array_float32_p lhs);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_log(bh_multi_array_float64_p lhs);

DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_log2(bh_multi_array_float32_p lhs);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_log2(bh_multi_array_float64_p lhs);

DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_log10(bh_multi_array_complex128_p lhs);
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_log10(bh_multi_array_complex64_p lhs);
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_log10(bh_multi_array_float32_p lhs);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_log10(bh_multi_array_float64_p lhs);

DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_log1p(bh_multi_array_float32_p lhs);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_log1p(bh_multi_array_float64_p lhs);

DLLEXPORT bh_multi_array_complex128_p bh_multi_array_complex128_sqrt(bh_multi_array_complex128_p lhs);
DLLEXPORT bh_multi_array_complex64_p bh_multi_array_complex64_sqrt(bh_multi_array_complex64_p lhs);
DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_sqrt(bh_multi_array_float32_p lhs);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_sqrt(bh_multi_array_float64_p lhs);

DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_ceil(bh_multi_array_float32_p lhs);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_ceil(bh_multi_array_float64_p lhs);

DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_trunc(bh_multi_array_float32_p lhs);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_trunc(bh_multi_array_float64_p lhs);

DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_floor(bh_multi_array_float32_p lhs);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_floor(bh_multi_array_float64_p lhs);

DLLEXPORT bh_multi_array_float32_p bh_multi_array_float32_rint(bh_multi_array_float32_p lhs);
DLLEXPORT bh_multi_array_float64_p bh_multi_array_float64_rint(bh_multi_array_float64_p lhs);


#ifdef __cplusplus
}
#endif

#endif
