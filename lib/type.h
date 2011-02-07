#ifndef __CPHVB_TYPE_H
#define __CPHVB_TYPE_H
/* Codes for known data types */
enum cphvb_type
{
    CPHVB_INT8,
    CPHVB_INT16,
    CPHVB_INT32,
    CPHVB_INT64,
    CPHVB_UINT8,
    CPHVB_UINT16,
    CPHVB_UINT32,
    CPHVB_UINT64,
    CPHVB_FLOAT16,
    CPHVB_FLOAT32,
    CPHVB_FLOAT64,
    CPHVB_PTR
};

/* Data size in bytes for the different types */
static const int cphvb_typesize[] = 
{ 
    [CPHVB_INT8] = 1,
    [CPHVB_INT16] = 2,
    [CPHVB_INT32] = 4,
    [CPHVB_INT64] = 8,
    [CPHVB_UINT8] = 1,
    [CPHVB_UINT16] = 2,
    [CPHVB_UINT32] = 4,
    [CPHVB_UINT64] = 8,
    [CPHVB_FLOAT16] = 2,
    [CPHVB_FLOAT32] = 4,
    [CPHVB_FLOAT64] = 8,
    [CPHVB_PTR] = sizeof(void*)
};
#define MAX_DATA_SIZE 8

/* Mapping of cphvb data types to C data types */ 
typedef int8_t   cphvb_int8;
typedef int16_t  cphvb_int16;
typedef int32_t  cphvb_int32;
typedef int64_t  cphvb_int64;
typedef uint8_t  cphvb_uint8;
typedef uint16_t cphvb_uint16;
typedef uint32_t cphvb_uint32;
typedef uint64_t cphvb_uint64;
typedef float    cphvb_float32;
typedef double   cphvb_float64;
typedef void**   cphvb_ptr;

typedef union
{
    cphvb_int8     int8;
    cphvb_int16    int16;
    cphvb_int32    int32;
    cphvb_int64    int64;
    cphvb_uint8    uint8;
    cphvb_uint16   uint16;
    cphvb_uint32   uint32;
    cphvb_uint64   uint64;
    cphvb_float32  float32;
    cphvb_float64  float64;
    cphvb_ptr      ptr;
} cphvb_constant;

#endif
