#ifndef DISTNUMPY_H
#define DISTNUMPY_H

#include <cphvb.h>
#include <cphvb_vem.h>

//ufunc definitions from numpy/ufuncobject.h.
//They are included here instead.
//Opcode is added to the original struct.
typedef void (*PyUFuncGenericFunction)
             (char **, npy_intp *, npy_intp *, void *);
typedef struct {
    PyObject_HEAD
    int nin, nout, nargs;
    int identity;
    PyUFuncGenericFunction *functions;
    void **data;
    int ntypes;
    int check_return;
    char *name, *types;
    char *doc;
    void *ptr;
    PyObject *obj;
    PyObject *userloops;
    int core_enabled;
    int core_num_dim_ix;
    int *core_num_dims;
    int *core_dim_ixs;
    int *core_offsets;
    char *core_signature;
    cphvb_opcode opcode;
} PyUFuncObject;

//Easy retrieval of dnduid
#define PyArray_DNDUID(obj) (((PyArrayObject *)(obj))->dnduid)

//Maximum number of allocated arrays
#define DNPY_MAX_NARRAYS 1024

//The maximum size of a instruction batch in bytes (should be power of 2).
#define DNPY_INSTRUCTION_BATCH_MAXSIZE 33554432 //32MB

//The maximum number of instructions in a batch.
#define DNPY_MAX_NINST 1


#endif
