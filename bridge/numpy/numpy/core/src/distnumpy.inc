#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include <cphvb.h>
#include <cphvb_vem.h>

#ifndef CBLAS_HEADER
#define CBLAS_HEADER "../blasdot/cblas.h"
#endif
#include CBLAS_HEADER

//Scheduled instruction. At the moment we only have one batch.
static cphvb_instruction inst_scheduled[DNPY_MAX_NINST];
static cphvb_intp ninst_scheduled=0;

//Pointer to the python module who has the ufunc operators.
static PyObject *ufunc_module;
//Pointers to the cblas functions.
typedef void (cblas_func_type)(const enum CBLAS_ORDER Order,...);
static cblas_func_type *cblas_dgemm_func = NULL;
static cblas_func_type *cblas_sgemm_func = NULL;
static cblas_func_type *cblas_zgemm_func = NULL;
static cblas_func_type *cblas_cgemm_func = NULL;

//Function pointers to the VEM.
static cphvb_vem_init vem_init;
static cphvb_vem_execute vem_execute;
static cphvb_vem_shutdown vem_shutdown;
static cphvb_vem_create_array vem_create_array;
static cphvb_vem_instruction_check vem_instruction_check;

//For now, we determent which VEM to use at compile time.
#ifdef CLUSTER
    #include <cphvb_vem_cluster.h>
#else
    #include <cphvb_vem_node.h>
#endif

//Check that the definitions in numpy are in accordance with cphVB.
#if NPY_BITSOF_SHORT != 16
#    error the NPY_BITSOF_INT not 16 bit
#endif
#if NPY_BITSOF_INT != 32
#    error the NPY_BITSOF_INT not 32 bit
#endif
#if NPY_BITSOF_LONG != 32 && NPY_BITSOF_LONG != 64
#    error the NPY_BITSOF_LONG not 32 or 64 bit
#endif
#if NPY_BITSOF_LONGLONG != 64
#    error the NPY_BITSOF_LONGLONG not 64 bit
#endif
#if NPY_BITSOF_FLOAT != 32
#    error the NPY_BITSOF_FLOAT not 32 bit
#endif
#if NPY_BITSOF_FLOAT == 64
#    error the NPY_BITSOF_FLOAT not 64 bit
#endif


/*===================================================================
 *
 * The data type conversion to and from NumPy and cphVB.
 * Private.
 */
static const cphvb_type const type_py2cph[] =
{
    [NPY_BOOL]   = CPHVB_BOOL,
    [NPY_BYTE]   = CPHVB_INT8,
    [NPY_UBYTE]  = CPHVB_UINT8,
    [NPY_SHORT]  = CPHVB_INT16,
    [NPY_USHORT] = CPHVB_UINT16,
    [NPY_INT]    = CPHVB_INT32,
    [NPY_UINT]   = CPHVB_UINT32,
    #if NPY_BITSOF_LONG == 32
        [NPY_LONG]  = CPHVB_INT32,
        [NPY_ULONG] = CPHVB_UINT32,
    #else
        [NPY_LONG]  = CPHVB_INT64,
        [NPY_ULONG] = CPHVB_UINT64,
    #endif
    [NPY_LONGLONG]    = CPHVB_INT64,
    [NPY_ULONGLONG]   = CPHVB_UINT64,
    [NPY_FLOAT]       = CPHVB_FLOAT32,
    [NPY_DOUBLE]      = CPHVB_FLOAT64,
    [NPY_LONGDOUBLE]  = CPHVB_UNKNOWN,
    [NPY_CFLOAT]      = CPHVB_UNKNOWN,
    [NPY_CDOUBLE]     = CPHVB_UNKNOWN,
    [NPY_CLONGDOUBLE] = CPHVB_UNKNOWN,
    [NPY_OBJECT]      = CPHVB_UNKNOWN,
    [NPY_STRING]      = CPHVB_UNKNOWN,
    [NPY_UNICODE]     = CPHVB_UNKNOWN,
    [NPY_VOID]        = CPHVB_UNKNOWN,
    [NPY_NTYPES]      = CPHVB_UNKNOWN,
    [NPY_NOTYPE]      = CPHVB_UNKNOWN,
    [NPY_CHAR]        = CPHVB_UNKNOWN,
    [NPY_USERDEF]     = CPHVB_UNKNOWN
};
static const cphvb_type const type_cph2py[] =
{
    [CPHVB_BOOL]    = NPY_BOOL,
    [CPHVB_INT8]    = NPY_BYTE,
    [CPHVB_UINT8]   = NPY_UBYTE,
    [CPHVB_INT16]   = NPY_SHORT,
    [CPHVB_UINT16]  = NPY_USHORT,
    [CPHVB_INT32]   = NPY_INT,
    [CPHVB_UINT32]  = NPY_UINT,
    #if NPY_BITSOF_LONG == 32
        [CPHVB_INT32]  = NPY_LONG,
        [CPHVB_UINT32] = NPY_ULONG,
    #else
        [CPHVB_INT64]  = NPY_LONG,
        [CPHVB_UINT64] = NPY_ULONG,
    #endif
    [CPHVB_INT64]   = NPY_LONGLONG,
    [CPHVB_UINT64]  = NPY_ULONGLONG,
    [CPHVB_FLOAT32] = NPY_FLOAT,
    [CPHVB_FLOAT64] = NPY_DOUBLE
};


/*
 *===================================================================
 * Schedule one instruction.
 * Private
 */
static void
util_schedule(cphvb_instruction *inst, int force_flush)
{
    cphvb_error error;
    if(inst != NULL)
        inst_scheduled[ninst_scheduled++] = *inst;

    if(ninst_scheduled >= DNPY_MAX_NINST || force_flush)
    {
        if(ninst_scheduled > 0)
        {
            error = vem_execute(ninst_scheduled, inst_scheduled);
            ninst_scheduled = 0;
            if(error)
            {
                fprintf(stderr, "Unhandled error returned by "
                                "vem_execute() in util_schedule()\n");
                exit(error);
            }
        }
    }
} /* util_schedule */


/*
 *===================================================================
 * Compose a release instruction and execute it.
 * Fatal on error.
 * Private
 */
static void
util_release(cphvb_array *ary)
{
    cphvb_instruction inst;
    //Tell the VEM to release the data to us.
    inst.opcode = CPHVB_RELEASE;
    inst.operand[0] = ary;
    util_schedule(&inst, 1);//Force a flush.
} /* util_release */


/*
 *===================================================================
 * Compose a sync instruction and execute it.
 * Fatal on error.
 * Private
 */
static void
util_sync(cphvb_array *ary)
{
    cphvb_instruction inst;
    //Tell the VEM to syncronize the data
    inst.opcode = CPHVB_SYNC;
    inst.operand[0] = ary;
    util_schedule(&inst, 1);//Force a flush.
} /* util_release */


/*NUMPY_API
 *===================================================================
 * Create dndarray.
 * Return 1 when the dtype is not supported else 0.
 * Public
 */
static intp
dnumpy_create_dndarray(int ndims, intp dims[NPY_MAXDIMS], int numpytype,
                       int zerofill, intp *uid)
{
    int i;
    cphvb_index s;
    cphvb_index stride[CPHVB_MAXDIM];
    cphvb_index shape[CPHVB_MAXDIM];
    cphvb_error error;
    cphvb_array *new_ary;
    cphvb_type dtype = type_py2cph[numpytype];

    if(dtype == CPHVB_UNKNOWN)
    {
        PyErr_SetString(PyExc_RuntimeError,
                        "cphVB does not support the datatype\n");
        return 1;//Datatype error.
    }

    //Compute the stride.
    s=1;
    for(i=ndims-1; i>=0; --i)
    {
        stride[i] = s;
        shape[i] = dims[i];//Convert from 'intp' to 'cphvb_index'.
        s *= shape[i];
    }

    error = vem_create_array(NULL, dtype, ndims, 0, shape, stride,
                             zerofill, (cphvb_constant)0L, &new_ary);

    //We save the array pointer in an 'intp'.
    *uid = (intp) new_ary;

    if(error == CPHVB_OUT_OF_MEMORY)
    {
        PyErr_NoMemory();
        return 1;
    }
    else if (error)
    {
        PyErr_SetString(PyExc_RuntimeError, "unknown cphVB error\n");
        return 1;
    }

    return 0;//Success.

} /* dnumpy_create_dndarray */

/*NUMPY_API
 *===================================================================
 * Destroy dndarray.
 * Public
 */
static void
dnumpy_destroy_dndarray(intp uid)
{
    cphvb_instruction inst;
    inst.opcode = CPHVB_DESTORY;
    inst.operand[0] = (cphvb_array*) uid;

    util_schedule(&inst, 0);

} /* dnumpy_destroy_dndarray */


/*NUMPY_API
 *===================================================================
 * Create a new view based on the 'org_ary'.
 * Public
 */
static intp
dnumpy_create_dndview(intp org_ary_uid, intp ndim,
                      intp dims[NPY_MAXDIMS],
                      intp stride_inbyte[NPY_MAXDIMS],
                      intp offset_inbyte)
{
    cphvb_intp i;
    cphvb_array *org_ary = (cphvb_array *) org_ary_uid;
    cphvb_array *base = cphvb_base_array(org_ary);
    cphvb_array *new_ary;
    cphvb_error error;
    cphvb_index stride[CPHVB_MAXDIM];
    cphvb_index shape[CPHVB_MAXDIM];
    intp dtypesize = cphvb_type_size(org_ary->type);
    cphvb_intp offset = offset_inbyte/dtypesize;

    //Check for memory alignment and convert to element indexes
    //rather than byte indexes.
    if(offset_inbyte % dtypesize > 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "cphVB does not support"
                        " views with an unaligned offset\n");
        return 0;
    }
    for(i=0; i<ndim; ++i)
    {
        shape[i]  = dims[i];
        stride[i] = stride_inbyte[i]/dtypesize;
        if(stride_inbyte[i] % dtypesize > 0)
        {
            PyErr_SetString(PyExc_RuntimeError, "cphVB does not support"
                            " views with unaligned strides\n");
            return 0;
        }
    }
    error = vem_create_array(base, org_ary->type, ndim,
                             offset+org_ary->start,
                             shape, stride, NPY_FALSE,
                             (cphvb_constant)0L, &new_ary);
    if(error == CPHVB_OUT_OF_MEMORY)
    {
        PyErr_NoMemory();
        return 0;
    }
    else if (error)
    {
        PyErr_SetString(PyExc_RuntimeError, "unknown cphVB error\n");
        return 0;
    }
    return (intp) new_ary;

} /* dnumpy_create_dndview */

/*NUMPY_API
 *===================================================================
 * Public
 * Executes all appending operations.
 */
static PyObject *
dnumpy_evalflush(void)
{
    util_schedule(NULL, 1);//Force a flush.
    Py_RETURN_NONE;
} /* dnumpy_evalflush */

/*NUMPY_API
 *===================================================================
 * Public
 * Returns True when the two arrays overlap.
 */
static char
dnumpy_data_overlap(intp A, intp B)
{
    cphvb_array *a = (cphvb_array *) A;
    cphvb_array *b = (cphvb_array *) B;
    assert(A != 0);
    assert(B != 0);

    if(a == b)//No conflict since it is the exact same array-view.
        return 0;
    else if(cphvb_base_array(a) == cphvb_base_array(b))
        return 1;

    return 0;
} /* dnumpy_data_overlap */

/*NUMPY_API
 *===================================================================
 * Assign the value to array at 'coordinate'.
 * 'coordinate' size must be the same as view->ndims.
 * Steals all reference to item. (Item is lost).
 * Public
 */
static int
dnumpy_dndarray_putitem(intp uid, intp coordinate[NPY_MAXDIMS],
                        PyObject *item)
{
    cphvb_array *ary = (cphvb_array*) uid;
    cphvb_array *base = cphvb_base_array(ary);
    cphvb_intp dtypesize = cphvb_type_size(ary->type);
    cphvb_intp d, offset;

    //Convert item to a compatible type.
    PyObject *item2 = PyArray_FROM_O(item);
    PyObject *citem2 = PyArray_Cast((PyArrayObject*)item2,
                                    type_cph2py[ary->type]);

    //Cleanup and return error if the cast failed.
    if(citem2 == NULL)
    {
        Py_DECREF(item2);
        fprintf(stderr,"Cast failed\n");
        return -1;
    }

    //Tell the VEM to release the data to us.
    util_release(ary);

    //We may have to allocate the array memory.
    if(cphvb_malloc_array_data(base) != CPHVB_SUCCESS)
    {
        fprintf(stderr,"Out of memory in dnumpy_dndarray_putitem\n");
        exit(CPHVB_OUT_OF_MEMORY);
    }

    //Compute offset. May be use cphvb_calc_offset()?
    offset = 0;
    for(d=ary->ndim-1; d>=0; --d)
        offset += coordinate[d] * ary->stride[d];
    offset += ary->start;
    offset *= dtypesize;

    //Write item to memory.
    memcpy(base->data + offset, PyArray_DATA(citem2), dtypesize);

    //Clean up.
    Py_DECREF(citem2);
    Py_DECREF(item2);

    return 0;//Succes
} /* dnumpy_dndarray_putitem */

/*NUMPY_API
 *===================================================================
 * Get a single value specified by 'coordinate' from the array.
 * 'coordinate' size must be the same as view->ndims.
 * Public
 */
static void
dnumpy_dndarray_getitem(char *retdata, intp uid,
                        const intp coordinate[NPY_MAXDIMS])
{
    cphvb_array *ary = (cphvb_array*) uid;
    cphvb_array *base = cphvb_base_array(ary);
    cphvb_intp dtypesize = cphvb_type_size(ary->type);
    cphvb_intp d, offset;

    //Tell the VEM to ensure that the data is syncronized.
    util_sync(ary);

    //We may have to allocate the array memory.
    if(cphvb_malloc_array_data(base) != CPHVB_SUCCESS)
    {
        fprintf(stderr,"Out of memory in dnumpy_dndarray_putitem\n");
        exit(CPHVB_OUT_OF_MEMORY);
    }

    //Compute offset. May be use cphvb_calc_offset()?
    offset = 0;
    for(d=ary->ndim-1; d>=0; --d)
        offset += coordinate[d] * ary->stride[d];
    offset += ary->start;
    offset *= dtypesize;

    //Write item to memory.
    memcpy(retdata, base->data + offset, dtypesize);

} /* dnumpy_dndarray_getitem */

/*NUMPY_API
 *===================================================================
 * Public
 * Apply ufunc on distributed arrays in args.
 * Returns -1 on failure.
 */
static int
dnumpy_ufunc(PyUFuncObject *ufunc, PyArrayObject *args[NPY_MAXARGS])
{
    cphvb_intp i;
    cphvb_instruction inst;

    if(ufunc->nargs > CPHVB_MAX_NO_OPERANDS)
    {
        PyErr_SetString(PyExc_RuntimeError, "Ufunc - number of "
                        "arguments exceeds CPHVB_MAX_NO_OPERANDS\n");
        return -1;
    }

    #ifdef DEBUG
        printf("[NumPy] ufunc (output last):");
        for(i=0; i<ufunc->nargs; ++i)
            printf(" %p", (cphvb_array*) args[i]->dnduid);
        printf("\n");
    #endif

    //Create an instruction that represent the ufunc operation.
    //Note that the outputs are the first operands.
    inst.opcode = ufunc->opcode;
    for(i=0; i<ufunc->nin; ++i)
    {
        cphvb_intp j = i + ufunc->nout;
        if(args[i]->dnduid == 0)
        {
            if(args[i]->nd != 0)//This us not a dndarray
            {
                PyErr_SetString(PyExc_RuntimeError,
                                "ufunc - distributed and non-"
                                "distributed arrays do not mix, "
                                "scalars is allowed though.\n");
                return -1;
            }
            //else this is a constant and we have to support the dtype.
            if(type_py2cph[args[i]->descr->type_num] == CPHVB_UNKNOWN)
            {
                PyErr_SetString(PyExc_RuntimeError,
                                "cphVB does not support the datatype\n");
                return -1;
            }
            inst.operand[j] = CPHVB_CONSTANT;
            inst.const_type[j] = type_py2cph[args[i]->descr->type_num];
            memcpy(&inst.constant[j], args[i]->data,
                   args[i]->descr->elsize);
        }
        else
            inst.operand[j] = (cphvb_array*) args[i]->dnduid;
    }
    for(i=0; i<ufunc->nout; ++i)
    {
        cphvb_intp j = i + ufunc->nin;
        if(args[j]->nd == 0)
        {
            PyErr_SetString(PyExc_RuntimeError,
                    "ufunc - scalar outputs is not allowed\n");
            return -1;
        }
        inst.operand[i] = (cphvb_array*) args[j]->dnduid;
    }

    //We may have to apply the ufunc ourselves.
    if(!vem_instruction_check(&inst))
    {
        //Get Python function.
        PyObject *PyOp = PyObject_GetAttrString(ufunc_module,
                                                ufunc->name);
        PyObject *tuple = PyTuple_New(ufunc->nargs);
        assert(PyOp != NULL);

        //Fill the tuple and apply the ufunc.
        for(i=0; i<ufunc->nargs; ++i)
        {
            char *data;
            if(args[i]->dnduid == 0)//Scalar.
            {
                assert(i < ufunc->nin);
                assert(args[i]->nd == 0);
                data = args[i]->data;
            }
            else
            {
                cphvb_array* a = (cphvb_array*) args[i]->dnduid;
                cphvb_array* b = cphvb_base_array(a);

                //We may have to allocate it.
                if(cphvb_malloc_array_data(b) != CPHVB_SUCCESS)
                {
                    fprintf(stderr, "Out of memory "
                                    "in dnumpy_ufunc\n");
                    exit(CPHVB_OUT_OF_MEMORY);
                }
                data = b->data + a->start * cphvb_type_size(a->type);
                //Tell the VEM to release the data to us.
                util_release(a);
            }
            PyObject *tary = PyArray_New(&PyArray_Type,
                                         args[i]->nd,
                                         args[i]->dimensions,
                                         args[i]->descr->type_num,
                                         args[i]->strides,
                                         data, 0,
                                         NPY_BEHAVED, NULL);
            assert(tary != NULL);
            PyTuple_SET_ITEM(tuple, i, tary);
        }
        PyObject *t = PyObject_CallObject(PyOp, tuple);
        assert(t != NULL);

        Py_DECREF(t);
        Py_DECREF(tuple);
        Py_DECREF(PyOp);
    }
    else // Tell the VEM to perform the instruction.
    {
        cphvb_error error;

        //Broadcast to match the number of dimensions of the output by
        //appending and extending 1-length dimensions.
        for(i=ufunc->nout; i<ufunc->nargs; ++i)
        {
            if(inst.operand[i] != CPHVB_CONSTANT )//&&
               //inst.operand[i]->ndim < inst.operand[0]->ndim)
            {
                cphvb_intp j;
                cphvb_intp shape[CPHVB_MAXDIM];
                cphvb_intp stride[CPHVB_MAXDIM];
                cphvb_array *tary;
                cphvb_intp nd_diff = inst.operand[0]->ndim -
                                     inst.operand[i]->ndim;

                for(j=inst.operand[0]->ndim-1; j>=nd_diff; --j)
                {
                    shape[j] = inst.operand[i]->shape[j-nd_diff];
                    if(shape[j] < inst.operand[0]->shape[j])
                    {
                        assert(shape[j] == 1);
                        shape[j] = inst.operand[0]->shape[j];
                        stride[j] = 0;
                    }
                    else
                        stride[j] = inst.operand[i]->stride[j-nd_diff];
                }
                for(j=nd_diff-1; j>=0; --j)
                {
                    shape[j] = 1;
                    stride[j] = 0;
                }
                error = vem_create_array(
                                    cphvb_base_array(inst.operand[i]),
                                    inst.operand[i]->type,
                                    inst.operand[0]->ndim,
                                    inst.operand[i]->start,
                                    shape,
                                    stride,
                                    NPY_FALSE,
                                    (cphvb_constant)0L,
                                    &tary);
                if(error)
                {
                    char errtxt[256];
                    sprintf(errtxt, "Ufunc - internal error when "
                            "calling vem_create_array(): %s.\n",
                            cphvb_error_text(error));
                    PyErr_SetString(PyExc_RuntimeError, errtxt);
                    return -1;
                }
                inst.operand[i] = tary;
            }
        }
        //Only execute the ufunc if the output is greater than zero.
        if(cphvb_nelements(inst.operand[0]->ndim,
                           inst.operand[0]->shape) > 0)
            util_schedule(&inst, 0);
    }

    return 0;
} /* dnumpy_ufunc */


/*NUMPY_API
 *===================================================================
 * Public
 * Apply ufunc method "reduce" on distributed array in_ary over axis.
 * Returns -1 on failure.
 */
static int
dnumpy_ufunc_reduce(PyUFuncObject *ufunc, PyArrayObject *in,
                    PyArrayObject *out, int axis)
{
    cphvb_intp i;

    #ifdef DEBUG
        printf("[NumPy] ufunc reduce in: %p, out: %p\n",
                (cphvb_array*) in->dnduid, (cphvb_array*) out->dnduid);

    #endif

    //For now we will just de the reduction our self.
    PyArrayObject *arys[2] = {in,out};

    //Get Python function.
    PyObject *PyOp = PyObject_GetAttrString(ufunc_module, ufunc->name);
    PyObject *tuple[2];
    assert(PyOp != NULL);

    //Fill the tuple and apply the ufunc.
    for(i=0; i<2; ++i)
    {
        char *data;
        if(arys[i]->dnduid == 0)//Scalar.
        {
            tuple[i] = PyArray_SimpleNewFromData(0, NULL,
                       arys[i]->descr->type_num, arys[i]->data);
        }
        else
        {
            cphvb_array* a = (cphvb_array*) arys[i]->dnduid;
            cphvb_array* b = cphvb_base_array(a);

            //We may have to allocate it.
            if(cphvb_malloc_array_data(b) != CPHVB_SUCCESS)
            {
                fprintf(stderr, "Out of memory "
                                "in dnumpy_ufunc\n");
                exit(CPHVB_OUT_OF_MEMORY);
            }
            data = b->data + a->start * cphvb_type_size(a->type);
            //Tell the VEM to release the data to us.
            util_release(a);

            tuple[i] = PyArray_New(&PyArray_Type,
                                   arys[i]->nd,
                                   arys[i]->dimensions,
                                   arys[i]->descr->type_num,
                                   arys[i]->strides,
                                   data, 0,
                                   NPY_BEHAVED, NULL);
            assert(tuple[i] != NULL);
        }
    }
    PyObject *t = PyObject_CallMethod(PyOp, "reduce", "(O,i,O,O)",
                                      tuple[0], axis, Py_None, tuple[1]);
    Py_XDECREF(t);
    Py_DECREF(PyOp);
    return 0;
} /* dnumpy_ufunc_reduce */

/*NUMPY_API
 *===================================================================
 * Public
 * Fill the distributed array/view with zeroes.
 */
static void
dnumpy_zerofill(intp ary_uid)
{
    fprintf(stderr,"dnumpy_zerofill not implemented.\n");
    exit(CPHVB_ERROR);
    //Get arrray structs.
    //dndview *view = get_dndview(ary_uid);

} /* dnumpy_zerofill */

/*NUMPY_API
 *===================================================================
 * Public
 * Fill the distributed array with data from file.
 */
static void
dnumpy_datafill(intp ary_uid, const char *filename, long datapos)
{
    fprintf(stderr,"dnumpy_datafill not implemented.\n");
    exit(CPHVB_ERROR);
    //Get arrray structs.
    //dndview *view = get_dndview(ary_uid);

    //Get filename length
    //int filename_length = strlen(filename);


} /* dnumpy_datafill */

/*NUMPY_API
 *===================================================================
 * Public
 * Save the distributed array to file.
 */
static void
dnumpy_datadump(intp ary_uid, const char *filename, long datapos)
{
    fprintf(stderr,"dnumpy_datadump not implemented.\n");
    exit(CPHVB_ERROR);

} /* dnumpy_datadump */

/*NUMPY_API
 *===================================================================
 * Public
 * Generate
 */
static PyObject *
dnumpy_random(PyObject *ary)
{
    intp uid;
    cphvb_instruction inst;
    cphvb_array *array;

    //We may have to create the array.
    if(!PyArray_ISDISTRIBUTED(ary))
    {
        if(dnumpy_create_dndarray(PyArray_NDIM(ary), PyArray_DIMS(ary),
                                  PyArray_TYPE(ary), 0, &uid))
            return NULL;
    }
    else
        uid = PyArray_DNDUID(ary);

    array = (cphvb_array*) uid;

    #ifdef DEBUG
        printf("[NumPy] random -- ary: %p\n", array);
    #endif

    inst.opcode = CPHVB_RANDOM;
    inst.operand[0] = array;

    if(!vem_instruction_check(&inst))
    {
        PyErr_SetString(PyExc_RuntimeError,
                        "Random is not supported by the VEM\n");
        return NULL;
    }
    else
        util_schedule(&inst, 1);


    Py_INCREF(ary);
    return ary;
} /* dnumpy_random */

/*NUMPY_API
 *===================================================================
 * Public
 * Returns a new 1-dim dist array filled with the diagonal from 'uid'.
 * Returns NULL on failure.
 */
static PyObject*
dnumpy_diagonal(intp uid, int offset, int axis1, int axis2)
{
    fprintf(stderr,"dnumpy_diagonal not implemented.\n");
    exit(CPHVB_ERROR);
    return NULL;
    /*
    PyObject *ret;
    intp n1, n2, start, stop, step, count;
    //Get arrray structs.
    dndview *view = get_dndview(uid);

    if(axis1 != 0 && axis2 != 1)
    {
        PyErr_Format(PyExc_ValueError, "axis1 and axis2 must be the "
                     "defualt values 0 and 1 on a distributed array.");
        return NULL;
    }
    if(view->ndims != 2)
    {
        PyErr_Format(PyExc_ValueError, "if array is distributed it "
                     "must have exactly two dimensions.");
        return NULL;
    }

    if(offset != 0)
    {
        PyErr_Format(PyExc_ValueError, "if array is distributed offset "
                     "must be zero.");
        return NULL;
    }

    if(view->alterations != 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "diagonal on a distributed "
                        "array must be a whole array and not a "
                        "partial view of a underlying array.");
        return NULL;
    }

    //Compute the size of the diagonal.
    n1 = view->base->dims[0];
    n2 = view->base->dims[1];
    step = n2 + 1;
    if (offset < 0) {
        start = -n2 * offset;
        stop = MIN(n2, n1+offset)*(n2+1) - n2*offset;
    }
    else {
        start = offset;
        stop = MIN(n1, n2-offset)*(n2+1) + offset;
    }
    //count = ceil((stop-start)/step)
    count = ((stop-start) / step) + (((stop-start) % step) != 0);

    //Create the returning distributed vector.
    ret = PyArray_New(&PyArray_Type, 1, &count, view->base->dtype,
                      NULL, NULL, 0,
                      NPY_BEHAVED | NPY_CARRAY | DNPY_DISTRIBUTED,
                      NULL);

    return ret;
    */
} /* dnumpy_diagonal */

/*NUMPY_API
 * ===================================================================
 * Public
 * Initialization of distnumpy.
 */
static void
dnumpy_init(void)
{
    int error;
    //Find the VEM
    #ifdef CLUSTER
        vem_init = &cphvb_vem_cluster_init;
        vem_execute = &cphvb_vem_cluster_execute;
        vem_shutdown = &cphvb_vem_cluster_shutdown;
        vem_create_array = &cphvb_vem_cluster_create_array;
        vem_instruction_check = &cphvb_vem_cluster_instruction_check;
    #else
        vem_init = &cphvb_vem_node_init;
        vem_execute = &cphvb_vem_node_execute;
        vem_shutdown = &cphvb_vem_node_shutdown;
        vem_create_array = &cphvb_vem_node_create_array;
        vem_instruction_check = &cphvb_vem_node_instruction_check;
    #endif

    error = vem_init();
    if(error)
    {
        printf("Error in vem_init()\n");
        exit(-1);
    }

} /* dnumpy_init */


/*NUMPY_API
 *===================================================================
 * Public
 * C = A * B.
 * All elements in C must be zero.
 */
static void
dnumpy_matrix_multiplication(intp Auid, intp Buid, intp Cuid)
{
    fprintf(stderr,"dnumpy_matrix_multiplication not implemented.\n");
    exit(CPHVB_ERROR);
    /*
    //Get arrray structs.
    dndview *A = get_dndview(Auid);
    dndview *B = get_dndview(Buid);
    dndview *C = get_dndview(Cuid);
    */

} /* dnumpy_matrix_multiplication */


/*NUMPY_API
 * ===================================================================
 * Public
 * De-initialization of distnumpy.
 */
static void
dnumpy_exit()
{
    dnumpy_evalflush();
    vem_shutdown();
} /* dnumpy_exit */

/*NUMPY_API
 * ===================================================================
 * Public
 * From this point on the master will continue with the pyton code
 * and the slaves will stay in C.
 * This only makes sense when combined with the VEM-CLUSTER.
 * If returning False the Python must call sys.exit(0) immediately.
 */
static PyObject *
dnumpy_master_slave_split(void)
{
    PyObject *ret = Py_True;
    #ifdef CLUSTER
        //This function is only for the VEM-CLUSTER.
        if(cphvb_vem_cluster_master_slave_split())
            ret = Py_False;
    #endif
        return ret;
} /* dnumpy_master_slave_split */

/*NUMPY_API
 * ===================================================================
 * Public
 * Registrete the python module in which ufunction's operators can be
 * found.
 */
static void
dnumpy_reg_ufunc_module(PyObject *module)
{
    ufunc_module = module;
} /* dnumpy_reg_ufunc_module */

/*NUMPY_API
 * ===================================================================
 * Public
 * Registrete the python module in which ufunction's operators can be
 * found.
 */
static PyObject*
dnumpy_get_reged_ufunc_module()
{
    return ufunc_module;
} /* dnumpy_get_reged_ufunc_module */

/*NUMPY_API
 * ===================================================================
 * Public
 * Registrete cblas. Used in _dotblas.c
 */
static void
dnumpy_reg_cblas(void *dgemm, void *sgemm, void *zgemm, void *cgemm)
{
    cblas_dgemm_func = dgemm;
    cblas_sgemm_func = sgemm;
    cblas_zgemm_func = zgemm;
    cblas_cgemm_func = cgemm;
} /* dnumpy_reg_Xgemm */
