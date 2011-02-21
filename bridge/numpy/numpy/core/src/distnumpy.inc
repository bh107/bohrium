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

//Pointer to the python module who has the ufunc operators.
static PyObject *ufunc_module;
//Pointers to the cblas functions.
typedef void (cblas_func_type)(const enum CBLAS_ORDER Order,...);
static cblas_func_type *cblas_dgemm_func = NULL;
static cblas_func_type *cblas_sgemm_func = NULL;
static cblas_func_type *cblas_zgemm_func = NULL;
static cblas_func_type *cblas_cgemm_func = NULL;



/*===================================================================
 *
 * The data type conversion to and from NumPy and cphVB.
 * Private.
 */
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
 static const cphvb_type const py2cph[] =
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
static const cphvb_type const cph2py[] =
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

/*NUMPY_API
 *===================================================================
 * Create dndarray.
 * Return 1 when the dtype is not supported else 0.
 * Public
 */
static int
dnumpy_create_dndarray(int ndims, intp dims[NPY_MAXDIMS], int numpytype,
                       intp *uid)
{
    printf("dnumpy_create_dndarray %d\n", numpytype);
    int i;
    cphvb_index s;
    cphvb_index stride[CPHVB_MAXDIM];
    cphvb_index shape[CPHVB_MAXDIM];
    cphvb_error error;
    cphvb_array *new_ary;
    cphvb_type dtype = py2cph[numpytype];

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

    error = cphvb_vem_create_array(NULL, dtype, ndims, 0, shape, stride,
                                NPY_FALSE, (cphvb_constant)0, &new_ary);

    //We save the array pointer in an 'intp'.
    *uid = (intp) new_ary;
    printf("uid (%ld)\n", *uid);

    if(error == CPHVB_OUT_OF_MEMORY)
        PyErr_NoMemory();
    else if (error)
        PyErr_SetString(PyExc_RuntimeError, "unknown cphVB error\n");

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
    printf("dnumpy_destroy_dndarray (%ld)\n", uid);

    cphvb_error error;
    cphvb_instruction inst;
    inst.opcode = CPHVB_DESTORY;
    inst.operand[0] = (cphvb_array*) uid;

    //We assume that the VEM supports the CPHVB_DESTORY opcode.
    error = cphvb_vem_execute(1, &inst);
    if(error)
    {
        fprintf(stderr,"Unhandled error returned by cphvb_vem_execute()"
                       " in dnumpy_destroy_dndarray()\n");
        exit(error);
    }

} /* dnumpy_destroy_dndarray */


/*NUMPY_API
 *===================================================================
 * Create a new view based on the 'org_ary'.
 * Public
 */
static int
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

    error = cphvb_vem_create_array(base, org_ary->type, ndim,
                                   offset+org_ary->start,
                                   shape, stride, NPY_FALSE,
                                   (cphvb_constant)0, &new_ary);

    if(error == CPHVB_OUT_OF_MEMORY)
        PyErr_NoMemory();
    else if (error)
        PyErr_SetString(PyExc_RuntimeError, "unknown cphVB error\n");

    printf("dnumpy_create_dndview(%ld)\n", (intp) new_ary);

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
    printf("dnumpy_evalflush\n");
    //The master should also do the operation.
    #ifdef DISTNUMPY_DEBUG
        printf("Rank 0 received msg: EVALFLUSH\n");
    #endif
    Py_RETURN_NONE;
} /* dnumpy_evalflush */

/*NUMPY_API
 *===================================================================
 * Public
 * Returns True when the two array-view overlap.
 */
static char
dnumpy_data_overlap(intp Auid, intp Buid)
{
    printf("dnumpy_data_overlap\n");
    /*
    dndview *A = get_dndview(Auid);
    dndview *B = get_dndview(Buid);
    //At the moment we only check at the array-base level.
    if(A->base->uid == B->base->uid)
        return 1;
    else
        return 0;
    */
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
    printf("dnumpy_dndarray_putitem\n");
    cphvb_error error;
    cphvb_instruction inst;
    cphvb_array *ary = (cphvb_array*) uid;
    cphvb_array *base = cphvb_base_array(ary);
    cphvb_intp dtypesize = cphvb_type_size(ary->type);
    cphvb_intp d, offset;

    //Convert item to a compatible type.
    PyObject *item2 = PyArray_FROM_O(item);
    PyObject *citem2 = PyArray_Cast((PyArrayObject*)item2,
                                    cph2py[ary->type]);

    //Cleanup and return error if the cast failed.
    if(citem2 == NULL)
    {
        Py_DECREF(item2);
        return -1;
    }

    //Tell the VEM to release the data to us.
    inst.opcode = CPHVB_RELEASE;
    inst.operand[0] = ary;
    //We assume that the VEM supports the CPHVB_RELEASE opcode.
    error = cphvb_vem_execute(1, &inst);
    if(error)
    {
        fprintf(stderr,"Unhandled error returned by cphvb_vem_execute()"
                       " in dnumpy_dndarray_putitem()\n");
        exit(error);
    }

    //We may have to allocate the array memory.
    if(base->data == NULL)
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

    printf("offset: %ld\n", offset);

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
    printf("dnumpy_dndarray_getitem(%ld)\n", uid);
    cphvb_error error;
    cphvb_instruction inst;
    cphvb_array *ary = (cphvb_array*) uid;
    cphvb_array *base = cphvb_base_array(ary);
    cphvb_intp dtypesize = cphvb_type_size(ary->type);
    cphvb_intp d, offset;

    //Tell the VEM to release the data to us.
    inst.opcode = CPHVB_RELEASE;
    inst.operand[0] = ary;
    //We assume that the VEM supports the CPHVB_RELEASE opcode.
    error = cphvb_vem_execute(1, &inst);
    if(error)
    {
        fprintf(stderr,"Unhandled error returned by cphvb_vem_execute()"
                       " in dnumpy_dndarray_putitem()\n");
        exit(error);
    }

    //We may have to allocate the array memory.
    if(base->data == NULL)
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

    printf("offset: %ld\n", offset);

    //Write item to memory.
    memcpy(retdata, base->data + offset, dtypesize);

} /* dnumpy_dndarray_getitem */

/*NUMPY_API
 *===================================================================
 * Public
 * Apply ufunc on distributed arrays in arylist.
 * op is the python name of the ufunction.
 * Returns -1 on failure.
 */
static int
dnumpy_ufunc(PyArrayObject *arylist[NPY_MAXARGS], int narys,
             int nout_arys, char *op, int appropriate_function)
{
    printf("dnumpy_ufunc\n");
    /*
    int i;
    //Tell slaves about the ufunc operations.
    msg[0] = DNPY_UFUNC;
    msg[1] = narys;
    msg[2] = nout_arys;
    msg[3] = appropriate_function;
    msg[4] = 0; //Scalar datatype.
    msg[5] = 0; //Scalar datasize, zero if no scalar is in arylist.

    //Copy scalar if one is in the arylist.
    for(i=0; i<narys;i++)
        if(arylist[i]->dnduid == 0)
        {
            if(arylist[i]->nd != 0)
            {
                PyErr_SetString(PyExc_RuntimeError,
                                "ufunc - distributed and non-"
                                "distributed arrays do not mix. "
                                "Scalars is allowed though.\n");
                return -1;
            }
            msg[4] = arylist[i]->descr->type_num;
            msg[5] = arylist[i]->descr->elsize;
            memcpy(msg+6, arylist[i]->data, msg[5]);
            break;
        }
    */
    return 0;
} /* dnumpy_ufunc */

/*NUMPY_API
 *===================================================================
 * Public
 * Apply ufunc method "reduce" on distributed array in_ary over axis.
 * op is the python name of the ufunction.
 * Returns -1 on failure.
 */
static int
dnumpy_ufunc_reduce(PyArrayObject *in_ary, PyArrayObject *out_ary,
                    int axis, char *op)
{
    printf("dnumpy_ufunc_reduce\n");
    /*
    int i;
    dndview *in = get_dndview(in_ary->dnduid);

    //We do not support reduce on views.
    for(i=0; i<in->nslice;i++)
    {
        if(in->slice[i].nsteps == SingleIndex ||
           in->slice[i].nsteps == PseudoIndex ||
           in->slice[i].start != 0 || in->slice[i].step != 1 ||
           in->slice[i].nsteps != in->base->dims[i])
        {
            PyErr_SetString(PyExc_RuntimeError, "reduce on distributed "
                            "arrays must be whole arrays and not a "
                            "partial view of a underlying array.");
            return -1;
        }
    }
    if(out_ary->dnduid > 0)
    {
        dndview *out = get_dndview(out_ary->dnduid);
        //Check if the views covers the whole array.
        if(in->alterations != 0 || out->alterations != 0)
        {
            PyErr_SetString(PyExc_RuntimeError,
                            "reduce on distributed arrays must"
                            "be whole arrays and not a partial"
                            " view of a underlying array.");
            return -1;
        }
    }

    //Tell slaves
    msg[0] = DNPY_UFUNC_REDUCE;
    msg[1] = in_ary->dnduid;
    msg[2] = out_ary->dnduid;
    //If out_ary is a scalar we will send the datatype and datatypesize.
    if(out_ary->dnduid == 0)
    {
        msg[3] = PyArray_TYPE(out_ary);
        msg[4] = PyArray_ITEMSIZE(out_ary);
    }
    */
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
    printf("dnumpy_zerofill\n");
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
    printf("dnumpy_datafill\n");
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
    printf("dnumpy_datadump\n");
    //Get array structs.
    //dndview *view = get_dndview(ary_uid);


} /* dnumpy_datadump */

/*NUMPY_API
 *===================================================================
 * Public
 * Returns a new 1-dim dist array filled with the diagonal from 'uid'.
 * Returns NULL on failure.
 */
static PyObject*
dnumpy_diagonal(intp uid, int offset, int axis1, int axis2)
{
    printf("dnumpy_diagonal\n");
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
    error = cphvb_vem_init();
    if(error)
    {
        printf("Error in cphvb_vem_init()\n");
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
    printf("dnumpy_matrix_multiplication\n");
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
    printf("dnumpy_exit\n");
    /*
    int i;
    if(myrank == 0)
    {
        //Shutdown slaves
        msg[0] = DNPY_SHUTDOWN;
        msg[1] = DNPY_MSG_END;
        #ifdef DISTNUMPY_DEBUG
            printf("Rank 0 received msg: SHUTDOWN\n");
        #endif
    }



    //Free buffers.
    free(workbuf);
    //Free Cartesian Information.
    for(i=0; i < NPY_MAXDIMS; i++)
    {
        free(cart_dim_strides[i]);
        free(cart_dim_sizes[i]);
    }
    int nleaks = 0;
    for(i=0; i < DNPY_MAX_NARRAYS; i++)
        if(dndviews_uid[i] != 0)
            nleaks++;

    if(nleaks > 0)
        printf("DistNumPy - Warning %d distributed arrays didn't get "
               "deallocated.\n", nleaks);
,   */
} /* dnumpy_exit */

/*NUMPY_API
 * ===================================================================
 * Public
 * From this point on the master will continue with the pyton code
 * and the slaves will stay in C.
 * This only makes sense when combined with MPI.
 * If returning False the Python must call sys.exit(0) immediately.
 */
static PyObject *
dnumpy_master_slave_split(void)
{
    printf("dnumpy_master_slave_split\n");
    return Py_True;

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
