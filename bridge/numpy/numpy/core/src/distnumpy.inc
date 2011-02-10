#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include <cphvb.h>
#include <svi.h>


#ifndef CBLAS_HEADER
#define CBLAS_HEADER "../blasdot/cblas.h"
#endif
#include CBLAS_HEADER

//Unique identification counter
static cphvb_int32 uid_count=0;


//Instructions info.
static cphvb_instruction* instructions[DNPY_MAX_NINST];//Inst memory.
static cphvb_instruction* next_inst; //Next instruction.

//Instruction batch info.
static int ninst_in_batch = 0; //Current number of instruction in batch.
static char *instruction_batch;//Batch memory.
static char *next_inst_batch;  //The next instruction in the batch.

//Array-views belonging to local MPI process
static dndview dndviews[DNPY_MAX_NARRAYS];
static npy_intp dndviews_uid[DNPY_MAX_NARRAYS];
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
 * Returns a string describing dtype.
 * Private
 */
#ifdef DISTNUMPY_DEBUG
static char *types_numpy2str(int dtype)
{
    switch(dtype)
    {
        case NPY_BOOL:
            return "NPY_BOOL";
        case NPY_BYTE:
            return "NPY_BYTE";
        case NPY_UBYTE:
            return "NPY_UBYTE";
        case NPY_SHORT:
            return "NPY_SHORT";
        case NPY_USHORT:
            return "NPY_USHORT";
        case NPY_INT:
            return "NPY_INT";
        case NPY_UINT:
            return "NPY_UINT";
        case NPY_LONG:
            return "NPY_LONG";
        case NPY_ULONG:
            return "NPY_ULONG";
        case NPY_LONGLONG:
            return "NPY_LONGLONG";
        case NPY_ULONGLONG:
            return "NPY_ULONGLONG";
        case NPY_FLOAT:
            return "NPY_FLOAT";
        case NPY_DOUBLE:
            return "NPY_DOUBLE";
        case NPY_LONGDOUBLE:
            return "NPY_LONGDOUBLE";
        case NPY_CFLOAT:
            return "NPY_CFLOAT";
        case NPY_CDOUBLE:
            return "NPY_CDOUBLE";
        case NPY_CLONGDOUBLE:
            return "NPY_CLONGDOUBLE";
        case NPY_OBJECT:
            return "NPY_OBJECT";
        case NPY_STRING:
            return "NPY_STRING";
        case NPY_UNICODE:
            return "NPY_UNICODE";
        case NPY_VOID:
            return "NPY_VOID";
        case NPY_NTYPES:
            return "NPY_NTYPES";
        case NPY_CHAR:
            return "NPY_CHAR";
        case NPY_USERDEF:
            return "NPY_USERDEF";
        default:
            return "\"Unknown data type\"";
    }
} /* types_numpy2str */
#endif

/*===================================================================
 *
 * Returns a string describing the operation type.
 * Private
 */
static char *optype2str(int optype)
{
    switch(optype)
    {
        case DNPY_CREATE_ARRAY:
            return "DNPY_CREATE_ARRAY";
        case DNPY_DESTROY_ARRAY:
            return "del";
        case DNPY_CREATE_VIEW:
            return "DNPY_CREATE_VIEW";
        case DNPY_PUT_ITEM:
            return "DNPY_PUT_ITEM";
        case DNPY_GET_ITEM:
            return "DNPY_GET_ITEM";
        case DNPY_UFUNC:
            return "ufunc";
        case DNPY_RECV:
            return "recv";
        case DNPY_SEND:
            return "send";
        case DNPY_BUF_RECV:
            return "Brecv";
        case DNPY_BUF_SEND:
            return "Bsend";
        case DNPY_APPLY:
            return "apply";
        case DNPY_UFUNC_REDUCE:
            return "DNPY_UFUNC_REDUCE";
        case DNPY_ZEROFILL:
            return "DNPY_ZEROFILL";
        case DNPY_DATAFILL:
            return "DNPY_DATAFILL";
        case DNPY_DIAGONAL:
            return "DNPY_DIAGONAL";
        case DNPY_MATMUL:
            return "DNPY_MATMUL";
        case DNPY_REDUCE_SEND:
            return "reduce_send";
        case DNPY_REDUCE_RECV:
            return "DNPY_REDUCE_RECV";
        default:
            return "\"Unknown data type\"";
    }
} /* optype2str */

/*===================================================================
 *
 * Put, get & remove views from local MPI process.
 * Private.
 */
static dndview *get_dndview(npy_intp uid)
{
    npy_intp i;
    if(uid)
        for(i=0; i < DNPY_MAX_NARRAYS; i++)
            if(dndviews_uid[i] == uid)
                return &dndviews[i];
    fprintf(stderr, "get_dndview, uid %ld does not exist\n", (long) uid);
    return NULL;
}
static dndview *put_dndview(dndview *view)
{
    npy_intp i;
    npy_intp ones[NPY_MAXDIMS];

    //Initiate the Python Array Object.
    for(i=0; i<view->ndims; i++)
        ones[i] = 1; //Just need ones for the dim.

    for(i=0; i < DNPY_MAX_NARRAYS; i++)
        if(dndviews_uid[i] == 0)
        {
            memcpy(&dndviews[i], view, sizeof(dndview));
            dndviews_uid[i] = view->uid;
            return &dndviews[i];
        }
    fprintf(stderr, "put_dndarray, MAX_NARRAYS is exceeded\n");
    return NULL;
}
//NB: rm_dndview will also free memory allocted for the dndarray
//if it is the last reference to the dndarray.
static void rm_dndview(npy_intp uid)
{
    npy_intp i;
    if(uid)
        for(i=0; i < DNPY_MAX_NARRAYS; i++)
            if(dndviews_uid[i] == uid)
            {
                dndview *view = &dndviews[i];
                //Cleanup base.
                dndviews_uid[i] = 0;
                if(view->base->ndims == 0)//Dummy Scalar.
                {
                    //Remove the array from to the lisked list used
                    //when statistic is defined.
                    #ifdef DNPY_STATISTICS
                        if(view->base->next != NULL)
                            view->base->next->prev = view->base->prev;
                        if(view->base->prev != NULL)
                            view->base->prev->next = view->base->next;
                        else
                            rootarray = view->base->next;
                    #endif
                    free(view->base->rootnodes);
                    free(view->base->data);
                    free(view->base);
                }
                else if(--view->base->refcount == 0)
                {
                    //Remove the array from to the lisked list used
                    //when statistic is defined.
                    #ifdef DNPY_STATISTICS
                        if(view->base->next != NULL)
                            view->base->next->prev = view->base->prev;
                        if(view->base->prev != NULL)
                            view->base->prev->next = view->base->next;
                        else
                            rootarray = view->base->next;
                    #endif
                    if(view->base->data != NULL)
                    {
                        #ifdef DNDY_TIME
                            unsigned long long tdelta;
                            DNDTIME(tdelta);
                        #endif

                        #ifdef DNDY_TIME
                            DNDTIME_SUM(tdelta, t_arydata_free)
                        #endif
                    }
                    free(view->base->rootnodes);
                    free(view->base);
                }
                return;
            }
    fprintf(stderr, "rm_dndarray, uid %ld does not exist\n", (long)uid);
    return;
}/* Put, get & rm dndview */


/*===================================================================
 *
 * Makes sure that the array's memory has been allocated.
 * Private.
 */
static void
delayed_array_allocation(dndarray *ary)
{
    #ifdef DNDY_TIME
        unsigned long long tdelta;
        DNDTIME(tdelta);
    #endif

    if(ary->data == NULL)
        ary->data = malloc(ary->nelements * ary->elsize);


    #ifdef DNDY_TIME
        DNDTIME_SUM(tdelta, t_arydata_malloc)
    #endif
}/* delayed_array_allocation */

/*===================================================================
 *
 * Convert visible vblock dimension index to base vblock
 * dimension index.
 * Private.
 */
static npy_intp
idx_v2b(const dndview *view, npy_intp vindex)
{
    assert(vindex < view->ndims);
    npy_intp i, bindex=0;
    for(i=0; i < view->nslice; i++)
    {
        if(view->slice[i].nsteps == SingleIndex)
        {
            if(view->base->ndims > 1)
                bindex++;
            continue;
        }
        if(vindex == 0)
            break;
        if(view->slice[i].nsteps != PseudoIndex)
            bindex++;
        vindex--;
    }
    //We need the MIN since bindex is too high when PseudoIndex is
    //used at the end of the view.
    return MIN(bindex, view->base->ndims-1);
} /* idx_v2b */

/*===================================================================
 *
 * Convert visible vblock dimension index to slice dimension index.
 * Private.
 */
static npy_intp
idx_v2s(const dndview *view, npy_intp vindex)
{
    npy_intp i;
    assert(vindex < view->ndims);
    for(i=0; i < view->nslice; i++)
    {
        if(view->slice[i].nsteps == SingleIndex)
            continue;
        if(vindex == 0)
            break;
        vindex--;
    }
    assert(i < view->nslice);
    return i;
} /* idx_v2s */


/*===================================================================
 *
 * Executes all instruction in the current batch and do some cleanup.
 * Private.
 */
static void
flush_inst_batch(void)
{
    printf("HEJ\n");
    //Execute instruction batch.
    svi_execute(42,ninst_in_batch,instruction_batch);
    //Resets counts and memory
    ninst_in_batch = 0;
    next_inst = instructions[0];
    next_inst_batch = instruction_batch;
} /* flush_inst_batch */



/*NUMPY_API
 *===================================================================
 * Create dndarray.
 * Public
 */
static intp
dnumpy_create_dndarray(int ndims, intp dims[NPY_MAXDIMS], int dtype,
                       intp dtypesize)
{
    int i;
    next_inst_batch = cphvb_init(next_inst,CPHVB_MALLOC,ndims,0,
                                 next_inst_batch);

    //Set instruction shape
    cphvb_int64 tdims[NPY_MAXDIMS];
    for(i=0; i<ndims; ++i)
        tdims[i] = (cphvb_int64) dims[i];
    cphvb_set_shape(next_inst,tdims);

    //Set instruction operand.
    cphvb_set_operand(next_inst,0,++uid_count,CPHVB_INT32,0,tdims);

    //Iterate to the memory of the next instruction.
    ++next_inst;

    //Check if we should flush the batched instructions.
    if(DNPY_MAX_NINST > ++ninst_in_batch)
        flush_inst_batch();


    return uid_count;
    /*
    int i;
    assert(ndims > 0);
    assert(ndims < NPY_MAXDIMS);

    //Create dndarray.
    dndarray *newarray = malloc(sizeof(dndarray));

    newarray->dtype = dtype;
    newarray->elsize = dtypesize;
    newarray->ndims = ndims;
    newarray->refcount = 1;
    for(i=0; i<ndims; i++)
        newarray->dims[i] = dims[i];

    //Create dndview. NB: the base will have to be set when 'newarray'
    //has found its final resting place. (Done by put_dndarray).
    dndview *newview = malloc(sizeof(dndview));
    newview->uid = ++uid_count;
    newview->nslice = ndims;
    newview->ndims = ndims;
    newview->alterations = 0;
    for(i=0; i<ndims; i++)
    {
        //Default the view will span over the whole array.
        newview->slice[i].start = 0;
        newview->slice[i].step = 1;
        newview->slice[i].nsteps = dims[i];
    }

    //Freeup memory.
    free(newarray);
    free(newview);

    return uid_count;
    */
} /* dnumpy_create_dndarray */

/*NUMPY_API
 *===================================================================
 * Destroy dndarray.
 * Public
 */
static void
dnumpy_destroy_dndarray(intp uid)
{
    printf("dnumpy_destroy_dndarray\n");
    //Get arrray structs.
    //dndview *ary = get_dndview(uid);


} /* dnumpy_destroy_dndarray */


/*NUMPY_API
 *===================================================================
 * Create a new view based on the 'org_view'.
 * Public
 */
static int
dnumpy_create_dndview(intp org_view_uid, int nslice,
                      dndslice slice[NPY_MAXDIMS])
{
    printf("dnumpy_create_dndview\n");
    return 0;
    /*
    //Get arrray structs.
    dndview *orgview = get_dndview(org_view_uid);

    //Create new view based on 'org_view' and the 'slice'.
    dndview newview;
    newview.uid = ++uid_count;
    newview.ndims = 0;
    newview.alterations = 0;

    //Merging the two views.
    int si = 0; //slice index.
    int ni = 0; //new index.
    int oi = 0; //old index.
    int di = 0; //dim index.
    while(si < nslice || oi < orgview->nslice)
    {
        //If we come to the end of the slices, that happens
        //if not all dimensions is included in the 'slices', we will
        //use the whole dimension.
        int vs = (si < nslice)?1:0;//Valid slice.

        //If dimension is invisible we will just copy it to 'newview'.
        if(oi < orgview->nslice &&
           orgview->slice[oi].nsteps == SingleIndex)
        {
            memcpy(&newview.slice[ni], &orgview->slice[oi],
                   sizeof(dndslice));
            ni++; oi++; di++;
            newview.alterations |= DNPY_NDIMS;
        }
        //A single index makes the dimension invisible.
        else if(vs && slice[si].nsteps == SingleIndex)
        {
            //If dimension is a Pseudo-dimension then just go to next
            //dimension.
            if(orgview->slice[oi].nsteps == PseudoIndex)
            {
                si++; oi++;
            }
            else
            {//Copy single index to 'newview'.
                newview.slice[ni].step = 0;
                newview.slice[ni].nsteps = SingleIndex;
                newview.slice[ni].start = orgview->slice[oi].start +
                                          (vs?slice[si].start:0) *
                                          orgview->slice[oi].step;
                si++; ni++; oi++; di++;
                newview.alterations |= DNPY_NDIMS;
            }
        }
        //If a extra pseudo index should be added we just copy the
        //slice to 'newview'.
        else if(vs && slice[si].nsteps == PseudoIndex)
        {
            memcpy(&newview.slice[ni], &slice[si],
                   sizeof(dndslice));
            ni++; si++;
            newview.ndims++;
            newview.alterations |= DNPY_NDIMS;
        }
        else if(orgview->slice[oi].nsteps == PseudoIndex)
        {
            memcpy(&newview.slice[ni], &orgview->slice[oi],
                   sizeof(dndslice));
            ni++; oi++; si++;
            newview.ndims++;
            newview.alterations |= DNPY_NDIMS;
        }
        //If no special slices we just merge the two views.
        else
        {
            if(vs)
            {
                newview.slice[ni].start = orgview->slice[oi].start +
                                           slice[si].start *
                                           orgview->slice[oi].step;
                newview.slice[ni].step = slice[si].step *
                                          orgview->slice[oi].step;
                newview.slice[ni].nsteps = slice[si].nsteps;
            }
            else
                memcpy(&newview.slice[ni], &orgview->slice[oi],
                       sizeof(dndslice));

            if(newview.slice[ni].step > 1)
                newview.alterations |= DNPY_STEP | DNPY_NSTEPS;
            else if(newview.slice[ni].nsteps < orgview->base->dims[di])
            {
                newview.alterations |= DNPY_NSTEPS;
            }
            newview.ndims++;
            si++; ni++; oi++; di++;
        }
    }
    //Save the total number of sliceses for the new view.
    newview.nslice = ni;

    return uid_count;
    */
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
    dndview *A = get_dndview(Auid);
    dndview *B = get_dndview(Buid);
    //At the moment we only check at the array-base level.
    if(A->base->uid == B->base->uid)
        return 1;
    else
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
    /*
    //Get arrray structs.
    dndview *view = get_dndview(uid);
    int ndims = view->ndims;
    int elsize = view->base->elsize;

    //Convert item to a compatible type.
    PyObject *item2 = PyArray_FROM_O(item);
    PyObject *citem2 = PyArray_Cast((PyArrayObject*)item2,
                                    view->base->dtype);

    //Cleanup and return error if the cast failed.
    if(citem2 == NULL)
    {
        Py_DECREF(item2);
        return -1;
    }

    //Clean up.
    Py_DECREF(citem2);
    Py_DECREF(item2);
    */
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
    printf("dnumpy_dndarray_getitem\n");
    //Get arrray structs.
    //dndview *view = get_dndview(uid);


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


void test_callback(cphvb_int32 batch_id,
                   cphvb_int32 instruction_count,
                   cphvb_error error_code)
{
    printf("[Test] Got callback: {batch_id: %d, instruction_count %d, error_code: %d}\n",batch_id,instruction_count, error_code);
}

/*NUMPY_API
 * ===================================================================
 * Public
 * Initialization of distnumpy.
 */
static void
dnumpy_init(void)
{
    int error;
    error = svi_init(&test_callback);
    if(error)
    {
        printf("Error in svi_init()\n");
        exit(-1);
    }

    //Allocate instruction batch.
    instruction_batch = malloc(DNPY_INSTRUCTION_BATCH_MAXSIZE);
    next_inst_batch = instruction_batch;
    if(instruction_batch == NULL)
    {
        printf("Memory error in dnumpy_init()\n");
        exit(-1);
    }
    next_inst = instructions[0];

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
