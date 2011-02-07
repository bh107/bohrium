#ifndef DISTNUMPY_H
#define DISTNUMPY_H
#include "mpi.h"

//ufunc definitions from numpy/ufuncobject.h.
//They are included here instead.
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
} PyUFuncObject;

//#define DISTNUMPY_DEBUG
//#define DNPY_STATISTICS
//#define DNDY_TIME
//#define DNDY_TIME_NODE 0


//Easy retrieval of dnduid
#define PyArray_DNDUID(obj) (((PyArrayObject *)(obj))->dnduid)

//Maximum message size (in bytes)
#define DNPY_MAX_MSG_SIZE 1024*4

//Maximum number of view block operations in the sub-view-block DAG.
#define DNPY_MAX_VB_IN_SVB_DAG 1000

//Maximum number of allocated arrays
#define DNPY_MAX_NARRAYS 1024

//Maximum number of operation merged together.
#define DNPY_MAX_OP_MERGES 10

//Default blocksize
#define DNPY_BLOCKSIZE 2

//Maximum number of nodes in the ready queue.
#define DNPY_RDY_QUEUE_MAXSIZE 1024*10

//The maximum size of the work buffer in bytes (should be power of 2).
#define DNPY_WORK_BUFFER_MAXSIZE 536870912 //½GB

//Operation types
enum opt {DNPY_MSG_END, DNPY_CREATE_ARRAY, DNPY_DESTROY_ARRAY,
          DNPY_CREATE_VIEW, DNPY_SHUTDOWN, DNPY_PUT_ITEM, DNPY_GET_ITEM,
          DNPY_UFUNC, DNPY_UFUNC_REDUCE, DNPY_ZEROFILL, DNPY_DATAFILL,
          DNPY_DATADUMP, DNPY_INIT_BLOCKSIZE, DNPY_DIAGONAL, DNPY_MATMUL,
          DNPY_RECV, DNPY_SEND, DNPY_BUF_RECV, DNPY_BUF_SEND, DNPY_APPLY,
          DNPY_EVALFLUSH, DNPY_READ, DNPY_WRITE, DNPY_COMM, DNPY_NONCOMM,
          DNPY_REDUCE_SEND, DNPY_REDUCE_RECV, DNPY_INIT_PROC_GRID};

//dndnode prototype.
typedef struct dndnode_struct dndnode;
typedef struct dndarray_struct dndarray;

//Type describing a distributed array.
struct dndarray_struct
{
    //Unique identification.
    npy_intp uid;
    //Reference count.
    int refcount;
    //Number of dimensions.
    int ndims;
    //Size of dimensions.
    npy_intp dims[NPY_MAXDIMS];
    //Size of block-dimensions.
    npy_intp blockdims[NPY_MAXDIMS];
    //Number of blocks (global).
    npy_intp nblocks;
    //Data type of elements in array.
    int dtype;
    //Size of an element in bytes.
    int elsize;
    //Pointer to local data.
    char *data;
    //Number of local elements (local to the MPI-process).
    npy_intp nelements;
    //Size of local dimensions (local to the MPI-process).
    npy_intp localdims[NPY_MAXDIMS];
    //Size of local block-dimensions (local to the MPI-process).
    npy_intp localblockdims[NPY_MAXDIMS];
    //MPI-datatype that correspond to an array element.
    MPI_Datatype mpi_dtype;
    //Root nodes (one per block).
    dndnode **rootnodes;
    //Next and prev are used for traversing all arrays.
    #ifdef DNPY_STATISTICS
        dndarray *next;
        dndarray *prev;
    #endif
};

//dndslice constants.
#define PseudoIndex -1//Adds a extra 1-dim - 'A[1,newaxis]'
#define RubberIndex -2//A[1,2,...] (Not used in distnumpy.inc)
#define SingleIndex -3//Dim not visible - 'A[1]'

//Type describing a slice of a dimension.
typedef struct
{
    //Start index.
    npy_intp start;
    //Elements between index.
    npy_intp step;
    //Number of steps (Length of the dimension).
    npy_intp nsteps;
} dndslice;

//View-alteration flags.
#define DNPY_NDIMS    0x001
#define DNPY_STEP     0x002
#define DNPY_NSTEPS   0x004

//Type describing a view of a distributed array.
typedef struct
{
    //Unique identification.
    npy_intp uid;
    //The array this view is a view of.
    dndarray *base;
    //Number of viewable dimensions.
    int ndims;
    //Number of sliceses. NB: nslice >= base->ndims.
    int nslice;
    //Sliceses - the global view of the base-array.
    dndslice slice[NPY_MAXDIMS];
    //A bit mask specifying which alterations this view represents.
    //Possible flags:
    //Zero        - no alterations.
    //DNPY_NDIMS  - number of dimensions altered.
    //DNPY_STEP   - 'step' altered.
    //DNPY_NSTEPS - 'nsteps' altered.
    int alterations;
    //Number of view-blocks.
    npy_intp nblocks;
    //Number of view-blocks in each viewable dimension.
    npy_intp blockdims[NPY_MAXDIMS];
} dndview;

//Type describing a sub-section of a view block.
typedef struct
{
    //The rank of the MPI-process that owns this sub-block.
    int rank;
    //Start index (one per base-dimension).
    npy_intp start[NPY_MAXDIMS];
    //Number of elements (one per base-dimension).
    npy_intp nsteps[NPY_MAXDIMS];
    //Number of elements to next dimension (one per base-dimension).
    npy_intp stride[NPY_MAXDIMS];
    //The MPI communication offset (in bytes).
    npy_intp comm_offset;
    //Number of elements in this sub-view-block.
    npy_intp nelem;
    //This sub-view-block's root node.
    dndnode **rootnode;
    //Pointer to data. NULL if data needs to be fetched.
    char *data;
    //The rank of the MPI process that have received this svb.
    //A negative value means that nothing has been received.
    int comm_received_by;
} dndsvb;

//Type describing a view block.
typedef struct
{
    //The id of the view block.
    npy_intp uid;
    //All sub-view-blocks in this view block (Row-major).
    dndsvb *sub;
    //Number of sub-view-blocks.
    npy_intp nsub;
    //Number of sub-view-blocks in each dimension.
    npy_intp svbdims[NPY_MAXDIMS];
} dndvb;

//The Super-type of a operation.
//refcount         - number of dependency nodes in the svb DAG.
//op               - the operation, e.g. DNPY_RECV and DNPY_UFUNC.
//optype           - the operation type, e.g. DNPY_COMM/_NONCOMM.
//narys & views    - list of array views involved.
//svbs             - list of sub-view-blocks involved (one per array),
//                   NULL when whole arrays are involved.
//accesstype       - access type e.g. DNPY_READ (one per array)
//uid              - unique identification - only used for statistics.
#define DNDOP_HEAD_BASE                     \
    npy_intp refcount;                      \
    char op;                                \
    char optype;                            \
    char narys;                             \
    dndview *views[NPY_MAXARGS];            \
    dndsvb *svbs[NPY_MAXARGS];              \
    char accesstypes[NPY_MAXARGS];
#ifdef DNPY_STATISTICS
    #define DNDOP_HEAD DNDOP_HEAD_BASE npy_intp uid;
#else
    #define DNDOP_HEAD DNDOP_HEAD_BASE
#endif
typedef struct dndop_struct dndop;
struct dndop_struct {DNDOP_HEAD};

//Type describing a communication DAG node.
typedef struct
{
    DNDOP_HEAD
    //The MPI tag used for the communication.
    npy_intp mpi_tag;
    //The MPI rank of the process that is the remote communication peer.
    int remote_rank;
} dndop_comm;

//Type describing an apply-sub-view-block, which is a subsection of a
//sub-view-block that is used in apply.
typedef struct
{
    npy_intp dims[NPY_MAXDIMS];
    npy_intp stride[NPY_MAXDIMS];
    npy_intp offset;
} dndasvb;

//Type describing a universal function DAG node.
typedef struct
{
    DNDOP_HEAD
    //List of apply-sub-view-block.
    dndasvb asvb[NPY_MAXARGS];
    //Number of output array views.
    char nout;
    //The operation described as a function, a data and a Python pointer.
    PyUFuncGenericFunction func;
    void *funcdata;
    PyObject *PyOp;
} dndop_ufunc;

//Type describing a DAG node.
struct dndnode_struct
{
    //The operation associated with this dependency.
    dndop *op;
    //The index to use when accessing op->views[] and op->svbs[].
    int op_ary_idx;
    //Next node in the dependency list.
    dndnode *next;
    //Unique identification used for statistics.
    #ifdef DNPY_STATISTICS
        npy_intp uid;
    #endif
};

#endif
