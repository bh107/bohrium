#ifndef CPHVB_VEM_CLUSTER_PRIVATE_H
#define CPHVB_VEM_CLUSTER_PRIVATE_H
#include <mpi.h>
#include <cphvb.h>

//#define CLUSTER_DEBUG
//#define CLUSTER_STATISTICS
//#define CLUSTER_TIME
//#define CLUSTER_TIME_NODE 0

//Maximum message size (in bytes)
#define CLUSTER_MSG_SIZE (1024*4)

//Maximum number of view block operations in the sub-view-block DAG.
#define CLUSTER_MAX_VB_IN_SVB_DAG (1000)

//Maximum number of allocated arrays
#define CLUSTER_MAX_NARRAYS (1024)

//Maximum number of operation merged together.
#define CLUSTER_MAX_OP_MERGES (10)

//Default blocksize
#define CLUSTER_BLOCKSIZE (2)

//Maximum number of nodes in the ready queue.
#define CLUSTER_RDY_QUEUE_MAXSIZE (1024*10)

//The maximum size of the work buffer in bytes (should be power of 2).
#define CLUSTER_WORK_BUFFER_MAXSIZE (536870912) //½GB

//Operation types
enum opt {CLUSTER_MSG_END, CLUSTER_INIT_BLOCKSIZE,
          CLUSTER_INIT_PROC_GRID, CLUSTER_RECV, CLUSTER_SEND,
          CLUSTER_BRECV, CLUSTER_BSEND, CLUSTER_APPLY, CLUSTER_COMM,
          CLUSTER_NONCOMM, CLUSTER_PUT_ITEM, CLUSTER_GET_ITEM};

//dndnode prototype.
typedef struct dndnode_struct dndnode;
typedef struct dndarray_struct dndarray;

//Type describing a distributed array.
struct dndarray_struct
{
    //Unique identification.
    cphvb_intp uid;
    //Reference count.
    int refcount;
    //Number of dimensions.
    int ndims;
    //Size of dimensions.
    cphvb_intp dims[CPHVB_MAXDIM];
    //Size of block-dimensions.
    cphvb_intp blockdims[CPHVB_MAXDIM];
    //Number of blocks (global).
    cphvb_intp nblocks;
    //Data type of elements in array.
    int dtype;
    //Size of an element in bytes.
    int elsize;
    //Pointer to local data.
    char *data;
    //Number of local elements (local to the MPI-process).
    cphvb_intp nelements;
    //Size of local dimensions (local to the MPI-process).
    cphvb_intp localdims[CPHVB_MAXDIM];
    //Size of local block-dimensions (local to the MPI-process).
    cphvb_intp localblockdims[CPHVB_MAXDIM];
    //MPI-datatype that correspond to an array element.
    MPI_Datatype mpi_dtype;
    //Root nodes (one per block).
    dndnode **rootnodes;
    //Next and prev are used for traversing all arrays.
    #ifdef CLUSTER_STATISTICS
        dndarray *next;
        dndarray *prev;
    #endif
};

//dndslice constants.
#define PseudoIndex -1//Adds a extra 1-dim - 'A[1,newaxis]'
#define SingleIndex -3//Dim not visible - 'A[1]'

//Type describing a slice of a dimension.
typedef struct
{
    //Start index.
    cphvb_intp start;
    //Elements between index.
    cphvb_intp step;
    //Number of steps (Length of the dimension).
    cphvb_intp nsteps;
} dndslice;

//View-alteration flags.
#define CLUSTER_NDIMS    0x001
#define CLUSTER_STEP     0x002
#define CLUSTER_NSTEPS   0x004

//Type describing a view of a distributed array.
typedef struct
{
    //Unique identification.
    cphvb_intp uid;
    //The array this view is a view of.
    dndarray *base;
    //Number of viewable dimensions.
    int ndims;
    //Number of sliceses. NB: nslice >= base->ndims.
    int nslice;
    //Sliceses - the global view of the base-array.
    dndslice slice[CPHVB_MAXDIM];
    //A bit mask specifying which alterations this view represents.
    //Possible flags:
    //Zero        - no alterations.
    //DNPY_NDIMS  - number of dimensions altered.
    //DNPY_STEP   - 'step' altered.
    //DNPY_NSTEPS - 'nsteps' altered.
    int alterations;
    //Number of view-blocks.
    cphvb_intp nblocks;
    //Number of view-blocks in each viewable dimension.
    cphvb_intp blockdims[CPHVB_MAXDIM];
} dndview;

//Type describing a sub-section of a view block.
typedef struct
{
    //The rank of the MPI-process that owns this sub-block.
    int rank;
    //Start index (one per base-dimension).
    cphvb_intp start[CPHVB_MAXDIM];
    //Number of elements (one per base-dimension).
    cphvb_intp nsteps[CPHVB_MAXDIM];
    //Number of elements to next dimension (one per base-dimension).
    cphvb_intp stride[CPHVB_MAXDIM];
    //The MPI communication offset (in bytes).
    cphvb_intp comm_offset;
    //Number of elements in this sub-view-block.
    cphvb_intp nelem;
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
    cphvb_intp uid;
    //All sub-view-blocks in this view block (Row-major).
    dndsvb *sub;
    //Number of sub-view-blocks.
    cphvb_intp nsub;
    //Number of sub-view-blocks in each dimension.
    cphvb_intp svbdims[CPHVB_MAXDIM];
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
    cphvb_intp refcount;                    \
    char op;                                \
    char optype;                            \
    char narys;                             \
    dndview *views[CPHVB_MAX_NO_OPERANDS];  \
    dndsvb *svbs[CPHVB_MAX_NO_OPERANDS];    \
    char accesstypes[CPHVB_MAX_NO_OPERANDS];
#ifdef DNPY_STATISTICS
    #define DNDOP_HEAD DNDOP_HEAD_BASE cphvb_intp uid;
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
    cphvb_intp mpi_tag;
    //The MPI rank of the process that is the remote communication peer.
    int remote_rank;
} dndop_comm;

//Type describing an apply-sub-view-block, which is a subsection of a
//sub-view-block that is used in apply.
typedef struct
{
    cphvb_intp dims[CPHVB_MAXDIM];
    cphvb_intp stride[CPHVB_MAXDIM];
    cphvb_intp offset;
} dndasvb;

//Type describing a universal function DAG node.
typedef struct
{
    DNDOP_HEAD
    //List of apply-sub-view-block.
    dndasvb asvb[CPHVB_MAX_NO_OPERANDS];
    //Number of output array views.
    char nout;
    //The operation described as a function, a data and a Python pointer.
    /*
    PyUFuncGenericFunction func;
    void *funcdata;
    PyObject *PyOp;*/
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
        cphvb_intp uid;
    #endif
};

#endif
