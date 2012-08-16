/*
This file is part of cphVB and copyright (c) 2012 the cphVB team:
http://cphvb.bitbucket.org

cphVB is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as 
published by the Free Software Foundation, either version 3 
of the License, or (at your option) any later version.

cphVB is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the 
GNU Lesser General Public License along with cphVB. 

If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef CPHVB_VEM_CLUSTER_H
#define CPHVB_VEM_CLUSTER_H
#include <mpi.h>
#include <cphvb.h>

#ifdef __cplusplus
extern "C" {
#endif

//#define DNPY_STATISTICS
//#define DNDY_TIME
//#define DNPY_SPMD

//Minimum jobsize for an OpenMP thread. >blocksize means no OpenMP.
#define DNPY_MIN_THREAD_JOBSIZE 10

//Maximum message size (in bytes)
#define DNPY_MAX_MSG_SIZE 1024*4

//Maximum number of memory allocations in the memory pool.
#define DNPY_MAX_MEM_POOL 10

//Maximum number of view block operations in the sub-view-block DAG.
#define DNPY_MAX_VB_IN_SVB_DAG 100

//Disable Lazy Evaluation by definding this macro.
#undef DNPY_NO_LAZY_EVAL

//Maximum number of allocated arrays
#define DNPY_MAX_NARRAYS 1024

//Maximum number of operation merged together.
#define DNPY_MAX_OP_MERGES 10

//Default blocksize
#define DNPY_BLOCKSIZE 2

//Maximum number of nodes in the ready queue.
#define DNPY_RDY_QUEUE_MAXSIZE 1024*10

//Maximum MPI tag.
#define DNPY_MAX_MPI_TAG 1048576

//The maximum size of the work buffer in bytes (should be power of 2).
#define DNPY_WORK_BUFFER_MAXSIZE 536870912 //½GB

//The work buffer memory alignment.
#define DNPY_WORK_BUFFER_MEM_ALIGNMENT 32

//Easy retrieval of dnduid
#define PyArray_DNDUID(obj) (((PyArrayObject *)(obj))->dnduid)


cphvb_intp blocksize = 2;

//dndnode prototype.
typedef struct dndnode_struct dndnode;
typedef struct dndarray_struct dndarray;
typedef struct dndmem_struct dndmem;

//Type describing a memory allocation.
struct dndmem_struct
{
    //Size of allocated memory.
    cphvb_intp size;
    //Pointer to the next free memory allocation.
    dndmem *next;
};

//Type describing a distributed array.

struct dndarray_struct
{
    CPHVB_ARRAY_HEAD__DET_MAA_JEG_IKKE_BRUG_array_database.h
    //The array that propagates down to the cphVB child component.
    cphvb_array *child_ary;
    //Size of block-dimensions.
    cphvb_intp blockdims[CPHVB_MAXDIM];
    //Number of blocks (global).
    cphvb_intp nblocks;
    //Number of local elements (local to the MPI-process).
    cphvb_intp localsize;
    //Size of local dimensions (local to the MPI-process).
    cphvb_intp localdims[CPHVB_MAXDIM];
    //Stride of local dimensions (local to the MPI-process).
    cphvb_intp localstride[CPHVB_MAXDIM];
    //Size of local block-dimensions (local to the MPI-process).
    cphvb_intp localblockdims[CPHVB_MAXDIM];
    //MPI-datatype that correspond to an array element.
    MPI_Datatype mpi_dtype;
    //Root nodes (one per block).
    dndnode **rootnodes;

};

//dndslice constants.
#define PseudoIndex -1//Adds a extra 1-dim - 'A[1,newaxis]'
#define RubberIndex -2//A[1,2,...] (Not used in distnumpy.inc)
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
#define DNPY_NDIMS      0x001
#define DNPY_STEP       0x002
#define DNPY_NSTEPS     0x004
#define DNPY_NONALIGNED 0x008

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
    //Zero            - no alterations.
    //DNPY_NDIMS      - number of dimensions altered.
    //DNPY_STEP       - 'step' altered.
    //DNPY_NSTEPS     - 'nsteps' altered.
    //DNPY_NONALIGNED - 'start % blocksize != 0' or 'step != 1'.
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
    cphvb_intp refcount;                      \
    char op;                                \
    char optype;                            \
    char narys;                             \
    dndview *views[CPHVB_MAX_NO_OPERANDS];            \
    dndsvb *svbs[CPHVB_MAX_NO_OPERANDS];              \
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

//Type describing the timing data.
typedef struct
{
    unsigned long long total;
    unsigned long long dag_svb_flush;
    unsigned long long dag_svb_rm;
    unsigned long long apply_ufunc;
    unsigned long long ufunc_comm;
    unsigned long long comm_init;
    unsigned long long arydata_free;
    unsigned long long reduce_1d;
    unsigned long long reduce_nd;
    unsigned long long reduce_nd_apply;
    unsigned long long zerofill;
    unsigned long long ufunc_svb;
    unsigned long long dag_svb_add;
    unsigned long long calc_vblock;
    unsigned long long arydata_malloc;
    unsigned long long msg2slaves;
    unsigned long long final_barrier;
    cphvb_intp mem_reused;
    cphvb_intp nconnect;
    cphvb_intp nconnect_max;
    cphvb_intp napply;
    cphvb_intp nflush;
} dndtime;

//Macro that increases the work buffer pointer.
#define WORKBUF_INC(bytes_taken)                                       \
{                                                                      \
    workbuf_nextfree += bytes_taken;                                   \
    workbuf_nextfree += DNPY_WORK_BUFFER_MEM_ALIGNMENT -               \
                        (((cphvb_intp)workbuf_nextfree)                  \
                        % DNPY_WORK_BUFFER_MEM_ALIGNMENT);             \
    if(workbuf_nextfree >= workbuf_max)                                \
    {                                                                  \
        fprintf(stderr, "Work buffer overflow - increase the maximum " \
                "work buffer size or decrease the maximum DAG size. "  \
                "The current values are %dMB and %d nodes,"            \
                "respectively.\n", DNPY_WORK_BUFFER_MAXSIZE / 1048576, \
                DNPY_MAX_VB_IN_SVB_DAG);                               \
        MPI_Abort(MPI_COMM_WORLD, -1);                                 \
    }                                                                  \
    assert(((cphvb_intp) workbuf_nextfree) %                             \
                       DNPY_WORK_BUFFER_MEM_ALIGNMENT == 0);           \
}

#ifdef __cplusplus
}
#endif

#endif
