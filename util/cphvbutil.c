#include "cphvbutil.h"

/* Reduce nDarray info to a base shape
 *
 * Remove dimentions that just indicate orientation in a 
 * higher dimention (Py:newaxis)
 *
 * @ndim          Number of dimentions 
 * @shape[]       Number of elements in each dimention.
 * @stride[]      Stride in each dimention.
 * @base_ndim     Placeholder for base number of dimentions 
 * @base_shape    Placeholder for base number of elements in each dimention.
 * @base_stride   Placeholder for base stride in each dimention.
 */
void cphvb_base_shape(cphvb_int32 ndim,
                      const cphvb_index shape[], 
                      const cphvb_index stride[],
                      cphvb_int32* base_ndim,
                      cphvb_index* base_shape, 
                      cphvb_index* base_stride)
{
    *base_ndim = 0;
    for (int i = 0; i < ndim; ++i)
    {
        // skipping (shape[i] == 1 && stride[i] == 0)        
        if (shape[i] != 1 || stride[i] != 0)
        {
            base_shape[*base_ndim] = shape[i];
            base_stride[*base_ndim] = stride[i];
            ++(*base_ndim);
        }
    }
}

/* Is the data accessed continuously, and only once
 *
 * @ndim     Number of dimentions 
 * @shape[]  Number of elements in each dimention.
 * @stride[] Stride in each dimention.
 * @return   Truth value indicating continuousity. 
 */
bool cphvb_is_continuous(cphvb_int32 ndim,
                         const cphvb_index shape[], 
                         const cphvb_index stride[])
{
    cphvb_int32 my_ndim = 0;
    cphvb_index my_shape[ndim]; 
    cphvb_index my_stride[ndim];
    cphvb_base_shape(ndim, shape, stride, &my_ndim, my_shape, my_stride);
    for (int i = 0; i < my_ndim - 1; ++i)
    {
        if (my_shape[i+1] != my_stride[i])
            return FALSE;
    }
    if (my_stride[my_ndim-1] != 1)
        return FALSE;
    
    return TRUE;
}


/* Number of element in a given shape
 *
 * @ndim     Number of dimentions 
 * @shape[]  Number of elements in each dimention.
 * @return   Number of element operations
 */
size_t cphvb_nelements(cphvb_int32 ndim,
                       const cphvb_index shape[])
{
    size_t res = 1;
    for (int i = 0; i < ndim; ++i)
    {
        res *= shape[i];
    }
    return res;
}


/* Calculate the offset into an array based on element index
 *
 * @ndim     Number of dimentions 
 * @shape[]  Number of elements in each dimention.
 * @stride[] Stride in each dimention.
 * @element  Index of element in question
 * @return   Truth value indicating continuousity. 
 */
cphvb_index cphvb_calc_offset(cphvb_int32 ndim,
                              const cphvb_index shape[],
                              const cphvb_index stride[],
                              const cphvb_index element)
{
    cphvb_int32 dim = ndim -1;
    cphvb_index dimbound = shape[dim];
    cphvb_index offset = (element % dimbound) * stride[dim];
    for (--dim; dim >= 0 ; --dim)
    {
        offset += (element / dimbound) * stride[dim];
        dimbound *= shape[dim];
    }
    return offset;
}
