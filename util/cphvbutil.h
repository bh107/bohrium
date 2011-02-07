#ifndef __CPHVBUTIL_H
#define __CPHVBUTIL_H

#include <cphvb.h>

#define bool int
#define FALSE 0
#define TRUE (!FALSE)

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
                      cphvb_index* base_stride);


/* Is the data accessed continuously, and only once
 *
 * @ndim     Number of dimentions 
 * @shape[]  Number of elements in each dimention.
 * @stride[] Stride in each dimention.
 * @return   Truth value indicating continuousity. 
 */
bool cphvb_is_continuous(cphvb_int32 ndim,
                         const cphvb_index shape[], 
                         const cphvb_index stride[]);


/* Number of element in a given shape
 *
 * @ndim     Number of dimentions 
 * @shape[]  Number of elements in each dimention.
 * @return   Number of element operations
 */
size_t cphvb_nelements(cphvb_int32 ndim,
                       const cphvb_index shape[]);


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
                              cphvb_index element);



#endif
