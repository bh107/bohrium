/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium
team <http://www.bh107.org>.

Bohrium is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3
of the License, or (at your option) any later version.

Bohrium is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the
GNU Lesser General Public License along with Bohrium.

If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __BH_ARRAY_H
#define __BH_ARRAY_H

#include <algorithm>
#include <stdbool.h>
#include <iostream>
#include <map>
#include <vector>
#include <cstring>
#include <cassert>
#include <tuple>
#include "bh_type.h"
#include <bh_constant.hpp>
#include "bh_error.h"
#include "bh_win.h"

#include <boost/serialization/split_member.hpp>

// Forward declaration of class boost::serialization::access
namespace boost {namespace serialization {class access;}}

#define BH_MAXDIM (16)

struct bh_base
{
    /// Pointer to the actual data.
    bh_data_ptr   data;

    /// The type of data in the array
    bh_type       type;

    /// The number of elements in the array
    bh_index      nelem;

    // Returns an unique ID of this base array
    unsigned int get_label() const;

    // Returns pprint string of this base array
    std::string str() const;

    template<class Archive>
    void save(Archive & ar, const unsigned int version) const
    {
        size_t tmp = (size_t)data;
        ar << tmp;
        ar & type;
        ar & nelem;
    }
    template<class Archive>
    void load(Archive & ar, const unsigned int version)
    {
        size_t tmp;
        ar >> tmp;
        data = (void*)tmp;
        ar & type;
        ar & nelem;
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()
};

//Implements pprint of base arrays
DLLEXPORT std::ostream& operator<<(std::ostream& out, const bh_base& b);

struct bh_view
{
    bh_view(){}
    bh_view(const bh_view& view)
    {
        base = view.base;
        if(base == NULL)
            return; //'view' is a constant thus the rest are garbage
        start = view.start;
        ndim = view.ndim;
        assert(ndim < BH_MAXDIM);
        std::memcpy(shape, view.shape, ndim * sizeof(bh_index));
        std::memcpy(stride, view.stride, ndim * sizeof(bh_index));
    }

    /// Pointer to the base array.
    bh_base*      base;

    /// Index of the start element
    bh_index      start;

    /// Number of dimensions
    bh_intp       ndim;

    /// Number of elements in each dimensions
    bh_index      shape[BH_MAXDIM];

    /// The stride for each dimensions
    bh_index      stride[BH_MAXDIM];

    // Returns a vector of tuples that describe the view using (almost)
    // Python Notation.
    // NB: in this notation the stride is always absolute eg. [2:4:3, 0:3:1]
    std::vector<std::tuple<int64_t, int64_t, int64_t> > python_notation() const;

    // Insert a new dimension at 'dim' with the size of 'size' and stride of 'stride'
    // NB: 0 <= 'dim' <= ndim
    void insert_dim(bh_index dim, bh_index size, bh_index stride);

    // Transposes by swapping the two axises 'axis1' and 'axis2'
    void transpose(int64_t axis1, int64_t axis2);

    bool operator<(const bh_view& other) const
    {
        if (base < other.base) return true;
        if (other.base < base) return false;
        if (start < other.start) return true;
        if (other.start < start) return false;
        if (ndim < other.ndim) return true;
        if (other.ndim < ndim) return false;
        for (bh_intp i = 0; i < ndim; ++i)
        {
            if (shape[i] < other.shape[i]) return true;
            if (other.shape[i] < shape[i]) return false;
        }
        for (bh_intp i = 0; i < ndim; ++i)
        {
            if (stride[i] < other.stride[i]) return true;
            if (other.stride[i] < stride[i]) return false;
        }
        return false;
    }

    bool operator==(const bh_view& other) const
    {
        if (base == NULL or this->base == NULL) return false;
        if (base != other.base) return false;
        if (ndim != other.ndim) return false;
        if (start != other.start) return false;
        for (bh_intp i = 0; i < ndim; ++i)
            if (shape[i] != other.shape[i]) return false;
        for (bh_intp i = 0; i < ndim; ++i)
            if (stride[i] != other.stride[i]) return false;
        return true;
    }

    bool operator!=(const bh_view& other) const
    {
        return !(*this == other);
    }

    template<class Archive>
    void save(Archive & ar, const unsigned int version) const
    {
        size_t tmp = (size_t)base;
        ar << tmp;
        ar & start;
        ar & ndim;
        for(bh_intp i=0; i<ndim; ++i)
        {
            ar & shape[i];
            ar & stride[i];
        }
    }
    template<class Archive>
    void load(Archive & ar, const unsigned int version)
    {
        size_t tmp;
        ar >> tmp;
        base = (bh_base*)tmp;
        ar & start;
        ar & ndim;
        for(bh_intp i=0; i<ndim; ++i)
        {
            ar & shape[i];
            ar & stride[i];
        }
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()
};

//Implements pprint of views
DLLEXPORT std::ostream& operator<<(std::ostream& out, const bh_view& v);

/** Create a new base array.
 *
 * @param type The type of data in the array
 * @param nelements The number of elements
 * @param new_base The handler for the newly created base
 * @return Error code (BH_SUCCESS, BH_OUT_OF_MEMORY)
 */
DLLEXPORT bh_error bh_create_base(bh_type    type,
                                  bh_index   nelements,
                                  bh_base**  new_base);

/** Destroy the base array.
 *
 * @param base  The base array in question
 */
DLLEXPORT void bh_destroy_base(bh_base**  base);

/* Returns the simplest view (fewest dimensions) that access
 * the same elements in the same pattern
 *
 * @view The view
 * @return The simplified view
 */
DLLEXPORT bh_view bh_view_simplify(const bh_view &view);

/* Simplifies the given view down to the given shape.
 * If that is not possible an std::invalid_argument exception is thrown
 *
 * @view The view
 * @return The simplified view
 */
DLLEXPORT bh_view bh_view_simplify(const bh_view& view, const std::vector<bh_index>& shape);

/* Find the base array for a given view
 *
 * @view   The view in question
 * @return The Base array
 */
#define bh_base_array(view) ((view)->base)

/* Number of non-broadcasted elements in a given view
 *
 * @view    The view in question.
 * @return  Number of elements.
 */
bh_index bh_nelements_nbcast(const bh_view *view);

/* Number of element in a given shape
 *
 * @ndim     Number of dimentions
 * @shape[]  Number of elements in each dimention.
 * @return   Number of element operations
 */
DLLEXPORT bh_index bh_nelements(bh_intp ndim,
                                const bh_index shape[]);
DLLEXPORT bh_index bh_nelements(const bh_view& view);

/* Size of the base array in bytes
 *
 * @base    The base in question
 * @return  The size of the base array in bytes
 */
bh_index bh_base_size(const bh_base *base);

/* Set the view stride to contiguous row-major
 *
 * @view    The view in question
 * @return  The total number of elements in view
 */
DLLEXPORT bh_intp bh_set_contiguous_stride(bh_view *view);

/* Updates the view with the complete base
 *
 * @view    The view to update (in-/out-put)
 * @base    The base assign to the view
 * @return  The total number of elements in view
 */
DLLEXPORT void bh_assign_complete_base(bh_view *view, bh_base *base);

/* Set the data pointer for the view.
 * Can only set to non-NULL if the data ptr is already NULL
 *
 * @view   The view in question
 * @data   The new data pointer
 * @return Error code (BH_SUCCESS, BH_ERROR)
 */
DLLEXPORT bh_error bh_data_set(bh_view* view, bh_data_ptr data);

/* Get the data pointer for the view.
 *
 * @view    The view in question
 * @result  Output data pointer
 * @return  Error code (BH_SUCCESS, BH_ERROR)
 */
DLLEXPORT bh_error bh_data_get(bh_view* view, bh_data_ptr* result);

/* Allocate data memory for the given base if not already allocated.
 * For convenience, the base is allowed to be NULL.
 *
 * @base    The base in question
 * @return  Error code (BH_SUCCESS, BH_ERROR, BH_OUT_OF_MEMORY)
 */
DLLEXPORT bh_error bh_data_malloc(bh_base* base);

/* Frees data memory for the given view.
 * For convenience, the view is allowed to be NULL.
 *
 * @base    The base in question
 * @return  Error code (BH_SUCCESS, BH_ERROR)
 */
bh_error bh_data_free(bh_base* base);

/* Determines whether the view is a scalar or a broadcasted scalar.
 *
 * @view The view
 * @return The boolean answer
 */
DLLEXPORT bool bh_is_scalar(const bh_view* view);

/* Determines whether the operand is a constant
 *
 * @o The operand
 * @return The boolean answer
 */
DLLEXPORT bool bh_is_constant(const bh_view* o);

/* Flag operand as a constant
 *
 * @o The operand
 */
DLLEXPORT void bh_flag_constant(bh_view* o);

/* Determines whether two views have same shape.
 *
 * @a The first view
 * @b The second view
 * @return The boolean answer
 */
DLLEXPORT bool bh_view_same_shape(const bh_view *a, const bh_view *b);

/* Determines whether two views are identical and points
 * to the same base array.
 *
 * @a The first view
 * @b The second view
 * @return The boolean answer
 */
DLLEXPORT bool bh_view_same(const bh_view *a, const bh_view *b);

/* Determines whether a view is contiguous
 *
 * @a The view
 * @return The boolean answer
 */
DLLEXPORT bool bh_is_contiguous(const bh_view *a);

/* Determines whether two views are aligned and points
 * to the same base array.
 *
 * @a The first view
 * @b The second view
 * @return The boolean answer
 */
DLLEXPORT bool bh_view_aligned(const bh_view *a, const bh_view *b);

/* Determines whether two views are aligned, points
 * to the same base array, and have same shape.
 *
 * @a The first view
 * @b The second view
 * @return The boolean answer
 */
DLLEXPORT bool bh_view_aligned_and_same_shape(const bh_view *a, const bh_view *b);

/* Determines whether two views access some of the same data points
 * NB: This functions may return True on non-overlapping views.
 *     But will always return False on overlapping views.
 *
 * @a The first view
 * @b The second view
 * @return The boolean answer
 */
DLLEXPORT bool bh_view_disjoint(const bh_view *a, const bh_view *b);


#endif
