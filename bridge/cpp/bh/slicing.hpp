/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium team:
http://bohrium.bitbucket.org

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
#ifndef __BOHRIUM_BRIDGE_CPP_SLICING
#define __BOHRIUM_BRIDGE_CPP_SLICING

namespace bh {

inline slice_range::slice_range() : begin(0), end(-1), stride(1) {}
inline slice_range::slice_range(int begin, int end, size_t stride) : begin(begin), end(end), stride(stride), inclusive_end(false) {}

inline slice_range& _(int begin, int end, size_t stride)
{
    slice_range* le_range = (new slice_range(begin, end, stride));
    le_range->inclusive_end = (0==end);

    return *le_range;
}

template <typename T>
slice<T>::slice(multi_array<T>& op) : op(&op), dims(0)
{
    for(int i=0; i<BH_MAXDIM; i++) {
        ranges[i] = slice_range();
    }
}

/**
 * Access a single element.
 */
template <typename T>
slice<T>& slice<T>::operator[](int rhs)
{
    ranges[dims].begin = rhs;
    ranges[dims].end   = rhs;
    dims++;

    return *this;
}

/**
 *  Access a range.
 */
template <typename T>
slice<T>& slice<T>::operator[](slice_range& rhs)
{
    ranges[dims] = rhs;
    dims++;

    return *this;
}

/**
 * Assign directly to a slice.
 * Such as:
 *
 * grid[_(1,-1,1)][_(1,-1,1)] = 1
 *
 * This should construct a temporary view and assign the value to it.
 *
 */
template <typename T>
multi_array<T>& slice<T>::operator=(T rhs)
{
    multi_array<T>* vv = &this->view();

    *vv = rhs;
    vv->setTemp(true);

    // TODO: Who collects the garbage here?

    return *vv;
}

/**
 *  Materialize the view based on the list of slice_ranges.
 */
template <typename T>
bh::multi_array<T>& slice<T>::view()
{
    multi_array<T>* lhs = &Runtime::instance().temp_view(*op);

    lhs->meta = op->meta;

    //lhs->meta.ndim   = op->meta.ndim;                    // Rank is maintained
    //lhs->meta.start  = op->meta.start;                   // Start is initialy the same
    int b, e;

    for(int i=0, lhs_dim=0; i < op->meta.ndim; ++i) {
                                                // Compute the "[beginning, end[" indexes
        b = ranges[i].begin < 0 ? op->meta.shape[i] + ranges[i].begin : ranges[i].begin;
        e = ranges[i].end   < 0 ? op->meta.shape[i] + ranges[i].end   : ranges[i].end;

        // NOTICE: e = 0 is special-case to make the last-element inclusive
        //         it is also used for single-element indexing but in that case
        //         inclusive_end should be false...
        if (ranges[i].inclusive_end && (ranges[i].end == 0)) {
            e = op->meta.shape[i];
        }

        if (b<e) {                              // Range
            lhs->meta.shape[lhs_dim]   = 1 + (((e-b) - 1) / ranges[i].stride); // ceil
            lhs->meta.stride[lhs_dim]  = ranges[i].stride * op->meta.stride[i];
            ++lhs_dim;
        } else if (b==e) {                      // Single-element
            lhs->meta.ndim   = lhs->meta.ndim -1;
        } else {
            throw std::runtime_error("Invalid range.");
        }
        lhs->meta.start  += b * op->meta.stride[i];
    }

    if (lhs->meta.ndim == 0) {                       // Fix up the pseudo-scalar
        lhs->meta.ndim = 1;
        lhs->meta.shape[0]   = 1;
        lhs->meta.stride[0]  = 1;
    }

    return *lhs;
}

}

#endif

