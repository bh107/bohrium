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

#include <map>
#include <string>
#include <algorithm>
#include <tuple>
#include <iostream>
#include <sstream>

#include <bh_view.hpp>
#include <bh_base.hpp>
#include <bh_pprint.hpp>

using namespace std;

bh_view::bh_view(const bh_view &view) {
    base = view.base;
    if (base == nullptr) {
        return; //'view' is a constant thus the rest are garbage
    }

    start = view.start;
    ndim = view.ndim;
    assert(ndim < BH_MAXDIM);
    slides = view.slides;
    shape = view.shape;
    stride = view.stride;
}

bh_view::bh_view(bh_base *base) {
    this->base = base;
    this->ndim = 1;
    this->start = 0;
    this->shape.push_back(this->base->nelem());
    this->stride.push_back(1);
}

void bh_view::insert_axis(int64_t dim, int64_t size, int64_t stride) {
    assert(dim <= ndim);
    this->shape.insert(this->shape.begin() + dim, size);
    this->stride.insert(this->stride.begin() + dim, size);
    ++ndim;
}

void bh_view::remove_axis(int64_t dim) {
    assert(1 < ndim);
    assert(dim < ndim);
    shape.erase(shape.begin() + dim);
    stride.erase(stride.begin() + dim);
    --ndim;
}

void bh_view::transpose(int64_t axis1, int64_t axis2) {
    assert(0 <= axis1 and axis1 < ndim);
    assert(0 <= axis2 and axis2 < ndim);
    assert(not isConstant());
    std::swap(shape[axis1], shape[axis2]);
    std::swap(stride[axis1], stride[axis2]);
    slides.transpose(axis1, axis2);
}

bool bh_view::isContiguous() const {
    if (isConstant()) {
        return false;
    }
    int64_t weight = 1;
    for (int64_t dim = ndim - 1; dim >= 0; --dim) {
        if (shape[dim] > 1 && stride[dim] != weight) {
            return false;
        }
        weight *= shape[dim];
    }
    return true;
}

vector<tuple<int64_t, int64_t, int64_t> > bh_view::python_notation() const {
    //stride&shape&index for each dimension (in that order)
    vector<tuple<int64_t, int64_t, int64_t> > sns;
    for (int64_t i = 0; i < this->ndim; ++i) {
        sns.push_back(make_tuple(this->stride[i], this->shape[i], i));
    }

    //Let's sort them such that the greatest strides comes first
    sort(sns.begin(), sns.end(), greater<tuple<int64_t, int64_t, int64_t> >());

    //Now let's compute start&end&stride of each dimension, which
    //makes up the python notation e.g. [2:4:1]
    vector<tuple<int64_t, int64_t, int64_t> > sne(sns.size());
    int64_t offset = this->start;
    for (size_t i = 0; i < sns.size(); ++i) {
        const int64_t stride = std::get<0>(sns[i]);
        const int64_t shape = std::get<1>(sns[i]);
        const int64_t index = std::get<2>(sns[i]);

        int64_t start = 0;

        if (stride > 0) {//avoid division by zero
            start = offset / stride;
        }

        int64_t end = start + shape;
        offset -= start * stride;
        assert(offset >= 0);
        sne[index] = make_tuple(start, end, stride);
    }

    // If 'offset' wasn't reduced to zero, we have to append a singleton dimension
    // with the stride of 'offset'
    if (offset > 0) {
        sne.push_back(make_tuple(1, 2, offset));
    }

    /*
    //Find the contiguous strides, that is the stride of each dimension
    //assuming the view is contiguous.
    int64_t cont_stride = 1;
    for(int64_t i=sns.size()-1; i >= 0; --i)
    {
        int64_t stride = std::get<0>(sns[i]);
        int64_t shape  = std::get<1>(sns[i]);
        int64_t index  = std::get<2>(sns[i]);

        if(stride == cont_stride)
            std::get<2>(sne[index]) = -1;//Flag it as unnecessary
        cont_stride *= shape;
    }
    */
    return sne;
}

string bh_view::pprint(bool py_notation) const {
    stringstream ss;
    ss << "a" << base->getLabel() << "[";
    if (isConstant()) {
        ss << "CONST";
    } else if (py_notation) {
        const vector<tuple<int64_t, int64_t, int64_t> > sne = python_notation();
        for (size_t i = 0; i < sne.size(); ++i) {
            int64_t start = std::get<0>(sne[i]);
            int64_t end = std::get<1>(sne[i]);
            int64_t stride = std::get<2>(sne[i]);
            ss << start << ":" << end << ":" << stride;

            if (i < sne.size() - 1) { //Not the last iteration
                ss << ",";
            }
        }
    } else {
        ss << "start: " << start;
        ss << ", ndim: " << ndim;
        ss << ", shape: " << shape;
        ss << ", stride: " << stride;
        ss << ", base: " << base;
    }
    ss << "]";
    return ss.str();
}

ostream &operator<<(ostream &out, const bh_view &v) {
    out << v.pprint(true);
    return out;
}

bool bh_view_same_shape(const bh_view *a, const bh_view *b) {
    if (a->ndim != b->ndim) {
        return false;
    }

    for (int i = 0; i < a->ndim; ++i) {
        if (a->shape[i] != b->shape[i]) {
            return false;
        }
    }

    return true;
}

bool bh_view_disjoint(const bh_view *a, const bh_view *b) {
    // TODO: In order to fixed BUG like <https://github.com/bh107/bohrium/issues/178>, we say that sharing
    //       the same base makes the views overlapping for now.
    return bh_base_array(a) != bh_base_array(b);
}
