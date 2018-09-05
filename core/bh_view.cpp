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

    std::memcpy(shape, view.shape, ndim * sizeof(int64_t));
    std::memcpy(stride, view.stride, ndim * sizeof(int64_t));
}

bh_view::bh_view(bh_base &base) {
    bh_assign_complete_base(this, &base);
}

void bh_view::insert_axis(int64_t dim, int64_t size, int64_t stride) {
    assert(dim <= ndim);

    if (dim == ndim) { // Appending
        this->shape[dim] = size;
        this->stride[dim] = stride;
    } else { // Inserting
        for (int64_t i = ndim - 1; i >= 0; --i) {
            if (i >= dim) { // Move shape and stride one to the right
                this->shape[i + 1] = this->shape[i];
                this->stride[i + 1] = this->stride[i];
                if (i == dim) { // Insert the new dimension
                    this->shape[i] = size;
                    this->stride[i] = stride;
                }
            }
        }
    }

    ++ndim;
}

void bh_view::remove_axis(int64_t dim) {
    assert(1 < ndim);
    assert(dim < ndim);

    for (int64_t i = dim; i < ndim - 1; ++i) {
        shape[i] = shape[i + 1];
        stride[i] = stride[i + 1];
    }

    --ndim;
}

void bh_view::transpose(int64_t axis1, int64_t axis2) {
    assert(0 <= axis1 and axis1 < ndim);
    assert(0 <= axis2 and axis2 < ndim);
    assert(not bh_is_constant(this));

    std::swap(shape[axis1], shape[axis2]);
    std::swap(stride[axis1], stride[axis2]);
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
    ss << "a" << base->get_label() << "[";
    if (bh_is_constant(this)) {
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
        ss << ", shape: " << pprint_carray(shape, ndim);
        ss << ", stride: " << pprint_carray(stride, ndim);
        ss << ", base: " << base;
    }

    ss << "]";
    return ss.str();
}

ostream &operator<<(ostream &out, const bh_view &v) {
    out << v.pprint(true);
    return out;
}

bh_view bh_view_simplify(const bh_view &view) {
    bh_view res;
    res.base = view.base;
    res.ndim = 0;
    res.start = view.start;
    int64_t i = 0;

    while (view.shape[i] == 1 && i < view.ndim - 1) {
        ++i;
    }

    res.shape[0] = view.shape[i];
    res.stride[0] = view.stride[i];

    for (++i; i < view.ndim; ++i) {
        if (view.shape[i] == 0) {
            res.ndim = 1;
            res.shape[0] = 0;
            return res;
        } else if (view.shape[i] == 1) {
            continue;
        }

        if (view.shape[i] * view.stride[i] == res.stride[res.ndim]) {
            res.shape[res.ndim] *= view.shape[i];
            res.stride[res.ndim] = view.stride[i];
        } else {
            ++res.ndim;
            res.shape[res.ndim] = view.shape[i];
            res.stride[res.ndim] = view.stride[i];
        }
    }

    if (res.ndim == 0 || res.shape[res.ndim] > 1) {
        ++res.ndim;
    }
    return res;
}

bh_view bh_view_simplify(const bh_view &view, const std::vector<int64_t> &shape) {
    assert(false); // TODO: complete rewrite under the assumption the cleandim has been run

    if (view.ndim < (int64_t) shape.size()) {
        std::stringstream ss;
        ss << "Can not simplify to more dimensions: ";
        ss << "shape: " << shape << " view: " << view;
        throw std::invalid_argument(ss.str());
    }

    bh_view res;
    res.base = view.base;
    res.ndim = 0;
    res.start = view.start;
    int64_t i = 0;

    while (view.shape[i] == 1 && i < view.ndim - 1) {
        ++i;
    }

    res.shape[0] = view.shape[i];
    res.stride[0] = view.stride[i];

    for (++i; i < view.ndim; ++i) {
        if (shape[res.ndim] == 0) {
            if (view.shape[i] != 0) {
                continue;
            } else {
                res.shape[res.ndim++] = 0;
                return res;
            }
        }

        if ((int64_t) shape.size() == res.ndim) {
            if (view.shape[i] == 1) {
                continue;
            } else {
                std::stringstream ss;
                ss << "Can not remove trailing dimensions of size > 1: ";
                ss << "shape: " << shape << " view: " << view;
                throw std::invalid_argument(ss.str());
            }
        }

        if (view.shape[i - 1] > shape[res.ndim]) {
            std::stringstream ss;
            ss << "Can not simplify to lower dimension size: ";
            ss << "shape: " << shape << " view: " << view;
            throw std::invalid_argument(ss.str());
        } else if (view.shape[i - 1] == shape[res.ndim]) {
            res.shape[++res.ndim] = view.shape[i];
            res.stride[res.ndim] = view.stride[i];
            continue;
        }

        if (view.shape[i] == 1) {
            continue;
        }

        if (view.shape[i] * view.stride[i] == res.stride[res.ndim]) {
            res.shape[res.ndim] *= view.shape[i];
            res.stride[res.ndim] = view.stride[i];
        } else {
            res.shape[++res.ndim] = view.shape[i];
            res.stride[res.ndim] = view.stride[i];
        }
    }

    if (res.ndim == 0 || res.shape[res.ndim] > 1) {
        ++res.ndim;
    }

    if (res.ndim != (int64_t) shape.size()) {
        std::stringstream ss;
        ss << "Can not simplify to given shape: ";
        ss << "shape: " << shape << " view: " << view;
        throw std::invalid_argument(ss.str());
    }

    return res;
}

int64_t bh_nelements(int64_t ndim, const int64_t shape[]) {
    assert (ndim > 0);
    int64_t res = 1;
    for (int i = 0; i < ndim; ++i) {
        res *= shape[i];
    }

    return res;
}

int64_t bh_nelements(const bh_view &view) {
    return bh_nelements(view.ndim, view.shape);
}

int64_t bh_set_contiguous_stride(bh_view *view) {
    int64_t s = 1;
    for (int64_t i = view->ndim - 1; i >= 0; --i) {
        view->stride[i] = s;
        s *= view->shape[i];
    }

    return s;
}

void bh_assign_complete_base(bh_view *view, bh_base *base) {
    view->base = base;
    view->ndim = 1;
    view->start = 0;
    view->shape[0] = view->base->nelem;
    view->stride[0] = 1;
}

bool bh_is_scalar(const bh_view *view) {
    return bh_nelements(*view) == 1;
}

bool bh_is_constant(const bh_view *o) {
    return (o->base == NULL);
}

void bh_flag_constant(bh_view *o) {
    o->base = NULL;
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

bool bh_is_contiguous(const bh_view *a) {
    if (bh_is_constant(a)) {
        return false;
    }

    int64_t weight = 1;
    for (int64_t dim = a->ndim - 1; dim >= 0; --dim) {
        if (a->shape[dim] > 1 && a->stride[dim] != weight) {
            return false;
        }

        weight *= a->shape[dim];
    }

    return true;
}

bool has_slides(const bh_view a) {
    return (not a.slides.dims.empty());
}


bool bh_view_disjoint(const bh_view *a, const bh_view *b) {
    // TODO: In order to fixed BUG like <https://github.com/bh107/bohrium/issues/178>, we say that sharing
    //       the same base makes the views overlapping for now.
    return bh_base_array(a) != bh_base_array(b);
}
