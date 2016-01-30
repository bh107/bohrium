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

#include <bh.h>
#include <map>
#include <string>
#include <algorithm>
#include <tuple>
#include <iostream>
#include <sstream>

using namespace std;

/** Create a new base array.
 *
 * @param type The type of data in the array
 * @param nelements The number of elements
 * @param new_base The handler for the newly created base
 * @return Error code (BH_SUCCESS, BH_OUT_OF_MEMORY)
 */
bh_error bh_create_base(bh_type    type,
                        bh_index   nelements,
                        bh_base**  new_base)
{

    bh_base *base = (bh_base *) malloc(sizeof(bh_base));
    if(base == NULL)
        return BH_OUT_OF_MEMORY;
    base->type = type;
    base->nelem = nelements;
    base->data = NULL;
    *new_base = base;

    return BH_SUCCESS;
}


/** Destroy the base array.
 *
 * @param base  The base array in question
 */
void bh_destroy_base(bh_base**  base)
{
    bh_base *b = *base;
    free(b);
    b = NULL;
}

// Returns the label of this base array
// NB: generated a new label if necessary
static map<const bh_base*, unsigned int>label_map;
unsigned int bh_base::get_label() const
{
   if(label_map.find(this) == label_map.end())
       label_map[this] = label_map.size();
   return label_map[this];
}

ostream& operator<<(ostream& out, const bh_base& b)
{
    unsigned int label = b.get_label();
    out << "a" << label << "{dtype: " << bh_type_text(b.type) << ", nelem: " \
        << b.nelem << ", address: " << &b << "}";
    return out;
}

string bh_base::str() const
{
    stringstream ss;
    ss << *this;
    return ss.str();
}

vector<tuple<int64_t, int64_t, int64_t> > bh_view::python_notation() const
{
    //stride&shape&index for each dimension (in that order)
    vector<tuple<int64_t, int64_t, int64_t> > sns;
    for(int64_t i=0; i<this->ndim; ++i)
    {
        sns.push_back(make_tuple(this->stride[i], this->shape[i], i));
    }
    //Let's sort them such that the greatest strides comes first
    sort(sns.begin(), sns.end(), greater<tuple<int64_t, int64_t, int64_t> >());

    //Now let's compute start&end&stride of each dimension, which
    //makes up the python notation e.g. [2:4:1]
    vector<tuple<int64_t, int64_t, int64_t> > sne(sns.size());
    int64_t offset = this->start;
    for(size_t i=0; i<sns.size(); ++i)
    {
        int64_t stride = std::get<0>(sns[i]);
        int64_t shape  = std::get<1>(sns[i]);
        int64_t index  = std::get<2>(sns[i]);

        int64_t start = 0;
        if (stride > 0)//avoid division by zero
            start = offset / stride;
        int64_t end = start + shape;
        offset -= start * stride;
        sne[index] = make_tuple(start, end, stride);
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

ostream& operator<<(ostream& out, const bh_view& v)
{
    unsigned int label = v.base->get_label();
    out << "a" << label << "[";

    const vector<tuple<int64_t, int64_t, int64_t> > sne = v.python_notation();
    for(size_t i=0; i<sne.size(); ++i)
    {
        int64_t start  = std::get<0>(sne[i]);
        int64_t end    = std::get<1>(sne[i]);
        int64_t stride = std::get<2>(sne[i]);
        out << start << ":" << end << ":" << stride;
        if(i < sne.size()-1)//Not the last iteration
            out << ",";
    }
    out << "]";
    return out;
}
