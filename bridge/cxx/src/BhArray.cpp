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

#include <bhxx/BhArray.hpp>
#include <bhxx/Runtime.hpp>
#include <bhxx/array_operations.hpp>
#include <bhxx/util.hpp>

using namespace std;

namespace bhxx {

// Note: This one line of code cannot move to the hpp file,
// since it requires the inclusion of Runtime.hpp, which in turn
// requires the inclusion of BhArray.hpp
void RuntimeDeleter::operator()(BhBase* ptr) const {
    // Simply hand the deletion over to Bohrium
    // including the ownership of the pointer to be deleted
    // by the means of a unique pointer.
    Runtime::instance().enqueue_deletion(std::unique_ptr<BhBase>(ptr));
}

//
// Properties and data access
//

template <typename T>
bool BhArray<T>::is_contiguous() const {
    assert(shape.size() == stride.size());

    auto itshape  = shape.rbegin();
    auto itstride = stride.rbegin();

    int64_t acc = 1;
    for (; itstride != stride.rend(); ++itstride, ++itshape) {
        if (*itstride > 1 && acc != *itstride) return false;
        acc *= static_cast<int64_t>(*itshape);
    }

    assert(acc == static_cast<int64_t>(n_elem()));
    return offset == 0;
}

//
// Routines
//

template <typename T>
void BhArray<T>::pprint(std::ostream& os) const {
    if (base == nullptr) {
        throw runtime_error("Cannot call pprint on array without base");
    }

    // Let's makes sure that the data we are reading is contiguous
    BhArray<T> contiguous = as_contiguous(*this);
    Runtime::instance().sync(contiguous.base);
    Runtime::instance().flush();

    // Get the data pointer and check for NULL
    const T* data = contiguous.data();
    if (data == nullptr) {
        os << "[<Uninitiated>]" << endl;
        return;
    }

    // Pretty print the content
    os << scientific;
    os << "[";
    for (size_t i = 0; i < static_cast<size_t>(contiguous.base->nelem); ++i) {
        if (i > 0) {
            os << ", ";
        }
        os << data[i];
    }
    os << "]" << endl;
}

// Instantiate all possible types of `BhArray`
#define INSTANTIATE(TYPE) template class BhArray<TYPE>

INSTANTIATE(bool);
INSTANTIATE(int8_t);
INSTANTIATE(int16_t);
INSTANTIATE(int32_t);
INSTANTIATE(int64_t);
INSTANTIATE(uint8_t);
INSTANTIATE(uint16_t);
INSTANTIATE(uint32_t);
INSTANTIATE(uint64_t);
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(std::complex<float>);
INSTANTIATE(std::complex<double>);

#undef INSTANTIATE

}  // namespace bhxx
