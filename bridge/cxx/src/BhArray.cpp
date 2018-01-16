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
    Runtime::instance().enqueueDeletion(std::unique_ptr<BhBase>(ptr));
}

//
// Properties and data access
//

template <typename T>
bool BhArray<T>::isContiguous() const {
    assert(shape.size() == stride.size());

    auto itshape  = shape.rbegin();
    auto itstride = stride.rbegin();

    int64_t acc = 1;
    for (; itstride != stride.rend(); ++itstride, ++itshape) {
        if (*itstride > 1 && acc != *itstride) return false;
        acc *= static_cast<int64_t>(*itshape);
    }

    assert(acc == static_cast<int64_t>(numberOfElements()));
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

    if (shape.size() == 2) {
        // If the shape is size 2, lets print a new 2d array.
        // The shape is (row, col)
        os << "[";
        for (size_t row = 0; row < shape[0]; ++row) {
            // The first row doesn't get a space
            if (row != 0) { os << " "; }

            os << "[";

            for (size_t col = 0; col < shape[1]; ++col) {
                // The first column doesn't get a comma
                if (col > 0) { os << ", "; }
                os << data[row * shape[1] + col];
            }

            os << "]";

            // The last row doesn't get a comma
            if (row != shape[0]-1) { os << ",\n"; }
        }

        os << "]" << endl;
    } else {
        os << "[";
        for (size_t i = 0; i < numberOfElements(); ++i) {
            if (i > 0) {
                os << ", ";
            }
            os << data[i];
        }
        os << "]" << endl;
    }
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
