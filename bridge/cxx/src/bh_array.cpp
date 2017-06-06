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

#include <bhxx/array_operations.hpp>
#include <bhxx/bh_array.hpp>
#include <bhxx/runtime.hpp>

using namespace std;

namespace bhxx {

template <typename T>
inline BhBase<T>::~BhBase() {
    //    cout << "Delete base " << this << endl;
    Runtime::instance().enqueue_free(*this);
}

template <typename T>
void BhArray<T>::pprint(std::ostream& os) const {

    // Let's makes sure that the data we are reading is contiguous
    BhArray<T> contiguous = BhArray<T>(shape);
    identity(contiguous, *this);
    sync(contiguous);
    Runtime::instance().flush();

    // Get the data pointer and check for NULL
    const T* data = static_cast<T*>(contiguous.base->base->data);
    if (data == nullptr) {
        os << "[<Uninitiated>]" << endl;
        return;
    }

    // Pretty print the content
    os << scientific;
    os << "[";
    for (size_t i = 0; i < static_cast<size_t>(contiguous.base->base->nelem); ++i) {
        if (i > 0) {
            os << ", ";
        }
        os << data[i];
    }
    os << "]" << endl;
}

// Instantiate all possible types of `BhArray` and `BhBase`, which makes it possible to
// implement many of
// their methods here
template class BhArray<bool>;
template class BhArray<int8_t>;
template class BhArray<int16_t>;
template class BhArray<int32_t>;
template class BhArray<int64_t>;
template class BhArray<uint8_t>;
template class BhArray<uint16_t>;
template class BhArray<uint32_t>;
template class BhArray<uint64_t>;
template class BhArray<float>;
template class BhArray<double>;
template class BhArray<std::complex<float>>;
template class BhArray<std::complex<double>>;
template class BhBase<bool>;
template class BhBase<int8_t>;
template class BhBase<int16_t>;
template class BhBase<int32_t>;
template class BhBase<int64_t>;
template class BhBase<uint8_t>;
template class BhBase<uint16_t>;
template class BhBase<uint32_t>;
template class BhBase<uint64_t>;
template class BhBase<float>;
template class BhBase<double>;
template class BhBase<std::complex<float>>;
template class BhBase<std::complex<double>>;

}  // namespace bhxx
