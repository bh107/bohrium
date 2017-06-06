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

using namespace std;

namespace bhxx {

template <typename T>
void BhArray<T>::pprint(std::ostream& os) const {

    // Let's makes sure that the data we are reading is contiguous
    BhArray<T> contiguous{shape};
    identity(contiguous, *this);
    sync(contiguous);
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
