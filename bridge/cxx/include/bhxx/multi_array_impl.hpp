//
// Copyright (C) 2017 by the linalgwrap authors
//
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

#ifndef __BHXX_MULTI_ARRAY_IMPL_H
#define __BHXX_MULTI_ARRAY_IMPL_H

#include <bhxx/array_operations.hpp>
#include "multi_array.hpp"
#include "runtime.hpp"


namespace bhxx {

template<typename T>
void BhArray<T>::pprint(std::ostream& os) const {
    using namespace std;

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
    for(size_t i=0; i < static_cast<size_t>(contiguous.base->base->nelem); ++i) {
        if (i > 0) {
            os << ", ";
        }
        os << data[i];
    }
    os << "]" << endl;
}

template <typename T>
std::ostream& operator<< (std::ostream& os, const BhArray<T>& ary) {
    ary.pprint(os);
    return os;
}

} // namespace bhxx

#endif