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

#include <algorithm>
#include <vector>
#include <type_traits>
#include <boost/filesystem.hpp>

#ifndef __BH_UTIL_H
#define __BH_UTIL_H

namespace util {

// Concatenate two vectors
template <typename T>
std::vector<T> vector_cat(const std::vector<T> &a, const std::vector<T> &b) {
    std::vector<T> ret(a);
    ret.insert(ret.end(), b.begin(), b.end());
    return ret;
}

// Use 'remove_all_const<T>' to remove 'const' from 'T' recursively (see http://stackoverflow.com/a/13479740)
template<typename T>
struct remove_all_const : std::remove_const<T> {
};
template<typename T>
struct remove_all_const<T *> {
    typedef typename remove_all_const<T>::type *type;
};
template<typename T>
struct remove_all_const<T *const> {
    typedef typename remove_all_const<T>::type *type;
};

// Checks if 'element' is in 'container'
// This function ignores 'const', which makes it possible to compare 'bh_base*' with 'const bh_base*' for example.
template <typename container_type, typename element_type>
bool exist_nconst(container_type &container, element_type &element) {
    return container.find(const_cast<typename remove_all_const<element_type>::type>(element)) != container.end();
}

// Checks if 'element' is in 'container'
template <typename container_type, typename element_type>
bool exist(container_type &container, element_type &element) {
    return container.find(element) != container.end();
}

// Checks if 'element' is in 'container' by searching through the container linearly
template <typename container_type, typename element_type>
bool exist_linearly(container_type &container, element_type &element) {
    return std::find(container.begin(), container.end(), element) != container.end();
}

// Remove all files in `dir` but keep some of the newest files.
void remove_old_files(const boost::filesystem::path &dir, int64_t num_of_newest_to_keep);

} // util

#endif
