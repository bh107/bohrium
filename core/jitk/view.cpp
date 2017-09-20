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

#include <sstream>

#include <jitk/base_db.hpp>


using namespace std;

namespace bohrium {
namespace jitk {

void write_array_index(const Scope &scope, const bh_view &view, stringstream &out,
                       int hidden_axis, const pair<int, int> axis_offset) {

    out << view.start + 1;
    if (not bh_is_scalar(&view)) { // NB: this optimization is required when reducing a vector to a scalar!
        for (int i = 0; i < view.ndim; ++i) {
            int t = i;
            if (i >= hidden_axis)
                ++t;
            if (view.stride[i] != 0) {
                if (axis_offset.first == t) {
                    out << " +(i" << t << "+(" << axis_offset.second << ")) ";
                } else {
                    out << " +i" << t;
                }
                if (view.stride[i] != 1) {
                    out << "*" << view.stride[i];
                }
            }
        }
    }
}

void write_array_index_variables(const Scope &scope, const bh_view &view, stringstream &out,
                                 int hidden_axis, const pair<int, int> axis_offset) {

    // Write view.start using the offset-and-strides variable
    out << "vo" << scope.symbols.offsetStridesID(view);

    if (not bh_is_scalar(&view)) { // NB: this optimization is required when reducing a vector to a scalar!
        for (int i = 0; i < view.ndim; ++i) {
            int t = i;
            if (i >= hidden_axis) {
                ++t;
            }
            if (axis_offset.first == t) {
                out << " +(i" << t << "+(" << axis_offset.second << ")) ";
            } else {
                out << " +i" << t;
            }
            out << "*vs" << scope.symbols.offsetStridesID(view) << "_" << i;
        }
    }
}

void write_array_subscription(const Scope &scope, const bh_view &view, stringstream &out, bool ignore_declared_indexes,
                              int hidden_axis, const pair<int, int> axis_offset) {
    assert(view.base != NULL); // Not a constant

    // Let's check if the index is already declared as a variable
    if (not ignore_declared_indexes) {
        if (scope.isIdxDeclared(view)) {
            out << "(";
            scope.getIdxName(view, out);
            out << ")";
            return;
        }
    }
    out << "(";
    if (scope.strides_as_variables and scope.isArray(view) and scope.symbols.existOffsetStridesID(view)) {
        write_array_index_variables(scope, view, out, hidden_axis, axis_offset);
    } else {
        write_array_index(scope, view, out, hidden_axis, axis_offset);
    }
    out << ")";
}

} // jitk
} // bohrium

