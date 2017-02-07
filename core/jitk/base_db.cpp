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

#include <bh_util.hpp>
#include <jitk/base_db.hpp>

using namespace std;

namespace bohrium {
namespace jitk {

void Scope::writeIdxDeclaration(const bh_view &view, const string &type_str, stringstream &out) {
    assert(not isIdxDeclared(view));
    _declared_idx.insert(view);
    out << "const " << type_str << " ";
    getIdxName(view, out);
    out << "=";
    bool empty_subscription = true;
    if (view.start > 0) {
        out << "(" << view.start;
        empty_subscription = false;
    } else {
        out << "(";
    }
    if (not bh_is_scalar(&view)) { // NB: this optimization is required when reducing a vector to a scalar!
        for (int i = 0; i < view.ndim; ++i) {
            int t = i;
            if (view.stride[i] > 0) {
                out << " +i" << t;
                if (view.stride[i] != 1) {
                    out << "*" << view.stride[i];
                }
                empty_subscription = false;
            }
        }
    }
    if (empty_subscription)
        out << "0";
    out << ");";
}

} // jitk
} // bohrium
