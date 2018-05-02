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
#include <jitk/scope.hpp>
#include <jitk/view.hpp>

using namespace std;

namespace bohrium {
namespace jitk {

void Scope::writeIdxDeclaration(const bh_view &view, const string &type_str, stringstream &out) {
    assert(not isIdxDeclared(view));
    out << "const " << type_str << " ";
    getIdxName(view, out);
    out << " = (";
    write_array_index(*this, view, out);
    out << ");";
    _declared_idx.insert(view);
}

} // jitk
} // bohrium
