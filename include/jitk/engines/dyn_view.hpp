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

void slide_views(BhIR *bhir) {
    // Iterate through all instructions and slide the relevant views
    for (bh_instruction &instr : bhir->instr_list) {
        for (bh_view &view : instr.operand) {
            if (not view.slide_strides.empty()) {
                // The relevant dimension in the view is updated by the given stride
                for (size_t i = 0; i < view.slide_strides.size(); i++) {
                    size_t off_dim    = view.slide_dimensions.at(i);
                    int off_stride = (int) view.slide_strides.at(i);
                    view.start += view.stride[off_dim] * off_stride;
                }
            }
        }
    }
}
