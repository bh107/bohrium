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
            if (has_slides(view)) {
                bool first_iter = view.slides.iteration_counter == 0;

                // The relevant dimension in the view is updated by the given stride
                for(const bh_slide_dim &dim: view.slides.dims) {
                    if (dim.step_delay == 1 || (view.slides.iteration_counter % dim.step_delay == dim.step_delay-1)) {
                        if (dim.stride) {
                            int64_t change = dim.offset_change*dim.stride;
                            int64_t max_rel_idx = dim.stride*dim.shape;
                            int64_t rel_idx = view.start % (dim.stride*dim.shape);
                            rel_idx += change;
                            if (rel_idx < 0) {
                                change += max_rel_idx;
                            } else if (rel_idx >= max_rel_idx) {
                                change -= max_rel_idx;
                            }

                            view.slides.changes_since_reset[dim.dim] += change;
                            view.start += change;

                            const auto search = view.slides.resets.find(dim.dim);
                            if (!first_iter && search != view.slides.resets.end() &&
                                (view.slides.iteration_counter / dim.step_delay) % search->second == search->second-1) {
                                int64_t reset = search->second;
                                view.start -= view.slides.changes_since_reset[dim.dim];
                                view.slides.changes_since_reset[dim.dim] = 0;
                                view.shape[dim.dim] -= reset*dim.shape_change;
                            }
                        }
                        view.shape[dim.dim] += dim.shape_change;
                        // We allow the user to make the shape negative, but we set it to zero here to prevent confusion
                        if(view.shape[dim.dim] < 0) {
                            view.shape[dim.dim] = 0;
                        }
                    }
                }
                view.slides.iteration_counter += 1;
            }
        }
    }
}
