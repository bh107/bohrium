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
                for (size_t i = 0; i < view.slides.offset_change.size(); i++) {
                    int64_t dim = view.slides.dim.at(i);
                    int64_t dim_stride = view.slides.dim_stride.at(i);
                    int64_t step_delay = view.slides.step_delay.at(i);
                    int64_t offset_change = view.slides.offset_change.at(i);
                    int64_t dim_shape = view.slides.dim_shape.at(i);
                    int64_t shape_change = view.slides.shape_change.at(i);
                    if (step_delay == 1 ||
                        (view.slides.iteration_counter % step_delay == step_delay-1)) {
                        if (dim_stride) {
                            int64_t change = offset_change*dim_stride;
                            int64_t max_rel_idx = dim_stride*dim_shape;
                            int64_t rel_idx = view.start % (dim_stride*dim_shape);
                            rel_idx += change;
                            if (rel_idx < 0) {
                                change += max_rel_idx;
                            } else if (rel_idx >= max_rel_idx) {
                                change -= max_rel_idx;
                            }

                            view.slides.changes_since_reset[dim] += change;
                            view.start += change;

                            auto search = view.slides.resets.find(dim);
                            if (!first_iter && search != view.slides.resets.end() &&
                                (view.slides.iteration_counter / step_delay) % search->second == search->second-1) {
                                int64_t reset = search->second;
                                view.start -= view.slides.changes_since_reset[dim];
                                view.slides.changes_since_reset[dim] = 0;
                                view.shape[dim] -= reset*shape_change;
                            }
                        }
                        view.shape[dim] += shape_change;
                    }
                }
                view.slides.iteration_counter += 1;
            }
        }
    }
}
