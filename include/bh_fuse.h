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

#ifndef __BH_IR_FUSE_H
#define __BH_IR_FUSE_H

#include <bh.h>
#include <string>

namespace bohrium {

/* Determines whether it is legal to fuse two instructions into one
 * kernel using the current selected fuse model, which it set through
 * the BH_FUSE_MODEL environment variable (see bh_fuse.cpp).
 *
 * @a The first instruction
 * @b The second instruction
 * @return The boolean answer
 */
bool check_fusible(const bh_instruction *a, const bh_instruction *b);

/* The possible fuse models */
enum FuseModel
{
/* The broadest possible model. I.e. a SIMD machine can
 * theoretically execute the two instructions in a single operation,
 * thus accepts broadcast, reduction, extension methods, etc. */
    BROADEST,

/* A very simple mode that only fuses same shaped arrays thus no
 * broadcast, reduction, extension methods, etc. */
    SAME_SHAPE,

/* Like same shape but includes range */
    SAME_SHAPE_RANGE,

/* Like same shape but includes random */
    SAME_SHAPE_RANDOM,

/* Like same shape but includes random */
    SAME_SHAPE_RANGE_RANDOM,

/* The number of models in this enum */
    NUM_OF_MODELS
};

/* Writes the name of the 'fuse_model' to the 'output' string
 *
 * @fuse_model  The fuse model
 * @output      The output string
 */
void fuse_model_text(FuseModel fuse_model, std::string &output);


/* Get the selected fuse model by reading the environment
 * variable 'BH_FUSE_MODEL' */
FuseModel fuse_get_selected_model();

} //namespace bohrium

#endif

