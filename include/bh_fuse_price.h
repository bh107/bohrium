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

#ifndef __BH_IR_FUSE_PRICE_H
#define __BH_IR_FUSE_PRICE_H

#include <bh.h>
#include <string>

namespace bohrium {

/* Returns the cost of the kernel 'kernel'
 *
 * @kernel The kernel in question
 * @return The kernel price
 */
uint64_t kernel_cost(const bh_ir_kernel &kernel);

/* Returns the cost of the kernel 'kernel' using the
 * Unique-Views cost model.
 *
 * @kernel The kernel in question
 * @return The kernel price
 */
uint64_t kernel_cost_unique_views(const bh_ir_kernel &kernel);

/* Returns the cost saving of fusing the two kernel 'k1' and 'k2' (in that order)
 *
 * @k1     The first kernel
 * @k2     The second kernel
 * @return The cost savings
 */
uint64_t cost_savings(const bh_ir_kernel &k1, const bh_ir_kernel &k2);

/* The possible fuse models */
enum FusePriceModel
{
/* The price of a kernel is the sum of unique views read and written */
    UNIQUE_VIEWS,
    TEMP_ELEMINATION,
    AMOS,
    MAX_SHARE,

/* The number of price models in this enum */
    NUM_OF_PRICE_MODELS
};

/* Writes the name of the 'fuse_model' to the 'output' string
 *
 * @fuse_model  The fuse model
 * @output      The output string
 */
void fuse_price_model_text(FusePriceModel fuse_price_model, std::string &output);

/* Get the selected price model by reading the environment
 * variable 'BH_PRICE_MODEL' */
FusePriceModel fuse_get_selected_price_model();

} //namespace bohrium

#endif

