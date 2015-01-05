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

#include <string>
#include <stdexcept>
#include <boost/algorithm/string/predicate.hpp> //For iequals()
#include <bh.h>
#include "bh_fuse.h"

using namespace std;

namespace bohrium {

/* The possible fuse models */
enum fuse_model
{
/* The broadest possible model. I.e. a SIMD machine can
 * theoretically execute the two instructions in a single operation,
 * thus accepts broadcast, reduction, extension methods, etc. */
    BROADEST,

/* A very simple mode that only fuses same shaped arrays thus no
 * broadcast, reduction, extension methods, etc. */
    SAME_SHAPE,

/* The sentinel */
    NONE
};

/* The default fuse model */
static const fuse_model default_fuse_model = BROADEST;

/* The current selected fuse model */
static fuse_model selected_fuse_model = NONE;


/************************************************************************/
/*************** Specific fuse model implementations ********************/
/************************************************************************/

static bool fuse_broadest(const bh_instruction *a, const bh_instruction *b)
{
    if(bh_opcode_is_system(a->opcode) || bh_opcode_is_system(b->opcode))
        return true;

    const int a_nop = bh_operands(a->opcode);
    for(int i=0; i<a_nop; ++i)
    {
        if(not bh_view_disjoint(&b->operand[0], &a->operand[i])
           && not bh_view_aligned(&b->operand[0], &a->operand[i]))
            return false;
    }
    const int b_nop = bh_operands(b->opcode);
    for(int i=0; i<b_nop; ++i)
    {
        if(not bh_view_disjoint(&a->operand[0], &b->operand[i])
           && not bh_view_aligned(&a->operand[0], &b->operand[i]))
            return false;
    }
    return true;
}

static bool fuse_same_shape(const bh_instruction *a, const bh_instruction *b)
{
    if(bh_opcode_is_system(a->opcode) || bh_opcode_is_system(b->opcode))
        return true;

    if(!bh_opcode_is_elementwise(a->opcode) || !bh_opcode_is_elementwise(b->opcode))
        return false;

    const int a_nop = bh_operands(a->opcode);
    const int b_nop = bh_operands(b->opcode);
    const bh_intp *shape = a->operand[0].shape;
    const bh_intp ndim = a->operand[0].ndim;
    for(int i=1; i<a_nop; ++i)
    {
        if(bh_is_constant(&a->operand[i]))
            continue;
        if(ndim != a->operand[i].ndim)
            return false;
        for(bh_intp j=0; j<ndim; ++j)
        {
            if(a->operand[i].shape[j] != shape[j])
                return false;
        }
    }
    for(int i=0; i<b_nop; ++i)
    {
        if(bh_is_constant(&b->operand[i]))
            continue;
        if(ndim != b->operand[i].ndim)
            return false;
        for(bh_intp j=0; j<ndim; ++j)
        {
            if(b->operand[i].shape[j] != shape[j])
                return false;
        }
    }
    return fuse_broadest(a, b);
}

/************************************************************************/
/*************** The public interface implementation ********************/
/************************************************************************/

/* Get the selected fuse model by reading the environment
 * variable 'BH_FUSE_MODEL' */
static fuse_model get_selected_fuse_model()
{
    using namespace boost;

    //Check enviroment variable
    const char *env = getenv("BH_FUSE_MODEL");
    if(env != NULL)
    {
        string e(env);
        if(iequals(e, string("broadest")))
        {
            //cout << "[FUSE] info: selected fuse model: 'BROADEST'" << endl;
            return BROADEST;
        }
        else if(iequals(e, string("same_shape")))
        {
            //cout << "[FUSE] info: selected fuse model: 'SAME_SHAPE'" << endl;
            return SAME_SHAPE;
        }
        else
        {
            cerr << "[FUSE] WARNING: unknown fuse model: '" << e;
            cerr << "', using the default model instead" << endl;
        }
    }
    //cout << "[FUSE] info: selected fuse model: 'BROADEST'" << endl;
    return default_fuse_model;
}

/* Determines whether it is legal to fuse two instructions into one
 * kernel using the 'selected_fuse_model'.
 *
 * @a The first instruction
 * @b The second instruction
 * @return The boolean answer
 */
bool check_fusible(const bh_instruction *a, const bh_instruction *b)
{
    switch(selected_fuse_model)
    {
        case NONE:
            selected_fuse_model = get_selected_fuse_model();
            return check_fusible(a, b);
        case BROADEST:
            return fuse_broadest(a,b);
        case SAME_SHAPE:
            return fuse_same_shape(a,b);
        default:
            throw runtime_error("No fuse module is selected!");
    }
}

} //namespace bohrium
