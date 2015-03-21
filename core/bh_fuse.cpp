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

/* The default fuse model */
static const FuseModel default_fuse_model = BROADEST;

/* The current selected fuse model */
static FuseModel selected_fuse_model = NUM_OF_MODELS;


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

static bool fuse_same_shape_range(const bh_instruction *a, const bh_instruction *b)
{
    if(bh_opcode_is_system(a->opcode) || bh_opcode_is_system(b->opcode))
        return true;

    if((a->opcode != BH_RANGE and not bh_opcode_is_elementwise(a->opcode)) or
       (b->opcode != BH_RANGE and not bh_opcode_is_elementwise(b->opcode)))
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

static bool fuse_same_shape_random(const bh_instruction *a, const bh_instruction *b)
{
    if(bh_opcode_is_system(a->opcode) || bh_opcode_is_system(b->opcode))
        return true;

    if((a->opcode != BH_RANDOM and not bh_opcode_is_elementwise(a->opcode)) or
       (b->opcode != BH_RANDOM and not bh_opcode_is_elementwise(b->opcode)))
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

static bool fuse_same_shape_range_random(const bh_instruction *a, const bh_instruction *b)
{
    if(bh_opcode_is_system(a->opcode) || bh_opcode_is_system(b->opcode))
        return true;

    if((a->opcode != BH_RANGE and a->opcode != BH_RANDOM and not bh_opcode_is_elementwise(a->opcode)) or
       (b->opcode != BH_RANGE and b->opcode != BH_RANDOM and not bh_opcode_is_elementwise(b->opcode)))
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

static bool is_scalar(const bh_view* view)
{
    return ((view->ndim == 1) and (view->shape[0]==1));
}

static bool fuse_same_shape_generate_1dreduce(const bh_instruction *a, const bh_instruction *b)
{
    if(bh_opcode_is_system(a->opcode) || bh_opcode_is_system(b->opcode))
        return true;

    if((a->opcode != BH_RANGE and a->opcode != BH_RANDOM \
        and not bh_opcode_is_elementwise(a->opcode)      \
        and not bh_opcode_is_reduction(a->opcode))
        or                                               \
       (b->opcode != BH_RANGE and b->opcode != BH_RANDOM \
        and not bh_opcode_is_elementwise(b->opcode)      \
        and not bh_opcode_is_reduction(b->opcode))) {
        return false;
    }

    if ((bh_opcode_is_reduction(a->opcode) and (a->operand[1].ndim>1))\
        or                                                          \
        (bh_opcode_is_reduction(b->opcode) and (b->operand[1].ndim>1))) {
        return false;
    }

    //  Check that the output of instruction "a" has the shape
    //  shape as all other operands.
    const int a_nop = bh_operands(a->opcode);
    const int b_nop = bh_operands(b->opcode);
    // a is reduction, b is reduction
    if (bh_opcode_is_reduction(a->opcode) and bh_opcode_is_reduction(b->opcode)) {
        return false;
    // a is NOT reduction, b is reduction
    } else if (not bh_opcode_is_reduction(a->opcode) and bh_opcode_is_reduction(b->opcode)) {
        const bh_intp *red_shape = b->operand[1].shape;
        const bh_intp red_ndim   = b->operand[1].ndim;

        // check that a does not depend on reduce-result of b
        for(int oidx=0; oidx<a_nop; ++oidx) {
            if(bh_is_constant(&a->operand[oidx])) {
                continue;
            }
            if (a->operand[oidx].base == b->operand[0].base) {
                return false;
            }
        }

        for(int oidx=0; oidx<a_nop; ++oidx) {
            if(bh_is_constant(&a->operand[oidx])) {
                continue;
            }
            if(red_ndim != a->operand[oidx].ndim) {
                return false;
            }
            for(bh_intp dim=0; dim<red_ndim; ++dim) {
                if(a->operand[oidx].shape[dim] != red_shape[dim]) {
                    return false;
                }
            }
        }
    // a is reduction, b is NOT reduction
    } else if (bh_opcode_is_reduction(a->opcode) and not bh_opcode_is_reduction(b->opcode)) {
        const bh_intp *red_shape = a->operand[1].shape;
        const bh_intp red_ndim   = a->operand[1].ndim;

        // check that b does not depend on reduce-result of a
        for(int oidx=0; oidx<b_nop; ++oidx) {
            if(bh_is_constant(&a->operand[oidx])) {
                continue;
            }
            if (b->operand[oidx].base == a->operand[0].base) {
                return false;
            }
        }

        for(int oidx=0; oidx<b_nop; ++oidx) {
            if(bh_is_constant(&b->operand[oidx])) {
                continue;
            }
            if(red_ndim != b->operand[oidx].ndim) {
                return false;
            }
            for(bh_intp dim=0; dim<red_ndim; ++dim) {
                if(b->operand[oidx].shape[dim] != red_shape[dim]) {
                    return false;
                }
            }
        }
    // everything else...
    } else {
        const bh_intp *shape = a->operand[0].shape;
        const bh_intp ndim = a->operand[0].ndim;

        if (not is_scalar(&a->operand[0])) {
            for(int i=0; i<b_nop; ++i) {
                if (bh_is_constant(&b->operand[i]))
                    continue;
                if (ndim != b->operand[i].ndim) {
                    return false;
                }
                for (bh_intp j=0; j<ndim; ++j) {
                    if(b->operand[i].shape[j] != shape[j]) {
                        return false;
                    }
                }
            }
        }
    }
    
    return fuse_broadest(a, b);
}

/************************************************************************/
/*************** The public interface implementation ********************/
/************************************************************************/

/* Get the selected fuse model by reading the environment
 * variable 'BH_FUSE_MODEL' */
FuseModel fuse_get_selected_model()
{
    using namespace boost;

    if(selected_fuse_model != NUM_OF_MODELS)
        return selected_fuse_model;

    string default_model;
    fuse_model_text(default_fuse_model, default_model);

    //Check enviroment variable
    const char *env = getenv("BH_FUSE_MODEL");
    if(env != NULL)
    {
        string e(env);
        //Iterate through the 'FuseModel' enum and find the enum that matches
        //the enviroment variable string 'e'
        for(FuseModel m = BROADEST; m < NUM_OF_MODELS; m = FuseModel(m + 1))
        {
            string model;
            fuse_model_text(m, model);
            if(iequals(e, model))
            {
//                cout << "[FUSE] info: selected fuse model: '" << model << "'" << endl;
                return m;
            }
        }
        cerr << "[FUSE] WARNING: unknown fuse model: '" << e;
        cerr << "', using the default model '" << default_model << "' instead" << endl;
        setenv("BH_FUSE_MODEL", default_model.c_str(), 1);
    }
//    cout << "[FUSE] info: selected fuse model: '" << default_model << "'" << endl;
    return default_fuse_model;
}

/* Writes the name of the 'fuse_model' to the 'output' string
 *
 * @fuse_model  The fuse model
 * @output      The output string
 */
void fuse_model_text(FuseModel fuse_model, string &output)
{
    switch(fuse_model)
    {
        case BROADEST:
            output = "broadest";
            break;
        case SAME_SHAPE:
            output = "same_shape";
            break;
        case SAME_SHAPE_RANGE:
            output = "same_shape_range";
            break;
        case SAME_SHAPE_RANDOM:
            output = "same_shape_random";
            break;
        case SAME_SHAPE_RANGE_RANDOM:
            output = "same_shape_range_random";
            break;
        case SAME_SHAPE_GENERATE_1DREDUCE:
            output = "same_shape_generate_1dreduce";
            break;
        default:
            output = "unknown";
    }
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
        case NUM_OF_MODELS:
            selected_fuse_model = fuse_get_selected_model();
            return check_fusible(a, b);
        case BROADEST:
            return fuse_broadest(a,b);
        case SAME_SHAPE:
            return fuse_same_shape(a,b);
        case SAME_SHAPE_RANGE:
            return fuse_same_shape_range(a,b);
        case SAME_SHAPE_RANDOM:
            return fuse_same_shape_random(a,b);
        case SAME_SHAPE_RANGE_RANDOM:
            return fuse_same_shape_range_random(a,b);
        case SAME_SHAPE_GENERATE_1DREDUCE:
            return fuse_same_shape_generate_1dreduce(a,b);
        default:
            throw runtime_error("No fuse module is selected!");
    }
}

} //namespace bohrium
