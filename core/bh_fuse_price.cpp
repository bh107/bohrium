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

#include <bh_fuse_price.hpp>
#include <bh_ir.hpp>
#include <bh_pprint.h>
#include <bh.hpp>

using namespace std;

namespace bohrium {

/* The default fuse model */
static const FusePriceModel default_price_model = UNIQUE_VIEWS;

/* The current selected fuse model */
static FusePriceModel selected_price_model = NUM_OF_PRICE_MODELS;


/************************************************************************/
/*************** Specific fuse model implementations ********************/
/************************************************************************/

/* Returns the bytes in a bh_view */
inline static uint64_t bytes_in_view(const bh_view &v)
{
    assert (!bh_is_constant(&v));
    return bh_nelements_nbcast(&v) * bh_type_size(v.base->type);
}

/* The cost of a kernel is the sum of unique views read and written */
static uint64_t cost_unique(const bh_ir_kernel &k)
{
    uint64_t sum = 0;
    for(const bh_view &v: k.get_input_set())
        sum += bytes_in_view(v);
    for(const bh_view &v: k.get_output_set())
        sum += bytes_in_view(v);
    return sum;
}
static uint64_t savings_unique(const bh_ir_kernel &k1, const bh_ir_kernel &k2)
{
    bh_ir_kernel tmp = k1;
    for(uint64_t instr_idx: k2.instr_indexes())
    {
        tmp.add_instr(instr_idx);
    }
    uint64_t old_cost = cost_unique(k1) + cost_unique(k2);
    uint64_t new_cost = cost_unique(tmp);
    assert(old_cost >= new_cost);
    return old_cost - new_cost;
}
/*
static uint64_t savings_unique_old(const bh_ir_kernel &a, const bh_ir_kernel &b)
{
    int64_t price_drop = 0;
    //Subtract inputs in 'a' that comes from 'b' or is already an input in 'b'
    for(const bh_view &i: a.get_input_set())
    {
        for(const bh_view &o: b.get_output_set())
        {
            if(bh_view_aligned(&i, &o))
                price_drop += bytes_in_view(i);
        }
        for(const bh_view &o: b.get_input_set())
        {
            if(bh_view_aligned(&i, &o))
                price_drop += bytes_in_view(i);
        }
    }
    //Subtract outputs from 'b' that are discared in 'a'
    for(const bh_view &o: b.get_output_set())
    {
        for(uint64_t a_instr_idx: a.instr_indexes())
        {
            const bh_instruction &a_instr = a.bhir->instr_list[a_instr_idx];
            if(a_instr.opcode == BH_DISCARD and a_instr.operand[0].base == o.base)
            {
                price_drop += bytes_in_view(o);
                break;
            }
        }
    }
    return price_drop;
}
*/

/* The cost of a kernel is 'number of instruction' * 3 - 'number of temp arrays' */
static uint64_t cost_temp_elemination(const bh_ir_kernel &k)
{
    return k.instr_indexes().size() * 3 - k.get_temps().size();
}
static uint64_t savings_temp_elemination(const bh_ir_kernel &k1, const bh_ir_kernel &k2)
{
    bh_ir_kernel tmp = k1;
    for(uint64_t instr_idx: k2.instr_indexes())
    {
        tmp.add_instr(instr_idx);
    }
    uint64_t old_cost = cost_temp_elemination(k1) + cost_temp_elemination(k2);
    uint64_t new_cost = cost_temp_elemination(tmp);
    assert(old_cost >= new_cost);
    return old_cost - new_cost;
}

static uint64_t cost_max_share(const bh_ir_kernel &k)
{
    if(k.instr_indexes().size() == 0)
        return 0;

    uint64_t shared_access = 0;
    for(uint64_t krn_idx: k.instr_indexes())
    {
        const bh_instruction &krn_instr = k.bhir->instr_list[krn_idx];
        if(bh_opcode_is_system(krn_instr.opcode))
            continue;
        //Find which instructions that reads and writes to arrays "visible" to
        //the instruction 'krn_idx'
        multiset<bh_view> read_access, read_access_outside;
        set<bh_view> write_access, write_access_outside;;
        for(uint64_t instr_idx=0; instr_idx < krn_idx; ++instr_idx)
        {
            const bh_instruction &instr = k.bhir->instr_list[instr_idx];
            if(bh_opcode_is_system(instr.opcode))
                continue;

            //Check if the instruction is in this kernel
            bool instr_in_kernel = false;
            if(std::find(k.instr_indexes().begin(),
                         k.instr_indexes().end(), instr_idx) != k.instr_indexes().end())
                instr_in_kernel = true;

            for(int i= bh_noperands(instr.opcode) - 1; i >= 0; --i)
            {
                const bh_view &v = instr.operand[i];
                if(bh_is_constant(&v))
                    continue;
                if(i > 0)
                {
                    read_access.insert(v);
                    if(not instr_in_kernel)
                        read_access_outside.insert(v);
                }
                else
                {
                    write_access.insert(v);
                    read_access.erase(v);
                    if(not instr_in_kernel)
                    {
                        write_access_outside.insert(v);
                    }
                    else
                        write_access_outside.erase(v);
                    read_access_outside.erase(v);
                }
            }
        }
        for(int i=1; i < bh_noperands(krn_instr.opcode); ++i)
        {
            const bh_view &v = krn_instr.operand[i];
            if(bh_is_constant(&v))
                continue;
            if(write_access_outside.find(v) != write_access_outside.end())
                shared_access++;
            shared_access += read_access_outside.count(v);
        }
    }
    return shared_access;
}
static uint64_t savings_max_share(const bh_ir_kernel &k1, const bh_ir_kernel &k2)
{
    bh_ir_kernel tmp = k1;
    for(uint64_t instr_idx: k2.instr_indexes())
    {
        tmp.add_instr(instr_idx);
    }
    uint64_t old_cost = cost_max_share(k1) + cost_max_share(k2);
    uint64_t new_cost = cost_max_share(tmp);
    assert(old_cost >= new_cost);
    return old_cost - new_cost;
}

static uint64_t cost_tmp_share(const bh_ir_kernel &k)
{
    uint64_t N = k.bhir->instr_list.size();
    if(k.instr_indexes().size() == 0)
        return 0;

    uint64_t not_tmp = k.get_parameters().size();
    uint64_t shared_access = cost_max_share(k);

    return shared_access*N*N+not_tmp*N;
}
static uint64_t savings_tmp_share(const bh_ir_kernel &k1, const bh_ir_kernel &k2)
{
    bh_ir_kernel tmp = k1;
    for(uint64_t instr_idx: k2.instr_indexes())
    {
        tmp.add_instr(instr_idx);
    }
    uint64_t old_cost = cost_tmp_share(k1) + cost_tmp_share(k2);
    uint64_t new_cost = cost_tmp_share(tmp);
    assert(old_cost >= new_cost);
    return old_cost - new_cost;
}

static uint64_t cost_amos(const bh_ir_kernel &k)
{
    uint64_t N = k.bhir->instr_list.size();
    if(k.instr_indexes().size() == 0)
        return 0;

    uint64_t loop_overhead = 1;
    uint64_t not_tmp = k.get_parameters().size();
    uint64_t shared_access = cost_max_share(k);

    return loop_overhead+not_tmp*N+shared_access*N*N;
}
static uint64_t savings_amos(const bh_ir_kernel &k1, const bh_ir_kernel &k2)
{
    bh_ir_kernel tmp = k1;
    for(uint64_t instr_idx: k2.instr_indexes())
    {
        tmp.add_instr(instr_idx);
    }
    uint64_t old_cost = cost_amos(k1) + cost_amos(k2);
    uint64_t new_cost = cost_amos(tmp);
    assert(old_cost >= new_cost);
    return old_cost - new_cost;
}

/************************************************************************/
/*************** The public interface implementation ********************/
/************************************************************************/

FusePriceModel fuse_get_selected_price_model()
{
    using namespace boost;

    if(selected_price_model != NUM_OF_PRICE_MODELS)
        return selected_price_model;

    string default_model;
    fuse_price_model_text(default_price_model, default_model);

    //Check enviroment variable
    const char *env = getenv("BH_PRICE_MODEL");
    if(env != NULL)
    {
        string e(env);
        //Iterate through the 'FusePriceModel' enum and find the enum that matches
        //the enviroment variable string 'e'
        for(FusePriceModel m = UNIQUE_VIEWS; m < NUM_OF_PRICE_MODELS; m = FusePriceModel(m + 1))
        {
            string model;
            fuse_price_model_text(m, model);
            if(iequals(e, model))
            {
                return m;
            }
        }
        cerr << "[FUSE] WARNING: unknown price model: '" << e;
        cerr << "', using the default model '" << default_model << "' instead" << endl;
        setenv("BH_PRICE_MODEL", default_model.c_str(), 1);
    }
    return default_price_model;
}

void fuse_price_model_text(FusePriceModel price_model, string &output)
{
    switch(price_model)
    {
    case UNIQUE_VIEWS:
        output = "unique_views";
        break;
    case TEMP_ELEMINATION:
        output = "temp_elemination";
        break;
    case MAX_SHARE:
        output = "max_share";
        break;
    case TEMP_SHARE:
        output = "temp_share";
        break;
    case AMOS:
        output = "amos";
        break;
    default:
        output = "unknown";
    }
}

uint64_t kernel_cost(const bh_ir_kernel &kernel, FusePriceModel model)
{
    FusePriceModel m = selected_price_model;

    if(model != ENV_DECIDE)
        m = model;

    switch(m)
    {
    case NUM_OF_PRICE_MODELS:
        selected_price_model = fuse_get_selected_price_model();
        return kernel_cost(kernel);
    case UNIQUE_VIEWS:
        return cost_unique(kernel);
    case TEMP_ELEMINATION:
        return cost_temp_elemination(kernel);
    case MAX_SHARE:
        return cost_max_share(kernel);
    case AMOS:
        return cost_amos(kernel);
    case TEMP_SHARE:
        return cost_tmp_share(kernel);
    default:
        throw runtime_error("No price module is selected!");
    }
}

uint64_t cost_savings(const bh_ir_kernel &k1, const bh_ir_kernel &k2)
{
    switch(selected_price_model)
    {
    case NUM_OF_PRICE_MODELS:
        selected_price_model = fuse_get_selected_price_model();
        return cost_savings(k1, k2);
    case UNIQUE_VIEWS:
        return savings_unique(k1, k2);
    case TEMP_ELEMINATION:
        return savings_temp_elemination(k1, k2);
    case MAX_SHARE:
        return savings_max_share(k1, k2);
    case AMOS:
        return savings_amos(k1, k2);
    case TEMP_SHARE:
        return savings_tmp_share(k1, k2);
    default:
        throw runtime_error("No price module is selected!");
    }
}

} //namespace bohrium
