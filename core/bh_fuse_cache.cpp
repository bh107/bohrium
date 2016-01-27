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
#include <cstring>
#include <bh.h>
#include "bh_fuse.h"
#include "bh_fuse_cache.h"
#include <fstream>
#include <exception>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/version.hpp>
#include <boost/algorithm/string/predicate.hpp> //For iequals()

using namespace std;
using namespace boost;
using namespace boost::filesystem;

namespace bohrium {

/* * OBS * OBS * OBS * OBS * OBS * OBS * OBS * OBS * OBS * OBS * OBS * OBS
 * When designing an instruction hash function REMEMBER:
 * The hash string should either be of fixed length and all feilds
 * contained also be of fixed legth OR unique seperators should be
 * used for each variable length field and to seperate instruction
 * hashed. The function hashOpcodeIdShapeSweepdim may be used as
 * inspiration.
 */

static const size_t inst_sep = SIZE_MAX;
static const size_t op_sep   = SIZE_MAX-1;

static void hashOpcodeOpidShapeidSweepdim(std::ostream& os, const bh_instruction& instr,
                                        BatchHash& batchHash)
{
    /* The Instruction hash consists of the following fields:
     * <opcode> (<operant-id> <ndim> <shape> <op_sep>)[1] <sweep-dim>[2] <inst_sep>
     * 1: for each operand
     * 2: if the operation is a sweep operation
     */
    int noperands = bh_operands(instr.opcode);
    os.write((const char*)&instr.opcode, sizeof(instr.opcode));         // <opcode>
    for(int oidx=0; oidx<noperands; ++oidx) {
        const bh_view& view = instr.operand[oidx];
        if (bh_is_constant(&view))
            continue;  // Ignore constants
        std::pair<size_t,bool> vid = batchHash.views.insert(view);
        size_t id = vid.first;
        os.write((char*)&id, sizeof(id));                               // <operant-id>
        os.write((char*)&view.ndim, sizeof(view.ndim));                 // <ndim>
        std::pair<size_t,bool> sidp = batchHash.shapes.insert(std::vector<bh_index>(view.shape,view.shape+view.ndim));
        os.write((char*)&sidp.first, sizeof(sidp.first));                   // <shape-id>
        os.write((char*)&op_sep, sizeof(op_sep));                       // <op_sep>
    }
    if (bh_opcode_is_sweep(instr.opcode))
        os.write((char*)&instr.constant.value.int64, sizeof(bh_int64)); // <sweep-dim>
    os.write((char*)&inst_sep, sizeof(inst_sep));                       // <inst_sep>
}

static void hashOpidSweepdim(std::ostream& os, const bh_instruction& instr, BatchHash& batchHash)
{
    /* The Instruction hash consists of the following fields:
     * (<operant-id>)[1] <op_sep> (<ndim> <sweep-dim>)[2] <seperator>
     * 1: for each operand
     * 2: if the operation is a sweep operation
     */
    int noperands = bh_operands(instr.opcode);
    for(int oidx=0; oidx<noperands; ++oidx) {
        const bh_view& view = instr.operand[oidx];
        if (bh_is_constant(&view))
            continue;  // Ignore constants
        std::pair<size_t,bool> vid = batchHash.views.insert(view);
        size_t id = vid.first;
        os.write((char*)&id, sizeof(id));                               // <operant-id>
    }
    os.write((char*)&op_sep, sizeof(op_sep));                           // <op_sep>
    if (bh_opcode_is_sweep(instr.opcode))
    {
        const bh_view& view = instr.operand[1];
        os.write((char*)&view.ndim, sizeof(view.ndim));                 // <ndim>
        os.write((char*)&instr.constant.value.int64, sizeof(bh_int64)); // <sweep-dim>
    }
    os.write((char*)&inst_sep, sizeof(inst_sep));                       // <inst_sep>
}

static void hashScalarOpidSweepdim(std::ostream& os, const bh_instruction& instr, BatchHash& batchHash)
{
    /* The Instruction hash consists of the following fields:
     * <is_scalar> (<operant-id>)[1] <op_sep> (<ndim> <sweep-dim>)[2] <seperator>
     * 1: for each operand
     * 2: if the operation is a sweep operation
     */
    int noperands = bh_operands(instr.opcode);
    if(noperands == 0)
    {
        os.write((char*)&inst_sep, sizeof(inst_sep));                       // <inst_sep>
        return;
    }
    bool scalar = (bh_is_scalar(&(instr.operand[0])) ||
                   (bh_opcode_is_accumulate(instr.opcode) && instr.operand[0].ndim == 1));
    os.write((char*)&scalar, sizeof(scalar));                           // <is_scalar>
    for(int oidx=0; oidx<noperands; ++oidx) {
        const bh_view& view = instr.operand[oidx];
        if (bh_is_constant(&view))
            continue;  // Ignore constants
        std::pair<size_t,bool> vid = batchHash.views.insert(view);
        size_t id = vid.first;
        os.write((char*)&id, sizeof(id));                               // <operant-id>
    }
    os.write((char*)&op_sep, sizeof(op_sep));                           // <op_sep>
    if (bh_opcode_is_sweep(instr.opcode))
    {
        const bh_view& view = instr.operand[1];
        os.write((char*)&view.ndim, sizeof(view.ndim));                 // <ndim>
        os.write((char*)&instr.constant.value.int64, sizeof(bh_int64)); // <sweep-dim>
    }
    os.write((char*)&inst_sep, sizeof(inst_sep));                       // <inst_sep>
}

static void hashScalarShapeidOpidSweepdim(std::ostream& os, const bh_instruction& instr, BatchHash& batchHash)
{
    /* The Instruction hash consists of the following fields:
     * <is_scalar> <shape-id> (<operant-id>)[1] <op_sep> (<ndim> <sweep-dim>)[2] <seperator>
     * 1: for each operand
     * 2: if the operation is a sweep operation
     * NB: but ignores instructions that takes no arguments, such as BH_NONE
     */
    int noperands = bh_operands(instr.opcode);
    if(noperands == 0)
    {
        os.write((char*)&inst_sep, sizeof(inst_sep));                       // <inst_sep>
        return;
    }
    bool scalar = (bh_is_scalar(&(instr.operand[0])) ||
                   (bh_opcode_is_accumulate(instr.opcode) && instr.operand[0].ndim == 1));
    os.write((char*)&scalar, sizeof(scalar));                           // <is_scalar>
    const bh_view& view = (bh_opcode_is_sweep(instr.opcode) ? instr.operand[1] : instr.operand[0]);
    std::pair<size_t,bool> sidp = batchHash.shapes.insert(std::vector<bh_index>(view.shape,view.shape+view.ndim));
    os.write((char*)&sidp.first, sizeof(sidp.first));                   // <shape-id>
    for(int oidx=0; oidx<noperands; ++oidx) {
        const bh_view& view = instr.operand[oidx];
        if (bh_is_constant(&view))
            continue;  // Ignore constants
        std::pair<size_t,bool> vid = batchHash.views.insert(view);
        size_t id = vid.first;
        os.write((char*)&id, sizeof(id));                               // <operant-id>
    }
    os.write((char*)&op_sep, sizeof(op_sep));                           // <op_sep>
    if (bh_opcode_is_sweep(instr.opcode))
    {
        const bh_view& view = instr.operand[1];
        os.write((char*)&view.ndim, sizeof(view.ndim));                 // <ndim>
        os.write((char*)&instr.constant.value.int64, sizeof(bh_int64)); // <sweep-dim>
    }
    os.write((char*)&inst_sep, sizeof(inst_sep));                       // <inst_sep>
}

static void hashOpid(std::ostream& os, const bh_instruction& instr, BatchHash& batchHash)
{
    /* The Instruction hash consists of the following fields:
     * (<operant-id>)[1]  <seperator>
     * 1: for each operand
     */
    int noperands = bh_operands(instr.opcode);
    for(int oidx=0; oidx<noperands; ++oidx) {
        const bh_view& view = instr.operand[oidx];
        if (bh_is_constant(&view))
            continue;  // Ignore constants
        std::pair<size_t,bool> vid = batchHash.views.insert(view);
        size_t id = vid.first;
        os.write((char*)&id, sizeof(id));                               // <operant-id>
    }
    os.write((char*)&inst_sep, sizeof(inst_sep));                       // <inst_sep>
}

#define __scalar(i) (bh_is_scalar(&(i)->operand[0]) || \
                     (bh_opcode_is_accumulate((i)->opcode) && (i)->operand[0].ndim == 1))

typedef void (*InstrHash)(std::ostream& os, const bh_instruction &instr, BatchHash& batchHash);

static InstrHash getInstrHash(FuseModel fuseModel)
{
    switch(fuseModel)
    {
    case BROADEST:
        return &hashOpid;
    case NO_XSWEEP:
        return &hashOpidSweepdim;
    case NO_XSWEEP_SCALAR_SEPERATE:
        return &hashScalarOpidSweepdim;
    case NO_XSWEEP_SCALAR_SEPERATE_SHAPE_MATCH:
        return &hashScalarShapeidOpidSweepdim;
    case SAME_SHAPE:
    case SAME_SHAPE_RANGE:
    case SAME_SHAPE_RANDOM:
    case SAME_SHAPE_RANGE_RANDOM:
    case SAME_SHAPE_GENERATE_1DREDUCE:
        return &hashOpcodeOpidShapeidSweepdim;
    default:
        throw runtime_error("Could not find valid hash function for fuse model.");
    }
}

//Constructor of the BatchHash class
BatchHash::BatchHash(const vector<bh_instruction> &instr_list)
{
    InstrHash hashFn = getInstrHash(fuse_get_selected_model());
    std::ostringstream data(std::ios_base::ate);
    for(const bh_instruction& instr: instr_list)
    {
        hashFn(data, instr, *this);
    }
    boost::hash<string> hasher;
    _hash = hasher(data.str());
}

InstrIndexesList &FuseCache::insert(const BatchHash &batch,
                                    const vector<bh_ir_kernel> &kernel_list)
{
    if(cache.find(batch.hash()) != cache.end())
    {
        throw runtime_error("Instruction list is already in the fuse cache!");
    }
    cache[batch.hash()] = InstrIndexesList(kernel_list, batch.hash(), fuser_name);
    return cache[batch.hash()];
}

bool FuseCache::lookup(const BatchHash &batch,
                       bh_ir &bhir,
                       vector<bh_ir_kernel> &kernel_list) const
{
    assert(kernel_list.size() == 0);
    assert(enabled);
    CacheMap::const_iterator it = cache.find(batch.hash());
    if(it == cache.end())
    {
        return false;
    }
    else
    {
        it->second.fill_kernel_list(bhir, kernel_list);
        return true;
    }
}

void FuseCache::write_to_files() const
{
    assert(enabled);
    if(dir_path == NULL or dir_path[0] == '\0')
    {
        cout << "[FUSE-CACHE] Couldn't find the 'cache_path' key in "   \
            "the configure file thus no cache files are written to disk!" << endl;
        return;
    }
    path cache_dir(dir_path);
    if(create_directories(cache_dir))
    {
        cout << "[FUSE-CACHE] Creating cache diretory " << cache_dir << endl;
#if BOOST_VERSION > 104900
            permissions(cache_dir, all_all);
#endif
    }

    path tmp_dir = cache_dir / unique_path();
    create_directories(tmp_dir);
    for(CacheMap::const_iterator it=cache.begin(); it != cache.end(); ++it)
    {
        string name;
        it->second.get_filename(name);
        path shared_name = cache_dir / name;

        if(exists(shared_name))
            continue;//No need to overwrite an existing file

        path unique_name = tmp_dir / name;
        std::ofstream ofs(unique_name.string().c_str());
        boost::archive::text_oarchive oa(ofs);
        oa << it->second;
        ofs.close();
#if BOOST_VERSION > 104900
        permissions(unique_name, all_all);
#endif
        rename(unique_name, shared_name);
    }
    remove(tmp_dir);
}

void FuseCache::load_from_files()
{
    assert(enabled);
    if(dir_path == NULL or dir_path[0] == '\0')
    {
        cout << "[FUSE-CACHE] Couldn't find the 'cache_path' key in "   \
            "the configure file thus no cache files are loaded from disk!" << endl;
        return;
    }
    path p(dir_path);
    if(not (exists(p) and is_directory(p)))
        return;

    string fuse_model_name;
    fuse_model_text(fuse_get_selected_model(), fuse_model_name);
    string fuse_price_name;
    fuse_price_model_text(fuse_get_selected_price_model(), fuse_price_name);

    //Iterate the 'dir_path' diretory and load each file
    directory_iterator it(p), eod;
    BOOST_FOREACH(const path &f, make_pair(it, eod))
    {
        if(is_regular_file(f))
        {
            int tries = 0;
            while(1)
            {
                try
                {
                    std::ifstream ifs(f.string().c_str());
                    boost::archive::text_iarchive ia(ifs);
                    InstrIndexesList t;
                    ia >> t;
                    if(iequals(t.fuser_name(), fuser_name) and
                       iequals(t.fuse_model(), fuse_model_name) and
                       iequals(t.price_model(), fuse_price_name))
                    {
                        if(cache.find(t.hash()) != cache.end())
                        {
                            throw runtime_error("Instruction list is already in the fuse cache!");
                        }
                        cache[t.hash()] = t;
                    }
                }
                catch(const std::exception &e)
                {
                    if(++tries >= 10)
                    {
                        cerr << "[FUSE-CACHE] failed to open file '" << f.string();
                        cerr << "' (" << tries << " tries): " << e.what() << endl;
                    }
                    else
                        continue;
                }
                break;
            }
        }
    }
}
} //namespace bohrium
