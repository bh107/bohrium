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

#include <vector>
#include <iostream>
#include <boost/functional/hash.hpp>

#include <jitk/fuser_cache.hpp>
#include <bh_seqset.hpp>


using namespace std;

namespace bohrium {
namespace jitk {

static boost::hash<string> hasher;
static const size_t SEP_INSTR = SIZE_MAX;
static const size_t SEP_OP    = SIZE_MAX-1;
static const size_t SEP_BLOCK = SIZE_MAX-2;
static const size_t SEP_SHAPE = SIZE_MAX-3;

void hash_view(const bh_view &view, seqset<bh_view> &views, std::stringstream& ss) {
    if (not bh_is_constant(&view)) {
        size_t view_id = views.insert(view).first;
        ss << " v" << view_id;
        ss << " s" << view.start;
        ss << " nd" << view.ndim << "(";
        for (int j=0; j<view.ndim; ++j) {
            ss << view.shape[j];
            ss << view.stride[j];
            ss << SEP_SHAPE;
        }
        ss << ")";
    }
}

/* The Instruction hash consists of the following fields:
 * <opcode> (<operand-id> <ndim> <shape> <SEP_OP>)[1] <sweep-dim>[2] <SEP_INSTR>
 * <opcode> (<operand-id><SEP_OP>)[1] <sweep-dim>[2] <SEP_INSTR>
 * 1: for each operand
 * 2: if the operation is a sweep operation or BH_MAXDIM
 */
void hash_instr(const bh_instruction& instr, seqset<bh_view> &views, std::stringstream& ss) {
    ss << instr.opcode; // <opcode>
    const int nop = bh_noperands(instr.opcode);
    for(int i=0; i<nop; ++i) {
        hash_view(instr.operand[i], views, ss);
        ss << " SEP_OP ";
    }
    ss << instr.sweep_axis();
    ss << " SEP_INSTR ";
}

size_t hash_instr_list(const vector<bh_instruction *> &instr_list) {
    stringstream ss;
    seqset<bh_view> views;
    for (const bh_instruction *instr: instr_list) {
        hash_instr(*instr, views, ss);
    }
    return hasher(ss.str());
}

/*
void hash_block(const Block &block, seqset<bh_view> &views, std::stringstream& ss) {
    if (block.isInstr()) {
        hash_instr(*block.getInstr(), views, ss);
    } else {
        map<const bh_instruction, size_t> instr_map;
        for (const InstrPtr instr: block.getAllInstr())
        for (const Block &b: block.getLoop()._block_list) {
            hash_block(b, views, ss);
        }
        ss << SEP_BLOCK;
    }
}
*/

static void updateWithOrigin(bh_view &view, const bh_view &origin) {
    view.base = origin.base;
}

static void updateWithOrigin(bh_instruction &instr, const bh_instruction *origin) {
    assert(instr.origin_id == origin->origin_id);
    assert(instr.opcode == origin->opcode);
    int nop = bh_noperands(instr.opcode);
    for(int i=0; i<nop; ++i) {
        if (bh_is_constant(&instr.operand[i]) and not bh_opcode_is_sweep(instr.opcode)) {
            // NB: sweeped axis values shouldn't be updated
            instr.constant = origin->constant;
        } else {
            updateWithOrigin(instr.operand[i], origin->operand[i]);
        }
    }
}

static void updateWithOrigin(Block &block, const map<int64_t, const bh_instruction *> &origin_id_to_instr) {
    if (block.isInstr()) {
        assert(block.getInstr()->origin_id >= 0);
        bh_instruction instr(*block.getInstr());
        updateWithOrigin(instr, origin_id_to_instr.at(instr.origin_id));
        block.setInstr(instr);
    } else {
        LoopB &loop = block.getLoop();
        for (Block &b: loop._block_list) {
            updateWithOrigin(b, origin_id_to_instr);
        }
        loop.metadata_update();
    }
}

pair<vector<Block>, bool> FuseCache::get(const vector<bh_instruction *> &instr_list) {
    const size_t lookup_hash = hash_instr_list(instr_list);
    ++num_lookups;
    if (_cache.find(lookup_hash) != _cache.end()) { // Cache hit!
        vector<Block> ret = _cache.at(lookup_hash);
        // Create a map: 'origin_id' => instruction
        map<int64_t, const bh_instruction *> origin_id_to_instr;
        for(const bh_instruction *instr: instr_list) {
            assert(instr->origin_id >= 0);
            assert(origin_id_to_instr.find(instr->origin_id) == origin_id_to_instr.end());
            origin_id_to_instr.insert(make_pair(instr->origin_id, instr));
        }
/*
        cout << "Instr request:" << endl;
        for(bh_instruction *instr: instr_list) {
                cout << "origin: " << instr->origin_id << " " << *instr << endl;
        }

        cout << "Instr in cache:" << endl;
        for(Block &block: ret) {
            for (InstrPtr instr: block.getAllInstr()) {
                cout << "origin: " << instr->origin_id << " " << *instr << endl;
            }
        }
*/
        // Let's update the cached blocks in 'ret' with the base data from origin
        for(Block &block: ret) {
            updateWithOrigin(block, origin_id_to_instr);
        }
/*
        cout << "Instr Returned:" << endl;
        for(Block &block: ret) {
            for (InstrPtr instr: block.getAllInstr()) {
                cout << "origin: " << instr->origin_id << " " << *instr << endl;
            }
        }
        cout << endl << endl;
*/
        return make_pair(ret, true);
    } else { // Cache miss!
        ++num_lookup_misses;
        return make_pair(vector<Block>(), false);
    }
}

void FuseCache::insert(const vector<bh_instruction *> &instr_list, const vector<Block> &block_list) {
    const size_t lookup_hash = hash_instr_list(instr_list);
    _cache.insert(make_pair(lookup_hash, block_list));
}

} // jitk
} // bohrium
