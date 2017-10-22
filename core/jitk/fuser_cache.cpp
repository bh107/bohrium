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


using namespace std;

namespace bohrium {
namespace jitk {

namespace {

// Handling view IDs
class ViewDB {
private:
    size_t maxid;
    std::map<bh_view,size_t> _map;
public:
    ViewDB() : maxid(0) {}

    // Insert an object
    std::pair<size_t,bool> insert(const bh_view &v) {
        auto it = _map.find(v);
        if (it == _map.end()) {
            size_t id = maxid++;
            _map.insert(std::make_pair(v,id));
            return std::make_pair(id,true);

        } else {
            return  std::make_pair(it->second,false);
        }
    }
};


boost::hash<string> hasher;
constexpr size_t SEP_INSTR = SIZE_MAX;
constexpr size_t SEP_OP = SIZE_MAX - 1;
constexpr size_t SEP_SHAPE = SIZE_MAX - 2;

/* The Instruction hash consists of the following fields:
 * <view_id><start><ndim>[<shape><stride><SEP_SHAPE>...]<SEP_OP>
 */
void hash_view(const bh_view &view, ViewDB &views, std::stringstream &ss) {
    if (not bh_is_constant(&view)) {
        size_t view_id = views.insert(view).first;
        ss << view_id;
        ss << view.start;
        ss << view.ndim;
        for (int j = 0; j < view.ndim; ++j) {
            ss << view.shape[j];
            ss << view.stride[j];
            ss << SEP_SHAPE;
        }
        ss << SEP_OP;
    }
}

/* The Instruction hash consists of the following fields:
 * <opcode[<hash_view>...]<sweep_axis()><SEP_INSTR>
 */
void hash_instr(const bh_instruction &instr, ViewDB &views, std::stringstream &ss) {
    ss << instr.opcode; // <opcode>
    for(const bh_view &op: instr.operand) {
        hash_view(op, views, ss);
    }
    ss << instr.sweep_axis();
    ss << SEP_INSTR;
}

// Hash of an instruction list
size_t hash_instr_list(const vector<bh_instruction *> &instr_list) {
    stringstream ss;
    ViewDB views;
    for (const bh_instruction *instr: instr_list) {
        hash_instr(*instr, views, ss);
    }
    return hasher(ss.str());
}

void updateWithOrigin(bh_view &view, const bh_view &origin) {
    view.base = origin.base;
}

void updateWithOrigin(bh_instruction &instr, const bh_instruction *origin) {
    assert(instr.origin_id == origin->origin_id);
    assert(instr.opcode == origin->opcode);

    for (size_t i = 0; i < instr.operand.size(); ++i) {
        if (bh_is_constant(&instr.operand[i]) and not bh_opcode_is_sweep(instr.opcode)) {
            // NB: sweeped axis values shouldn't be updated
            instr.constant = origin->constant;
        } else {
            updateWithOrigin(instr.operand[i], origin->operand[i]);
        }
    }
}

void updateWithOrigin(Block &block, const map<int64_t, const bh_instruction *> &origin_id_to_instr) {
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

} // Anon namespace

pair<vector<Block>, bool> FuseCache::get(const vector<bh_instruction *> &instr_list) {
    const size_t lookup_hash = hash_instr_list(instr_list);
    ++stat.fuser_cache_lookups;
    if (_cache.find(lookup_hash) != _cache.end()) { // Cache hit!
        vector<Block> ret = _cache.at(lookup_hash);
        // Create a map: 'origin_id' => instruction
        map<int64_t, const bh_instruction *> origin_id_to_instr;
        for(const bh_instruction *instr: instr_list) {
            assert(instr->origin_id >= 0);
            assert(origin_id_to_instr.find(instr->origin_id) == origin_id_to_instr.end());
            origin_id_to_instr.insert(make_pair(instr->origin_id, instr));
        }
        // Let's update the cached blocks in 'ret' with the base data from origin
        for(Block &block: ret) {
            updateWithOrigin(block, origin_id_to_instr);
        }
        return make_pair(ret, true);
    } else { // Cache miss!
        ++stat.fuser_cache_misses;
        return make_pair(vector<Block>(), false);
    }
}

void FuseCache::insert(const vector<bh_instruction *> &instr_list, const vector<Block> &block_list) {
    const size_t lookup_hash = hash_instr_list(instr_list);
    _cache.insert(make_pair(lookup_hash, block_list));
}

} // jitk
} // bohrium
