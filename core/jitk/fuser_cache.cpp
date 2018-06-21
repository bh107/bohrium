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


constexpr size_t SEP_INSTR = SIZE_MAX;
constexpr size_t SEP_OP = SIZE_MAX - 1;
constexpr size_t SEP_SHAPE = SIZE_MAX - 2;
constexpr size_t SEP_CONSTANT = SIZE_MAX - 3;

/* The Instruction hash consists of the following fields:
 * <view_id><start><ndim>[<shape><stride><SEP_SHAPE>...]<SEP_OP>
 */
void hash_view(const bh_view &view, ViewDB &views, std::stringstream &ss) {
    if (not bh_is_constant(&view)) {
        size_t view_id = views.insert(view).first;
        ss << view_id;
        // Sliding views has identical hashes across iterations
        if (view.slide.empty()) {
            ss << view.start;
        } else {
            // Check whether the shape of the sliding view is a single value
            bool single_index = true;
            for (int i = 0; i < view.ndim; i++) {
                if (view.shape[i] != 1) {
                    single_index = false;
                    break;
                }
            }
            if (!single_index) {
                ss << view.start;
            }
        }

        ss << view.ndim;
        for (int j = 0; j < view.ndim; ++j) {
            ss << view.shape[j];
            ss << view.stride[j];
            ss << SEP_SHAPE;
        }
        ss << SEP_OP;
    } else {
        // Notice, we can ignore the value of the constant but we need to hash the location of the constant
        ss << SEP_CONSTANT;
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
    return util::hash(ss.str());
}

// Replace the cached values of constants and bases arrays in `instr` with their original values
void update_with_origin(bh_instruction &instr, const bh_instruction *origin,
                        const std::map<bh_base*, bh_base*> &base_cached2new) {
    assert(instr.origin_id == origin->origin_id);
    assert(instr.opcode == origin->opcode);
    for (size_t i = 0; i < instr.operand.size(); ++i) {
        if (not instr.operand[i].slide.empty()) {
            instr.operand[i].start = origin->operand[i].start;
        }

        if (bh_is_constant(&instr.operand[i])) {
            // NB: sweeped axis values shouldn't be updated
            if (not bh_opcode_is_sweep(instr.opcode)) {
                instr.constant = origin->constant;
            }
        } else {
            instr.operand[i].base = base_cached2new.at(instr.operand[i].base);
            assert(instr.operand[i].base == origin->operand[i].base);
        }
    }
}

// Replace the cached values of constants and bases arrays in `block` with their original values
void update_with_origin(Block &block, const std::map<bh_base*, bh_base*> &base_cached2new,
                        const map<int64_t, const bh_instruction *> &origin_id_to_instr) {
    if (block.isInstr()) {
        assert(block.getInstr()->origin_id >= 0);
        bh_instruction instr(*block.getInstr());
        update_with_origin(instr, origin_id_to_instr.at(instr.origin_id), base_cached2new);
        block.setInstr(instr);
    } else {
        LoopB &loop = block.getLoop();
        for (Block &b: loop._block_list) {
            update_with_origin(b, base_cached2new, origin_id_to_instr);
        }
        set<bh_base*> frees;
        for (bh_base *b: loop._frees) {
            frees.insert(base_cached2new.at(b));
        }
        loop._frees = std::move(frees);
        loop.metadataUpdate();
    }
}

// We assign the base IDs in the order they appear in the 'instr_list'
// Notice, the base IDs corresponds to their position in the returned vector
std::vector<bh_base*> calc_base_ids(const vector<bh_instruction *> &instr_list) {
    std::vector<bh_base*> ret;
    std::set<bh_base*> unique_bases;
    for (const auto &instr: instr_list) {
        for (const bh_view *view: instr->get_views()) {
            if (not util::exist(unique_bases, view->base)) {
                unique_bases.insert(view->base);
                ret.push_back(view->base);
            }
        }
    }
    return ret;
}
} // Anon namespace

pair<vector<Block>, bool> FuseCache::get(const vector<bh_instruction *> &instr_list) {
    const size_t lookup_hash = hash_instr_list(instr_list);
    ++stat.fuser_cache_lookups;

    if (_cache.find(lookup_hash) != _cache.end()) { // Cache hit!
        // Create a map: 'origin_id' => instruction for updating the constants
        map<int64_t, const bh_instruction *> origin_id_to_instr;
        for(const bh_instruction *instr: instr_list) {
            assert(instr->origin_id >= 0);
            assert(not util::exist(origin_id_to_instr, instr->origin_id));
            origin_id_to_instr.insert(make_pair(instr->origin_id, instr));
        }
        // Create a map: 'cached bases' => 'new bases' for updating the base arrays
        const CachePayload &cached = _cache.at(lookup_hash);
        std::map<bh_base*, bh_base*> base_cached2new;
        {
            size_t id = 0;
            for(auto &base: calc_base_ids(instr_list)) {
                assert(id < cached.base_ids.size());
                base_cached2new[cached.base_ids[id++]] = base;
            }
        }
        // Let's make a copy of the cached block list and update the bases
        vector<Block> ret = cached.block_list;
        for(Block &block: ret) {
            update_with_origin(block, base_cached2new, origin_id_to_instr);
        }
        return make_pair(std::move(ret), true);
    } else { // Cache miss!
        ++stat.fuser_cache_misses;
        return make_pair(vector<Block>(), false);
    }
}

void FuseCache::insert(const vector<bh_instruction *> &instr_list, vector<Block> block_list) {
    const size_t lookup_hash = hash_instr_list(instr_list);
    CachePayload payload = {std::move(block_list), calc_base_ids(instr_list)};
    _cache.insert(make_pair(lookup_hash, std::move(payload)));
}

} // jitk
} // bohrium
