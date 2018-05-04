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

#include <jitk/codegen_cache.hpp>

using namespace std;

namespace bohrium {
namespace jitk {

namespace {

/* The View hash consists of the following fields:
 * <view_id><start><ndim>[<shape><stride><SEP_SHAPE>...]<SEP_OP>
 */
void hash_stream(const bh_view &view, const SymbolTable &symbols, std::stringstream &ss) {
    ss << "dtype: " << static_cast<uint32_t>(view.base->type);
    ss << "baseid: " << symbols.baseID(view.base);

    if (symbols.strides_as_var) {
        ss << "strideid: " << symbols.offsetStridesID(view);
    } else {
        ss << "vstart: " << view.start;
        for (int j = 0; j < view.ndim; ++j) {
            ss << "dim: " << j;
            ss << "shape: " << view.shape[j];
            ss << "stride: " << view.stride[j];
        }
    }
    if (symbols.index_as_var) {
        ss << "indexid: " << symbols.idxID(view);
        if (bh_is_scalar(&view)) { // We optimize indexes into 1-sized arrays, which we need the hash to reflect
            ss << "is-1-elem: " << endl;
        }
    }
}

/* The Instruction hash consists of the following fields:
 * <opcode[<hash_view>...]<sweep_axis()><SEP_INSTR>
 */
void hash_stream(const bh_instruction &instr, const SymbolTable &symbols, std::stringstream &ss) {
    ss << "opcode: " << instr.opcode;
    for (const bh_view &op: instr.operand) {
        if (bh_is_constant(&op)) {
            int64_t id = symbols.constID(instr);
            if (id >= 0 and symbols.const_as_var) {
                ss << "const: " << symbols.constID(instr);
            } else {
                ss << "const: " << instr.constant;
            }
            ss << "const dtype: " << static_cast<uint32_t >(instr.constant.type);
        } else {
            hash_stream(op, symbols,  ss);
        }
    }
    ss << "sweep: " << instr.sweep_axis();
}

/* The Block hash consists of the following fields:
 * <block_rank><instr_hash><SEP_BLOCK>
 */
void hash_stream(const LoopB &block, const SymbolTable &symbols, std::stringstream &ss) {
    ss << "rank: " << block.rank;
    ss << "size: " << block.size;
    {  // The order of BH_FREE within a block doesn't matter, thus we sort the freed base IDs here
        ss << "freed: ";
        set<uint64_t>sorted_freed_bases;
        for (const bh_base *b: block._frees) {
            sorted_freed_bases.insert(symbols.baseID(b));
        }
        for(uint64_t b_id: sorted_freed_bases) {
            ss << b_id << ",";
        }
    }
    for (const Block &b: block._block_list) {
        if (b.isInstr()) {
            if (b.getInstr()->opcode != BH_FREE) {
                hash_stream(*b.getInstr(), symbols, ss);
            }
        } else {
            hash_stream(b.getLoop(), symbols, ss);
        }
    }
}

/* The Block hash from above as an uint64_t */
uint64_t hash_stream(const LoopB &block, const SymbolTable &symbols) {
    stringstream ss;
    hash_stream(block, symbols, ss);
    return util::hash(ss.str());
}
} // Anonymous Namespace

std::pair<std::string, uint64_t> CodegenCache::lookup(const LoopB &kernel, const SymbolTable &symbols) {
    ++stat.codegen_cache_lookups;
    const uint64_t lookup_hash = hash_stream(kernel, symbols);
    auto lookup = _cache.find(lookup_hash);
    if (lookup != _cache.end()) { // Cache hit!
        return make_pair(lookup->second, lookup_hash);
    } else {
        ++stat.codegen_cache_misses;
        return make_pair("", lookup_hash);
    }
}

void CodegenCache::insert(std::string source, const LoopB &kernel, const SymbolTable &symbols) {
    const uint64_t lookup_hash = hash_stream(kernel, symbols);
    assert(_cache.find(lookup_hash) == _cache.end()); // The source shouldn't exist in the cache already
    _cache[lookup_hash] = std::move(source);
}

} // jitk
} // bohrium
