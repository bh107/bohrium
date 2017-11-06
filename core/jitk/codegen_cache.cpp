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

#include <jitk/codegen_cache.hpp>

using namespace std;

namespace bohrium {
namespace jitk {

namespace {

boost::hash<string> hasher;

/* The View hash consists of the following fields:
 * <view_id><start><ndim>[<shape><stride><SEP_SHAPE>...]<SEP_OP>
 */
void hash_stream(const bh_view &view, const SymbolTable &symbols, std::stringstream &ss) {
    ss << "dtype: " << static_cast<size_t>(view.base->type);
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
            ss << "const dtype: " << static_cast<size_t>(instr.constant.type);
        } else {
            hash_stream(op, symbols,  ss);
        }
    }
    ss << "sweep: " << instr.sweep_axis();
}

/* The Block hash consists of the following fields:
 * <block_rank><instr_hash><SEP_BLOCK>
 */
void hash_stream(const Block &block, const SymbolTable &symbols, std::stringstream &ss) {
    if (block.isInstr()) {
        hash_stream(*block.getInstr(), symbols, ss);
    } else {
        ss << "rank: " << block.rank();
        ss << "size: " << block.getLoop().size;
        for (const Block &b: block.getLoop()._block_list) {
            hash_stream(b, symbols, ss);
        }
    }
}

/* The Block list hash consists of the following fields:
 * <block_rank><SEP_BLOCK>
 */
size_t block_list_hash(const std::vector<Block> &block_list, const SymbolTable &symbols) {
    stringstream ss;
    for (const Block &b: block_list) {
        hash_stream(b, symbols, ss);
        ss << "<block>";
    }
    return hasher(ss.str());
}
} // Anonymous Namespace

std::pair<std::string, bool> CodegenCache::get(const std::vector<Block> &block_list, const SymbolTable &symbols) {
    ++stat.codegen_cache_lookups;
    const size_t lookup_hash = block_list_hash(block_list, symbols);
    auto lookup = _cache.find(lookup_hash);
    if (lookup != _cache.end()) { // Cache hit!
        return make_pair(lookup->second, true);
    } else {
        ++stat.codegen_cache_misses;
        return make_pair("", false);
    }
}

void CodegenCache::insert(std::string source, const std::vector<Block> &block_list, const SymbolTable &symbols) {
    const size_t lookup_hash = block_list_hash(block_list, symbols);
    assert(_cache.find(lookup_hash) == _cache.end()); // The source shouldn't exist in the cache already
    _cache[lookup_hash] = std::move(source);
}

} // jitk
} // bohrium
