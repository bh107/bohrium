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

#include <sstream>
#include <cassert>

#include <jitk/block.hpp>
#include <jitk/instruction.hpp>

using namespace std;

namespace bohrium {
namespace jitk {

namespace {
void spaces(stringstream &out, int num) {
    for (int i = 0; i < num; ++i) {
        out << " ";
    }
}
} // Anonymous name space

Block create_nested_block(vector<bh_instruction *> &instr_list, int rank, int64_t size_of_rank_dim) {

    if (instr_list.empty()) {
        throw runtime_error("create_nested_block: 'instr_list' is empty!");
    }

    for (bh_instruction *instr: instr_list) {
        if (instr->max_ndim() <= rank)
            throw runtime_error("create_nested_block() was given an instruction with max_ndim <= 'rank'");
    }

    // Let's reshape instructions to match 'size_of_rank_dim'
    for (bh_instruction *instr: instr_list) {
        if (instr->reshapable() and instr->operand[0].shape[rank] != size_of_rank_dim) {
            vector<int64_t> shape((size_t) rank + 1);
            // The dimensions up til 'rank' (not including 'rank') are unchanged
            for (int64_t r = 0; r < rank; ++r) {
                shape[r] = instr->operand[0].shape[r];
            }
            int64_t size = 1; // The size of the reshapeable block
            for (int64_t r = rank; r < instr->operand[0].ndim; ++r) {
                size *= instr->operand[0].shape[r];
            }
            assert(size >= size_of_rank_dim);
            shape[rank] = size_of_rank_dim;
            if (size != size_of_rank_dim) { // We might have to add an extra dimension
                if (size % size_of_rank_dim != 0)
                    throw runtime_error("create_nested_block: shape is not divisible with 'size_of_rank_dim'");
                shape.push_back(size / size_of_rank_dim);
            }
            instr->reshape(shape);
        }
    }

    for (const bh_instruction *instr: instr_list) {
        vector<int64_t> shape = instr->dominating_shape();
        assert(shape.size() > (uint64_t) rank);
        if (shape[rank] != size_of_rank_dim)
            throw runtime_error("create_nested_block() was given an instruction where shape[rank] != size_of_rank_dim");
    }

    // Let's build the nested block from the 'rank' level to the instruction block
    vector <Block> block_list;
    set<bh_instruction*> sweeps;
    set<bh_base *> news;
    set<bh_base *> frees;
    for (bh_instruction *instr: instr_list) {
        const int64_t max_ndim = instr->max_ndim();
        assert(max_ndim > rank);

        // Create the rest of the dimension as a singleton block
        if (max_ndim > rank + 1) {
            vector<int64_t> shape = instr->dominating_shape();
            assert(shape.size() > 0);
            vector<bh_instruction *> single_instr = {instr};
            block_list.push_back(create_nested_block(single_instr, rank + 1, shape[rank + 1]));
        } else { // No more dimensions -- let's write the instruction block
            assert(max_ndim == rank + 1);
            block_list.emplace_back(instr, rank + 1);

            // Since 'instr' execute at this 'rank' level, we can calculate news, frees, and temps.
            if (instr->constructor) {
                if (not bh_opcode_is_accumulate(instr->opcode))// TODO: Support array contraction of accumulated output
                    news.insert(instr->operand[0].base);
            }
            if (instr->opcode == BH_FREE) {
                frees.insert(instr->operand[0].base);
            }
        }
        if (sweep_axis(*instr) == rank) {
            sweeps.insert(instr);
        }
    }
    return Block(rank, size_of_rank_dim, std::move(block_list), std::move(sweeps), std::move(news), std::move(frees));
}

Block *Block::findInstrBlock(const bh_instruction *instr) {
    assert(validation());
    if (isInstr()) {
        if (_instr != NULL and _instr == instr)
            return this;
    } else {
        for (Block &b: this->_block_list) {
            Block *ret = b.findInstrBlock(instr);
            if (ret != NULL)
                return ret;
        }
    }
    return NULL;
}

string Block::pprint(const char *newline) const {
    stringstream ss;
    spaces(ss, rank * 4);
    if (isInstr()) {
        if (_instr != NULL) {
            ss << *_instr << newline;
        }
    } else {
        ss << "rank: " << rank << ", size: " << size;
        if (_sweeps.size() > 0) {
            ss << ", sweeps: { ";
            for (const bh_instruction *i : _sweeps) {
                ss << *i << ",";
            }
            ss << "}";
        }
        if (_reshapable) {
            ss << ", reshapable";
        }
        if (_news.size() > 0) {
            ss << ", news: {";
            for (const bh_base *b : _news) {
                ss << "a" << b->get_label() << ",";
            }
            ss << "}";
        }
        if (_frees.size() > 0) {
            ss << ", frees: {";
            for (const bh_base *b : _frees) {
                ss << "a" << b->get_label() << ",";
            }
            ss << "}";
        }
        const set<bh_base *> temps = getLocalTemps();
        if (temps.size() > 0) {
            ss << ", temps: {";
            for (const bh_base *b : temps) {
                ss << "a" << b->get_label() << ",";
            }
            ss << "}";
        }
        if (_block_list.size() > 0) {
            ss << ", block list:" << newline;
            for (const Block &b : _block_list) {
                ss << b.pprint(newline);
            }
        }
    }
    return ss.str();
}

void Block::getAllSubBlocks(std::vector<const Block *> &out) const {
    for (const Block &b : _block_list) {
        if (not b.isInstr()) {
            out.push_back(&b);
            b.getAllSubBlocks(out);
        }
    }
}
void Block::getAllSubBlocks(std::vector<Block *> &out) {
    for (Block &b : _block_list) {
        if (not b.isInstr()) {
            out.push_back(&b);
            b.getAllSubBlocks(out);
        }
    }
}
vector<const Block *> Block::getAllSubBlocks() const {
    vector<const Block *> ret;
    getAllSubBlocks(ret);
    return ret;
}
vector<Block *> Block::getAllSubBlocks() {
    vector<Block *> ret;
    getAllSubBlocks(ret);
    return ret;
}

vector<const Block *> Block::getLocalSubBlocks() const {
    vector<const Block *> ret;
    for (const Block &b : _block_list) {
        if (not b.isInstr()) {
            ret.push_back(&b);
        }
    }
    return ret;
}

void Block::getAllInstr(vector<bh_instruction *> &out) const {
    if (isInstr()) {
        if (_instr != NULL)
            out.push_back(_instr);
    } else {
        for (const Block &b : _block_list) {
            b.getAllInstr(out);
        }
    }
}
vector<bh_instruction *> Block::getAllInstr() const {
    vector<bh_instruction *> ret;
    getAllInstr(ret);
    return ret;
}

void Block::getLocalInstr(vector<bh_instruction *> &out) const {
    for (const Block &b : _block_list) {
        if (b.isInstr() and b._instr != NULL) {
            out.push_back(b._instr);
        }
    }
}
vector<bh_instruction *> Block::getLocalInstr() const {
    vector<bh_instruction *> ret;
    getLocalInstr(ret);
    return ret;
}

void Block::getAllNews(set<bh_base *> &out) const {
    out.insert(_news.begin(), _news.end());
    for (const Block &b: _block_list) {
        b.getAllNews(out);
    }
}

set<bh_base *> Block::getAllNews() const {
    set<bh_base *> ret;
    getAllNews(ret);
    return ret;
}

void Block::getAllFrees(set<bh_base *> &out) const {
    out.insert(_frees.begin(), _frees.end());
    for (const Block &b: _block_list) {
        b.getAllFrees(out);
    }
}

set<bh_base *> Block::getAllFrees() const {
    set<bh_base *> ret;
    getAllFrees(ret);
    return ret;
}

void Block::getLocalTemps(set<bh_base *> &out) const {
    const set<bh_base *> frees = getAllFrees();
    std::set_intersection(_news.begin(), _news.end(), frees.begin(), frees.end(), std::inserter(out, out.begin()));
    const set<bh_base *> news = getAllNews();
    std::set_intersection(_frees.begin(), _frees.end(), news.begin(), news.end(), std::inserter(out, out.begin()));
}

set<bh_base *> Block::getLocalTemps() const {
    set<bh_base *> ret;
    getLocalTemps(ret);
    return ret;
}

void Block::getAllTemps(set<bh_base *> &out) const {
    getLocalTemps(out);
    for (const Block &b: _block_list) {
        b.getAllTemps(out);
    }
}

set<bh_base *> Block::getAllTemps() const {
    set<bh_base *> ret;
    getAllTemps(ret);
    return ret;
}

bool Block::validation() const {
    if (isInstr()) {
        if (_instr == NULL) {
            assert(1 == 2);
            return false;
        }
        if (_block_list.size() != 0) {
            assert(1 == 2);
            return false;
        }
        return true;
    }
    if (size < 0 or rank < 0) {
        assert(1 == 2);
        return false;
    }
    const vector<bh_instruction *> allInstr = getAllInstr();
    if (allInstr.empty()) {
        assert(1 == 2);
        return false;
    }
    for (const bh_instruction *instr: allInstr) {
        if (bh_opcode_is_system(instr->opcode))
            continue;
        if (instr->max_ndim() <= rank) {
            assert(1 == 2);
            return false;
        }
        if (instr->dominating_shape()[rank] != size) {
            assert(1 == 2);
            return false;
        }
    }
    if (_block_list.size() == 0) {
        assert(1 == 2);
        return false;
    }
    for (const Block &b: _block_list) {
        if (not b.validation())
            return false;
    }
    set<bh_base*> frees;
    for (const bh_instruction *instr: getLocalInstr()) {
        if (instr->opcode == BH_FREE) {
            frees.insert(instr->operand[0].base);
        }
    }
    if (frees != _frees) {
        assert(1 == 2);
        return false;
    }
    return true;
}

pair<Block*, int64_t> Block::findLastAccessBy(const bh_base *base) {
    assert(validation());
    assert(not isInstr());
    for (int64_t i=_block_list.size()-1; i >= 0; --i) {
        if (_block_list[i].isInstr()) {
            if (base == NULL) { // Searching for any access
                return make_pair(this, i);
            } else {
                const set<bh_base*> bases = _block_list[i]._instr->get_bases();
                if (bases.find(const_cast<bh_base*>(base)) != bases.end()) {
                    return make_pair(this, i);
                }
            }
        } else {
            // Check if the sub block accesses 'base'
            pair<Block*, int64_t> block = _block_list[i].findLastAccessBy(base);
            if (block.first != NULL) {
                return block; // We found the block and instruction that accesses 'base'
            }
        }
    }
    return make_pair((Block *)NULL, -1); // Not found
}

void Block::insert_system_after(bh_instruction *instr, const bh_base *base) {
    assert(validation());
    assert(not isInstr());

    Block *block;
    int64_t index;
    tie(block, index) = findLastAccessBy(base);
    // If no instruction accesses 'base' we insert it after the last instruction
    if (block == NULL) {
        tie(block, index) = findLastAccessBy(NULL); //NB: findLastAccessBy(NULL) will always find an instruction
        assert(block != NULL);
    }

    instr->reshape_force(block->_block_list[index]._instr->dominating_shape());
    block->_block_list.insert(block->_block_list.begin()+index+1, Block(instr, block->rank+1));

    // Let's update the '_free' set
    if (instr->opcode == BH_FREE) {
        block->_frees.insert(instr->operand[0].base);
    }
    assert(validation());
}

Block merge(const Block &a, const Block &b, bool based_on_block_b) {
    assert(not a.isInstr());
    assert(not b.isInstr());
    // For convenient such that the new block always depend on 't1'
    const Block &t1 = based_on_block_b ? b : a;
    const Block &t2 = based_on_block_b ? a : b;
    Block ret(t1);
    // The block list should always be in order: 'a' before 'b'
    ret._block_list.clear();
    ret._block_list.insert(ret._block_list.end(), a._block_list.begin(), a._block_list.end());
    ret._block_list.insert(ret._block_list.end(), b._block_list.begin(), b._block_list.end());
    // The order of the sets doesn't matter
    ret._sweeps.insert(t2._sweeps.begin(), t2._sweeps.end());
    ret._news.insert(t2._news.begin(), t2._news.end());
    ret._frees.insert(t2._frees.begin(), t2._frees.end());
    ret._reshapable = is_reshapeable(ret.getAllInstr());
    return ret;
}

// Unnamed namespace for all some merge help functions
namespace {
// Check if 'a' and 'b' supports data-parallelism when merged
bool data_parallel_compatible(const bh_instruction *a, const bh_instruction *b)
{
    if(bh_opcode_is_system(a->opcode) || bh_opcode_is_system(b->opcode))
        return true;

    const int a_nop = bh_noperands(a->opcode);
    for(int i=0; i<a_nop; ++i)
    {
        if(not bh_view_disjoint(&b->operand[0], &a->operand[i])
           && not bh_view_aligned(&b->operand[0], &a->operand[i]))
            return false;
    }
    const int b_nop = bh_noperands(b->opcode);
    for(int i=0; i<b_nop; ++i)
    {
        if(not bh_view_disjoint(&a->operand[0], &b->operand[i])
           && not bh_view_aligned(&a->operand[0], &b->operand[i]))
            return false;
    }
    return true;
}

// Check if 'b1' and 'b2' supports data-parallelism when merged
bool data_parallel_compatible(const Block &b1, const Block &b2) {
    for (const bh_instruction *i1 : b1.getAllInstr()) {
        for (const bh_instruction *i2 : b2.getAllInstr()) {
            if (not data_parallel_compatible(i1, i2))
                return false;
        }
    }
    return true;
}

// Check if 'block' accesses the output of a sweep in 'sweeps'
bool sweeps_accessed_by_block(const set<bh_instruction*> &sweeps, const Block &block) {
    for (bh_instruction *instr: sweeps) {
        assert(bh_noperands(instr->opcode) > 0);
        auto bases = block.getAllBases();
        if (bases.find(instr->operand[0].base) != bases.end())
            return true;
    }
    return false;
}
} // Unnamed namespace

pair<vector<const Block *>, uint64_t> find_threaded_blocks(const Block &block) {
    pair<vector<const Block*>, uint64_t> ret;

    if (block.isInstr()) {
        ret.second = 0;
        return ret; // Instruction blocks are not threaded
    }

    // We should search in 'this' block and its sub-blocks
    vector<const Block *> block_list = {&block};
    block.getAllSubBlocks(block_list);

    // Find threaded blocks
    constexpr int MAX_NUM_OF_THREADED_BLOCKS = 3;
    ret.second = 1;
    for (const Block *b: block_list) {
        if (b->_sweeps.size() == 0 and not b->isSystemOnly()) {
            ret.first.push_back(b);
            ret.second *= b->size;
        }
        // Multiple blocks or mixing instructions and blocks at the same level is not thread compatible
        if (not (b->getLocalSubBlocks().size() == 1 and b->getLocalInstr().size() == 0)) {
            break;
        }
        if (ret.first.size() == MAX_NUM_OF_THREADED_BLOCKS) {
            break;
        }
    }
    return ret;
}

bool merge_possible(const Block &a, const Block &b) {

    vector<bh_instruction> instr_list;
    instr_list.reserve(a.getAllInstr().size() + b.getAllInstr().size());
    Block a1;
    {
        vector<bh_instruction*> instr_list_ptr;
        for (bh_instruction *instr: a.getAllInstr()) {
            instr_list.push_back(*instr);
            instr_list_ptr.push_back(&instr_list.back());
        }
        a1 = create_nested_block(instr_list_ptr, a.rank, a.size);
    }
    Block b1;
    {
        vector<bh_instruction*> instr_list_ptr;
        for (bh_instruction *instr: b.getAllInstr()) {
            instr_list.push_back(*instr);
            instr_list_ptr.push_back(&instr_list.back());
        }
        b1 = create_nested_block(instr_list_ptr, b.rank, b.size);
    }
    return merge_if_possible(a1, b1).second;
}

std::pair<Block, bool> merge_if_possible(Block &a, Block &b) {
    assert(a.validation());
    assert(b.validation());

    // First we check for data incompatibility
    if (a.isInstr() or b.isInstr() or not data_parallel_compatible(a, b) or sweeps_accessed_by_block(a._sweeps, b)) {
        return make_pair(Block(), false);
    }
    // Check for perfect match, which is directly mergeable
    if (a.size == b.size) {
        return make_pair(merge(a, b), true);
    }

    // System-only blocks are very flexible because they array sizes does not have to match when reshaping
    // thus we can simply append system instructions without further checks.
    if (b.isSystemOnly()) {
        Block block(a);
        for (bh_instruction *instr: b.getAllInstr()) {
            if (bh_noperands(instr->opcode) > 0) {
                block.insert_system_after(instr, instr->operand[0].base);
            }
        }
        return make_pair(block, true);
    }

    // Check fusibility of reshapable blocks
    if (b._reshapable && b.size % a.size == 0) {
        vector<bh_instruction *> cur_instr = a.getAllInstr();
        vector<bh_instruction *> it_instr = b.getAllInstr();
        cur_instr.insert(cur_instr.end(), it_instr.begin(), it_instr.end());
        return make_pair(create_nested_block(cur_instr, b.rank, a.size), true);
    }
    if (a._reshapable && a.size % b.size == 0) {
        vector<bh_instruction *> cur_instr = a.getAllInstr();
        vector<bh_instruction *> it_instr = b.getAllInstr();
        cur_instr.insert(cur_instr.end(), it_instr.begin(), it_instr.end());
        return make_pair(create_nested_block(cur_instr, a.rank, b.size), true);
    }
    return make_pair(Block(), false);
}

ostream &operator<<(ostream &out, const Block &b) {
    out << b.pprint();
    return out;
}

ostream &operator<<(ostream &out, const vector<Block> &block_list) {
    out << "Block list: " << endl;
    for (const Block &b: block_list) {
        out << b;
    }
    return out;
}


} // jitk
} // bohrium
