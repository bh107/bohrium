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

// Returns true if a block consisting of 'instr_list' is reshapable
bool is_reshapeable(const std::vector<InstrPtr> &instr_list) {
    assert(instr_list.size() > 0);

    // In order to be reshapeable, all instructions must have the same rank and be reshapeable
    int64_t rank = instr_list[0]->max_ndim();
    for (InstrPtr instr: instr_list) {
        if (not instr->reshapable())
            return false;
        if (instr->max_ndim() != rank)
            return false;
    }
    return true;
}

// Append 'instr' to the loop block 'block'
void add_instr_to_block(LoopB &block, InstrPtr instr, int rank, int64_t size_of_rank_dim) {
    if (instr->max_ndim() <= rank)
        throw runtime_error("add_instr_to_block() was given an instruction with max_ndim <= 'rank'");

    // Let's reshape the instruction to match 'size_of_rank_dim'
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
                throw runtime_error("add_instr_to_block(): shape is not divisible with 'size_of_rank_dim'");
            shape.push_back(size / size_of_rank_dim);
        }
        bh_instruction instr_reshaped = bh_instruction(*instr);
        instr_reshaped.reshape(shape);
        instr.reset(new bh_instruction(instr_reshaped));
    }

    vector<int64_t> shape = instr->dominating_shape();

    // Sanity check
    assert(shape.size() > (uint64_t) rank);
    if (shape[rank] != size_of_rank_dim)
        throw runtime_error("create_nested_block() was given an instruction where shape[rank] != size_of_rank_dim");
    const int64_t max_ndim = instr->max_ndim();
    assert(max_ndim > rank);

    // Create the rest of the dimension as a singleton block
    if (max_ndim > rank + 1) {
        assert(shape.size() > 0);
        vector<InstrPtr> single_instr = {instr};
        block._block_list.push_back(create_nested_block(single_instr, rank + 1, shape[rank + 1]));
    } else { // No more dimensions -- let's write the instruction block
        assert(max_ndim == rank + 1);
        block._block_list.emplace_back(*instr, rank + 1);

        // Since 'instr' execute at this 'rank' level, we can calculate news, frees, and temps.
        if (instr->constructor) {
            if (not bh_opcode_is_accumulate(instr->opcode))// TODO: Support array contraction of accumulated output
                block._news.insert(instr->operand[0].base);
        }
        if (instr->opcode == BH_FREE) {
            block._frees.insert(instr->operand[0].base);
        }
    }
    if (sweep_axis(*instr) == rank) {
        block._sweeps.insert(instr);
    }
}

} // Anonymous name space

// *** LoopB Methods *** //

void LoopB::replaceInstr(InstrPtr subject, const bh_instruction &replacement) {
    for (Block &b: _block_list) {
        if (b.isInstr()) {
            if (*b.getInstr() == *subject) {
                b.setInstr(replacement);
            }
        } else {
            b.getLoop().replaceInstr(subject, replacement);
        }
    }
}

bool LoopB::isInnermost() const {
    for (const Block &b: _block_list) {
        if (not b.isInstr()) {
            return false;
        }
    }
    return true;
}

bool LoopB::isSystemOnly() const {
    for (const Block &b: _block_list) {
        if (not b.isSystemOnly()) {
            return false;
        }
    }
    return true;
}

void LoopB::getAllSubBlocks(std::vector<const LoopB *> &out) const {
    for (const Block &b : _block_list) {
        if (not b.isInstr()) {
            out.push_back(&b.getLoop());
            b.getLoop().getAllSubBlocks(out);
        }
    }
}

vector<const Block *> LoopB::getLocalSubBlocks() const {
    vector<const Block *> ret;
    for (const Block &b : _block_list) {
        if (not b.isInstr()) {
            ret.push_back(&b);
        }
    }
    return ret;
}

void LoopB::getAllInstr(vector<InstrPtr> &out) const {
    for (const Block &b : _block_list) {
        b.getAllInstr(out);
    }
}
vector<InstrPtr> LoopB::getAllInstr() const {
    vector<InstrPtr> ret;
    getAllInstr(ret);
    return ret;
}

void LoopB::getLocalInstr(vector<InstrPtr> &out) const {
    for (const Block &b : _block_list) {
        if (b.isInstr() and b.getInstr() != NULL) {
            out.push_back(b.getInstr());
        }
    }
}
vector<InstrPtr> LoopB::getLocalInstr() const {
    vector<InstrPtr> ret;
    getLocalInstr(ret);
    return ret;
}

void LoopB::getAllNews(set<bh_base *> &out) const {
    out.insert(_news.begin(), _news.end());
    for (const Block &b: _block_list) {
        if (not b.isInstr()){
            b.getLoop().getAllNews(out);
        }
    }
}

set<bh_base *> LoopB::getAllNews() const {
    set<bh_base *> ret;
    getAllNews(ret);
    return ret;
}

void LoopB::getAllFrees(set<bh_base *> &out) const {
    out.insert(_frees.begin(), _frees.end());
    for (const Block &b: _block_list) {
        if (not b.isInstr()){
            b.getLoop().getAllFrees(out);
        }
    }
}

set<bh_base *> LoopB::getAllFrees() const {
    set<bh_base *> ret;
    getAllFrees(ret);
    return ret;
}

void LoopB::getLocalTemps(set<bh_base *> &out) const {
    const set<bh_base *> frees = getAllFrees();
    std::set_intersection(_news.begin(), _news.end(), frees.begin(), frees.end(), std::inserter(out, out.begin()));
    const set<bh_base *> news = getAllNews();
    std::set_intersection(_frees.begin(), _frees.end(), news.begin(), news.end(), std::inserter(out, out.begin()));
}

set<bh_base *> LoopB::getLocalTemps() const {
    set<bh_base *> ret;
    getLocalTemps(ret);
    return ret;
}

void LoopB::getAllTemps(set<bh_base *> &out) const {
    getLocalTemps(out);
    for (const Block &b: _block_list) {
        if (not b.isInstr()){
            b.getLoop().getAllTemps(out);
        }
    }
}

set<bh_base *> LoopB::getAllTemps() const {
    set<bh_base *> ret;
    getAllTemps(ret);
    return ret;
}

pair<LoopB*, int64_t> LoopB::findLastAccessBy(const bh_base *base) {
    assert(validation());
    for (int64_t i=_block_list.size()-1; i >= 0; --i) {
        if (_block_list[i].isInstr()) {
            if (base == NULL) { // Searching for any access
                return make_pair(this, i);
            } else {
                const set<const bh_base*> bases = _block_list[i].getInstr()->get_bases();
                if (bases.find(base) != bases.end()) {
                    return make_pair(this, i);
                }
            }
        } else {
            // Check if the sub block accesses 'base'
            pair<LoopB*, int64_t> block = _block_list[i].getLoop().findLastAccessBy(base);
            if (block.first != NULL) {
                return block; // We found the block and instruction that accesses 'base'
            }
        }
    }
    return make_pair((LoopB *)NULL, -1); // Not found
}

bool LoopB::validation() const {
    if (size < 0 or rank < 0) {
        assert(1 == 2);
        return false;
    }
    const vector<InstrPtr> allInstr = getAllInstr();
    if (allInstr.empty()) {
        assert(1 == 2);
        return false;
    }
    for (const InstrPtr instr: allInstr) {
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
    for (const InstrPtr instr: getLocalInstr()) {
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

void LoopB::insert_system_after(InstrPtr instr, const bh_base *base) {
    assert(validation());

    LoopB *block;
    int64_t index;
    tie(block, index) = findLastAccessBy(base);
    // If no instruction accesses 'base' we insert it after the last instruction
    if (block == NULL) {
        tie(block, index) = findLastAccessBy(NULL); //NB: findLastAccessBy(NULL) will always find an instruction
        assert(block != NULL);
    }
    bh_instruction instr_reshaped(*instr);
    instr_reshaped.reshape_force(block->_block_list[index].getInstr()->dominating_shape());
    Block instr_block(instr_reshaped, block->rank+1);
    block->_block_list.insert(block->_block_list.begin()+index+1, instr_block);

    // Let's update the '_free' set
    if (instr->opcode == BH_FREE) {
        block->_frees.insert(instr->operand[0].base);
    }
    assert(validation());
}

string LoopB::pprint(const char *newline) const {
    stringstream ss;
    spaces(ss, rank * 4);
    ss << "rank: " << rank << ", size: " << size;
    if (_sweeps.size() > 0) {
        ss << ", sweeps: { ";
        for (const InstrPtr instr : _sweeps) {
            ss << *instr << ",";
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
    return ss.str();
}


// *** Block Methods *** //

bool Block::validation() const {
    if (isInstr()) {
        return true;
    } else {
        return getLoop().validation();
    }
}

void Block::getAllInstr(vector<InstrPtr> &out) const {
    if (isInstr()) {
        out.push_back(getInstr());
    } else {
        for (const Block &b : getLoop()._block_list) {
            b.getAllInstr(out);
        }
    }
}

vector<InstrPtr> Block::getAllInstr() const {
    vector<InstrPtr> ret;
    getAllInstr(ret);
    return ret;
}

string Block::pprint(const char *newline) const {

    if (isInstr()) {
        stringstream ss;
        if (getInstr() != NULL) {
            spaces(ss, rank() * 4);
            ss << *getInstr() << newline;
        }
        return ss.str();
    } else {
        return getLoop().pprint(newline);
    }
}


// *** Block Functions *** //

LoopB merge(const LoopB &l1, const LoopB &l2) {
    LoopB ret(l1);
    // The block list should always be in order: 'a' before 'b'
    ret._block_list.clear();
    ret._block_list.insert(ret._block_list.end(), l1._block_list.begin(), l1._block_list.end());
    ret._block_list.insert(ret._block_list.end(), l2._block_list.begin(), l2._block_list.end());
    // The order of the sets doesn't matter
    ret._sweeps.insert(l2._sweeps.begin(), l2._sweeps.end());
    ret._news.insert(l2._news.begin(), l2._news.end());
    ret._frees.insert(l2._frees.begin(), l2._frees.end());
    ret._reshapable = is_reshapeable(ret.getAllInstr());
    return ret;
}

Block create_nested_block(const vector<InstrPtr> &instr_list, int rank, int64_t size_of_rank_dim) {
    if (instr_list.empty()) {
        throw runtime_error("create_nested_block: 'instr_list' is empty!");
    }

    // Let's build the nested block from the 'rank' level to the instruction block
    LoopB ret;
    for (InstrPtr instr: instr_list) {
        add_instr_to_block(ret, instr, rank, size_of_rank_dim);
    }
    ret.rank = rank;
    ret.size = size_of_rank_dim;
    ret._reshapable = is_reshapeable(ret.getAllInstr());
    assert(ret.validation());
    return Block(std::move(ret));
}

pair<vector<const LoopB *>, uint64_t> find_threaded_blocks(const LoopB &block) {
    pair<vector<const LoopB*>, uint64_t> ret;

    // We should search in 'this' block and its sub-blocks
    vector<const LoopB *> block_list = {&block};
    block.getAllSubBlocks(block_list);

    // Find threaded blocks
    constexpr int MAX_NUM_OF_THREADED_BLOCKS = 3;
    ret.second = 1;
    for (const LoopB *b: block_list) {
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


// Unnamed namespace for all some merge help functions
namespace {
// Check if 'a' and 'b' supports data-parallelism when merged
bool data_parallel_compatible(const InstrPtr a, const InstrPtr b)
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

// Check if loop block 'l1' and 'l2' supports data-parallelism when merged
bool data_parallel_compatible(const LoopB &l1, const LoopB &l2) {
    for (const InstrPtr i1 : l1.getAllInstr()) {
        for (const InstrPtr i2 : l2.getAllInstr()) {
            if (not data_parallel_compatible(i1, i2))
                return false;
        }
    }
    return true;
}

// Check if 'block' accesses the output of a sweep in 'sweeps'
bool sweeps_accessed_by_block(const set<InstrPtr> &sweeps, const LoopB &loop_block) {
    for (InstrPtr instr: sweeps) {
        assert(bh_noperands(instr->opcode) > 0);
        auto bases = loop_block.getAllBases();
        if (bases.find(instr->operand[0].base) != bases.end())
            return true;
    }
    return false;
}
} // Unnamed namespace

bool merge_possible(const Block &b1, const Block &b2) {
    return merge_if_possible(b1, b2).second;
}

std::pair<Block, bool> merge_if_possible(const Block &b1, const Block &b2) {
    if (b1.isInstr() or b2.isInstr()) {
        return make_pair(Block(), false);
    }
    const LoopB &l1 = b1.getLoop();
    const LoopB &l2 = b2.getLoop();

    // First we check for data incompatibility
    if (not data_parallel_compatible(l1, l2) or sweeps_accessed_by_block(l1._sweeps, l2)) {
        return make_pair(Block(), false);
    }
    // Check for perfect match, which is directly mergeable
    if (l1.size == l2.size) {
        return make_pair(Block(merge(l1, l2)), true);
    }

    // System-only blocks are very flexible because they array sizes does not have to match when reshaping
    // thus we can simply append system instructions without further checks.
    if (l2.isSystemOnly()) {
        LoopB block(l1);
        for (const InstrPtr instr: l2.getAllInstr()) {
            if (bh_noperands(instr->opcode) > 0) {
                block.insert_system_after(instr, instr->operand[0].base);
            }
        }
        return make_pair(Block(std::move(block)), true);
    }

    // Check fusibility of reshapable blocks
    if (l2._reshapable && l2.size % l1.size == 0) {
        vector<InstrPtr> cur_instr = l1.getAllInstr();
        vector<InstrPtr> it_instr = l2.getAllInstr();
        cur_instr.insert(cur_instr.end(), it_instr.begin(), it_instr.end());
        return make_pair(create_nested_block(cur_instr, l2.rank, l1.size), true);
    }
    if (l1._reshapable && l1.size % l2.size == 0) {
        vector<InstrPtr> cur_instr = l1.getAllInstr();
        vector<InstrPtr> it_instr = l2.getAllInstr();
        cur_instr.insert(cur_instr.end(), it_instr.begin(), it_instr.end());
        return make_pair(create_nested_block(cur_instr, l1.rank, l2.size), true);
    }
    return make_pair(Block(), false);
}

ostream &operator<<(ostream &out, const LoopB &b) {
    out << b.pprint();
    return out;
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
