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

#include <bh_util.hpp>
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
    int64_t rank = instr_list[0]->ndim();
    for (InstrPtr instr: instr_list) {
        if (not instr->reshapable())
            return false;
        if (instr->ndim() != rank)
            return false;
    }
    return true;
}

// Append 'instr' to the loop block 'block'
void add_instr_to_block(LoopB &block, InstrPtr instr, int rank, int64_t size_of_rank_dim) {
    if (instr->ndim() <= rank)
        throw runtime_error("add_instr_to_block() was given an instruction with ndim <= 'rank'");

    // Let's reshape the instruction to match 'size_of_rank_dim'
    if (instr->reshapable() and instr->operand[0].shape[rank] != size_of_rank_dim) {
        instr = reshape_rank(instr, rank, size_of_rank_dim);
    }

    vector<int64_t> shape = instr->shape();

    // Sanity check
    assert(shape.size() > (uint64_t) rank);
    if (shape[rank] != size_of_rank_dim)
        throw runtime_error("create_nested_block() was given an instruction where shape[rank] != size_of_rank_dim");
    const int64_t max_ndim = instr->ndim();
    assert(max_ndim > rank);

    // Create the rest of the dimension as a singleton block
    if (max_ndim > rank + 1) {
        assert(shape.size() > 0);
        vector<InstrPtr> single_instr = {instr};
        block._block_list.push_back(create_nested_block(single_instr, rank + 1, shape[rank + 1]));
    } else { // No more dimensions -- let's write the instruction block
        assert(max_ndim == rank + 1);
        block._block_list.emplace_back(*instr, rank + 1);
    }
    block.metadata_update();
}

} // Anonymous name space

// *** LoopB Methods *** //


int LoopB::replaceInstr(InstrPtr subject, const bh_instruction &replacement) {
    int ret = 0;
    for (Block &b: _block_list) {
        if (b.isInstr()) {
            if (*b.getInstr() == *subject) {
                b.setInstr(replacement);
                ++ret;
            }
        } else {
            ret += b.getLoop().replaceInstr(subject, replacement);
        }
    }
    return ret;
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

vector<const LoopB *> LoopB::getLocalSubBlocks() const {
    vector<const LoopB *> ret;
    for (const Block &b : _block_list) {
        if (not b.isInstr()) {
            ret.push_back(&b.getLoop());
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
                const set<const bh_base*> bases = _block_list[i].getInstr()->get_bases_const();
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
        if (instr->ndim() <= rank) {
            assert(1 == 2);
            return false;
        }
        if (instr->shape()[rank] != size) {
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
    instr_reshaped.reshape_force(block->_block_list[index].getInstr()->shape());
    Block instr_block(instr_reshaped, block->rank+1);
    block->_block_list.insert(block->_block_list.begin()+index+1, instr_block);

    // Let's update the '_free' set
    if (instr->opcode == BH_FREE) {
        block->_frees.insert(instr->operand[0].base);
    }
    assert(validation());
}

uint64_t LoopB::localThreading() const {
    if (_sweeps.size() == 0 and not isSystemOnly()) {
        assert (size >= 0);
        return static_cast<uint64_t>(size);
    }
    return 0;
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

void LoopB::metadata_update() {
    _news.clear();
    _frees.clear();
    _sweeps.clear();
    for (const InstrPtr instr: getLocalInstr()) {
        if (instr->constructor) {
            if (not bh_opcode_is_accumulate(instr->opcode))// TODO: Support array contraction of accumulated output
                _news.insert(instr->operand[0].base);
        }
        if (instr->opcode == BH_FREE) {
            _frees.insert(instr->operand[0].base);
        }
    }
    for (const InstrPtr instr: getAllInstr()) {
        if (instr->sweep_axis() == rank) {
            _sweeps.insert(instr);
        }
    }
    _reshapable = is_reshapeable(getAllInstr());
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


Block create_nested_block(const vector<InstrPtr> &instr_list, int rank) {
    if (instr_list.empty()) {
        throw runtime_error("create_nested_block: 'instr_list' is empty!");
    }
    if (bh_opcode_is_system(instr_list[0]->opcode)) {
        throw runtime_error("create_nested_block: first instruction is a sysop!");
    }
    const int ndim = (int) instr_list[0]->ndim();
    const vector<int64_t> shape = instr_list[0]->shape();
    assert(ndim > rank);

    LoopB ret_loop;
    ret_loop.rank = rank;
    ret_loop.size = shape[rank];
    if (rank == ndim - 1) { // The innermost rank
        ret_loop.rank = ndim-1;
        ret_loop.size = shape[ndim-1];
        for (InstrPtr instr: instr_list) {
            if (bh_opcode_is_system(instr->opcode)) {
                bh_instruction tmp = *instr;
                tmp.reshape_force(shape);
                ret_loop._block_list.emplace_back(tmp, ndim);
            } else {
                ret_loop._block_list.emplace_back(*instr, ndim);
            }
            assert(ret_loop._block_list.back().getInstr()->shape() == shape);
        }
    } else {
        ret_loop._block_list.emplace_back(create_nested_block(instr_list, rank+1));
    }
    ret_loop.metadata_update();
    assert(ret_loop.validation());
    return Block(std::move(ret_loop));
}

Block create_nested_block(const vector<InstrPtr> &instr_list, int rank, int64_t size_of_rank_dim) {
    if (instr_list.empty()) {
        throw runtime_error("create_nested_block: 'instr_list' is empty!");
    }

    // Let's build the nested block from the 'rank' level to the instruction block
    LoopB ret;
    ret.rank = rank;
    ret.size = size_of_rank_dim;
    for (InstrPtr instr: instr_list) {
        add_instr_to_block(ret, instr, rank, size_of_rank_dim);
    }
    assert(ret.validation());
    return Block(std::move(ret));
}

pair<vector<const LoopB *>, uint64_t> util_find_threaded_blocks(const LoopB &block) {
    pair<vector<const LoopB*>, uint64_t> ret;

    // We should search in 'this' block and its sub-blocks
    vector<const LoopB *> block_list = {&block};
    block.getAllSubBlocks(block_list);

    // Find threaded blocks
    constexpr int MAX_NUM_OF_THREADED_BLOCKS = 3;
    ret.second = 1;
    for (const LoopB *b: block_list) {
        if (b->_sweeps.size() > 0) {
            break;
        }
        const uint64_t thds = b->localThreading();
        if (thds > 0) {
            ret.first.push_back(b);
            ret.second *= thds;
        }
        // Multiple blocks or mixing instructions and blocks makes all the following blocks non-threadable
        if (not (b->getLocalSubBlocks().size() == 1 and b->getLocalInstr().size() == 0)) {
            break;
        }
        // Too much threading can be a bad thing e.g. OpenCL only supports three dimensions
        if (ret.first.size() == MAX_NUM_OF_THREADED_BLOCKS) {
            break;
        }
    }
    return ret;
}

// Unnamed namespace for all some merge help functions
namespace {

// Check if 'writer' and 'reader' supports data-parallelism when merged
bool data_parallel_compatible(const bh_view &writer,
                              const bh_view &reader) {

    // Disjoint views or constants are obviously compatible
    if (bh_is_constant(&writer) or bh_is_constant(&reader) or writer.base != reader.base) {
        return true;
    }

    // The views must have the same offset
    if (writer.start != reader.start) {
        return false;
    }

    // Same dimensionally requires same shape and strides
    if (writer.ndim == reader.ndim) {
        // TODO: if the 'reader' never accesses the 'rank' dimension of the 'writer'
        //       the 'reader' is actually allowed to have 0-stride even when the 'writer' does not
        return std::equal(writer.shape, writer.shape + writer.ndim, reader.shape) and \
               std::equal(writer.stride, writer.stride + writer.ndim, reader.stride);;
    }

    // Finally, two equally sized contiguous arrays are also parallel compatible
    return bh_nelements(writer) == bh_nelements(reader) and \
           bh_is_contiguous(&writer) and bh_is_contiguous(&reader);
}

// Check if 'a' and 'b' (in that order) supports data-parallelism when merged
bool data_parallel_compatible(const InstrPtr a, const InstrPtr b) {
    if(bh_opcode_is_system(a->opcode) || bh_opcode_is_system(b->opcode))
        return true;

    // Gather reads its first input in arbitrary order
    if (b->opcode == BH_GATHER) {
        if (a->operand[0].base == b->operand[1].base) {
            return false;
        }
    }

    // Scatter writes in arbitrary order
    if (a->opcode == BH_SCATTER or a->opcode == BH_COND_SCATTER) {

        for(size_t i=0; i<b->operand.size(); ++i) {
            if ((not bh_is_constant(&b->operand[i])) and a->operand[0].base == b->operand[i].base) {
                return false;
            }
        }
    } else if (b->opcode == BH_SCATTER or b->opcode == BH_COND_SCATTER) {
        for(size_t i=0; i<a->operand.size(); ++i) {
            if ((not bh_is_constant(&a->operand[i])) and b->operand[0].base == a->operand[i].base) {
                return false;
            }
        }
    }

    {// The output of 'a' cannot conflict with the input and output of 'b'
        const bh_view &src = a->operand[0];
        for(size_t i=0; i<b->operand.size(); ++i) {
            if (not data_parallel_compatible(src, b->operand[i])) {
                return false;
            }
        }
    }
    {// The output of 'b' cannot conflict with the input and output of 'a'
        const bh_view &src = b->operand[0];
        for(const bh_view &a_op: a->operand) {
            if (not data_parallel_compatible(src, a_op)) {
                return false;
            }
        }
    }
    return true;
}

// Check if 'b1' and 'b2' (in that order) supports data-parallelism when merged
bool data_parallel_compatible(const LoopB &b1, const LoopB &b2) {
    assert(b1.rank == b2.rank);
    for (const InstrPtr i1 : b1.getAllInstr()) {
        for (const InstrPtr i2 : b2.getAllInstr()) {
            if (i1.get() != i2.get()) {
                if (not data_parallel_compatible(i1, i2)) {
                    return false;
                }
            }
        }
    }
    return true;
}

// Check if 'block' accesses the output of a sweep in 'sweeps'
bool sweeps_accessed_by_block(const set<InstrPtr> &sweeps, const LoopB &loop_block) {
    for (InstrPtr instr: sweeps) {
        assert(instr->operand.size() > 0);
        auto bases = loop_block.getAllBases();
        if (bases.find(instr->operand[0].base) != bases.end())
            return true;
    }
    return false;
}

// Reshape 'l1' to match 'size_of_rank_dim' at 'l1.rank'
Block reshape(const LoopB &l1, int64_t size_of_rank_dim) {
    assert(l1._reshapable);
    vector<InstrPtr> instr_list;
    for (const InstrPtr &instr: l1.getAllInstr()) {
        instr_list.push_back(reshape_rank(instr, l1.rank, size_of_rank_dim));
    }
    return create_nested_block(instr_list, l1.rank);
}
} // Unnamed namespace


// Reshape and merges the two loop blocks 'l1' and 'l2' (in that order).
// NB: the loop blocks must be mergeable!
Block reshape_and_merge(const LoopB &l1, const LoopB &l2) {
    // Check for perfect match, which is directly mergeable
    if (l1.size == l2.size) {
        return Block(merge(l1, l2));
    }
    // Let's try to reshape 'l2' to see if it can match the shape of 'l1'
    if (l2._reshapable && l2.size % l1.size == 0) {
        const LoopB new_l2 = reshape(l2, l1.size).getLoop();
        return Block(merge(l1, new_l2));
    }
    // Let's try to reshape 'l1' to see if it can match the shape of 'l2'
    if (l1._reshapable && l1.size % l2.size == 0) {
        const LoopB new_l1 = reshape(l1, l2.size).getLoop();
        return Block(merge(new_l1, l2));
    }
    throw runtime_error("reshape_and_merge: the blocks are not mergeable!");
}


bool mergeable(const Block &b1, const Block &b2, bool avoid_rank0_sweep) {
    if (b1.isInstr() or b2.isInstr()) {
        return false;
    }
    const LoopB &l1 = b1.getLoop();
    const LoopB &l2 = b2.getLoop();

    // System-only blocks are very flexible because they array sizes does not have to match when reshaping.
    if (l2.isSystemOnly()) {
        return true;
    }

    // We might have to avoid fusion when one of the (root) blocks are sweeping
    if (avoid_rank0_sweep and l1.rank == 0 and l2.rank == 0) {
        if ((l1._sweeps.size() > 0) != (l2._sweeps.size() > 0)) {
            return false;
        }
    }

    // If instructions in 'b2' reads the sweep output of 'b1' than we cannot merge them
    if (sweeps_accessed_by_block(l1._sweeps, l2)) {
        return false;
    }

    if (l1.size == l2.size or // Perfect match
        (l2._reshapable && l2.size % l1.size == 0) or // 'l2' is reshapable to match 'l1'
        (l1._reshapable && l1.size % l2.size == 0)) { // 'l1' is reshapable to match 'l2'
        return data_parallel_compatible(l1, l2);
    } else {
        return false;
    }
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
