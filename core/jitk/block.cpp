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
bool is_reshapeable(const vector<bh_instruction *> &instr_list) {
    assert(instr_list.size() > 0);

    // In order to be reshapeable, all instructions must have the same rank and be reshapeable
    int64_t rank = instr_list[0]->max_ndim();
    for (auto instr: instr_list) {
        if (not instr->reshapable())
            return false;
        if (instr->max_ndim() != rank)
            return false;
    }
    return true;
}
} // Anonymous name space

Block create_nested_block(vector<bh_instruction *> &instr_list, int rank, int64_t size_of_rank_dim,
                          const set<bh_instruction *> &news, const set<bh_base *> &temps) {

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
    Block ret;
    set<bh_base *> syncs; // Set of all sync'ed bases
    for (bh_instruction *instr: instr_list) {
        const int64_t max_ndim = instr->max_ndim();
        assert(max_ndim > rank);

        // Create the rest of the dimension as a singleton block
        if (max_ndim > rank + 1) {
            vector<int64_t> shape = instr->dominating_shape();
            assert(shape.size() > 0);
            vector<bh_instruction *> single_instr = {instr};
            ret._block_list.push_back(create_nested_block(single_instr, rank + 1, shape[rank + 1], news));
        } else { // No more dimensions -- let's write the instruction block
            assert(max_ndim == rank + 1);
            ret._block_list.emplace_back(instr, rank + 1);

            // Since 'instr' execute at this 'rank' level, we can calculate news, syncs, frees, and temps.
            if (news.find(instr) != news.end()) {
                if (not bh_opcode_is_accumulate(instr->opcode))// TODO: Support array contraction of accumulated output
                    ret._news.insert(instr->operand[0].base);
            }
            if (instr->opcode == BH_SYNC) {
                syncs.insert(instr->operand[0].base);
            } else if (instr->opcode == BH_FREE) {
                if (syncs.find(instr->operand[0].base) == syncs.end()) {
                    // If the array is free'ed and not sync'ed, it can be destroyed
                    ret._frees.insert(instr->operand[0].base);
                }
            }
        }
        if (sweep_axis(*instr) == rank) {
            ret._sweeps.insert(instr);
        }
    }
    ret.rank = rank;
    ret.size = size_of_rank_dim;
    ret._reshapable = is_reshapeable(ret.getAllInstr());
    assert(ret.validation());
    return ret;
}

Block *Block::findInstrBlock(const bh_instruction *instr) {
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

string Block::pprint() const {
    stringstream ss;
    if (isInstr()) {
        if (_instr != NULL) {
            spaces(ss, rank * 4);
            ss << *_instr << endl;
        }
    } else {
        spaces(ss, rank * 4);
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
            ss << ", block list:" << endl;
            for (const Block &b : _block_list) {
                ss << b.pprint();
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
vector<const Block *> Block::getAllSubBlocks() const {
    vector<const Block *> ret;
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
    return true;
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
