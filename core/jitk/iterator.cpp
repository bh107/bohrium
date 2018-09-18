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

#include <bh_util.hpp>
#include <jitk/iterator.hpp>


using namespace std;
using namespace bohrium::jitk;

namespace bohrium {
namespace jitk {
namespace iterator {


BlockList::BlockList(const vector<Block> &block_list) {
    if (not block_list.empty()) {
        _bottom(block_list);
    }
}

BlockList::BlockList(const Block &block) {
    if (block.isInstr()) {
        _stack.push_back(make_pair(nullptr, &block));
    } else {
        const vector<Block> &block_list = block.getLoop()._block_list;
        if (not block_list.empty()) {
            _bottom(block_list);
        }
    }
}

void BlockList::_bottom(const vector<Block> &block_list) {
    if (block_list.empty()) {
        throw runtime_error("BlockList::_bottom() - `block_list` is empty!");
    }
    _stack.push_back(make_pair(&block_list, &block_list[0]));
    if (block_list[0].isInstr()) {
        _stack.push_back(make_pair(nullptr, &block_list[0]));
    } else {
        _bottom(block_list[0].getLoop()._block_list);
    }
}

void BlockList::increment() {
    if (_stack.empty()) {
        return;
    }
    auto &block_list = _stack.back().first;
    auto &block = _stack.back().second;
    if (block_list == nullptr) {
        _stack.resize(_stack.size() - 1);
        increment();
    } else {
        // We use a while-loop here for the unlikely case where a child's block_list is empty
        while (true) {
            if (block == &block_list->back()) {
                _stack.resize(_stack.size() - 1);
                increment();
                break;
            } else {
                ++block;
                if (block->isInstr()) {
                    _stack.push_back(make_pair(nullptr, block));
                    break;
                } else {
                    if (not block->getLoop()._block_list.empty()) {
                        _bottom(block->getLoop()._block_list);
                        break;
                    }
                    // If `block->getLoop()._block_list` is empty, we continue to next block item.
                }
            }
        }
    }
}

bool BlockList::equal(BlockList const &other) const {
    return this->_stack == other._stack;
}

InstrPtr const &BlockList::dereference() const {
    auto &block_list = _stack.back().first;
    auto &block = _stack.back().second;
    assert(block_list == nullptr);
    assert(block->isInstr());
    return block->getInstr();
}


BlockList::Range allInstr(const LoopB &loop) {
    BlockList end, begin(loop._block_list);
    auto ret = boost::make_iterator_range(begin, end);
    {
        auto instr_list = loop.getAllInstr();
        size_t i = 0;
        for (const InstrPtr &iter: ret) {
            if (instr_list[i] != iter) {
                cout << "allInstr(const LoopB &loop) not equal: " << endl;
                cout << *iter << endl;
                cout << *instr_list[i] << endl;
                exit(-1);
            }
            i++;
        }
    }
    return ret;
}

BlockList::Range allInstr(const Block &block) {
    BlockList end, begin(block);
    auto ret = boost::make_iterator_range(begin, end);
    {
        auto instr_list = block.getAllInstr();
        size_t i = 0;
        for (const InstrPtr &iter: ret) {
            if (instr_list[i] != iter) {
                cout << "allInstr(const Block &block) not equal: " << endl;
                cout << *iter << endl;
                cout << *instr_list[i] << endl;
                exit(-1);
            }
            i++;
        }
    }
    return ret;
}

BaseList::Range allBases(const bh_instruction &instr) {
    BaseList begin{instr.operand}, end{instr.operand.end()};
    auto ret = boost::make_iterator_range(begin, end);
    {
        set<const bh_base *> _bases;
        for (const bh_view &op: instr.getViews()) {
            _bases.insert(op.base);
        }
        for (const bh_base *b: ret) {
            if (not util::exist(_bases, b)) {
                cout << instr << endl;
                cout << *b << endl;
                assert(1==2);
                exit(-1);
            } else {
                _bases.erase(b);
            }
        }
    }
    return ret;
}


} // iterator
} // jitk
} // bohrium
