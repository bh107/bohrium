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
#pragma once

/** In this file we declare the bohrium::jitk::iterator namespace, which is a collection of use full iterators **/

#include <boost/iterator/iterator_facade.hpp>
#include <bohrium/jitk/block.hpp>


namespace bohrium {
namespace jitk {
namespace iterator {

/// An iterator over instruction in a block list
class BlockList
        : public boost::iterator_facade<BlockList, const bohrium::jitk::InstrPtr, boost::forward_traversal_tag> {
private:
    friend class boost::iterator_core_access;

    /** Design Overview
     *  The state of the iterator is based on the `_stack`, which consist of pairs of a block_list and a block pointer.
     *  Valid states:
     *    - `_stack` is empty: the iterator is at the end
     *    - `_stack` is not empty: the last item in `_stack` points to the block that are the current instruction.
     *      The block_list is nullptr
     */

    // Notice, we use a static vector with maximum size of `BH_MAXDIM+2` to support all possible ranks plus rank -1
    // and the rank of the bottom level that consist of a single instruction.
    boost::container::static_vector<std::pair<const std::vector<bohrium::jitk::Block> *, const bohrium::jitk::Block *>,
            BH_MAXDIM + 2> _stack;

    // Go to the bottom of the `block_list` -- pushing each traversed level on the `_stack`
    void _bottom(const std::vector<bohrium::jitk::Block> &block_list);

    // Increase iterator
    void increment();

    // Iterator equality
    bool equal(BlockList const &other) const;

    // Deference the iterator
    bohrium::jitk::InstrPtr const &dereference() const;

public:
    BlockList() = default;

    explicit BlockList(const std::vector<bohrium::jitk::Block> &block_list);

    explicit BlockList(const bohrium::jitk::Block &block);

    typedef boost::iterator_range<BlockList> Range;
};


/// Return a range of all instructions in the loop (incl. nested loops)
BlockList::Range allInstr(const bohrium::jitk::LoopB &loop);

/// Return a range of all instructions in the Block (incl. nested Blocks)
BlockList::Range allInstr(const bohrium::jitk::Block &block);


/// An iterator over bases in an instruction (skipping duplicates)
class BaseList
        : public boost::iterator_facade<BaseList, const bh_base *, boost::forward_traversal_tag, const bh_base *> {
private:
    friend class boost::iterator_core_access;

    std::vector<bh_view>::const_iterator cur, begin, end;

    // Iterate `it` to the next instruction
    inline void _next(std::vector<bh_view>::const_iterator &it) {
        assert(it != end);
        while (++it != end and it->isConstant()) {}
    }

    // Increase iterator
    void increment() {
        if (cur != end) {
            _next(cur);
            if (cur != end) {
                for (auto it = begin; it != cur; ++it) {
                    // In order to avoid duplicates, we call increment() again if `cur->base` has already been iterated
                    if ((not it->isConstant()) and it->base == cur->base) {
                        increment();
                        break;
                    }
                }
            }
        }
    }

    // Iterator equality
    bool equal(BaseList const &other) const {
        return cur == other.cur;
    }

    // Deference the iterator
    const bh_base *dereference() const {
        assert(cur != end);
        assert(not cur->isConstant());
        return cur->base;
    }

public:

    /// Construct an "end" pointer
    explicit BaseList(std::vector<bh_view>::const_iterator end) : cur(end), begin(end), end(end) {}

    /// Construct based on an instruction list
    explicit BaseList(const std::vector<bh_view> &view_list) : cur(view_list.begin()), begin(view_list.begin()),
                                                               end(view_list.end()) {
        // If the first item is a constant, we iterate to the first non-constant
        if ((not view_list.empty()) and view_list.front().isConstant()) {
            increment();
        }
    }

    typedef boost::iterator_range<BaseList> Range;
};

/// Return a range of all bases in the instruction (skipping duplicates)
BaseList::Range allBases(const bh_instruction &instr);


/// An iterator over local instruction in a block list
class LocalInstrList : public boost::iterator_facade<LocalInstrList, const bohrium::jitk::InstrPtr, boost::forward_traversal_tag> {
private:
    friend class boost::iterator_core_access;

    std::vector<Block>::const_iterator cur, end;

    // Increase iterator
    void increment() {
        assert(cur != end);
        while(++cur != end and not cur->isInstr()) {}
    }

    // Iterator equality
    bool equal(LocalInstrList const &other) const {
        return cur == other.cur;
    }

    // Deference the iterator
    InstrPtr const &dereference() const {
        assert(cur != end);
        assert(cur->isInstr());
        return cur->getInstr();
    }

public:
    /// Construct an "end" pointer
    explicit LocalInstrList(std::vector<Block>::const_iterator end) : cur(end), end(end) {}

    /// Construct based on a block
    explicit LocalInstrList(const std::vector<Block> &block_list) : cur(block_list.begin()), end(block_list.end()) {
        if (cur != end) {
            if (not cur->isInstr()) {
                increment(); // Get to the first instruction
            }
        }
    }

    typedef boost::iterator_range<LocalInstrList> Range;
};

/// Return a range of all local instructions in the block list
LocalInstrList::Range allLocalInstr(const std::vector<Block> &block_list);

/// Return a range of all local instructions in the loop
LocalInstrList::Range allLocalInstr(const LoopB &loop);

} // iterator
} // jitk
} // bohrium
