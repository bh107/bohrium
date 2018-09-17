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
#include <jitk/block.hpp>


namespace bohrium {
namespace jitk {
namespace iterator {

/// An iterator over instruction in a block list
class BlockList : public boost::iterator_facade<BlockList, const bohrium::jitk::InstrPtr, boost::forward_traversal_tag> {
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
    boost::container::static_vector<std::pair<const std::vector<bohrium::jitk::Block> *, const bohrium::jitk::Block *>, BH_MAXDIM+2> _stack;

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



} // iterator
} // jitk
} // bohrium
