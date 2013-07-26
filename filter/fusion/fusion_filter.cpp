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
#include <string>
#include <vector>
#include <set>
#include <bh.h>

#define NODE_LOOKUP(x) (((bh_graph_node*)bhir->nodes->data)[(x)])
#define LEFT_C(x) (((bh_graph_node*)bhir->nodes->data)[(x)].left_child)
#define NODE_LOOKUP(x) (((bh_graph_node*)bhir->nodes->data)[(x)])
#define INSTRUCTION_LOOKUP(x) (((bh_instruction*)bhir->instructions->data)[(x)])

using namespace std;

// Assumes that the given node is valid
bool only_free(bh_ir* bhir, bh_node_index idx)
{
    bh_node_index left_c    = NODE_LOOKUP(idx).left_child;
    bh_node_index right_c   = NODE_LOOKUP(idx).right_child;

    switch(NODE_LOOKUP(idx).type) {
        case BH_COLLECTION:             // Go deeper
            return (((left_c  == INVALID_NODE) || (only_free(bhir, left_c))) && \
                    ((right_c == INVALID_NODE) || (only_free(bhir, right_c))));
        case BH_INSTRUCTION:            // If am a free instruction then we are happy
            return INSTRUCTION_LOOKUP(NODE_LOOKUP(idx).instruction).opcode == BH_FREE;
        default:
            return false;
    }
}

/**
 *  Search the graph for reductions.
 *  Assumes that given node is valid and not previously visited.
 */
void find_fusion(bh_ir* bhir, bh_node_index idx,
                    set<bh_node_index> &hits, vector<bool> &visited)
{
    visited[idx] = true;    // Update to avoid revisiting this node.
    bh_node_index left  = NODE_LOOKUP(idx).left_child;
    bh_node_index right = NODE_LOOKUP(idx).right_child;

    bool left_free  = only_free(bhir, left);
    bool right_free = only_free(bhir, right);

    bool able = true;
    if ((NODE_LOOKUP(idx).type == BH_INSTRUCTION)) {
        switch(INSTRUCTION_LOOKUP(NODE_LOOKUP(idx).instruction).opcode) {
            case BH_ADD_REDUCE:
            case BH_MULTIPLY_REDUCE:
            case BH_MINIMUM_REDUCE:
            case BH_MAXIMUM_REDUCE:
            case BH_LOGICAL_AND_REDUCE:
            case BH_LOGICAL_OR_REDUCE:
            case BH_LOGICAL_XOR_REDUCE:
            case BH_BITWISE_AND_REDUCE:
            case BH_BITWISE_OR_REDUCE:
            case BH_BITWISE_XOR_REDUCE:
            case BH_USERFUNC:
            case BH_FREE:
            case BH_SYNC:
            case BH_DISCARD:
                able = false;
                break;
        }
    }

    if (left_free && (!right_free)) {
        hits.insert(idx);
    } else if (right_free && (!left_free)) {
        hits.insert(idx);
    }

    // A collection or an instruction
    if ((left!=INVALID_NODE) && (!visited[left])) {
        find_fusion(bhir, left, hits, visited);
    }
    if ((right!=INVALID_NODE) && (!visited[right])) {
        find_fusion(bhir, right, hits, visited);
    }

}

void fusion_filter(bh_ir* bhir)
{
    set<bh_node_index> hits;
    vector<bool> visited(bhir->nodes->count, false);
    find_fusion(bhir, 0, hits, visited);
    // At this point we know the sub-graph... so what to do now...
}

