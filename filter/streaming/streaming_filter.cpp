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
#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <bh.h>

#define NODE_LOOKUP(x) (((bh_graph_node*)bhir->nodes->data)[(x)])
#define INSTRUCTION_LOOKUP(x) (((bh_instruction*)bhir->instructions->data)[(x)])

using namespace std;

/**
 *  Ensure that the given node and its children are either:
 *
 *  BH_COLLECTION
 *  or
 *  BH_INSTRUCTION and opcode == FREE
 */
bool fods(bh_ir* bhir, bh_node_index idx)
{
    if (idx==INVALID_NODE) {
        return false;
    }

    switch(NODE_LOOKUP(idx).type) {
        case BH_COLLECTION:
            return fods(bhir, NODE_LOOKUP(idx).left_child) && fods(bhir, NODE_LOOKUP(idx).right_child);
        case BH_INSTRUCTION:
            return INSTRUCTION_LOOKUP(NODE_LOOKUP(idx).instruction).opcode == BH_FREE;
        default:
            return false;
    }
}

/**
 *  Search up the graph for nodes which can be added to the streamable sub-graph.
 */
void up(bh_ir* bhir, bh_node_index idx, bh_node_index child, set<bh_node_index>& inputs)
{
    if (idx==INVALID_NODE) {            // This node is no good
        inputs.insert(child);           // so we use its child
        return;
    }

    bh_node_index left_c    = NODE_LOOKUP(idx).left_child;
    bh_node_index right_c   = NODE_LOOKUP(idx).right_child;
    bh_node_index other_child = (child == left_c) ? right_c : left_c;

    if (!fods(bhir, other_child)) {     // This node is no good
        inputs.insert(child);           // so we use its child
        return;
    }

    // At this point 'idx' is "streamable" so we continue further up
    // up the graph until we reach something which is not "streamable".

    bh_node_index left_p  = NODE_LOOKUP(idx).left_parent;
    bh_node_index right_p = NODE_LOOKUP(idx).right_parent;

    if (left_p != INVALID_NODE) {
        up(bhir, left_p, idx, inputs);
    }
    if (right_p != INVALID_NODE) {
        up(bhir, right_p, idx, inputs);
    }
}

/**
 *  Search the graph for reductions.
 *  Assumes that given node is valid and not previously visited.
 */
void find_reductions(bh_ir* bhir, bh_node_index idx, set<bh_node_index> &hits, vector<bool> &visited)
{
    visited[idx] = true;    // Update to avoid revisiting this node.

    if ((NODE_LOOKUP(idx).type == BH_INSTRUCTION)) {    // Found one
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
                hits.insert(idx);
        }
    }

    bh_node_index left  = NODE_LOOKUP(idx).left_child;  // Go deeper
    bh_node_index right = NODE_LOOKUP(idx).right_child;

    if ((left!=INVALID_NODE) && (!visited[left])) {
        find_reductions(bhir, left, hits, visited);
    }

    if ((right!=INVALID_NODE) && (!visited[right])) {
        find_reductions(bhir, right, hits, visited);
    }
}

void streaming_filter(bh_ir *bhir)
{
    set<bh_node_index> potentials;
    vector<bool> visited(bhir->nodes->count, false);
    cout << "### Streaming filter, searching through " << bhir->nodes->count << " nodes." << endl;
    find_reductions(bhir, 0, potentials, visited);
    cout << "# found = " << potentials.size() << " potentials." << endl;

    for(set<bh_node_index>::iterator output=potentials.begin();
        output != potentials.end();
        ++output) {

        set<bh_node_index> inputs;
        up(bhir, NODE_LOOKUP(*output).left_parent, *output, inputs);
        cout << "## Potential" << endl;
        cout << "# inputs: [";
        bool first = true;
        for(set<bh_node_index>::iterator input=inputs.begin();
             input!=inputs.end();
             ++input) {

            if (!first) {
                cout << ", ";
            }
            first = false;
            cout << *input; 
        }
        cout << "]" << endl;
        cout << "# output: [" << *output << "]" << endl;
        cout << "##" << endl;
    }
    cout << "###" << endl;
}
