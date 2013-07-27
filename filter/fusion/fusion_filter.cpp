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

using namespace std;

// Assumes that the given node is valid
bool only_free(bh_ir* bhir, bh_node_index idx)
{
    if (idx == INVALID_NODE) {
        return true;
    }
    switch(NODE_LOOKUP(idx).type) {
        case BH_COLLECTION:             // Go deeper
            return (((LEFT_C(idx) == INVALID_NODE) || (only_free(bhir, LEFT_C(idx)))) && \
                   ((RIGHT_C(idx) == INVALID_NODE) || (only_free(bhir, RIGHT_C(idx)))));
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
void find_fusion(bh_ir* bhir,
                    bh_node_index idx,
                    bh_node_index parent,
                    size_t count,
                    size_t score,
                    vector<bh_node_index> &hits,
                    vector<bool> &visited)
{
    visited[idx] = true;    // Update to avoid revisiting this node.
    bh_node_index left  = LEFT_C(idx);
    bh_node_index right = RIGHT_C(idx);
    bh_node_index other_parent = (LEFT_P(idx) == parent) ? RIGHT_P(idx) : parent;

    if ((only_free(bhir, left) || only_free(bhir, right)) && \
        (other_parent == INVALID_NODE)) {

        if (NODE_LOOKUP(idx).type == BH_INSTRUCTION) {
            ++score;
        }
        ++count;
        parent = idx;
        hits.push_back(idx);

        if (((left!=INVALID_NODE) && (!visited[left])) && \
            (!only_free(bhir, left))) {
            find_fusion(bhir, left, parent, count, score, hits, visited);
        } else if (((right!=INVALID_NODE) && (!visited[right])) && \
                   (!only_free(bhir, right))) {
            find_fusion(bhir, right, parent, count, score, hits, visited);
        } else {
            if (parent != INVALID_NODE) {
                if (score<2) {
                    for(size_t i=0; i<count; ++i) {
                        hits.pop_back();
                    }
                } else {
                    hits.push_back(INVALID_NODE);
                }
                count = 0;
                score = 0;
            }
        }
    } else {
        if (parent != INVALID_NODE) {
            if (score<2) {
                for(size_t i=0; i<count; ++i) {
                    hits.pop_back();
                }
            } else {
                hits.push_back(INVALID_NODE);
            }
            count = 0;
            score = 0;
        }
        parent = INVALID_NODE;
        if ((left!=INVALID_NODE) && (!visited[left])) {
            find_fusion(bhir, left, parent, count, score, hits, visited);
        }
        if ((right!=INVALID_NODE) && (!visited[right])) {
            find_fusion(bhir, right, parent, count, score, hits, visited);
        }
    }
}

void fusion_filter(bh_ir* bhir)
{
    vector<bool> visited(bhir->nodes->count, false);
    vector<bh_node_index> hits;
    cout << "### Fusion filter, searching through " << bhir->nodes->count << " nodes." << endl;
    find_fusion(bhir, 0, INVALID_NODE, 0, 0, hits, visited);
    std::cout << "# found = [" << std::endl << "  ";

    bh_node_index prev = INVALID_NODE;
    for(vector<bh_node_index>::iterator it=hits.begin(); it != hits.end(); ++it) {
        if (*it == INVALID_NODE) {
            cout << endl << "  ";
        } else {
            if (prev != INVALID_NODE) {
                std::cout << ", ";
            }
            std::cout << *it;
        }
        prev = *it;
    }
    std::cout << "]" << std::endl;
    std::cout << "###" << std::endl;
}

