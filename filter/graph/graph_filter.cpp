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
#include <bh.h>
#include <fstream>
#include <sstream>
#include <set>
#include "boost/graph/adjacency_list.hpp"
#include "boost/graph/transitive_reduction.hpp"
#include "boost/graph/topological_sort.hpp"
#include <boost/graph/graphviz.hpp>

static uint64_t execcount = 0;

using namespace std;
using namespace boost;

// Underlying graph representation and implementation
typedef adjacency_list_traits<vecS, vecS, directedS> Traits;

// Graph representation
typedef subgraph< adjacency_list<vecS, vecS, directedS,
    property<vertex_color_t, int>, property<edge_index_t, int> > > Graph;

// Iterating over vertices and edges
typedef graph_traits<Graph>::vertex_iterator vertex_iter;
typedef graph_traits<Graph>::edge_iterator edge_iter;

/**
 *  Create a graph with instructions as vertices and edges as data-dependencies.
 *
 *  @param bhir The bhir containing list of instructions.
 *  @return The graph.
 */
Graph construct_graph(bh_ir *bhir)
{
    Graph graph(bhir->instr_list.size());

    cout << bhir->instr_list.size() << endl;
    for(int64_t idx=0; idx<bhir->instr_list.size(); ++idx) {

        // The instruction to find data-dependencies for
        bh_instruction* instr = &bhir->instr_list[idx];

        // Bases associated with the instruction
        set<bh_base*> inputs;
        bh_base* output = NULL;
        for(int64_t op_idx=0; op_idx<bh_operands(instr->opcode); ++op_idx) {

            if (0 == op_idx) {
                output = instr->operand[op_idx].base;
            } else {
                if (!bh_is_constant(&instr->operand[op_idx])) {
                    inputs.insert(instr->operand[op_idx].base);
                }
            }
        }

        //
        // Look for dependencies
        bool found = false;
        for(int64_t other=idx+1; (other<bhir->instr_list.size()) && (!found); ++other) {
            bh_instruction* other_instr = &bhir->instr_list[other];

            // Search operands of other instruction
            int64_t noperands = bh_operands(other_instr->opcode);
            for(int64_t op_idx=0; op_idx<noperands; ++op_idx) {
                // Array operations dependent on the output
                bh_view* other_op   = &other_instr->operand[op_idx];
                bh_base* other_base = other_op->base;

                // System operations
                switch(other_instr->opcode) {
                    case BH_FREE:
                        break;

                    default:
                        if (!bh_is_constant(other_op)) {
                            if (other_base == output) {
                                found = true;
                                add_edge(idx, other, graph);
                                break;
                            }
                        }
                        break;
                }
            }
        }

        found = false;
        for(int64_t other=idx+1; (other<bhir->instr_list.size()) && (!found); ++other) {
            bh_instruction* other_instr = &bhir->instr_list[other];

            // Search operands of other instruction
            int64_t noperands = bh_operands(other_instr->opcode);
            for(int64_t op_idx=0; op_idx<noperands; ++op_idx) {
                // Array operations dependent on the output
                bh_view* other_op   = &other_instr->operand[op_idx];
                bh_base* other_base = other_op->base;

                // System operations
                switch(other_instr->opcode) {
                    case BH_FREE:
                        if(inputs.find(other_base) != inputs.end()) {
                            found = true;
                            add_edge(idx, other, graph);
                        }
                    default:
                        break;
                }
            }
        }
    }
    return graph;
}

/**
 *  Return a textual representation in dot-format of the given Graph.
 *
 *  @param graph The graph to create textual dot-representation for.
 *  @return String with dot-representation of graph.
 */
string generate_dot(Graph& graph, bh_instruction* instr_list)
{
    stringstream ss;
    ss << "digraph {" << endl;
    
    // Vertices
    std::pair<vertex_iter, vertex_iter> vp = vertices(graph);
    for(vertex_iter it = vp.first; it != vp.second; ++it) {
        bh_intp opcode = instr_list[*it].opcode;
        
        stringstream operands;
        for(int64_t op_idx=0; op_idx<bh_operands(opcode); ++op_idx) {
            if (bh_is_constant(&(instr_list[*it].operand[op_idx]))) {
                operands << " K";
            } else {
                operands << " " << instr_list[*it].operand[op_idx].base;
            }
        }

        ss << *it << " [";
        ss << "shape=box ";
        ss << "style=filled,rounded ";
        ss << "fillcolor=\"#CBD5E9\" ";
        ss << "label=\"" << *it << " - ";
        ss << bh_opcode_text(opcode);
        ss << "(";
        ss << operands.str();
        ss << ")";
        ss << "\"";
        ss << "]";
        ss << endl;
    }
    
    // Edges
    std::pair<edge_iter, edge_iter> ep = edges(graph);
    for(edge_iter it = ep.first; it != ep.second; ++it) {
        ss << source(*it, graph) << "->" << target(*it, graph) << ";" <<  endl;
    }
    
    ss << "}" << endl;
    return ss.str();
}

void graph_filter(bh_ir *bhir)
{
    Graph graph = construct_graph(bhir);
    stringstream filename;
    filename << "graph" << execcount << ".dot";

    std::ofstream fout(filename.str());
    fout << generate_dot(graph, bhir->instr_list) << std::endl;

    execcount++;
}



