#include <sstream>
#include <algorithm>
#include <map>
#include "dag.hpp"
#include "symbol_table.hpp"
#include "utils.hpp"

//
// Mostly boiler-plate code, (de)constructor, getters, etc.
//

using namespace std;
using namespace boost;
namespace bohrium{
namespace core {

void sequential_dependencies(vector<Graph*>& subgraphs)
{
    for(vector<Graph*>::iterator gi=subgraphs.begin(); gi!=subgraphs.end(); ++gi) {
        for(size_t idx=1; idx < num_vertices(**gi); ++idx) {
            add_edge(idx-1, idx, **gi);
        }
    }
}

/**
 *  Construct a subgraph for each instruction.
 */
void trivial_partition(Graph& graph, vector<Graph*>& subgraphs, vector<uint32_t>& omasks, const vector<tac_t>& program, SymbolTable& symbol_table) {
    size_t nsubs =0;
    std::pair<vertex_iter, vertex_iter> svp = vertices(graph);
    for(vertex_iter vi = svp.first; vi != svp.second; ++vi) {
        Graph* subgraph = &(graph.create_subgraph());
        add_vertex(*vi, *subgraph);
        subgraphs.push_back(subgraph);
        omasks[nsubs] = program[*vi].op;
        nsubs++;
    }
}


void greedy_partition(Graph& graph, vector<Graph*>& subgraphs, vector<uint32_t>& omasks, const vector<tac_t>& program, SymbolTable& symbol_table)
{
    vector<vector<size_t>* > operands;
    subgraphs.push_back(&(graph.create_subgraph()));
    operands.push_back(new vector<size_t>());
    omasks.push_back(0);
    size_t graph_idx = 0;

    const char TAG[] = "greedy_patition";

    for(size_t idx=0; idx < program.size(); ++idx) {
        const tac_t& tac = program[idx];

        if (((tac.op & NON_FUSABLE)>0) || ((omasks[graph_idx] & NON_FUSABLE) > 0)) {
            goto new_subgraph;
        } else {
            // Check for compatibility and conflicts
            if ((tac.op & ARRAY_OPS)>0) {
                
                // Compatible with self
                switch(tac_noperands(tac)) {
                    case 3:
                        if (!compatible(symbol_table[tac.in1], symbol_table[tac.in2])) {
                            goto new_subgraph;
                        }
                        if (!compatible(symbol_table[tac.in2], symbol_table[tac.out])) {
                            goto new_subgraph;
                        }
                    case 2:
                        if (!compatible(symbol_table[tac.in1], symbol_table[tac.out])) {
                            goto new_subgraph;
                        }
                    case 1:
                    default:
                        break;
                }

                // Compare with existing operands
                for(vector<size_t>::iterator it=operands[graph_idx]->begin(); it!=operands[graph_idx]->end(); ++it) {
                    if (!compatible(symbol_table[*it], symbol_table[tac.out])) {
                        DEBUG(TAG, "Incompatible: " << tac.out << " and " <<  *it << ".");
                        goto new_subgraph;
                    }
                }
            }

            DEBUG(TAG, "Adding to existing...");
            DEBUG(TAG, tac_text(tac));
            goto add_instruction;
        }
        
        //  Fallout
        //
        //  1. Create a new subgraph and add the vertex to it.
        //  2. Add the vertex to the current subgraph
        //

        new_subgraph:   // Create a new subgraph unless the current is empty
            if (0 != num_vertices(*subgraphs[graph_idx])) {
                subgraphs.push_back(&(graph.create_subgraph()));
                operands.push_back(new vector<size_t>());
                omasks.push_back(0);
                graph_idx = subgraphs.size()-1;
            }

        add_instruction:
            add_vertex(idx, *subgraphs[graph_idx]);     // Add the tac / vertex
            omasks[graph_idx] |= tac.op;                // Notate the operation-mask
            switch(tac_noperands(tac)) {                // Add the operands
                case 3:
                    if ((symbol_table[tac.in2].layout & SCALAR_CONST) == 0) {
                        operands[graph_idx]->push_back(tac.in2);
                    }
                case 2:
                    if ((symbol_table[tac.in1].layout & SCALAR_CONST) == 0) {
                        operands[graph_idx]->push_back(tac.in1);
                    }
                case 1:
                    operands[graph_idx]->push_back(tac.out);
                    break;
            }
    }
}

Dag::Dag(SymbolTable& symbol_table, std::vector<tac_t>& program)
    : symbol_table_(symbol_table), program_(program),
      graph_(program.size()), subgraphs_(), omask_(program.size())
{
    //array_deps();   // Construct dependencies based on array operations
    //system_deps();  // Construct dependencies based on system operations
    
    // These dependencies are for visualization purpososes only...
    sequential_dependencies(subgraphs_);
    //trivial_partition(graph_, subgraphs_, omask_, program, symbol_table);
    greedy_partition(graph_, subgraphs_, omask_, program, symbol_table);
}

Dag::~Dag(void)
{
}

tac_t& Dag::tac(size_t tac_idx)
{
    return program_[tac_idx];
}

vector<Graph*>& Dag::subgraphs(void)
{
    return subgraphs_;
}

}}
