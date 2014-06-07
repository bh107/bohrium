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

/*
void bottomup(Graph& graph, Graph& subgraph,
                        vector<tac_t>& program, SymbolTable& symbol_table,
                        uint32_t& omask,
                        vector<bool>& visited,
                        size_t r_idx, size_t p_idx, size_t v_idx)
{
    if (!visited[v_idx]) {
        size_t n_in  = in_degree(v_idx, graph);
        size_t n_out = out_degree(v_idx, graph);

        bool first      = (r_idx == v_idx);
        bool last       = (n_out == 0);
        bool inbetween  = !(first || last);

        tac_t& tac = program[v_idx];

        if ((n_in > 1) && inbetween) {  // Cannot fuse something with multiple deps
            return;
        }
                                        // Check the operations
        if (!first) {
            if ((program[v_idx].op & NON_FUSABLE)>0) {
                return;
            }
            if ((program[p_idx].op & NON_FUSABLE)>0) {
                return;
            }
         }

        visited[v_idx] = true;
        add_vertex(v_idx, subgraph);
        omask |= program[v_idx].op;
        
        // Visit children
        pair<out_edge_iter, out_edge_iter> oeip = out_edges(v_idx, graph);
        for(out_edge_iter oei = oeip.first; oei != oeip.second; ++oei) {
            bottomup(
                graph, subgraph,
                program, symbol_table,
                omask,
                visited, r_idx, v_idx, target(*oei, graph)
            );
        }
    }
}

void up(Graph& graph, Graph& subgraph,
                        vector<tac_t>& program, SymbolTable& symbol_table,
                        uint32_t& omask,
                        vector<bool>& visited,
                        size_t r_idx, size_t p_idx, size_t v_idx)
{
    tac_t& tac = program[v_idx];
    in_edges(v_idx, graph);

}

void down(Graph& graph, Graph& subgraph,
                        vector<tac_t>& program, SymbolTable& symbol_table,
                        uint32_t& omask,
                        vector<bool>& visited,
                        size_t r_idx, size_t p_idx, size_t v_idx)
{
}

void topdown(Graph& graph, Graph& subgraph,
                        vector<tac_t>& program, SymbolTable& symbol_table,
                        uint32_t& omask,
                        vector<bool>& visited,
                        size_t r_idx, size_t p_idx, size_t v_idx, multimap<bh_base*, size_t>& operands)
{
    if (!visited[v_idx]) {
        size_t n_in  = in_degree(v_idx, graph);
        size_t n_out = out_degree(v_idx, graph);

        bool first      = (r_idx == v_idx);
        bool last       = (n_out == 0);
        bool inbetween  = !(first || last);

        tac_t& tac = program[v_idx];

        if ((n_in > 1) && inbetween) {  // Cannot fuse something with multiple deps
            return;
        }
                                        // Cannot fuse anything but element-wise ops
        if (!first) {

            if ((program[p_idx].op & NON_FUSABLE)>0) {
                return;
            }

            if ((program[v_idx].op & NON_FUSABLE)>0) {
                return;
            }

            if ((program[v_idx].op & ARRAY_OPS)>0) {
                bh_base* base = symbol_table[tac.out].base;
                for(multimap<bh_base*, size_t>::iterator it=operands.find(base); it!=operands.end(); ++it) {
                    if (!equivalent(symbol_table[(*it).second], symbol_table[tac.out])) {
                        return;
                    }
                }
            }
        }

        // Check operands
        visited[v_idx] = true;
        add_vertex(v_idx, subgraph);
        omask |= program[v_idx].op;

        switch(tac_noperands(tac)) {
            case 3:
                if ((symbol_table[tac.in2].layout & SCALAR_LAYOUT) >0) {
                    operands.insert(pair<bh_base*, size_t>(symbol_table[tac.in2].base, tac.in2));
                }
            case 2:
                if ((symbol_table[tac.in1].layout & SCALAR_LAYOUT) >0) {
                    operands.insert(pair<bh_base*, size_t>(symbol_table[tac.in1].base, tac.in1));
                }
            case 1:
                operands.insert(pair<bh_base*, size_t>(symbol_table[tac.out].base, tac.out));
                break;
        }

        // Add operands
        
        // Visit children
        pair<out_edge_iter, out_edge_iter> oeip = out_edges(v_idx, graph);
        for(out_edge_iter oei = oeip.first; oei != oeip.second; ++oei) {
            topdown(
                graph, subgraph,
                program, symbol_table,
                omask,
                visited, r_idx, v_idx, target(*oei, graph), operands
            );
        }
    }
}

void partitioned(Graph& graph, vector<Graph*>& subgraphs, vector<uint32_t>& omasks, vector<tac_t>& program, SymbolTable& symbol_table)
{
    size_t nsubs =0;
    vector<bool> visited(program.size());
    std::pair<vertex_iter, vertex_iter> svp = vertices(graph);
    for(vertex_iter vi = svp.first; vi != svp.second; ++vi) {
        if (!visited[*vi]) {
            Graph* subgraph = &(graph.create_subgraph());
            multimap<bh_base*, size_t> operands;
            subgraphs.push_back(subgraph);
            uint32_t omask = 0;
            topdown(graph, *subgraph, program, symbol_table,
                        omask,visited, *vi, *vi, *vi, operands);
            omasks[nsubs] = omask;
            nsubs++;
        }
    }
}



bool fusable_first(SymbolTable& symbol_table, tac_t& cur, tac_t& prev)
{
    // But only map and zip array operations
    if (!((cur.op == MAP) || (cur.op == ZIP) || (cur.op == SYSTEM))) {
        return false;
    }

    // But only map and zip array operations
    if (!((prev.op == MAP) || (prev.op == ZIP) || (prev.op == SYSTEM))) {
        return false;
    }

    //
    // Check for compatible operands
    bool compat_operands = true;
    bool conflicting = false;

    // Do conflict checks with all operands!

    operand_t cur_op = symbol_table[cur.out];
    switch(core::tac_noperands(prev)) {
        case 3:
            // Second input
            compat_operands = compat_operands && (core::compatible(
                symbol_table[prev.in2],
                cur_op
            ));
            conflicting = conflicting || \
                            ((cur_op.base    == symbol_table[prev.in2].base) && \
                            (cur_op.start   != symbol_table[prev.in2].start));
        case 2:
            // First input
            compat_operands = compat_operands && (core::compatible(
                symbol_table[prev.in1],
                cur_op
            ));
            conflicting = conflicting || \
                            ((cur_op.base    == symbol_table[prev.in1].base) && \
                            (cur_op.start   != symbol_table[prev.in1].start));

            // Output operand
            compat_operands = compat_operands && (core::compatible(
                symbol_table[prev.out],
                cur_op
            ));
            conflicting = conflicting || \
                            ((cur_op.base    == symbol_table[prev.out].base) && \
                            (cur_op.start   != symbol_table[prev.out].start));
        default:
            break;
    }

    return compat_operands && (!conflicting);
}

void part_first(Graph& graph, vector<Graph*>& subgraphs, vector<uint32_t>& omasks, vector<tac_t>& program, SymbolTable& symbol_table)
{
    int64_t graph_idx=0;

    for(int64_t idx=0; idx < program.size();) {    // Then look at the remaining

        // Look only at sequences of element-wise and system operations
        int64_t sub_begin   = idx,
                sub_end     = idx;
        tac_t* prev = &program[idx];
        for(int64_t inner=idx; inner <program.size(); ++inner) {
            tac_t* cur = &program[inner];

            if (!fusable_first(symbol_table, *cur, *prev)) {
                break;
            }
            
            sub_end = inner;
        }

        // Stuff them into a subgraph
        subgraphs.push_back(&(graph.create_subgraph()));
        // Annotate operation mask for the subgraph
        omasks.push_back(0);
        for(int64_t sub_idx=sub_begin; sub_idx<=sub_end; ++sub_idx) {
            add_vertex(sub_idx, *subgraphs[graph_idx]);
            omasks[graph_idx] |= program[sub_idx].op;
        }
        graph_idx++;
        idx = sub_end+1;
    }
}


*/

void sequential_dependencies(vector<Graph*>& subgraphs)
{
    for(vector<Graph*>::iterator gi=subgraphs.begin(); gi!=subgraphs.end(); ++gi) {
        for(size_t idx=1; idx < num_vertices(**gi); ++idx) {
            add_edge(idx-1, idx, **gi);
        }
    }
}

void trivial_partition(Graph& graph, vector<Graph*>& subgraphs, vector<uint32_t>& omasks, vector<tac_t>& program, SymbolTable& symbol_table) {
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

void greedy_partition(Graph& graph, vector<Graph*>& subgraphs, vector<uint32_t>& omasks, vector<tac_t>& program, SymbolTable& symbol_table)
{
    vector<multimap<bh_base*, size_t>* > operands;
    subgraphs.push_back(&(graph.create_subgraph()));
    operands.push_back(new multimap<bh_base*, size_t>());
    omasks.push_back(0);
    size_t graph_idx = 0;


    const char TAG[] = "greedy_patition";

    for(size_t idx=0; idx < program.size(); ++idx) {
        tac_t& tac = program[idx];

        if (((tac.op & NON_FUSABLE)>0) || ((omasks[graph_idx] & NON_FUSABLE) > 0)) {
            goto new_subgraph;
        } else {
            // Check for compatibility and conflicts
            if ((tac.op & ARRAY_OPS)>0) {
                bh_base* base = symbol_table[tac.out].base;
                //for(multimap<bh_base*, size_t>::iterator it=operands[graph_idx]->find(base); it!=operands[graph_idx]->end(); ++it) {
                for(multimap<bh_base*, size_t>::iterator it=operands[graph_idx]->begin(); it!=operands[graph_idx]->end(); ++it) {
                    if (!compatible(symbol_table[(*it).second], symbol_table[tac.out])) {
                        DEBUG(TAG, "Incompatible: " << tac.out << " and " <<  (*it).second << ".");
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
                operands.push_back(new multimap<bh_base*, size_t>());
                omasks.push_back(0);
                graph_idx = subgraphs.size()-1;
            }

        add_instruction:
            add_vertex(idx, *subgraphs[graph_idx]);     // Add the tac / vertex
            omasks[graph_idx] |= tac.op;                // Notate the operation-mask
            switch(tac_noperands(tac)) {                // Add the operands
                case 3:
                    if ((symbol_table[tac.in2].layout & SCALAR_CONST) == 0) {
                        operands[graph_idx]->insert(pair<bh_base*, size_t>(symbol_table[tac.in2].base, tac.in2));
                    }
                case 2:
                    if ((symbol_table[tac.in1].layout & SCALAR_CONST) == 0) {
                        operands[graph_idx]->insert(pair<bh_base*, size_t>(symbol_table[tac.in1].base, tac.in1));
                    }
                case 1:
                    operands[graph_idx]->insert(pair<bh_base*, size_t>(symbol_table[tac.out].base, tac.out));
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
