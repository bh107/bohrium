#include <sstream>
#include <algorithm>
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

void topdown_visit(Graph& graph, Graph& subgraph,
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
            topdown_visit(
                graph, subgraph,
                program, symbol_table,
                omask,
                visited, r_idx, v_idx, target(*oei, graph)
            );
        }
    }
}



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
            /*            
            // Check for contractable
            if ((program[p_idx].op & ARRAY_OPS>0) && (!symbol_table.is_temp(program[p_idx].out))) {
                return;
            }
            if ((program[v_idx].op & ARRAY_OPS>0) && (!symbol_table.is_temp(program[v_idx].out))) {
                return;
            }*/
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

void partitioned(Graph& graph, vector<Graph*>& subgraphs, vector<uint32_t>& omasks, vector<tac_t>& program, SymbolTable& symbol_table)
{
    size_t nsubs =0;
    vector<bool> visited(program.size());
    std::pair<vertex_iter, vertex_iter> svp = vertices(graph);
    for(vertex_iter vi = svp.first; vi != svp.second; ++vi) {
        if (!visited[*vi]) {
            Graph* subgraph = &(graph.create_subgraph());
            subgraphs.push_back(subgraph);
            uint32_t omask = 0;
            bottomup(graph, *subgraph, program, symbol_table,
                        omask,visited, *vi, *vi, *vi);
            omasks[nsubs] = omask;
            nsubs++;
        }
    }
}

Dag::Dag(SymbolTable& symbol_table, std::vector<tac_t>& program)
    : symbol_table_(symbol_table), program_(program),
      graph_(program.size()), subgraphs_(), omask_(program.size())
{
    array_deps();   // Construct dependencies based on array operations
    system_deps();  // Construct dependencies based on system operations

    partitioned(graph_, subgraphs_, omask_, program, symbol_table);
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
