#include <fstream>
#include <sstream>
#include <set>
#include "dag.hpp"
#include "utils.hpp"
#include "boost/graph/depth_first_search.hpp"
#include "boost/graph/visitors.hpp"

using namespace std;
using namespace boost;
namespace bohrium{
namespace core {

const char Dag::TAG[] = "Dag";

struct partitioner : public base_visitor<partitioner> {
    partitioner(vector<tac_t>& program, vector<Graph*>& subgraphs) : program_(program), subgraphs_(subgraphs) {}
    typedef on_discover_vertex event_filter;

    template <class T, class Graph>
    void operator()(T t, Graph& g) {
        cout << t << endl;
    }

    vector<tac_t>& program_;
    vector<Graph*>& subgraphs_;
};

uint32_t Dag::omask(size_t subgraph_idx)
{
    return omask_[subgraph_idx];
}

void Dag::array_deps(void)
{
    //
    // Find dependencies on array operations
    for(int64_t idx=0; idx < program_.size(); ++idx) {
        // The instruction to find data-dependencies for
        tac_t& tac = program_[idx];

        // Ignore sys-ops
        if ((tac.op == SYSTEM) || (NOOP == tac.op)) {
            continue;
        }

        // Bases associated with the instruction
        bh_base* output = symbol_table_[tac.out].base;

        bool found = false;
        for(int64_t other=idx+1; (other<program_.size()) && (!found); ++other) {
            tac_t& other_tac = program_[other];

            // Ignore sys and noops
            if ((other_tac.op == SYSTEM) || (NOOP == other_tac.op)) {
                continue;
            }

            // Search operands of other instruction
            switch(tac_noperands(other_tac)) {
                case 3:
                    if (symbol_table_[other_tac.in2].layout != SCALAR_CONST) {
                        if (symbol_table_[other_tac.in2].base == output) {
                            found = true;
                            add_edge(idx, other, graph_);
                            break;
                        }
                    }
                case 2:
                    if (symbol_table_[other_tac.in1].layout != SCALAR_CONST) {
                        if (symbol_table_[other_tac.in1].base == output) {
                            found = true;
                            add_edge(idx, other, graph_);
                            break;
                        }
                    }
                case 1:
                    if (symbol_table_[other_tac.out].base == output) {
                        found = true;
                        add_edge(idx, other, graph_);
                        break;
                    }
                default:
                    break;
            }
        }
    }
}

void Dag::system_deps(void)
{
    //
    // Find dependencies on system operations
    for(int64_t idx=program_.size()-1; idx>=0; --idx) {

        // The tac to find data-dependencies for
        tac_t& tac = program_[idx];

        if (tac.op != SYSTEM) {
            continue;
        }

        bh_base* output = symbol_table_[tac.out].base;

        bool found = false;
        for(int64_t other=idx-1; (other>=0) && (!found); --other) {
            tac_t& other_tac = program_[other];

            switch(tac_noperands(other_tac)) {
                case 3:
                    if (symbol_table_[other_tac.in2].base == output) {
                        found = true;
                        add_edge(other, idx, graph_);
                        break;
                    }
                case 2:
                    if (symbol_table_[other_tac.in1].base == output) {
                        found = true;
                        add_edge(other, idx, graph_);
                        break;
                    }
                case 1:
                    if (symbol_table_[other_tac.out].base == output) {
                        found = true;
                        add_edge(other, idx, graph_);
                        break;
                    }
                default:
                    break;
            }
        }
    }
}

}}

