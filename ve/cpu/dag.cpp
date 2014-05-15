#include <fstream>
#include <sstream>
#include <set>
#include "dag.hpp"
#include "utils.hpp"

using namespace std;
using namespace boost;
namespace bohrium{
namespace core {

const char Dag::TAG[] = "Dag";

bool Dag::fusable(tac_t& prev, tac_t& cur)
{
    bool compatible = false;

    if ((prev.op == MAP) || (prev.op == ZIP) || (prev.op == SYSTEM)) {
        compatible = true;
    }

    // Check shapes and dependencies

    return compatible;
}

/**
 *  Construct a list of subgraphs...
 */
void Dag::partition(void)
{
    DEBUG(TAG,"partition(...)");

    int64_t graph_idx=0;

    subgraphs_.push_back(&(graph_.create_subgraph()));    // Create the first subgraph
    tac_t* prev = &(tacs_[0]);
    add_vertex(0, *(subgraphs_[graph_idx]));            // Add the first instruction

    for(int64_t idx=1; idx < ninstr_; ++idx) {    // Then look at the remaining
        tac_t* cur = &(tacs_[idx]);

        if (!fusable(*prev, *cur)) {
            subgraphs_.push_back(&(graph_.create_subgraph()));
            ++graph_idx;
        }
        add_vertex(idx, *(subgraphs_[graph_idx]));

        prev = cur;
    }

    DEBUG(TAG,"partition(...);");
}

void Dag::array_deps(void)
{
    DEBUG(TAG,"array_deps(...)");
    //
    // Find dependencies on array operations
    for(int64_t idx=0; idx < ninstr_; ++idx) {
        // The instruction to find data-dependencies for
        tac_t& tac = tacs_[idx];

        // Ignore sys-ops
        if ((tac.op == SYSTEM) || (NOOP == tac.op)) {
            DEBUG(TAG, "Ignoring system...");
            continue;
        }

        // Bases associated with the instruction
        bh_base* output = symbol_table_[tac.out].base;

        bool found = false;
        for(int64_t other=idx+1; (other<ninstr_) && (!found); ++other) {
            tac_t& other_tac = tacs_[other];

            // Ignore sys and noops
            if ((other_tac.op == SYSTEM) || (NOOP == other_tac.op)) {
                DEBUG(TAG, "Ignoring system...inside...");
                continue;
            }

            // Search operands of other instruction
            switch(tac_noperands(other_tac)) {
                case 3:
                    DEBUG(TAG, "Comparing" << symbol_table_[other_tac.in2].base << " == " << output);
                    if (symbol_table_[other_tac.in2].layout != CONSTANT) {
                        if (symbol_table_[other_tac.in2].base == output) {
                            found = true;
                            add_edge(idx, other, graph_);
                            break;
                        }
                    }
                case 2:
                    DEBUG(TAG, "Comparing" << symbol_table_[other_tac.in1].base << " == " << output);
                    if (symbol_table_[other_tac.in1].layout != CONSTANT) {
                        if (symbol_table_[other_tac.in1].base == output) {
                            found = true;
                            add_edge(idx, other, graph_);
                            break;
                        }
                    }
                case 1:
                    DEBUG(TAG, "Comparing" << symbol_table_[other_tac.out].base << " == " << output);
                    if (symbol_table_[other_tac.out].base == output) {
                        found = true;
                        add_edge(idx, other, graph_);
                        break;
                    }
                default:
                    break;
            }
        }
        DEBUG(TAG, "Found=" << found << ".");
    }
    DEBUG(TAG,"array_deps(...);");
}

void Dag::system_deps(void)
{
    DEBUG(TAG,"system_deps(...)");
    //
    // Find dependencies on system operations
    for(int64_t idx=ninstr_-1; idx>=0; --idx) {

        // The tac to find data-dependencies for
        tac_t& tac = tacs_[idx];

        if (tac.op != SYSTEM) {
            continue;
        }

        bh_base* output = symbol_table_[tac.out].base;

        bool found = false;
        for(int64_t other=idx-1; (other>=0) && (!found); --other) {
            tac_t& other_tac = tacs_[other];

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
    DEBUG(TAG,"system_deps(...);");
}

}}

