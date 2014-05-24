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

bool Dag::fusable(tac_t& cur, tac_t& prev)
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
    switch(core::tac_noperands(cur)) {
        case 3:
            // Second input
            compat_operands = compat_operands && (core::compatible(
                symbol_table_[prev.out],
                symbol_table_[cur.in2]
            ));
        case 2:
            // First input
            compat_operands = compat_operands && (core::compatible(
                symbol_table_[prev.out],
                symbol_table_[cur.in1]
            ));

            // Output operand
            compat_operands = compat_operands && (core::compatible(
                symbol_table_[prev.out],
                symbol_table_[cur.out]
            ));
        default:
            break;
    }

    // Check if instructions share an operand and that shared operands are temp.
    bool uses = false;
    switch(tac_noperands(cur)) {
        case 3: // Binary
            if ((prev.out == cur.in2) && \
                (symbol_table_.temp().find(prev.out) != symbol_table_.temp().end())) {
                uses = true;
                break;
            }
        case 2: // Unary
            if ((prev.out == cur.in1) && \
                (symbol_table_.temp().find(prev.out) != symbol_table_.temp().end())) {
                uses = true;
                break;
            }
        case 1: // System
            switch(tac_noperands(prev)) {
                case 3:
                    if (cur.out == prev.in2) {
                        uses = true;
                        break;
                    }
                case 2:
                    if (cur.out == prev.in1) {
                        uses = true;
                        break;
                    }
                case 1:
                    if (cur.out == prev.out) {
                        uses = true;
                        break;
                    }
            }
            break;
    }
    compat_operands = compat_operands && uses;

    return compat_operands;
}

uint32_t Dag::omask(size_t subgraph_idx)
{
    DEBUG(TAG, "Quering omask #" << subgraph_idx << " out of #"<< omask_.size() << " val="<< omask_[subgraph_idx] << ".");
    return omask_[subgraph_idx];
}

/**
 *  Construct a list of subgraphs... annotate the operation-mask of the subgraph.
 */
void Dag::partition(void)
{
    DEBUG(TAG,"partition(...)");

    int64_t graph_idx=0;

    for(int64_t idx=0; idx < program_.size();) {    // Then look at the remaining

        // Look only at sequences of element-wise and system operations
        int64_t sub_begin   = idx,
                sub_end     = idx;
        tac_t* prev = &program_[idx];
        for(int64_t inner=idx; inner <program_.size(); ++inner) {
            tac_t* cur = &program_[inner];

            if (!fusable(*cur, *prev)) {
                break;
            }
            
            sub_end = inner;
        }

        // Stuff them into a subgraph
        subgraphs_.push_back(&(graph_.create_subgraph()));
        // Annotate operation mask for the subgraph
        omask_.push_back(0);
        for(int64_t sub_idx=sub_begin; sub_idx<=sub_end; ++sub_idx) {
            add_vertex(sub_idx, *subgraphs_[graph_idx]);
            omask_[graph_idx] |= program_[sub_idx].op;
        }
        graph_idx++;
        idx = sub_end+1;
    }

    DEBUG(TAG,"partition(...);");
}

void Dag::array_deps(void)
{
    DEBUG(TAG,"array_deps(...)");
    //
    // Find dependencies on array operations
    for(int64_t idx=0; idx < program_.size(); ++idx) {
        // The instruction to find data-dependencies for
        tac_t& tac = program_[idx];

        // Ignore sys-ops
        if ((tac.op == SYSTEM) || (NOOP == tac.op)) {
            DEBUG(TAG, "Ignoring system...");
            continue;
        }

        // Bases associated with the instruction
        bh_base* output = symbol_table_[tac.out].base;

        bool found = false;
        for(int64_t other=idx+1; (other<program_.size()) && (!found); ++other) {
            tac_t& other_tac = program_[other];

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
    DEBUG(TAG,"system_deps(...);");
}

}}

