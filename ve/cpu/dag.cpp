#include <fstream>
#include <sstream>
#include <set>
#include "dag.hpp"
#include "utils.hpp"

using namespace std;
using namespace boost;
namespace bohrium{
namespace core {
namespace dag {

const char Dag::TAG[] = "Dag";

Dag::Dag(bh_ir* bhir) : _bhir(bhir), _dag(bhir->ninstr), _subgraphs()
{
    DEBUG(TAG,"Dag(...)");
    array_deps();
    system_deps();
    partition();
    DEBUG(TAG,"Dag(...);");
}

Dag::~Dag(void)
{
}

bool Dag::fusable(bh_instruction* prev, bh_instruction* cur)
{
    bool compatible = false;

    switch(prev->opcode) {               // Verify the opcode
        case BH_RANDOM:
        case BH_RANGE:
        case BH_ADD_ACCUMULATE:
        case BH_MULTIPLY_ACCUMULATE:
        case BH_ADD_REDUCE:
        case BH_MULTIPLY_REDUCE:
        case BH_MINIMUM_REDUCE:
        case BH_MAXIMUM_REDUCE:
        case BH_LOGICAL_AND_REDUCE:
        case BH_BITWISE_AND_REDUCE:
        case BH_LOGICAL_OR_REDUCE:
        case BH_BITWISE_OR_REDUCE:
        case BH_LOGICAL_XOR_REDUCE:
        case BH_BITWISE_XOR_REDUCE:
            return compatible;
        default:
            break;
    }

    // Check shapes and dependencies

    compatible = true;
    return compatible;
}

/**
 *  Construct a list of subgraphs...
 */
void Dag::partition(void)
{
    DEBUG(TAG,"partition(...)");

    int64_t graph_idx=0;

    _subgraphs.push_back(&(_dag.create_subgraph()));    // Create the first subgraph
    bh_instruction* prev = &(_bhir->instr_list[0]);
    add_vertex(0, *(_subgraphs[graph_idx]));            // Add the first instruction

    for(int64_t idx=1; idx < _bhir->ninstr; ++idx) {    // Then look at the remaining
        bh_instruction* cur = &(_bhir->instr_list[idx]);

        if (!fusable(prev, cur)) {
            _subgraphs.push_back(&(_dag.create_subgraph()));
            ++graph_idx;
        }
        add_vertex(idx, *(_subgraphs[graph_idx]));

        prev = cur;
    }

    DEBUG(TAG,"partition(...);");
}

void Dag::array_deps(void)
{
    DEBUG(TAG,"array_deps(...)");
    //
    // Find dependencies on array operations
    for(int64_t idx=0; idx < _bhir->ninstr; ++idx) {

        // The instruction to find data-dependencies for
        bh_instruction* instr = &_bhir->instr_list[idx];

        // Ignore these
        if (((instr->opcode == BH_FREE)     ||  \
             (instr->opcode == BH_DISCARD)  ||  \
             (instr->opcode == BH_SYNC)     ||  \
             (instr->opcode == BH_NONE))) {
            continue;
        }

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

        bool found = false;
        for(int64_t other=idx+1; (other<_bhir->ninstr) && (!found); ++other) {
            bh_instruction* other_instr = &_bhir->instr_list[other];

            // Search operands of other instruction
            int64_t noperands = bh_operands(other_instr->opcode);
            for(int64_t op_idx=0; (op_idx<noperands) && (!found); ++op_idx) {
                bh_view* other_op   = &other_instr->operand[op_idx];
                bh_base* other_base = other_op->base;

                // System operations
                switch(other_instr->opcode) {
                    case BH_FREE:
                    case BH_SYNC:
                    case BH_DISCARD:
                    case BH_NONE:
                        break;

                    default:
                        if (!bh_is_constant(other_op)) {
                            if (other_base == output) {
                                found = true;
                                add_edge(idx, other, _dag);
                                break;
                            }
                        }
                        break;
                }
            }
        }
    }
    DEBUG(TAG,"array_deps(...);");
}

void Dag::system_deps(void)
{
    DEBUG(TAG,"system_deps(...)");
    //
    // Find dependencies on system operations
    for(int64_t idx=_bhir->ninstr-1; idx>=0; --idx) {

        // The instruction to find data-dependencies for
        bh_instruction* instr = &_bhir->instr_list[idx];

        if (!((instr->opcode == BH_FREE)     ||  \
              (instr->opcode == BH_DISCARD)  ||  \
              (instr->opcode == BH_SYNC)     ||  \
              (instr->opcode == BH_NONE))) {
            continue;
        }

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

        bool found = false;
        for(int64_t other=idx-1; (other>=0) && (!found); --other) {
            bh_instruction* other_instr = &_bhir->instr_list[other];

            // Search operands of other instruction
            int64_t noperands = bh_operands(other_instr->opcode);
            for(int64_t op_idx=0; op_idx<noperands; ++op_idx) {
                // Array operations dependent on the output
                bh_view* other_op   = &other_instr->operand[op_idx];
                bh_base* other_base = other_op->base;

                if (other_base == output) {
                    found = true;
                    add_edge(other, idx, _dag);
                    break;
                }
            }
        }
    }
    DEBUG(TAG,"system_deps(...);");
}

string Dag::text(void)
{
    return "text(void);";
}

string Dag::dot(bh_instruction* instr, int64_t nr)
{
    DEBUG(TAG,"dot(instr*(" << instr->opcode << "), " << nr << ")");

    int64_t opcode = instr->opcode;

    stringstream operands;
    for(int64_t op_idx=0; op_idx<bh_operands(instr->opcode); ++op_idx) {
        if (bh_is_constant(&(instr->operand[op_idx]))) {
            operands << "K";
        } else {
            operands << instr->operand[op_idx].base;
        }
        if ((op_idx+1) != bh_operands(instr->opcode)) {
            operands << ", ";
        }
    }

    stringstream style, label;
    switch(opcode) {
        case BH_FREE:
            style << "shape=parallelogram, ";
            style << "fillcolor=\"#FDAE61\", ";
            break;

        case BH_DISCARD:
            style << "shape=trapezium, ";
            style << "fillcolor=\"#FFFFBF\", ";
            break;

        case BH_SYNC:
            style << "shape=circle, ";
            style << "fillcolor=\"#D7191C\", ";
            break;

        case BH_NONE:
            style << "shape=square, ";
            style << "fillcolor=\"#A6D96A\", ";
            break;

        default:
            label << "label=\"" << nr << " - ";
            label << bh_opcode_text(opcode);
            label << "(";
            label << operands.str();
            label << ")";
            label << "\"";
            break;
    }

    stringstream rep;
    rep << nr << " [" << style.str() << label.str() << "];";
    
    DEBUG(TAG,"dot(...);");
    return rep.str();
}

/**
 *  Return a textual representation in dot-format of the given Graph.
 *
 *  @param graph The graph to create textual dot-representation for.
 *  @return String with dot-representation of graph.
 */
string Dag::dot(void)
{
    DEBUG(TAG,"dot(void)");
    stringstream ss;
    ss << "digraph {" << endl;

    ss << "graph [";
    ss << "nodesep=0.8, ";
    ss << "sep=\"+25,25\", ";
    ss << "splines=false];" << endl;

    ss << "node [";
    ss << "shape=box, ";
    ss << "fontname=\"Courier\",";
    ss << "fillcolor=\"#CBD5E9\" ";
    ss << "style=filled,";
    ss << "];" << endl;
    
    // Vertices
    std::pair<vertex_iter, vertex_iter> vip = vertices(_dag);
    for(vertex_iter vi = vip.first; vi != vip.second; ++vi) {
        ss << dot(&_bhir->instr_list[*vi], *vi) << endl;
    }
    
    // Edges
    std::pair<edge_iter, edge_iter> eip = edges(_dag);
    for(edge_iter ei = eip.first; ei != eip.second; ++ei) {
        ss << source(*ei, _dag) << "->" << target(*ei, _dag) << ";" <<  endl;
    }

    // Subgraphs
    uint64_t subgraph_count = 0;
    for(vector<Graph*>::iterator gi=_subgraphs.begin(); gi!=_subgraphs.end(); gi++) {
        // Dot-text for subgraph begin
        ss << "subgraph cluster_" << subgraph_count << " { " << endl;
        ss << "style=filled;";
        ss << "color=lightgrey;";
        ss << endl;
        
        // Vertices in the given subgraph
        std::pair<vertex_iter, vertex_iter> svp = vertices(**gi);
        for(vertex_iter vi = svp.first; vi != svp.second; ++vi) {
            ss << (**gi).local_to_global(*vi) << ";";
        }
        
        // Dot-text for subgraph end
        ss << endl << "}" << endl;
        
        subgraph_count++;
    }
    
    ss << "}" << endl;
    DEBUG(TAG,"dot(void);");
    return ss.str();
}

}}}

