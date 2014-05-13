#include <fstream>
#include <sstream>
#include <set>
#include "dag.hpp"

using namespace std;
using namespace boost;
namespace bohrium{
namespace core {
namespace dag {

/**
 *  Create a graph with instructions as vertices and edges as data-dependencies.
 *
 *  @param bhir The bhir containing list of instructions.
 *  @return The graph.
 */
Dag::Dag(bh_ir* bhir) : _bhir(bhir), _dag(bhir->ninstr)
{
    init();
}

void Dag::init(void)
{
    //
    // Find dependencies on array operations
    for(int64_t idx=0; idx < _bhir->ninstr; ++idx) {

        // The instruction to find data-dependencies for
        bh_instruction* instr = &_bhir->instr_list[idx];
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

        //
        // Create edges for system
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
}

string Dag::text(void)
{
    return "Dag::text(void);";
}

string dot_instr(bh_instruction* instr, int64_t nr)
{
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
    stringstream ss;
    ss << "digraph {" << endl;

    ss << "graph [";
    //ss << "rankdir=LR, ";
    //ss << "overlap=false, ";
    ss << "layout=dot, ";
    ss << "nodesep=0.8, ";
    ss << "sep=\"+25,25\", ";
    ss << "overlap=scalexy, ";
    ss << "splines=false];" << endl;

    ss << "node [";
    ss << "shape=box, ";
    ss << "fontname=\"Courier\",";
    ss << "fillcolor=\"#CBD5E9\" ";
    ss << "style=filled,";
    //ss << "margin=0, pad=0";
    ss << "];" << endl;
    
    // Vertices
    std::pair<vertex_iter, vertex_iter> vp = vertices(_dag);
    for(vertex_iter it = vp.first; it != vp.second; ++it) {
        ss << dot_instr(&_bhir->instr_list[*it], *it) << endl;
    }
    
    // Edges
    std::pair<edge_iter, edge_iter> ep = edges(_dag);
    for(edge_iter it = ep.first; it != ep.second; ++it) {
        ss << source(*it, _dag) << "->" << target(*it, _dag) << ";" <<  endl;
    }
    
    ss << "}" << endl;
    return ss.str();
}

}}}

