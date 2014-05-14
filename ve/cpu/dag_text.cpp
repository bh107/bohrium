#include <sstream>
#include "dag.hpp"
#include "utils.hpp"

using namespace std;
using namespace boost;
namespace bohrium{
namespace core {

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
    std::pair<vertex_iter, vertex_iter> vip = vertices(graph_);
    for(vertex_iter vi = vip.first; vi != vip.second; ++vi) {
        ss << dot(&instr_[*vi], *vi) << endl;
    }
    
    // Edges
    std::pair<edge_iter, edge_iter> eip = edges(graph_);
    for(edge_iter ei = eip.first; ei != eip.second; ++ei) {
        ss << source(*ei, graph_) << "->" << target(*ei, graph_) << ";" <<  endl;
    }

    // Subgraphs
    uint64_t subgraph_count = 0;
    for(vector<Graph*>::iterator gi=subgraphs_.begin(); gi!=subgraphs_.end(); gi++) {
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


}}
