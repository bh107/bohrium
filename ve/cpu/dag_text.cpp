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

string Dag::dot_operand(int64_t idx)
{
    operand_t& opr = symbol_table_[idx];
    stringstream txt;
    if (opr.layout == CONSTANT) {
        txt << "K";
    } else {
        if (NULL == opr.base) {
            txt << "V:" << idx;
        } else {
            txt << "B:" << idx;
        }
    }

    return txt.str();
}

string Dag::dot(const tac_t& tac, int64_t nr)
{
    stringstream operands;
    switch(tac_noperands(tac)) {
        case 3:
            operands << "(";
            operands << dot_operand(tac.out) << ", ";
            operands << dot_operand(tac.in1) << ", ";
            operands << dot_operand(tac.in2) << ")";
            break;
        case 2:
            operands << "(";
            operands << dot_operand(tac.out) << ", ";
            operands << dot_operand(tac.in1) << ")";
            break;
        case 1:
            operands << "(" << dot_operand(tac.out) << ")";
            break;
        default:
            break;
    }

    stringstream style, label;

    switch(tac.op) {
        case MAP:
            style << "fillcolor=\"#b2df8a\", ";
            break;
        case ZIP:
            style << "fillcolor=\"#a5c966\", ";
            break;
        case REDUCE:
            style << "fillcolor=\"#a6cee3\", ";
            break;
        case SCAN:
            style << "fillcolor=\"#6baccd\", ";
            break;
        case GENERATE:
            style << "fillcolor=\"#33a02c\", ";
            break;
        case SYSTEM:
            switch(tac.oper) {
                case FREE:
                    style << "shape=parallelogram, ";
                    style << "fillcolor=\"#FDAE61\", ";
                    break;

                case DISCARD:
                    style << "shape=trapezium, ";
                    style << "fillcolor=\"#FFFFBF\", ";
                    break;

                case SYNC:
                    style << "shape=circle, ";
                    style << "fillcolor=\"#D7191C\", ";
                    break;

                case NONE:
                    style << "shape=square, ";
                    style << "fillcolor=\"#A6D96A\", ";
                    break;
            }
            break;
        case EXTENSION:
            style << "fillcolor=\"d0c2e5\", ";
            break;
    }

    switch(tac.op) {
        case SYSTEM:
            label << "label=\"" << nr << ": ";
            label << operands.str();
            label << "\"";
            break;
        default:
            label << "label=\"" << nr << ": ";
            label << operation_text(tac.op);
            label << "_";
            label << operator_text(tac.oper);
            label << operands.str();
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
        //ss << dot(&instr_[*vi], *vi) << endl;
        ss << dot(tacs_[*vi], *vi) << endl;
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
        ss << "label=\"Subgraph #" << subgraph_count << "\";";
        ss << "fontname=\"Courier\";";
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
