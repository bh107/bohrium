/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium
team <http://www.bh107.org>.

Bohrium is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3
of the License, or (at your option) any later version.

Bohrium is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the
GNU Lesser General Public License along with Bohrium.

If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef __BH_PPRINT_H
#define __BH_PPRINT_H

#include "bh_opcode.h"
#include "bh_array.h"
#include "bh_error.h"
#include "bh_ir.h"
#include "bh_type.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Pretty print an base.
 *
 * @op      The base in question
 * @buf     Output buffer (must have sufficient size)
 */
DLLEXPORT void bh_sprint_base(const bh_base *base, char buf[]);

/* Pretty print an array.
 *
 * @view  The array view in question
 */
DLLEXPORT void bh_pprint_array(const bh_view *view);

/* Pretty print an view.
 *
 * @op      The view in question
 * @buf     Output buffer (must have sufficient size)
 */
DLLEXPORT void bh_sprint_view(const bh_view *op, char buf[]);

/* Pretty print an instruction.
 *
 * @instr  The instruction in question
 */
DLLEXPORT void bh_pprint_instr(const bh_instruction *instr);

/* Pretty print an instruction.
 *
 * @instr   The instruction in question
 * @buf     Output buffer (must have sufficient size)
 * @newline The new line string
 */
DLLEXPORT void bh_sprint_instr(const bh_instruction *instr, char buf[],
                               const char newline[]);

/* Pretty print an instruction list.
 *
 * @instr_list  The instruction list in question
 * @ninstr      Number of instructions
 * @txt         Text prepended the instruction list,
 *              ignored when NULL
 */
DLLEXPORT void bh_pprint_instr_list(const bh_instruction instr_list[],
                                    bh_intp ninstr, const char* txt);

/* Pretty print an array view.
 *
 * @view  The array view in question
 */
DLLEXPORT void bh_pprint_array(const bh_view *view);

/* Pretty print an array base.
 *
 * @base  The array base in question
 */
DLLEXPORT void bh_pprint_base(const bh_base *base);

/* Pretty print an coordinate.
 *
 * @coord  The coordinate in question
 * @ndims  Number of dimensions
 */
DLLEXPORT void bh_pprint_coord(const bh_index coord[], bh_index ndims);

/* Pretty print an instruction trace of the BhIR.
 *
 * @bhir The BhIR in question
 *
 */
DLLEXPORT void bh_pprint_trace_file(const bh_ir *bhir, char trace_fn[]);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

#include <boost/foreach.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/graphviz.hpp>
#include <iostream>
#include <fstream>

/* Writes the DOT file of a boost DAG where
 * each vertex is a pointer a bh_ir_kernel.
 *
 * @dag       The DAG to write (of type 'Graph')
 * @filename  The name of DOT file
 */
template <typename Graph>
void bh_pprint_dag_file(Graph &dag, const char filename[])
{
    using namespace std;
    using namespace boost;

    //We define a graph and a kernel writer for graphviz
    struct graph_writer
    {
        void operator()(std::ostream& out) const
        {
            out << "graph [bgcolor=white, fontname=\"Courier New\"]" << endl;
            out << "node [shape=box color=black, fontname=\"Courier New\"]" << endl;
        }
    };
    typedef typename graph_traits<Graph>::vertex_descriptor Vertex;
    struct kernel_writer
    {
        const Graph &graph;
        kernel_writer(const Graph &g) : graph(g) {};
        void operator()(std::ostream& out, const Vertex& v) const
        {
            char buf[1024*10];
            out << "[label=\"Kernel " << v << "\\n";
            out << "Input views: \\l";
            BOOST_FOREACH(const bh_view &i, graph[v]->input_list())
            {
                bh_sprint_view(&i, buf);
                out << buf << "\\l";
            }
            out << "Output views: \\l";
            BOOST_FOREACH(const bh_view &i, graph[v]->output_list())
            {
                bh_sprint_view(&i, buf);
                out << buf << "\\l";
            }
            out << "Temp base-arrays: \\l";
            BOOST_FOREACH(const bh_base *i, graph[v]->temp_list())
            {
                bh_sprint_base(i, buf);
                out << buf << "\\l";
            }
            out << "Instruction list: \\l";
            BOOST_FOREACH(const bh_instruction &i, graph[v]->instr_list())
            {
                bh_sprint_instr(&i, buf, "\\l");
                out << buf << "\\l";
            }
            out << "\"]";
        }
    };
    //cout << "Writing file " << filename << endl;
    ofstream file;
    file.open(filename);
    write_graphviz(file, dag, kernel_writer(dag), default_writer(), graph_writer());
    file.close();
}
#endif
#endif
