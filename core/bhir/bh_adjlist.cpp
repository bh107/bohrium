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

#include <bh.h>
#include <assert.h>
#include <bh_adjlist.h>
#include <bh_vector.h>
#include <map>
#include <set>
#include <vector>

/* Creates an adjacency list based on a instruction list
 * where an index in the instruction list refer to a row or
 * a column index in the adjacency matrix.
 *
 * @adjmat      The adjacency list in question
 * @ninstr      Number of instructions
 * @instr_list  The instruction list
 * May throw exception (std::bad_alloc)
 */
void bh_adjlist_create_from_instr(bh_adjlist &adjlist, bh_intp ninstr,
                                  const bh_instruction instr_list[])
{
    assert(adjlist.node.size() == 0 && adjlist.sub_dag.size() == 0);

    //Record over which instructions (identified by indexes in the instruction list)
    //are reading to a specific array. We use a std::vector since multiple instructions
    //may read to the same array.
    std::map<bh_base*, std::vector<bh_intp> > reads;

    //Record over the last instruction (identified by indexes in the instruction list)
    //that wrote to a specific array.
    //We only need the most recent write instruction since that instruction will depend on
    //all preceding write instructions.
    std::map<bh_base*, bh_intp> writes;

    for(bh_intp i=0; i<ninstr; ++i)
    {
        const bh_instruction *inst = &instr_list[i];
        const bh_view *ops = bh_inst_operands((bh_instruction *)inst);
        int nops = bh_operands_in_instruction(inst);

        if(nops == 0)//Instruction does nothing.
            continue;

        //Find the instructions that the i'th instruction depend on and insert them into
        //the sorted set 'deps'.
        bh_adjlist_node node;
        std::set<bh_intp> &deps(node.adj);
        for(bh_intp j=0; j<nops; ++j)
        {
            if(bh_is_constant(&ops[j]))
                continue;//Ignore constants
            bh_base *base = bh_base_array(&ops[j]);
            //When we are accessing an array, we depend on the instruction that wrote
            //to it previously (if any).
            std::map<bh_base*, bh_intp>::iterator w = writes.find(base);
            if(w != writes.end())
                deps.insert(w->second);
        }
        //When we are writing to an array, we depend on all previous reads that hasn't
        //already been overwritten
        bh_base *base = bh_base_array(&ops[0]);
        std::vector<bh_intp> &r(reads[base]);
        deps.insert(r.begin(), r.end());

        //Now all previous reads is overwritten
        r.clear();

        //The i'th instruction is now the newest write to array 'ops[0]'
        writes[base] = i;
        //and among the reads to arrays 'ops[1:]'
        for(bh_intp j=1; j<nops; ++j)
        {
            if(bh_is_constant(&ops[j]))
                continue;//Ignore constants
            bh_base *base = bh_base_array(&ops[j]);
            reads[base].push_back(i);
        }

        //For now all nodes gets its own sub-DAG
        node.sub_dag = i;
        adjlist.node.push_back(node);
        bh_adjlist_sub_dag sub_dag;
        sub_dag.adj.insert(deps.begin(), deps.end());
        sub_dag.node.insert(i);
        adjlist.sub_dag.push_back(sub_dag);
    }
}

/* Fills the dag_list in the ‘bhir’ based on the adjacency list ‘adjlist’
 * and the instruction list in the bhir.
 * NB: The dag_list within the bhir should be uninitialized (NULL).
 *
 * @adjmat  The adjacency list in question
 * @bhir    The bihr to update
 * May throw exception (std::bad_alloc)
 */
void bh_adjlist_fill_bhir(const bh_adjlist &adjlist, bh_ir *bhir)
{
    assert(bhir->dag_list == NULL);

    //Allocate the DAG list
    bh_intp ndags = adjlist.sub_dag.size()+1;//One root DAG plus the sub-DAGs
    bhir->dag_list = (bh_dag*) bh_vector_create(sizeof(bh_dag), ndags, ndags);
    if(bhir->dag_list == NULL)
        throw std::bad_alloc();
    bhir->ndag = ndags;

    //Lets build all sub-DAGs
    for(bh_intp i=0; i<(bh_intp)adjlist.sub_dag.size(); i++)
    {
        const std::set<bh_intp> &instr_idx(adjlist.sub_dag[i].node);//Instructions in the i'th sub-DAG
        bh_dag *dag = &bhir->dag_list[i+1];
        dag->node_map = (bh_intp*) bh_vector_create(sizeof(bh_intp), instr_idx.size(), instr_idx.size());
        if(dag->node_map == NULL)
            throw std::bad_alloc();
        dag->nnode = instr_idx.size();
        dag->tag = 0;
        dag->adjmat = bh_adjmat_create(instr_idx.size());
        if(dag->adjmat == NULL)
            throw std::bad_alloc();

        //Fill the adjmat sequentially starting at row zero
        bh_intp node_count = 0;
        std::map<bh_intp,bh_intp> instr2node;
        for(std::set<bh_intp>::iterator it=instr_idx.begin(); it!=instr_idx.end(); it++)
        {
            const std::set<bh_intp> &deps(adjlist.node[*it].adj);//The instruction's dependencies
            //Note that the order of 'it' is ascending thus the topological order is preserved.
            dag->node_map[node_count] = *it;
            //Mapping from the original instruction to local node index within the sub-DAG.
            //(i.e. the inverse of the node_map)
            instr2node[*it] = node_count;

            if(deps.size() > 0)
            {
                //Convert instruction indices to indices in the local sub-DAG
                std::vector<bh_intp> sorted_vector;
                for(std::set<bh_intp>::iterator it = deps.begin(); it != deps.end(); it++)
                {
                    std::map<bh_intp,bh_intp>::iterator n = instr2node.find(*it);
                    //If 'it' is not in 'instr2node' it must be a dependency to another sub-DAG,
                    //which we will handle later.
                    if(n != instr2node.end())
                        sorted_vector.push_back(n->second);
                }
                bh_error e = bh_adjmat_fill_empty_col(dag->adjmat, node_count,
                                                      sorted_vector.size(),
                                                      &sorted_vector[0]);
                if(e != BH_SUCCESS)
                    throw std::bad_alloc();
            }
            node_count++;
        }
        if(bh_adjmat_finalize(dag->adjmat) != BH_SUCCESS)
            throw std::bad_alloc();
    }

    //Lets build the root DAG
    {
        bh_dag *dag = &bhir->dag_list[0];
        dag->node_map = (bh_intp*) bh_vector_create(sizeof(bh_intp), ndags-1, ndags-1);
        if(dag->node_map == NULL)
            throw std::bad_alloc();
        for(bh_intp i=0; i<ndags-1; ++i)
            dag->node_map[i] = (-1*(i+1)-1);
        dag->nnode = ndags-1;
        dag->tag = 0;
        dag->adjmat = bh_adjmat_create(ndags-1);
        if(dag->adjmat == NULL)
            throw std::bad_alloc();


        //Fill each row in the adjacency matrix with the dependencies between sub-DAGs
        for(bh_intp i=0; i < ndags-1; i++)
        {
            const std::set<bh_intp> &deps(adjlist.sub_dag[i].adj);
            if(deps.size() > 0)
            {
                std::vector<bh_intp> sorted_vector(deps.begin(), deps.end());
                bh_error e = bh_adjmat_fill_empty_col(dag->adjmat, i,
                                                      deps.size(),
                                                      &sorted_vector[0]);
                if(e != BH_SUCCESS)
                    throw std::bad_alloc();
            }
        }
        if(bh_adjmat_finalize(dag->adjmat) != BH_SUCCESS)
            throw std::bad_alloc();
    }
}

/* Pretty Print the adjlist
 *
 * @adjmat  The adjacency list in question
 */
void bh_adjlist_pprint(const bh_adjlist &adjlist)
{
    printf("Adjacency list - nodes (%d){\n", (int) adjlist.node.size());
    printf("instr:\tDAG:\tdeps:\n");
    int i=0;
    for(std::vector<bh_adjlist_node>::const_iterator it=adjlist.node.begin();
        it!=adjlist.node.end(); it++)
    {
        printf("%2d,\t%2ld,\t[", i, it->sub_dag);
        for(std::set<bh_intp>::iterator dep=it->adj.begin(); dep!=it->adj.end(); dep++)
        {
            if(dep == it->adj.begin())//First iteration
                printf("%ld", *dep);
            else
                printf(",%ld", *dep);
        }
        printf("]\n");
        i++;
    }
    printf("}\n");
    printf("Adjacency list - sub-DAGs (%d){\n", (int) adjlist.sub_dag.size());
    printf("DAG:\tdeps:\t\tnodes:\n");
    i=0;
    for(std::vector<bh_adjlist_sub_dag>::const_iterator it=adjlist.sub_dag.begin();
        it!=adjlist.sub_dag.end(); it++)
    {
        printf("%2d,\t[", i);
        for(std::set<bh_intp>::iterator dep=it->adj.begin(); dep!=it->adj.end(); dep++)
        {
            if(dep == it->adj.begin())//First iteration
                printf("%ld", *dep);
            else
                printf(",%ld", *dep);
        }
        printf("],\t\t[");
        for(std::set<bh_intp>::iterator node=it->node.begin(); node!=it->node.end(); node++)
        {
            if(node == it->node.begin())//First iteration
                printf("%ld", *node);
            else
                printf(",%ld", *node);
        }
        printf("]\n");
        i++;
    }
    printf("}\n");
}
