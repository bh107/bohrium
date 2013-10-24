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
#include <map>
#include <set>
#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>
#include "bh_ir.h"
#include "bh_vector.h"


/* Creates a Bohrium Internal Representation (BhIR)
 * based on a instruction list. It will consist of one DAG.
 *
 * @bhir        The BhIR handle
 * @ninstr      Number of instructions
 * @instr_list  The instruction list
 * @return      Error code (BH_SUCCESS, BH_OUT_OF_MEMORY)
 */
bh_error bh_ir_create(bh_ir *bhir, bh_intp ninstr,
                      const bh_instruction instr_list[])
{
    bh_intp instr_nbytes = sizeof(bh_instruction)*ninstr;

    //Make a copy of the instruction list
    bhir->instr_list = (bh_instruction*) bh_vector_create(instr_nbytes, ninstr, ninstr);
    if(bhir->instr_list == NULL)
        return BH_OUT_OF_MEMORY;
    memcpy(bhir->instr_list, instr_list, instr_nbytes);
    bhir->ninstr = ninstr;

    //Allocate the DAG list
    bhir->dag_list = (bh_dag*) bh_vector_create(sizeof(bh_dag), 1, 10);
    if(bhir->dag_list == NULL)
        return BH_OUT_OF_MEMORY;
    bhir->ndag = 1;

    //Initiate the first (and only) DAG
    bh_dag *dag = &bhir->dag_list[0];
    dag->node_map = (bh_intp*) bh_vector_create(sizeof(bh_intp), ninstr, ninstr);
    if(dag->node_map == NULL)
        return BH_OUT_OF_MEMORY;
    for(bh_intp i=0; i<ninstr; ++i)
        dag->node_map[i] = i;//A simple 1:1 map
    dag->nnode = ninstr;
    dag->tag = 0;
    dag->adjmat = bh_adjmat_create_from_instr(ninstr, instr_list);
    if(dag->adjmat == NULL)
        return BH_OUT_OF_MEMORY;
    return BH_SUCCESS;
}


/* Destory a Bohrium Internal Representation (BhIR).
 *
 * @bhir        The BhIR handle
 */
void bh_ir_destroy(bh_ir *bhir)
{
    for(bh_intp i=0; i < bhir->ndag; ++i)
    {
        bh_dag *dag = &bhir->dag_list[i];
        bh_vector_destroy(dag->node_map);
        bh_adjmat_destroy(&dag->adjmat);
    }
    bh_vector_destroy(bhir->instr_list);
    bh_vector_destroy(bhir->dag_list);
}


/* Resets the node to the first node in the BhIR (topologically).
 *
 * @node    The node to reset
 * @bhir    The BhIR handle
*/
void bh_node_reset(bh_node *node, const bh_ir *bhir)
{
    node->bhir = bhir;
    node->dag_idx = 0;
    node->node_map_idx = 0;
}


/* Iterate to the next node in a DAG (topological order)
 * NB: it will not iterate into a sub-DAG.
 *
 * @node     The BhIR node
 * @return   BH_ERROR when at the end, BH_SUCCESS otherwise
 */
bh_error bh_node_next(bh_node *node)
{
    if(node->dag_idx >= node->bhir->ndag)
        return BH_ERROR;

    bh_dag *dag = &node->bhir->dag_list[node->dag_idx];//Current DAG
    if(++node->node_map_idx >= dag->nnode)
    {
        node->node_map_idx = 0;
        if(++node->dag_idx >= node->bhir->ndag)
            return BH_ERROR;
    }
    return BH_SUCCESS;
}


/* Go to the root node of a specific DAG.
 *
 * @dag_node  The node that points to the DAG
 * @new_node  The BhIR node (output)
 * @return    BH_ERROR when the 'dag_node' dosn't points
 *            to a DAG in which case the 'new_node' is
 *            untouched, BH_SUCCESS otherwise.
*/
bh_error bh_node_goto_dag(const bh_node *dag_node,
                          bh_node *new_node)
{
    bh_dag *dag = &dag_node->bhir->dag_list[dag_node->dag_idx];
    bh_intp idx = dag->node_map[dag_node->node_map_idx];
    if(idx > 0)//Positive indexes are instruction list references.
        return BH_ERROR;

    new_node->bhir = dag_node->bhir;
    new_node->node_map_idx = 0;//We go to the root of the DAG
    new_node->dag_idx = -1*idx-1;
    return BH_SUCCESS;
}


/* Splits the DAG into an updated version of itself and a new sub-DAG that
 * consist of the nodes in 'nodes_idx'. Instead of the nodes in sub-DAG,
 * the updated DAG will have a new node that represents the sub-DAG.
 *
 * @bhir        The BhIR node
 * @nnodes      Number of nodes in the new sub-DAG
 * @nodes_idx   The nodes in the original DAG that will constitute the new sub-DAG.
 *              NB: this list will be sorted inplace
 * @dag_idx     The original DAG to split and thus modified
 * @sub_dag_idx The new sub-DAG which will be overwritten. -1 indicates that the
 *              new sub-DAG should be appended the DAG list
 *
 * @return      Error code (BH_SUCCESS, BH_OUT_OF_MEMORY)
*/
bh_error bh_dag_split(bh_ir *bhir, bh_intp nnodes, bh_intp nodes_idx[],
                      bh_intp dag_idx, bh_intp sub_dag_idx)
{
    bh_error e;
    assert(bhir->ndag == bh_vector_nelem(bhir->dag_list));
    assert(dag_idx < bhir->ndag);
    bh_dag *dag = &bhir->dag_list[dag_idx];

    if(sub_dag_idx == -1)
    {   //Lets make room for the new DAG
        bhir->dag_list = (bh_dag*) bh_vector_resize(bhir->dag_list, ++bhir->ndag);
        if(bhir->dag_list == NULL)
            return BH_OUT_OF_MEMORY;
        sub_dag_idx = bhir->ndag - 1;
    }
    else
    {   //Lets cleanup the DAG that might already exist at the index
        assert(sub_dag_idx < bhir->ndag);
        bh_dag *d = &bhir->dag_list[sub_dag_idx];
        bh_vector_destroy(d->node_map);
        bh_adjmat_destroy(&d->adjmat);
    }

    //Create the sub-DAG and the associated adjacency matrix
    bh_dag *sub_dag = &bhir->dag_list[sub_dag_idx];
    sub_dag->nnode = nnodes;
    sub_dag->node_map = (bh_intp*) bh_vector_create(sizeof(bh_intp), nnodes, nnodes);
    if(sub_dag->node_map == NULL)
        return BH_OUT_OF_MEMORY;
    sub_dag->adjmat->m = bh_boolmat_create(nnodes);
    if(sub_dag->adjmat->m == NULL)
        return BH_OUT_OF_MEMORY;
    sub_dag->tag = 0;

    //Just by sorting the nodes we find the topological order
    std::sort(nodes_idx, nodes_idx+nnodes);

    //Lets create a map from the original node indexes to the new sub-DAG node
    //indexes and create a node map for the sub-DAG
    std::map<bh_intp, bh_intp> org2sub;
    for(bh_intp i=0; i<nnodes; ++i)
    {
        org2sub[nodes_idx[i]] = i;
        sub_dag->node_map[i] = dag->node_map[nodes_idx[i]];
    }

    //In order to update the original DAG, we need to find the new sub-DAG's
    //input and output nodes. Additionally, we need to convert the node indexes
    //in 'nodes_idx' into sub-DAG node indexes.
    std::set<bh_intp> input, output;
    for(bh_intp i=0; i<nnodes; ++i)
    {
        bh_intp nchildren, nparents;
        //Check all the i'th nodes's children
        const bh_intp *children = bh_adjmat_get_row(dag->adjmat, nodes_idx[i], &nchildren);
        std::vector<bh_intp> children_in_sub_dag;
        for(bh_intp j=0; j<nchildren; ++j)
        {
            bh_intp child = children[j];
            if(org2sub.find(child) != org2sub.end())//The child is part of the sub-DAG
                children_in_sub_dag.push_back(org2sub[child]);
            else
                output.insert(child);
        }
        //Check all the i'th nodes's parents
        const bh_intp *parents = bh_adjmat_get_col(dag->adjmat, nodes_idx[i], &nparents);
        for(bh_intp j=0; j<nparents; ++j)
        {
            bh_intp parent = parents[j];
            if(org2sub.find(parent) == org2sub.end())//The parent is NOT part of the sub-DAG
                input.insert(parent);
        }

        //Now we know the i'th node's children that are within the sub-DAG
        e = bh_boolmat_fill_empty_row(sub_dag->adjmat->m, i, children_in_sub_dag.size(),
                                      &children_in_sub_dag[0]);
        if(e != BH_SUCCESS)
            return e;
    }
    //To finish the sub-DAG we have to save the transposed matrix as well
    sub_dag->adjmat->mT = bh_boolmat_transpose(sub_dag->adjmat->m);
    if(sub_dag->adjmat->mT == NULL)
        return BH_OUT_OF_MEMORY;

    //Save a copy of the original DAG and create a new DAG
    bh_dag org_dag = *dag;
    dag->nnode = org_dag.nnode - nnodes + 1;
    dag->adjmat->mT = bh_boolmat_create(dag->nnode);
    if(dag->adjmat->mT == NULL)
        return BH_OUT_OF_MEMORY;
    dag->node_map = (bh_intp*) bh_vector_create(sizeof(bh_intp), dag->nnode, dag->nnode);
    if(dag->node_map == NULL)
        return BH_OUT_OF_MEMORY;
    dag->tag = 0;

/* We fill the new DAG in three phases:
 *   1) Fill the initial rows in the new DAG with all nodes that do
 *      NOT depend on the sub-DAG (preserving their topological order)
 *   2) Fill the next row in the new DAG with a new node that represents
 *      the sub-DAG. The topological order is preserved since no previous
 *      nodes depend on the sub-DAG.
 *   3) Fill the last rows in the new DAG with all nodes that depend on
 *      the sub-DAG (preserving their topological order)
 *
 * Additionally, while inserting a row in the new DAG we convert the node’s
 * dependency indexes to match their new location.
 */

    //Phases 1)
    std::map<bh_intp, bh_intp> org2new;
    bh_intp nrows=0;//Current number of filled rows in the new DAG
    for(bh_intp org_idx=0; org_idx < org_dag.nnode; ++org_idx)
    {
        if(org2sub.find(org_idx) != org2sub.end())//The original node is part of the sub-DAG
            continue;//Ignore all sub-DAG nodes

        if(output.find(org_idx) == output.end())//The original node dosn't depend on the sub-DAG
        {
            org2new[org_idx] = nrows;
            bh_intp nparents;
            const bh_intp *parents = bh_adjmat_get_col(org_dag.adjmat, org_idx, &nparents);
            std::vector<bh_intp> parents_in_new_dag;
            //Lets save the parents as indexes in the new DAG
            for(bh_intp i=0; i<nparents; ++i)
            {
                bh_intp parent = parents[i];
                if(org2sub.find(parent) == org2sub.end())//The parent is NOT part of the sub-DAG
                    parents_in_new_dag.push_back(org2new[parent]);
            }
            e = bh_boolmat_fill_empty_row(dag->adjmat->mT, nrows, parents_in_new_dag.size(),
                                          &parents_in_new_dag[0]);
            if(e != BH_SUCCESS)
                return e;

            dag->node_map[nrows] = org_dag.node_map[org_idx];
            ++nrows;
        }
        else//The original node depend on the sub-DAG
        {
            output.insert(org_idx);
        }
    }

    //Phases 2)
    //Insert the new sub-DAG and at the nrows
    bh_intp sub_dag_location = nrows;
    {
        std::vector<bh_intp> parents_in_new_dag;
        //Lets save the parents as indexes in the new DAG
        for(std::set<bh_intp>::iterator it=input.begin(); it != input.end(); ++it)
        {
            parents_in_new_dag.push_back(org2new[*it]);
        }
        e = bh_boolmat_fill_empty_row(dag->adjmat->mT, nrows, parents_in_new_dag.size(),
                                      &parents_in_new_dag[0]);
        if(e != BH_SUCCESS)
            return e;

        dag->node_map[nrows] = -1*(sub_dag_idx+1);
        ++nrows;
    }

    //Phases 3)
    for(bh_intp org_idx=0; org_idx < org_dag.nnode; ++org_idx)
    {
        if(org2sub.find(org_idx) != org2sub.end())//The original node is part of the sub-DAG
            continue;//Ignore all sub-DAG nodes

        if(output.find(org_idx) != output.end())//The original node depends on the sub-DAG
        {
            org2new[org_idx] = nrows;
            bh_intp nparents;
            const bh_intp *parents = bh_adjmat_get_col(org_dag.adjmat, org_idx, &nparents);
            std::vector<bh_intp> parents_in_new_dag;
            //Lets save the parents as indexes in the new DAG
            for(bh_intp i=0; i<nparents; ++i)
            {
                bh_intp parent = parents[i];
                if(org2sub.find(parent) == org2sub.end())//The parent is NOT part of the sub-DAG
                    parents_in_new_dag.push_back(org2new[parent]);
            }
            parents_in_new_dag.push_back(sub_dag_location);
            e = bh_boolmat_fill_empty_row(dag->adjmat->mT, nrows, parents_in_new_dag.size(),
                                          &parents_in_new_dag[0]);
            if(e != BH_SUCCESS)
                return e;

            dag->node_map[nrows] = org_dag.node_map[org_idx];
            ++nrows;
        }
    }
    assert(dag->nnode == nrows);
    //Finally we need the transposed matrix aswell
    dag->adjmat->m = bh_boolmat_transpose(dag->adjmat->mT);
    if(dag->adjmat->m == NULL)
        return BH_OUT_OF_MEMORY;

    //Cleanup the original DAG
    bh_vector_destroy(org_dag.node_map);
    bh_adjmat_destroy(&org_dag.adjmat);
    return BH_SUCCESS;
}


//Private function to write a DAG in the DOT format
static void _dag2dot(const bh_ir* bhir, bh_intp dag_idx,
                     std::ofstream &fs)
{
    const bh_dag *dag = &bhir->dag_list[dag_idx];

    fs << "subgraph clusterDAG" << dag_idx << " {" << std::endl;
    fs << "label=\"ID: " << dag_idx << " TAG: " << dag->tag << "\";" << std::endl;

    for(bh_intp node_idx=0; node_idx<dag->nnode; ++node_idx)
    {
        bh_intp idx = dag->node_map[node_idx];
        fs << "d" << dag_idx << "_n" << node_idx;
        if(idx >= 0)    // An instruction
        {
            bh_intp opcode = bhir->instr_list[idx].opcode;
            const char* style;
            const char* color;

            if (opcode == BH_DISCARD || opcode == BH_FREE) {
                style = "dashed,rounded";
                color = "#ffffE8";
            } else {
                style = "filled,rounded";
                color = "#CBD5E8";
            }

            fs << " [shape=box ";
            fs << "style=\"" << style << "\" ";
            fs << "fillcolor=\"" << color << "\" ";
            fs << "label=\"I_" << idx << " - " << bh_opcode_text(opcode) << "\"]";
        }
        else            // A subgraph
        {
            fs << " [label=\"D" << -1*(idx+1) << "_sub-DAG\"]";
        }
        fs << ";" << std::endl;

        bh_intp nparents;
        const bh_intp *children = bh_adjmat_get_row(dag->adjmat,
                                                    node_idx, &nparents);
        for(bh_intp i=0; i<nparents; ++i)
        {
            bh_intp child = children[i];
            fs << "d" << dag_idx << "_n" << node_idx;
            fs << " -> ";
            fs << "d" << dag_idx << "_n" << child;
            fs << ";" << std::endl;
        }
    }
    fs << "}" << std::endl;
}


/* Write the BhIR in the DOT format.
 *
 * @bhir      The graph to print
 * @filename  Name of the written dot file
 */
void bh_bhir2dot(const bh_ir* bhir, const char* filename)
{
    std::ofstream fs(filename);

    fs << "digraph {" << std::endl;
    fs << "compound=true;" << std::endl;
    for(bh_intp dag_idx=0; dag_idx<bhir->ndag; ++dag_idx)
    {
        _dag2dot(bhir, dag_idx, fs);
    }
    fs << "}" << std::endl;
    fs.close();
}

