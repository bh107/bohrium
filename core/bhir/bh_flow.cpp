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
#include <bh_flow.h>
#include <bh_vector.h>
#include <assert.h>
#include <iostream>
#include <string>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <string>
#include <stdexcept>
//using namespace std;


//Create a new flow_node. Use this function for creating flow nodes exclusively
bh_flow::flow_node &bh_flow::create_node(bool readonly, flow_instr *instr, const bh_view *view)
{
    //Create the new node at the base it accesses
    bases[view->base].push_back(flow_node(nnodes++, readonly, instr, view));
    flow_node &ret = bases[view->base].back();
    if(readonly)
        instr->reads.insert(ret);
    else
        instr->writes.insert(ret);
    ++nnodes;
    return ret;
}

//Add accesses that conflicts with the 'node' to 'conflicts'
set<bh_flow::flow_node> bh_flow::get_conflicting_access(const flow_node &node) 
{
    //Search through all nodes with the same base as 'node'
    const vector<flow_node> &same_base = bases[node.view->base];

    //Iterate 'it' to where the 'node' is in 'same_base'
    vector<flow_node>::const_reverse_iterator it = same_base.rbegin();
    for(it = same_base.rbegin(); it != same_base.rend(); ++it)
    {
        if(*it == node)
            break;
    }
    assert(it != same_base.rend());//'node' must be found in 'same_base'

    set<flow_node> conflicts;
    //Now continue iterating through possible conflicts
    for(++it; it != same_base.rend(); ++it)
    {
        assert(node.id != it->id);//We should not be here multiple times

        if(node.readonly && it->readonly)
            continue;//No possible conflict when both is read only

        if(node.instr->idx == it->instr->idx)
            continue;//No possible conflict within the same instruction

        if(!bh_view_disjoint(node.view, it->view))
        {
            conflicts.insert(*it);
        }
    }
    return conflicts;
}

//Create a new flow object based on an instruction list
bh_flow::bh_flow(bh_intp ninstr, const bh_instruction *instr_list):
                 ninstr(ninstr), instr_list(instr_list), flow_instr_list(ninstr), nnodes(0)
{
    for(bh_intp i=0; i<ninstr; i++)
    {
        flow_instr &instr = flow_instr_list[i];
        instr.idx = i;

        for(bh_intp o=0; o < bh_operands_in_instruction(&instr_list[i]); o++)
        {
            const bh_view *op = &instr_list[i].operand[o];
            if(bh_is_constant(op))
                continue;

            //Create a new access node
            bool readonly = (o==0)?false:true;
            flow_node &node = create_node(readonly, &instr, op);

            //The timestep of the instruction must be greater than any conflicting instructions
            set<flow_node> conflicts = get_conflicting_access(node);
            for(set<flow_node>::const_iterator it=conflicts.begin(); it!=conflicts.end(); ++it)
            {
                if(node.instr->timestep <= it->instr->timestep)
                    node.instr->timestep = it->instr->timestep+1;
            }
        }
        //Add the final timestep of the instruction to 'timesteps'
        if(instr.timestep >= (bh_intp)timesteps.size())
        {
            assert(instr.timestep == (bh_intp)timesteps.size());
            timesteps.resize(instr.timestep+1);
        }
        timesteps[instr.timestep].push_back(&instr);
    }
    sub_dag_clustering();
}


//Fill the uninitialized 'bhir' based on the flow object
void bh_flow::bhir_fill(bh_ir *bhir)
{
    assert(bhir->dag_list == NULL);

    //A set of dependencies for each instruction index in the flow
    vector<set<flow_instr*> > instr_deps(ninstr);
    //Map between flow and bhir sub-DAG indices
    map<bh_intp, bh_intp> dag_f2b;
    //Number of sub-DAGs
    map<bh_intp, bh_intp>::size_type ndags = 0;

    //Initiate the sub-DAGs
    for(vector<flow_instr>::const_iterator i=flow_instr_list.begin(); i!=flow_instr_list.end(); i++)
    {
        if(i->sub_dag == -1)
        {
            throw runtime_error("All nodes must be assigned to a sub_dag (no -1 indices) "
                                "before calling bhir_fill()");
        }
        if(dag_f2b.count(i->sub_dag) == 0)
            dag_f2b[i->sub_dag] = ndags++;
    }
    //A set of dependencies for each sub-DAG in the bhir
    vector<set<bh_intp> > dag_deps(ndags);

    //A set of instructions for each sub-DAG in the flow
    vector<set<flow_instr *> > dag_nodes(ndags);

    //Compute dependencies both between nodes and sub-DAGs
    for(vector<flow_instr>::iterator i=flow_instr_list.begin(); i!=flow_instr_list.end(); i++)
    {
        //For each instruction, we find all dependencies
        set<flow_node> deps;
        for(set<flow_node>::const_iterator n=i->writes.begin(); n!=i->writes.end(); ++n)
        {
            set<flow_node> ndeps = get_conflicting_access(*n);
            deps.insert(ndeps.begin(),ndeps.end());
        }
        for(set<flow_node>::const_iterator n=i->reads.begin(); n!=i->reads.end(); ++n)
        {
            set<flow_node> ndeps = get_conflicting_access(*n);
            deps.insert(ndeps.begin(),ndeps.end());
        }
        for(set<flow_node>::const_iterator d=deps.begin(); d != deps.end(); d++)
        {
            if(i->sub_dag == d->instr->sub_dag)//The dependency is within a sub-DAG
            {
                if(i->idx != d->instr->idx)//We cannot conflict with ourself
                    instr_deps[i->idx].insert(d->instr);
            }
            else//The dependency is to another sub-DAG
            {
                dag_deps[dag_f2b[i->sub_dag]].insert(dag_f2b[d->instr->sub_dag]);
            }
        }
        dag_nodes[dag_f2b[i->sub_dag]].insert(&(*i));
    }

    //Allocate the DAG list
    bhir->ndag = ndags+1;//which includes the root DAG
    bhir->dag_list = (bh_dag*) bh_vector_create(sizeof(bh_dag), bhir->ndag, bhir->ndag);
    if(bhir->dag_list == NULL)
        throw std::bad_alloc();

    //Create the root DAG where all nodes a sub-DAGs
    {
        bh_dag *dag = &bhir->dag_list[0];
        dag->node_map = (bh_intp*) bh_vector_create(sizeof(bh_intp), ndags, ndags);
        if(dag->node_map == NULL)
            throw std::bad_alloc();
        for(map<bh_intp, bh_intp>::size_type i=0; i<ndags; ++i)
            dag->node_map[i] = (-1*(i+1)-1);
        dag->nnode = ndags;
        dag->tag = 0;
        dag->adjmat = bh_adjmat_create(ndags);
        if(dag->adjmat == NULL)
            throw std::bad_alloc();

        //Fill each row in the adjacency matrix with the dependencies between sub-DAGs
        for(vector<set<bh_intp> >::size_type i=0; i < ndags; i++)
        {
            const std::set<bh_intp> &deps = dag_deps[i];
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
    //Create all sub-DAGs
    for(vector<set<bh_intp> >::size_type dag_idx=0; dag_idx<dag_nodes.size(); ++dag_idx)
    {
        const set<flow_instr*> &nodes = dag_nodes[dag_idx];
        bh_dag *dag = &bhir->dag_list[dag_idx+1];
        dag->node_map = (bh_intp*) bh_vector_create(sizeof(bh_intp), nodes.size(), nodes.size());
        if(dag->node_map == NULL)
            throw std::bad_alloc();
        dag->nnode = nodes.size();
        dag->tag = 0;
        dag->adjmat = bh_adjmat_create(nodes.size());
        if(dag->adjmat == NULL)
            throw std::bad_alloc();

        //Fill the adjmat sequentially starting at row zero
        map<bh_intp,bh_intp> instr2row;//instruction index to row in the adjmat
        set<flow_instr*>::size_type row=0;
        for(set<flow_instr*>::const_iterator instr=nodes.begin(); instr!=nodes.end(); ++instr, ++row)
        {
            const set<flow_instr*> &deps = instr_deps[(*instr)->idx];

            //Note that the order of 'row' is ascending thus the topological order is preserved.
            dag->node_map[row] = (*instr)->idx;
            //Mapping from flow index to node index within the sub-DAG.
            instr2row[(*instr)->idx] = row;

            if(deps.size() > 0)
            {
                //Convert flow indices to indices in the local sub-DAG
                vector<bh_intp> sorted_vector;
                for(set<flow_instr*>::const_iterator d = deps.begin(); d != deps.end(); d++)
                {
                    sorted_vector.push_back(instr2row[(*d)->idx]);
                }
                bh_error e = bh_adjmat_fill_empty_col(dag->adjmat, row,
                                                      sorted_vector.size(),
                                                      &sorted_vector[0]);
                if(e != BH_SUCCESS)
                    throw std::bad_alloc();
            }
        }
        if(bh_adjmat_finalize(dag->adjmat) != BH_SUCCESS)
            throw std::bad_alloc();
    }
}

bh_intp bh_flow::get_sub_dag_id(flow_instr* instr)
{
    if (instr->sub_dag == -1)
    {
        instr->sub_dag = instr->idx;
        assert(sub_dags.insert(std::make_pair(instr->sub_dag,std::vector<flow_instr*>(1,instr))).second);
    } else {
        assert(sub_dags.find(instr->sub_dag) != sub_dags.end());
    }
    return instr->sub_dag;
}

bool bh_flow::sub_dag_merge(bh_intp sub_dag_id1, bh_intp sub_dag_id2)
{
    auto sdi1 = sub_dags.find(sub_dag_id1);
    auto sdi2 = sub_dags.find(sub_dag_id2);
    assert(sdi1 != sub_dags.end() && sdi2 != sub_dags.end());
    if (sub_dag_id1 == sub_dag_id2)
    {
        return true;
    }
    for (flow_instr* instr1: sdi1->second)
    {
        if (instr_list[instr1->idx].opcode == BH_FREE || instr_list[instr1->idx].opcode == BH_DISCARD)
            continue;
        for (flow_instr* instr2: sdi2->second)
        {
            if (instr_list[instr2->idx].opcode == BH_FREE || instr_list[instr2->idx].opcode == BH_DISCARD)
                continue;
            for (const flow_node& o1: instr1->writes)
            {
                for (const flow_node& i2: instr2->reads)
                {
                    if (!(bh_view_disjoint(o1.view, i2.view) || bh_view_aligned(o1.view, i2.view)))
                        return false;
                }
            }
            for (const flow_node& o1: instr1->writes)
            {
                for (const flow_node& o2: instr2->writes)
                {
                    if (!(bh_view_disjoint(o1.view, o2.view) || bh_view_aligned(o1.view, o2.view)))
                        return false;
                }
            }
            for (const flow_node& i1: instr1->reads)
            {
                for (const flow_node& o2: instr2->writes)
                {
                    if (!(bh_view_disjoint(i1.view, o2.view) || bh_view_aligned(i1.view, o2.view)))
                        return false;
                }
            }
        }
    }
    for (flow_instr* instr2: sdi2->second)
    {
        instr2->sub_dag = sdi1->first;
        sdi1->second.push_back(instr2);
    }
    sub_dags.erase(sdi2);
    return true;
}

//Cluster the flow object into sub-DAGs suitable as kernals
void bh_flow::sub_dag_clustering(void)
{
    for (auto &base: bases)
    {
        auto fni = base.second.begin();
        auto instr1 = fni->instr;
        
        for (++fni; fni != base.second.end(); ++fni)
        {
            auto instr2 = fni->instr;
            sub_dag_merge(get_sub_dag_id(instr1), get_sub_dag_id(instr2));
            instr1 = instr2;
        }
    }
}

// Write the flow object in the DOT format.
void bh_flow::dot(const char* filename)
{
    ofstream fs(filename);

    fs << "digraph {" << std::endl;
    fs << "compound=true;" << std::endl;

    //Write all nodes and conflict edges
    map<const bh_base *, vector<flow_node> >::const_iterator b;
    for(b=bases.begin(); b != bases.end(); b++)
    {
        fs << "subgraph clusterBASE" << b->first << " {" << endl;
        fs << "label=\"" << b->first << "\";" << endl;

        //Define all nodes
        vector<flow_node>::const_iterator n,m;
        for(n=b->second.begin(); n != b->second.end(); n++)
        {
            fs << "n" << n->id << "[label=\"" << n->id << "T" << n->instr->timestep;
            if(n->readonly)
                fs << "R";
            else
                fs << "W";
            fs << n->instr->sub_dag;
            fs << "_" << bh_opcode_text(instr_list[n->instr->idx].opcode)+3;
            fs << "(" << n->instr->idx << ")\"";
            fs << " shape=box style=filled,rounded";
            if(n->instr->sub_dag >= 0)
                fs << " colorscheme=paired12 fillcolor=" << n->instr->sub_dag%12+1;
            fs << "]" << endl;
        }
        //Write invisible edges in order to get correct layout
        for(n=b->second.begin(); n != b->second.end()-1; n++)
        {
            fs << "n" << n->id << " -> n" << (n+1)->id << "[style=\"invis\"];" << endl;
        }
        //Write conflict edges
        for(n=b->second.begin(); n != b->second.end()-1; n++)
        {
            for(m=n+1; m != b->second.end(); m++)
            {
                if(!bh_view_identical(n->view, m->view))
                    fs << "n" << n->id << " -> n" << m->id << " [color=red];" << endl;
            }
        }
        fs << "}" << endl;
    }
    fs << "}" << std::endl;
    fs.close();
}

//In order to remove duplicates, we need a view compare function
struct view_compare {
  bool operator() (const bh_view *v1, const bh_view *v2) const
  {
      //First we compare base, ndim, and start
      int ret = memcmp(v1, v2, 3*sizeof(bh_intp));
      if(ret == 0)
      {//Then we compare shape and stride
        ret = memcmp(v1->shape, v2->shape, v1->ndim*sizeof(bh_index));
        if(ret == 0)
            ret = memcmp(v1->stride, v2->stride, v1->ndim*sizeof(bh_index));
      }
      if(ret < 0)
          return true;
      else
          return false;
  }
};

// Write the flow object in the DOT format.
void bh_flow::html(const char* filename)
{
    ofstream fs(filename);
    fs << "<!DOCTYPE html><html><body>" << endl << "<div style=\"float:left;\">";
    fs << "<table border=\"1\" cellpadding=\"5\" style=\"text-align:center\">" << endl;

    //'table' contains a string for each cell in the html table (excl. the header)
    //such that table[x][y] returns the string at coordinate (x,y).
    //We write to 'table' before writing to file.
    map<uint64_t, map<uint64_t, string> > table;

    //Create a map over all views in each base
    map<const bh_base *, set<const bh_view*, view_compare> > view_in_base;
    for(map<const bh_base *, vector<flow_node> >::const_iterator b=bases.begin();
        b != bases.end(); b++)
    {
        vector<flow_node>::const_iterator n;
        for(n=b->second.begin(); n != b->second.end(); n++)
            view_in_base[b->first].insert(n->view);
    }

    //Fill 'table'
    uint64_t ncol=0;
    map<const bh_base *, set<const bh_view*, view_compare> >::const_iterator b;
    for(b=view_in_base.begin(); b != view_in_base.end(); b++)
    {
        set<const bh_view*, view_compare>::const_iterator v;
        for(v=b->second.begin(); v!= b->second.end(); v++, ncol++)
        {
            vector<flow_node>::const_iterator n;
            for(n=bases[b->first].begin(); n != bases[b->first].end(); n++)
            {
                if(bh_view_identical(n->view, *v))
                {
                    char str[100];
                    if(n->readonly)
                        snprintf(str, 100, "%ld<sub>R</sub><sup>%ld</sup>", (long) n->instr->idx, (long) n->instr->sub_dag);
                    else
                        snprintf(str, 100, "%ld<sub>W</sub><sup>%ld</sup>", (long) n->instr->idx, (long) n->instr->sub_dag);
                    table[n->instr->timestep][ncol] += str;
                }
            }
        }
    }

    //Write base header
    fs << "<tr>" << endl;
    for(b=view_in_base.begin(); b != view_in_base.end(); b++)
    {
        fs << "\t<td colspan=\"" << b->second.size() << "\">" << b->first << "<br>";
        fs << " (#elem: " << b->first->nelem << ", dtype: " << bh_type_text(b->first->type);
        fs << ")</td>" << endl;
    }
    fs << "</tr>" << endl;

    //Write view header
    fs << "<tr>" << endl;
    for(b=view_in_base.begin(); b != view_in_base.end(); b++)
    {
        set<const bh_view*, view_compare>::const_iterator v;
        for(v=b->second.begin(); v!= b->second.end(); v++)
        {
            fs << "\t<td>";
            fs << (*v)->start << "(";
            for(bh_intp i=0; i<(*v)->ndim; ++i)
            {
                fs << (*v)->shape[i];
                if(i < (*v)->ndim-1)
                    fs << ",";
            }
            fs << ")(";
            for(bh_intp i=0; i<(*v)->ndim; ++i)
            {
                fs << (*v)->stride[i];
                if(i < (*v)->ndim-1)
                    fs << ",";
            }
            fs << ")</td>" << endl;
        }
    }
    fs << "</tr>" << endl;

    //Write 'table'
    for(uint64_t row=0; row<timesteps.size(); ++row)
    {
        fs << "<tr>" << endl;
        for(uint64_t col=0; col<ncol; ++col)
        {
            if(table[row].count(col) == 1)
            {
                fs << "\t<td>" << table[row][col] << "</td>" << endl;
            }
            else
            {
                fs << "\t<td></td>" << endl;
            }
        }
        fs << "</tr>" << endl;
    }
    fs << "</table></div>" << endl;
    //Write the instruction list
    fs << "<div style=\"float:right;\"><table  border=\"1\">" << endl;
    for (const auto &sub_dag: sub_dags)
    {
        fs << "<tr><td>" << endl;
        for (const flow_instr* instr: sub_dag.second)
        {
            char buf[100000];
            bh_sprint_instr(&instr_list[instr->idx], buf, "<br>");
            fs << "<b>" << instr->idx << "</b>)" << buf << "<br>";
        }
        fs << "</tr></td>" << endl;
    }
    fs << "</table></div></body></html>" << endl;
    fs.close();
}
