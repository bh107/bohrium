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

#ifndef __BH_FLOW_H
#define __BH_FLOW_H

#ifdef __cplusplus
#include <stdint.h>
#include <bh.h>
#include <set>
#include <vector>
#include <map>

using namespace std;

class bh_flow
{
private:
    class flow_node;//forward declaration

    class flow_instr
    {
    public:
        //The bh_instruction index this instruction is referring
        bh_intp idx;
        //Set of read and write nodes
        set<flow_node> reads;
        set<flow_node> writes;
        //The time step of this instruction
        bh_intp timestep;
        //The sub-DAG index this instruction is part of (-1 means none)
        bh_intp sub_dag;
        //The constructor
        flow_instr(void):idx(-1), timestep(0), sub_dag(-1){};
    };

    class flow_node
    {
    public:
        //The id of this node
        uint64_t id;
        //Whether the node is read only
        bool readonly;
        //The instruction this node is referring
        flow_instr *instr;
        //The access pattern of this node
        const bh_view *view;

        //Comparison of flow nodes
        bool operator<(const flow_node& rhs) const
        {
            return this->id < rhs.id;
        }
        bool operator==(const flow_node& rhs) const
        {
            return this->id == rhs.id;
        }

        //The constructor
        flow_node(uint64_t id, bool r, flow_instr *i, const bh_view *v):
                  id(id), readonly(r), instr(i), view(v){};
    };

    //Create a new flow_node. Use this function for creating flow nodes exclusively
    flow_node &create_node(bool readonly, flow_instr *instr, const bh_view *view);

    //The original instruction list
    const bh_intp ninstr;
    const bh_instruction * const instr_list;

    //The flow instruction list. NB: the length of this vector and
    //the associated memory addresses are fixed through the execution
    vector<flow_instr> flow_instr_list;

    //Collection of vectors that contains nodes that all access the same
    //base array in topological order
    map<const bh_base *, vector<flow_node> > bases;

    map<bh_intp, vector<flow_instr*> > sub_dags;
    bh_intp get_sub_dag_id(flow_instr* instr);
    bool sub_dag_merge(bh_intp sub_dag_id1, bh_intp sub_dag_id2);

    //Vector of flow instructions in each timestep
    vector<vector<flow_instr *> > timesteps;

    //Number of flow nodes
    uint64_t nnodes;

    //Get conflicting nodes
    set<flow_node> get_conflicting_access(const flow_node &node);

    //Cluster the flow object into sub-DAGs suitable as kernels
    void sub_dag_clustering(void);

public:
    //Create a new flow object based on an instruction list
    bh_flow(bh_intp ninstr, const bh_instruction *instr_list);

    //Fill 'bhir' based on the flow object
    void bhir_fill(bh_ir *bhir);

    //Write the flow object in the DOT format
    void dot(const char* filename);

    //Write the flow object in the HTML format
    void html(const char* filename);
};




#endif
#endif
