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

#include<set>
#include<vector>
#include<map>


class bh_flow_node
{
public:
    //The time step of this node
    bh_intp timestep;
    //Whether the node is read only
    bool readonly;
    //The instruction index this node is referring
    bh_intp instr_idx;
    //The access pattern of this node
    const bh_view *view;
    //The origin of the incoming flow (node indices)
    std::set<bh_intp> parents;
    //The constructor
    bh_flow_node(bh_intp t, bool r, bh_intp i, const bh_view *v):
                 timestep(t), readonly(r), instr_idx(i), view(v),
                 parents() {}
};


class bh_flow
{
private:
    //List of nodes in a topological order
    std::vector<bh_flow_node> node_list;
    //Collection of vectors that contains indices to
    //flow-nodes that all access the same base array
    std::map<const bh_base *, std::vector<bh_intp> > bases;
    //The original instruction list
    bh_intp ninstr; const bh_instruction *instr_list;

void add_access(bh_intp node_idx);
bh_intp get_latest_conflicting_access(const bh_view *view, bool readonly);

public:
    bh_flow(bh_intp ninstr, const bh_instruction *instr_list);
    void pprint(void);

};




#endif
#endif
