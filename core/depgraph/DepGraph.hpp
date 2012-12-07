/*
This file is part of cphVB and copyright (c) 2012 the cphVB team:
http://cphvb.bitbucket.org

cphVB is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as 
published by the Free Software Foundation, either version 3 
of the License, or (at your option) any later version.

cphVB is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the 
GNU Lesser General Public License along with cphVB. 

If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __DEPGRAPH_HPP
#define __DEPGRAPH_HPP

#include <cphvb.h>
#include <map>
#include "DepSubGraph.hpp"

class DepGraph
{
    typedef std::map<cphvb_array*, DepSubGraph*> LastModifiedByMap;
private:
    std::list<DepSubGraph*> subGraphs;
    LastModifiedByMap lastModifiedBy;
    void ufunc(cphvb_instruction* inst);
public:
    DepGraph(cphvb_intp instruction_count,
             cphvb_instruction instruction_list[]);
};

#endif
