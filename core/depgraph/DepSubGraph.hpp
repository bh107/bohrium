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

#ifndef __DEPSUBGRAPH_HPP
#define __DEPSUBGRAPH_HPP

#include <cphvb.h>
#include <map>
#include <list>
#include <set>

/*
 * A SubGraph is a part of the dependenci graph within which:
 * Any data point, partial or complete, can be written without
 * invalidating any datapoint that is read.
 */
class DepSubGraph
{
    // A map from base array to instruction where it is modified
    typedef std::multimap<cphvb_array* , cphvb_instruction*> ModificationMap;
private:
    std::list<cphvb_instruction*> instructions;
    ModificationMap modificationMap;
    std::list<DepSubGraph*> dependsOn;
public:
    DepSubGraph(cphvb_instruction* inst);
    DepSubGraph(cphvb_instruction* inst, std::list<DepSubGraph*> _dependsOn);
    static DepSubGraph* merge(cphvb_instruction* inst, std::list<DepSubGraph*> dependsOn);
    void add(cphvb_instruction* inst);
    std::set<cphvb_array*> getModified();
};

class DepSubGraphException 
{
private:
    int code;
public:
    DepSubGraphException(int code_) : code(code_) {}
};

#endif
