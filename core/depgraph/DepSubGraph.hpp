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

#ifndef __DEPSUBGRAPH_HPP
#define __DEPSUBGRAPH_HPP

#include <bh.h>
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
    typedef std::multimap<bh_array* , bh_instruction*> ModificationMap;
    struct atributes{bool sync; bool discard; bool free;};
    typedef std::map<bh_array*, atributes> ArrayAtributes;
private:
    std::list<bh_instruction*> instructions;
    ModificationMap modificationMap;
    std::list<DepSubGraph*> dependsOn;
    ArrayAtributes arrayAtributes;
    friend DepSubGraph* merge(bh_instruction* inst, std::list<DepSubGraph*> subGraphs);
public:
<<<<<<< HEAD
    DepSubGraph(bh_instruction* inst);
    DepSubGraph(bh_instruction* inst, std::list<DepSubGraph*> _dependsOn);
    static DepSubGraph* merge(bh_instruction* inst, std::list<DepSubGraph*> subGraphs);
    void add(bh_instruction* inst);
    bool depends(DepSubGraph* subGraph);
    std::set<bh_array*> getModified();
    void sync(bh_array a);
    void free(bh_array a);
    void discard(bh_array a);
};

class DepSubGraphException 
{
private:
    int code;
public:
    DepSubGraphException(int code_) : code(code_) {}
};

#endif
