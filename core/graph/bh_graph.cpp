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
#include <list>
#include <queue>
#include <iostream>
#include <fstream>
#include <tr1/unordered_map>

#include "bh_graph.hpp"

#ifdef __GNUC__
#include <ext/hash_map>
namespace std { using namespace __gnu_cxx; }
#else
#include <hash_map>
#endif

#define BASIC_HASHER(T) template<> struct hash<T> { size_t operator()(T __x) const { return (size_t)__x; } };

static bh_intp print_graph_filename = 0;

/* Creates a new graph node.
 *
 * @type The node type
 * @instruction The instruction attached to the node, or NULL
 * @return Error codes (BH_SUCCESS)
 */
bh_graph_node* bh_graph_new_node(bh_intp type, bh_instruction* instruction)
{
    // TODO: Fix memory management
    bh_graph_node* node = (bh_graph_node*)malloc(sizeof(bh_graph_node));
    node->type = type;
    node->instruction = instruction;
    return node;
}

/* Destroys a new graph node.
 *
 * @node The node to free
 */
void bh_graph_free_node(bh_graph_node* node)
{
}


/* Parses a list of instructions into a graph representation.
 *
 * @bhir The root node extracted from the parsing
 * @return Error codes (BH_SUCCESS)
 */
bh_error bh_graph_from_list(bh_ir* bhir)
{
	std::tr1::unordered_map<bh_array*, bh_graph_node*> map;
	std::tr1::unordered_map<bh_array*, bh_graph_node*>::iterator it;
	std::queue<bh_graph_node*> exploration;
	
    print_graph_filename++;
    
	if (getenv("BH_PRINT_INSTRUCTION_GRAPH") != NULL)
	{
	    //Debug case only!
        char filename[8000];
        
        snprintf(filename, 8000, "%sinstlist-%lld.dot", getenv("BH_PRINT_INSTRUCTION_GRAPH"), (bh_int64)print_graph_filename);
        bh_graph_print_from_instructions(bhir, filename);
	}
	
#ifdef DEBUG
	bh_intp instrCount = 0;
#endif
	
	bh_graph_node* root = bh_graph_new_node(BH_COLLECTION, NULL);
	
	for(bh_intp i =0; i < bhir->instruction_count; i++)
	{
        bh_array* selfId = bh_base_array(bhir->instructions[i].operand[0]);
        bh_array* leftId = bh_base_array(bhir->instructions[i].operand[1]);
        bh_array* rightId = bh_base_array(bhir->instructions[i].operand[2]);
        
        if (selfId == NULL)
        {
            bh_userfunc* uf = bhir->instructions[i].userfunc;
            if (uf == NULL || uf->nout != 1 || (uf->nin != 0 && uf->nin != 1 && uf->nin != 2))
            {
                printf("Bailing because the userfunc is weird :(");
                return BH_ERROR;
            }
            bh_array** operands = (bh_array**)uf->operand;
        
            selfId = bh_base_array(operands[0]);
            if (uf->nin >= 1)
                leftId = bh_base_array(operands[1]);
            if (uf->nin >= 2)
                rightId = bh_base_array(operands[2]);
        }
        
        bh_graph_node* selfNode = bh_graph_new_node(BH_INSTRUCTION, &bhir->instructions[i]);
        if (selfNode == NULL)
            return BH_ERROR;
            
#ifdef DEBUG
        snprintf(selfNode.tag, 1000, "I%d - %s", instrCountr++, bh_opcode_text(bhir->instructions[i].opcode));
#endif
    
        if (bhir->instructions[i].opcode == BH_DISCARD || bhir->instructions[i].opcode == BH_SYNC)
        {
             while (!exploration.empty())
                 exploration.pop();
                 
            bh_graph_node* cur = NULL;
            it = map.find(selfId);
            if (it != map.end())
                cur = it->second;
            map[selfId] = selfNode;

            if (cur == NULL)
            {
                if (bh_grap_node_add_child(root, selfNode) != BH_SUCCESS)
                    return BH_ERROR;
            }
            else
            {
                if (cur->left_child != NULL)
                {
                    cur = cur->left_child;
                    if (cur->right_child != NULL)
                        exploration.push(cur->right_child);
                }
                else if (cur->right_child != NULL)
                    cur = cur->right_child;
            
                while (cur != NULL)
                {
                    if (cur->type == BH_INSTRUCTION)
                    {
                        if (bh_grap_node_add_child(cur, selfNode) != BH_SUCCESS)
                            return BH_ERROR;
                            
                        if (exploration.empty())
                            cur = NULL;
                        else
                        {
                            cur = exploration.front();
                            exploration.pop();
                        }
                    }
                    else
                    {
                        if (cur->left_child != NULL)
                        {
                            cur = cur->left_child;
                            if (cur->right_child != NULL)
                                exploration.push(cur->right_child);
                        }
                        else if (cur->right_child != NULL)
                            cur = cur->right_child;
                        else if (exploration.empty())
                            cur = NULL;
                        else
                        {
                            cur = exploration.front();
                            exploration.pop();
                        }
                    }
                }
            }
        }
        else
        {				
            bh_graph_node* oldTarget = NULL;
            it = map.find(selfId);
            if (it != map.end())
            {
                oldTarget = it->second;
                if (bh_grap_node_add_child(oldTarget, selfNode) != BH_SUCCESS)
                    return BH_ERROR;
            }
        
            map[selfId] = selfNode;
        
            bh_graph_node* leftDep = NULL;
            bh_graph_node* rightDep = NULL;
            it = map.find(leftId);
            if (it != map.end())
                leftDep = it->second;
            it = map.find(rightId);
            if (it != map.end())
                rightDep = it->second;

            if (leftDep != NULL)
            {
                if (bh_grap_node_add_child(leftDep, selfNode) != BH_SUCCESS)
                    return BH_ERROR;
            }
            
            if (rightDep != NULL && rightDep != leftDep)
            {
                if (bh_grap_node_add_child(rightDep, selfNode) != BH_SUCCESS)
                    return BH_ERROR;
            }
        
            if (leftDep == NULL && rightDep == NULL && oldTarget == NULL)
            {
                if (bh_grap_node_add_child(root, selfNode) != BH_SUCCESS)
                    return BH_ERROR;
            }
        }
	}
	
	bhir->node = root;

	if (getenv("BH_PRINT_NODE_INPUT_GRAPH") != NULL)
	{
	    //Debug case only!
        char filename[8000];
        
        snprintf(filename, 8000, "%sinput-graph-%lld.dot", getenv("BH_PRINT_NODE_INPUT_GRAPH"), (bh_int64)print_graph_filename);
        bh_graph_print_graph(root, filename);
	}

	return BH_SUCCESS;
}

/* Creates a list of instructions from a graph representation.
 *
 * @root The root node
 * instructions Storage for retrieving the instructions
 * instruction_count Input the size of the list, outputs the number of instructions
 * @return BH_SUCCESS if all nodes are added to the list, BH_ERROR if the storage was insufficient
 */
bh_error bh_graph_serialize(bh_graph_node* root, bh_instruction* instructions, bh_intp* instruction_count)
{
    // Keep track of already scheduled nodes
    std::tr1::unordered_map<bh_graph_node*, bh_graph_node*> scheduled;
    // Keep track of items that have unsatisfied dependencies
    std::queue<bh_graph_node*> blocked;

	if (getenv("BH_PRINT_NODE_OUTPUT_GRAPH") != NULL)
	{
	    //Debug case only!
        char filename[8000];
        
        snprintf(filename, 8000, "%soutput-graph-%lld.dot", getenv("BH_PRINT_NODE_OUTPUT_GRAPH"), (bh_int64)print_graph_filename);
        bh_graph_print_graph(root, filename);
	}
    
    blocked.push(root);
    
    bh_intp instr = 0;
    
    while (!blocked.empty())
    {
        bh_graph_node* n = blocked.front();
        blocked.pop();
        if (scheduled.find(n) == scheduled.end())
        {
            // Check if dependencies are met
            if ((n->left_parent == NULL || scheduled.find(n->left_parent) != scheduled.end()) && (n->right_parent == NULL || scheduled.find(n->right_parent) != scheduled.end()))
            {
                if (n->type == BH_INSTRUCTION)
                {
                    //TODO: Don't copy instruction?
                    if (instr < *instruction_count)
                        instructions[instr] = *n->instruction;
                        
                    //Keep counting, so we can return the required amount 
                    instr++;
                }
                
                scheduled[n] = n;
                
                //Examine child nodes
                if (n->left_child != NULL)
                    blocked.push(n->left_child);
                if (n->right_child != NULL && n->right_child != n->left_child)
                    blocked.push(n->right_child);
            }
            else
            {
                // Re-insert at bottom of work queue
                blocked.push(n);
            }
        }
    }
    
    if (instr < *instruction_count)
    {
        *instruction_count = instr;
        return BH_SUCCESS;
    }
    else
    {
        *instruction_count = instr;
        return BH_ERROR;
    }
}

/* Cleans up all memory used by the graph
 *
 * @bhir The entry to remove
 * @return Error codes (BH_SUCCESS) 
 */
bh_error bh_graph_destroy(bh_ir* bhir)
{
    return BH_SUCCESS;
}

/* Inserts a node into the graph
 *
 * @self The node to insert before
 * @other The node to insert 
 * @return Error codes (BH_SUCCESS)
 */
bh_error bh_grap_node_insert_before(bh_graph_node* self, bh_graph_node* other)
{
    self->left_child = other;
    if (other->left_parent != NULL)
    {
        if (other->left_parent->left_child == other)
            other->left_parent->left_child = self;
        else if (other->left_parent->right_child == other)
            other->left_parent->right_child = self;
        else
        {
            printf("Bad graph");
            return BH_ERROR;
        }
        
        self->left_parent = other->left_parent;
    }
    
    if (other->right_parent != NULL)
    {
        if (other->right_parent->left_child == other)
            other->right_parent->left_child = self;
        else if (other->right_parent->right_child == other)
            other->right_parent->right_child = self;
        else
        {
            printf("Bad graph");
            return BH_ERROR;
        }
        
        self->right_parent = other->right_parent;
    }
    
    other->left_parent = self;
    other->right_parent = NULL;
    
    return BH_SUCCESS;
}

/* Appends a node onto another node in the graph
 *
 * @self The node to append to
 * @newchild The node to append 
 * @return Error codes (BH_SUCCESS)
 */
bh_error bh_grap_node_add_child(bh_graph_node* self, bh_graph_node* newchild)
{
    if (self->left_child == NULL)
    {
        self->left_child = newchild;
        bh_grap_node_add_parent(newchild, self);
    }
    else if (self->right_child == NULL)
    {
        self->right_child = newchild;
        bh_grap_node_add_parent(newchild, self);
    }
    else
    {
        bh_graph_node* cn = bh_graph_new_node(BH_COLLECTION, NULL);
        if (cn == NULL)
            return BH_ERROR;
        cn->left_child = self->left_child;
        cn->right_child = newchild;
        self->left_child = cn;

        if (cn->left_child->left_parent == self)
            cn->left_child->left_parent = cn;
        else if (cn->left_child->right_parent == self)
            cn->left_child->right_parent = cn;
        else
        {
            printf("Bad graph");
            return BH_ERROR;
        }
            
        if (bh_grap_node_add_parent(newchild, cn) != BH_SUCCESS)
            return BH_ERROR;
            
        cn->left_parent = self;
    }
    
    return BH_SUCCESS;
}

/* Inserts a node into the graph
 *
 * @self The node to update
 * @newparent The node to append 
 * @return Error codes (BH_SUCCESS)
 */
bh_error bh_grap_node_add_parent(bh_graph_node* self, bh_graph_node* newparent)
{
    if (self->left_parent == newparent || self->right_parent == newparent || newparent == NULL)
        return BH_SUCCESS;
    else if (self->left_parent == NULL)
        self->left_parent = newparent;
    else if (self->right_parent == NULL)
        self->right_parent = newparent;
    else
    {
        bh_graph_node* cn = bh_graph_new_node(BH_COLLECTION, NULL);
        if (cn == NULL)
            return BH_ERROR;
            
        cn->left_parent = self->left_parent;
        cn->right_parent = self->right_parent;
        
        if (self->left_parent->left_child == self)
            cn->left_parent->left_child = cn;
        else if (self->left_parent->right_child == self)
            cn->left_parent->right_child = cn;
         
        if (self->right_parent->left_child == self)
            cn->right_parent->left_child = cn;
        else if (self->right_parent->right_child == self)
            cn->right_parent->right_child = cn;

        self->left_parent = cn;
        self->right_parent = newparent;
        cn->left_child = self;
    }
    
    return BH_SUCCESS;
}

/* Uses the instruction list to calculate dependencies and print a graph in DOT format.
 *
 * @bhir The input instructions
 * @return Error codes (BH_SUCCESS)
 */
bh_error bh_graph_print_from_instructions(bh_ir* bhir, const char* filename)
{
    std::tr1::unordered_map<bh_array*, bh_intp> nameDict;
    std::tr1::unordered_map<bh_array*, bh_intp>::iterator it;

    bh_intp lastName = 0;
    bh_intp constName = 0;
    
    std::ofstream fs(filename);
  
    fs << "digraph {" << std::endl;
    
    for(bh_intp i = 0; i < bhir->instruction_count; i++)
    {
        bh_instruction* n = &bhir->instructions[i];
        
        if (n->opcode != BH_USERFUNC)
        {
            bh_array* baseID = bh_base_array(n->operand[0]);
            bh_array* leftID = bh_base_array(n->operand[1]);
            bh_array* rightID = bh_base_array(n->operand[2]);
                                
            bh_intp parentName;
            bh_intp leftName;
            bh_intp rightName;
        
            it = nameDict.find(baseID);
            if (it == nameDict.end())
            {
                parentName = lastName++;
                nameDict[baseID] = parentName;
            }
            else
                parentName = it->second;
        
            it = nameDict.find(leftID);
            if (it == nameDict.end())
            {
                leftName = lastName++;
                nameDict[leftID] = leftName;
            }
            else
                leftName = it->second;
        
            it = nameDict.find(rightID);
            if (it == nameDict.end())
            {
                rightName = lastName++;
                nameDict[rightID] = rightName;
            }
            else
                rightName = it->second;
        
            bh_intp nops = bh_operands(n->opcode);
                                
            if (nops == 3)
            {
                if (leftID == NULL)
                {
                    bh_intp constid = constName++;
                    fs << "const_" << constid << "[shape=pentagon, style=filled, fillcolor=\"#ff0000\", label=\"" << n->constant.value.float64 << "\"];" << std::endl;
                    fs << "const_" << constid << " -> " << "I_" << i << ";" << std::endl; 
                }
                else
                {
                    fs << "B_" << leftName << "[shape=ellipse, style=filled, fillcolor=\"#0000ff\", label=\"B_" << leftName << " - " << bh_base_array(n->operand[1]) << "\"];" << std::endl;
                    fs << "B_" << leftName << " -> " << "I_" << i << ";" << std::endl;
                }
                
                if (rightID == NULL)
                {
                    bh_intp constid = constName++;
                    fs << "const_" << constid << "[shape=pentagon, style=filled, fillcolor=\"#ff0000\", label=\"" << n->constant.value.float64 << "\"];" << std::endl;
                    fs << "const_" << constid << " -> " << "I_" << i << ";" << std::endl;
                }
                else
                {
                    fs << "B_" << rightName << "[shape=ellipse, style=filled, fillcolor=\"#0000ff\", label=\"B_" << rightName << " - " << bh_base_array(n->operand[2]) << "\"];" << std::endl;
                    fs << "B_" << rightName << " -> " << "I_" << i << ";" << std::endl;
                }
            
            }
            else if (nops == 2)
            {
                if (leftID == NULL)
                {
                    bh_intp constid = constName++;
                    fs << "const_" << constid << "[shape=pentagon, style=filled, fillcolor=\"#ff0000\", label=\"" << n->constant.value.float64 << "\"];" << std::endl;
                    fs << "const_" << constid << " -> " << "I_" << i << ";" << std::endl;
                }
                else
                {
                    fs << "B_" << leftName << "[shape=ellipse, style=filled, fillcolor=\"#0000ff\", label=\"B_" << leftName << " - " << bh_base_array(n->operand[1]) << "\"];" << std::endl;
                    fs << "B_" << leftName << " -> " << "I_" << i << ";" << std::endl;
                }
            }
        
            fs << "I_" << i << "[shape=box, style=filled, fillcolor=\"#CBD5E8\", label=\"I_" << i << " - " << bh_opcode_text(n->opcode) << "\"];" << std::endl;
            fs << "B_" << parentName << "[shape=ellipse, style=filled, fillcolor=\"#0000ff\", label=\"" << "B_" << parentName << " - " << bh_base_array(n->operand[0]) << "\"];" << std::endl;
        
            fs << "I_" << i << " -> " << "B_" << parentName << ";" << std::endl;
        }
    }

    fs << "}" << std::endl;
    fs.close();
    
    return BH_SUCCESS;
}

/* Prints a graph representation of the node.
 *
 * @root The root node to draw
 * @return Error codes (BH_SUCCESS)
 */
bh_error bh_graph_print_graph(bh_graph_node* root, const char* filename)
{
    std::tr1::unordered_map<bh_graph_node*, bh_graph_node*> visited;
    std::queue<bh_graph_node*> queue;
    std::tr1::unordered_map<bh_graph_node*, bh_intp> nameTable;
    std::tr1::unordered_map<bh_graph_node*, bh_intp>::iterator it;
    
    std::ofstream fs(filename);
  
    fs << "digraph {" << std::endl;

    queue.push(root);

    bh_intp lastName = 0L;

    while(!queue.empty())
    {
        bh_graph_node* node = queue.front();
        queue.pop();
        
        if (visited.find(node) != visited.end())
            continue;

        visited[node] = node;
            
        if (node->left_child != NULL)
            queue.push(node->left_child);
        if (node->right_child != NULL)
            queue.push(node->right_child);
            
        const char T = node->type == BH_INSTRUCTION ? 'I' : 'C';
        bh_intp nodeName;
    
        it = nameTable.find(node);
        if (it == nameTable.end())
        {
            nodeName = lastName++;
            nameTable[node] = nodeName;
        } 
        else
            nodeName = it->second;
    
        if (node->type == BH_INSTRUCTION)
        {
            const char* color = "#CBD5E8"; // = roots.Contains(node) ? "#CBffff" : "#CBD5E8";
            const char* style = node->instruction->opcode == BH_DISCARD ? "dashed,rounded" : "filled,rounded";
            fs << T << "_" << nodeName << " [shape=box style=" << style << " fillcolor=\"" << color << "\" label=\"" << T << "_" << nodeName << " - " << bh_opcode_text(node->instruction->opcode) << "\"];" << std::endl;
        }
        else if (node->type == BH_COLLECTION)
        {
            fs << T << "_" << nodeName << " [shape=box, style=filled, fillcolor=""#ffffE8"", label=\"" << T << nodeName << " - COLLECTION\"];" << std::endl;
        }
    
        if (node->left_child != NULL)
        {
            const char T2 = node->left_child->type == BH_INSTRUCTION ? 'I' : 'C';
            bh_intp childName;
            it = nameTable.find(node->left_child);
            if (it == nameTable.end())
            {
                childName = lastName++;
                nameTable[node->left_child] = childName;
            } 
            else
                childName = it->second;
 
            fs << T << "_" << nodeName << " -> " << T2 << "_" << childName << ";" << std::endl;
        }
        if (node->right_child != NULL)
        {
            const char T2 = node->right_child->type == BH_INSTRUCTION ? 'I' : 'C';
            bh_intp childName;
            it = nameTable.find(node->right_child);
            if (it == nameTable.end())
            {
                childName = lastName++;
                nameTable[node->right_child] = childName;
            } 
            else
                childName = it->second;
 
            fs << T << "_" << nodeName << " -> " << T2 << "_" << childName << ";" << std::endl;
        }
    }

    fs << "}" << std::endl;
    fs.close();
    
    return BH_SUCCESS;
}