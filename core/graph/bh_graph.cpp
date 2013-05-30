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

#include "bh_graph.hpp"

#define NODE_LOOKUP(x) (((bh_graph_node*)bhir->nodes->data)[(x)])
#define INSTRUCTION_LOOKUP(x) (((bh_instruction*)bhir->instructions->data)[(x)])


#ifdef __GNUC__
#include <ext/hash_map>
namespace std { using namespace __gnu_cxx; }
#define hashmap std::hash_map
#else
#include <hash_map>
#define hashmap std::hash_map
#endif

struct hash_bh_intp{
  size_t operator()(const bh_intp x) const{
    return (size_t)x;
  }
};
struct hash_bh_graph_node_p{
  size_t operator()(const bh_graph_node* x) const{
    return (size_t)x;
  }
};
struct hash_bh_array_p{
  size_t operator()(const bh_array* x) const{
    return (size_t)x;
  }
};

// Static counter, used to generate sequential filenames when printing multiple batches
static bh_intp print_graph_filename = 0;

/* Creates a new graph storage element
 *
 * @bhir A pointer to the result
 * @instructions The initial instruction list (can be NULL if @instruction_count is 0)
 * @instruction_count The number of instructions in the initial list
 * @return BH_ERROR on allocation failure, otherwise BH_SUCCESS
 */
bh_error bh_graph_create(bh_ir** bhir, bh_instruction* instructions, bh_intp instruction_count)
{
    bh_ir* ir = (bh_ir*)malloc(sizeof(bh_ir));
    *bhir = NULL;
    if (ir == NULL)
        return BH_ERROR;
    
    ir->root = INVALID_NODE;
    
    ir->nodes = bh_dynamic_list_create(sizeof(bh_graph_node), 4000);
    if (ir->nodes == NULL)
    {
        free(ir);
        return BH_ERROR;
    }
    ir->instructions = bh_dynamic_list_create(sizeof(bh_instruction), 2000);
    if (ir->instructions == NULL)
    {
        bh_dynamic_list_destroy(ir->nodes);
        free(ir);
        return BH_ERROR;
    }
    
    if (instruction_count > 0)
        if (bh_graph_append(ir, instructions, instruction_count) != BH_SUCCESS)
        {
            bh_dynamic_list_destroy(ir->nodes);
            bh_dynamic_list_destroy(ir->instructions);
            free(ir);
            return BH_ERROR;            
        }
    
    *bhir = ir;
    return BH_SUCCESS;
}

/* Removes all allocated nodes
 *
 * @bhir The graph to update
 * @return Error codes (BH_SUCCESS) 
 */
bh_error bh_graph_delete_all_nodes(bh_ir* bhir)
{
    while(bhir->nodes->count > 0)
        bh_dynamic_list_remove(bhir->nodes, bhir->nodes->count - 1);
    bhir->root = INVALID_NODE;
    
    return BH_SUCCESS;
}

/* Cleans up all memory used by the graph
 *
 * @bhir The graph to destroy
 * @return Error codes (BH_SUCCESS) 
 */
bh_error bh_graph_destroy(bh_ir* bhir)
{
    if (bhir == NULL || bhir->nodes == NULL || bhir->instructions == NULL)
        return BH_ERROR;
    
    bh_dynamic_list_destroy(bhir->nodes);
    bh_dynamic_list_destroy(bhir->instructions);
    bhir->nodes = NULL;
    bhir->instructions = NULL;
    bhir->root = INVALID_NODE;
    
    return BH_SUCCESS;
}

/* Appends a new instruction to the current graph
 *
 * @bhir The graph to update
 * @instructions The instructions to append
 * @instruction_count The number of instructions in the list
 * @return Error codes (BH_SUCCESS) 
 */
bh_error bh_graph_append(bh_ir* bhir, bh_instruction* instructions, bh_intp instruction_count)
{
    if (bhir->root >= 0)
    {
        // Updating is not supported,
        // we need to maintain the map for that to work
        return BH_ERROR;
    }

    for(bh_intp i = 0; i < instruction_count; i++)
    {
        bh_intp ix = bh_dynamic_list_append(bhir->instructions);
        if (ix < 0)
            return BH_ERROR;
        ((bh_instruction*)bhir->instructions->data)[ix] = instructions[i];
    }
    
    return BH_SUCCESS;
}


/* Creates a new graph node.
 *
 * @type The node type
 * @instruction The instruction attached to the node, or NULL
 * @return Error codes (BH_SUCCESS)
 */
bh_node_index bh_graph_new_node(bh_ir* bhir, bh_intp type, bh_instruction_index instruction)
{
    bh_node_index ix = (bh_node_index)bh_dynamic_list_append(bhir->nodes);
    if (ix < 0)
        return INVALID_NODE;

    NODE_LOOKUP(ix).type = type;
    NODE_LOOKUP(ix).instruction = instruction;
    NODE_LOOKUP(ix).left_child = INVALID_NODE;
    NODE_LOOKUP(ix).right_child = INVALID_NODE;
    NODE_LOOKUP(ix).left_parent = INVALID_NODE;
    NODE_LOOKUP(ix).right_parent = INVALID_NODE;
    
    return ix;
}

/* Destroys a new graph node.
 *
 * @bhir The bh_ir structure
 * @node The node to free
 */
void bh_graph_free_node(bh_ir* bhir, bh_node_index node)
{
    bh_dynamic_list_remove(bhir->nodes, node);
}

/* Parses the instruction list and creates a new graph
 *
 * @bhir The graph to update
 * @return Error codes (BH_SUCCESS) 
 */
bh_error bh_graph_parse(bh_ir* bhir)
{
	hashmap<bh_array*, bh_intp, hash_bh_array_p> map;
	hashmap<bh_array*, bh_intp, hash_bh_array_p>::iterator it;
	std::queue<bh_intp> exploration;
	
	// If already parsed, just return
	if (bhir->root >= 0)
	    return BH_SUCCESS;
	
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
	
	bh_intp root = bh_graph_new_node(bhir, BH_COLLECTION, INVALID_INSTRUCTION);
	
	for(bh_intp i = 0; i < bhir->instructions->count; i++)
	{
	    // We can keep a reference pointer as we do not need to update the list
	    // while traversing it
	    bh_instruction* instr = &(((bh_instruction*)bhir->instructions->data)[i]);
        bh_array* selfId = bh_base_array(instr->operand[0]);
        bh_array* leftId = bh_base_array(instr->operand[1]);
        bh_array* rightId = bh_base_array(instr->operand[2]);
        
        if (selfId == NULL)
        {
            bh_userfunc* uf = instr->userfunc;
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
        
        bh_node_index selfNode = bh_graph_new_node(bhir, BH_INSTRUCTION, i);
        if (selfNode == INVALID_NODE)
        {
            bh_graph_delete_all_nodes(bhir);
            return BH_ERROR;
        }
            
#ifdef DEBUG
        snprintf(selfNode.tag, 1000, "I%d - %s", instrCountr++, bh_opcode_text(instr->opcode));
#endif
    
        if (instr->opcode == BH_DISCARD || instr->opcode == BH_SYNC)
        {
             while (!exploration.empty())
                 exploration.pop();
                 
            bh_node_index cur = INVALID_NODE;
            it = map.find(selfId);
            if (it != map.end())
                cur = it->second;
            map[selfId] = selfNode;

            if (cur == INVALID_NODE)
            {
                if (bh_grap_node_add_child(bhir, root, selfNode) != BH_SUCCESS)
                {
                    bh_graph_delete_all_nodes(bhir);
                    return BH_ERROR;
                }
            }
            else
            {
                if (NODE_LOOKUP(cur).left_child != INVALID_NODE)
                {
                    cur = NODE_LOOKUP(cur).left_child;
                    if (NODE_LOOKUP(cur).right_child != INVALID_NODE)
                        exploration.push(NODE_LOOKUP(cur).right_child);
                }
                else if (NODE_LOOKUP(cur).right_child != INVALID_NODE)
                    cur = NODE_LOOKUP(cur).right_child;
            
                while (cur != INVALID_NODE)
                {
                    if (NODE_LOOKUP(cur).type == BH_INSTRUCTION)
                    {
                        if (bh_grap_node_add_child(bhir, cur, selfNode) != BH_SUCCESS)
                        {
                            bh_graph_delete_all_nodes(bhir);
                            return BH_ERROR;
                        }
                            
                        if (exploration.empty())
                            cur = INVALID_NODE;
                        else
                        {
                            cur = exploration.front();
                            exploration.pop();
                        }
                    }
                    else
                    {
                        if (NODE_LOOKUP(cur).left_child != INVALID_NODE)
                        {
                            cur = NODE_LOOKUP(cur).left_child;
                            if (NODE_LOOKUP(cur).right_child != INVALID_NODE)
                                exploration.push(NODE_LOOKUP(cur).right_child);
                        }
                        else if (NODE_LOOKUP(cur).right_child != INVALID_NODE)
                            cur = NODE_LOOKUP(cur).right_child;
                        else if (exploration.empty())
                            cur = INVALID_NODE;
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
            bh_node_index oldTarget = INVALID_NODE;
            it = map.find(selfId);
            if (it != map.end())
            {
                oldTarget = it->second;
                if (bh_grap_node_add_child(bhir, oldTarget, selfNode) != BH_SUCCESS)
                {
                    bh_graph_delete_all_nodes(bhir);
                    return BH_ERROR;
                }
            }
        
            map[selfId] = selfNode;
        
            bh_node_index leftDep = INVALID_NODE;
            bh_node_index rightDep = INVALID_NODE;
            it = map.find(leftId);
            if (it != map.end())
                leftDep = it->second;
            it = map.find(rightId);
            if (it != map.end())
                rightDep = it->second;

            if (leftDep != INVALID_NODE)
            {
                if (bh_grap_node_add_child(bhir, leftDep, selfNode) != BH_SUCCESS)
                {
                    bh_graph_delete_all_nodes(bhir);
                    return BH_ERROR;
                }
            }
            
            if (rightDep != INVALID_NODE && rightDep != leftDep)
            {
                if (bh_grap_node_add_child(bhir, rightDep, selfNode) != BH_SUCCESS)
                {
                    bh_graph_delete_all_nodes(bhir);
                    return BH_ERROR;
                }
            }
        
            if (leftDep == INVALID_NODE && rightDep == INVALID_NODE && oldTarget == INVALID_NODE)
            {
                if (bh_grap_node_add_child(bhir, root, selfNode) != BH_SUCCESS)
                {
                    bh_graph_delete_all_nodes(bhir);
                    return BH_ERROR;
                }
            }
        }
	}
	
	bhir->root = root;

	if (getenv("BH_PRINT_NODE_INPUT_GRAPH") != NULL)
	{
	    //Debug case only!
        char filename[8000];
        
        snprintf(filename, 8000, "%sinput-graph-%lld.dot", getenv("BH_PRINT_NODE_INPUT_GRAPH"), (bh_int64)print_graph_filename);
        bh_graph_print_graph(bhir, filename);
	}

	return BH_SUCCESS;
}

struct bh_graph_iterator {
    // Keep track of already scheduled nodes
    hashmap<bh_node_index, bh_node_index, hash_bh_intp>* scheduled;
    // Keep track of items that have unsatisfied dependencies
    std::queue<bh_node_index>* blocked;
    // The graph we are iterating
    bh_ir* bhir;
    // The currently visited node
    bh_node_index current;
};

/* Creates a new iterator for visiting nodes in the graph
 *
 * @bhir The graph to iterate
 * @iterator The new iterator
 * @return BH_SUCCESS if the iterator is create, BH_ERROR otherwise
 */
bh_error bh_graph_iterator_create(bh_ir* bhir, bh_graph_iterator** iterator)
{
	if (getenv("BH_PRINT_NODE_OUTPUT_GRAPH") != NULL)
	{
	    //Debug case only!
        char filename[8000];
        
        snprintf(filename, 8000, "%soutput-graph-%lld.dot", getenv("BH_PRINT_NODE_OUTPUT_GRAPH"), (bh_int64)print_graph_filename);
        bh_graph_print_graph(bhir, filename);
	}
	
    struct bh_graph_iterator* t = (struct bh_graph_iterator*)malloc(sizeof(struct bh_graph_iterator));
    if (t == NULL)
    {
        *iterator = NULL;
        return BH_ERROR;
    }
    
    t->scheduled = new hashmap<bh_node_index, bh_node_index, hash_bh_intp>();
    t->blocked = new std::queue<bh_node_index>();
    t->bhir = bhir;
    t->current = t->bhir->root;
    if (t->current != INVALID_NODE)
        t->blocked->push(t->current);
    
    *iterator = t;
    return BH_SUCCESS;
}

/* Resets a graph iterator 
 *
 * @iterator The iterator to reset
 * @return BH_SUCCESS if the iterator is reset, BH_ERROR otherwise
 */
bh_error bh_graph_iterator_reset(bh_graph_iterator* iterator)
{
    delete iterator->scheduled;
    delete iterator->blocked;
    
    iterator->scheduled = new hashmap<bh_node_index, bh_node_index, hash_bh_intp>();
    iterator->blocked = new std::queue<bh_node_index>();
    iterator->current = iterator->bhir->root;
    if (iterator->current != INVALID_NODE)
        iterator->blocked->push(iterator->current);

    return BH_SUCCESS;
}

/* Move a graph iterator 
 *
 * @iterator The iterator to move
 * @instruction The next instruction
 * @return BH_SUCCESS if the iterator moved, BH_ERROR otherwise
 */
bh_error bh_graph_iterator_next_instruction(bh_graph_iterator* iterator, bh_instruction** instruction)
{
    bh_ir* bhir = iterator->bhir;

    // If we have not parsed, just give the instruction list as-is
    if (iterator->bhir->root == INVALID_NODE)
    {
        if (iterator->current == INVALID_NODE)
            iterator->current = -1;
        iterator->current++;
        
        if (iterator->current < bhir->instructions->count)
        {
            *instruction = &INSTRUCTION_LOOKUP(iterator->current);
            return BH_SUCCESS;
        }
        else
        {
            *instruction = NULL;
            return BH_ERROR;
        }
    }

    bh_node_index ix;
    while(bh_graph_iterator_next_node(iterator, &ix) == BH_SUCCESS)
        if (NODE_LOOKUP(ix).type == BH_INSTRUCTION)
        {
            *instruction = &(INSTRUCTION_LOOKUP(NODE_LOOKUP(ix).instruction));
            return BH_SUCCESS;
        }

    *instruction = NULL;
    return BH_ERROR;
}

/* Move a graph iterator 
 *
 * @iterator The iterator to move
 * @instruction The next instruction
 * @return BH_SUCCESS if the iterator moved, BH_ERROR otherwise
 */
bh_error bh_graph_iterator_next_node(bh_graph_iterator* iterator, bh_node_index* node)
{
    bh_ir* bhir = iterator->bhir;

    while (!iterator->blocked->empty())
    {
        bh_node_index n = iterator->blocked->front();
        iterator->blocked->pop();
        if (n != INVALID_NODE && iterator->scheduled->find(n) == iterator->scheduled->end())
        {
            // Check if dependencies are met
            if ((NODE_LOOKUP(n).left_parent == INVALID_NODE || iterator->scheduled->find(NODE_LOOKUP(n).left_parent) != iterator->scheduled->end()) && (NODE_LOOKUP(n).right_parent == INVALID_NODE || iterator->scheduled->find(NODE_LOOKUP(n).right_parent) != iterator->scheduled->end()))
            {                
                (*(iterator->scheduled))[n] = n;
                
                //Examine child nodes
                if (NODE_LOOKUP(n).left_child != INVALID_NODE)
                    iterator->blocked->push(NODE_LOOKUP(n).left_child);
                if (NODE_LOOKUP(n).right_child != INVALID_NODE && NODE_LOOKUP(n).right_child != NODE_LOOKUP(n).left_child)
                    iterator->blocked->push(NODE_LOOKUP(n).right_child);
                    
                *node = n;
                return BH_SUCCESS;
            }
            else
            {
                // Re-insert at bottom of work queue
                iterator->blocked->push(n);
            }
        }
    }
    
    *node = INVALID_NODE;
    return BH_ERROR;
}

/* Destroys a graph iterator 
 *
 * @iterator The iterator to destroy
 * @return BH_SUCCESS if the iterator is destroyed, BH_ERROR otherwise
 */
bh_error bh_graph_iterator_destroy(bh_graph_iterator* iterator)
{
    delete iterator->scheduled;
    delete iterator->blocked;
    iterator->scheduled = NULL;
    iterator->blocked = NULL;
    iterator->bhir = NULL;
    iterator->current = INVALID_NODE;
    free(iterator);
    
    return BH_SUCCESS;
}

/* Creates a list of instructions from a graph representation.
 *
 * @bhir The graph to serialize
 * @instructions Storage for retrieving the instructions
 * @instruction_count Input the size of the list, outputs the number of instructions
 * @return BH_SUCCESS if all nodes are added to the list, BH_ERROR if the storage was insufficient
 */
bh_error bh_graph_serialize(bh_ir* bhir, bh_instruction* instructions, bh_intp* instruction_count)
{
    bh_graph_iterator* it;        
    bh_instruction dummy;
    bh_instruction* cur;
    bh_intp count = 0;

    if (bh_graph_iterator_create(bhir, &it) != BH_SUCCESS)
        return BH_ERROR;        

    cur = *instruction_count == 0 ? &dummy : instructions;
    
    while(bh_graph_iterator_next_instruction(it, &cur))
    {
        count++;
        if (count > *instruction_count)
            cur = &dummy;
        else
            cur = &instructions[count];
    }
    
    if (count > *instruction_count)
    {
        *instruction_count = count;
        return BH_ERROR;
    }
    else
    {
        *instruction_count = count;
        return BH_SUCCESS;
    }
    
    bh_graph_iterator_destroy(it);
}

/* Inserts a node into the graph
 *
 * @bhir The graph to update
 * @self The node to insert before
 * @other The node to insert 
 * @return Error codes (BH_SUCCESS)
 */
bh_error bh_grap_node_insert_before(bh_ir* bhir, bh_node_index self, bh_node_index other)
{
    NODE_LOOKUP(self).left_child = other;
    if (NODE_LOOKUP(other).left_parent != INVALID_NODE)
    {
        if (NODE_LOOKUP(NODE_LOOKUP(other).left_parent).left_child == other)
            NODE_LOOKUP(NODE_LOOKUP(other).left_parent).left_child = self;
        else if (NODE_LOOKUP(NODE_LOOKUP(other).left_parent).right_child == other)
            NODE_LOOKUP(NODE_LOOKUP(other).left_parent).right_child = self;
        else
        {
            printf("Bad graph");
            return BH_ERROR;
        }
        
        NODE_LOOKUP(self).left_parent = NODE_LOOKUP(other).left_parent;
    }
    
    if (NODE_LOOKUP(other).right_parent != INVALID_NODE)
    {
        if (NODE_LOOKUP(NODE_LOOKUP(other).right_parent).left_child == other)
            NODE_LOOKUP(NODE_LOOKUP(other).right_parent).left_child = self;
        else if (NODE_LOOKUP(NODE_LOOKUP(other).right_parent).right_child == other)
            NODE_LOOKUP(NODE_LOOKUP(other).right_parent).right_child = self;
        else
        {
            printf("Bad graph");
            return BH_ERROR;
        }
        
        NODE_LOOKUP(self).right_parent = NODE_LOOKUP(other).right_parent;
    }
    
    NODE_LOOKUP(other).left_parent = self;
    NODE_LOOKUP(other).right_parent = INVALID_NODE;
    
    return BH_SUCCESS;
}

/* Appends a node onto another node in the graph
 *
 * @bhir The graph to update
 * @self The node to append to
 * @newchild The node to append 
 * @return Error codes (BH_SUCCESS)
 */
bh_error bh_grap_node_add_child(bh_ir* bhir, bh_node_index self, bh_node_index newchild)
{
    if (NODE_LOOKUP(self).left_child == INVALID_NODE)
    {
        NODE_LOOKUP(self).left_child = newchild;
        bh_grap_node_add_parent(bhir, newchild, self);
    }
    else if (NODE_LOOKUP(self).right_child == INVALID_NODE)
    {
        NODE_LOOKUP(self).right_child = newchild;
        bh_grap_node_add_parent(bhir, newchild, self);
    }
    else
    {
        bh_node_index cn = bh_graph_new_node(bhir, BH_COLLECTION, INVALID_INSTRUCTION);
        if (cn == INVALID_NODE)
            return BH_ERROR;
        NODE_LOOKUP(cn).left_child = NODE_LOOKUP(self).left_child;
        NODE_LOOKUP(cn).right_child = newchild;
        NODE_LOOKUP(self).left_child = cn;

        if (NODE_LOOKUP(NODE_LOOKUP(cn).left_child).left_parent == self)
            NODE_LOOKUP(NODE_LOOKUP(cn).left_child).left_parent = cn;
        else if (NODE_LOOKUP(NODE_LOOKUP(cn).left_child).right_parent == self)
            NODE_LOOKUP(NODE_LOOKUP(cn).left_child).right_parent = cn;
        else
        {
            printf("Bad graph");
            return BH_ERROR;
        }
            
        if (bh_grap_node_add_parent(bhir, newchild, cn) != BH_SUCCESS)
            return BH_ERROR;
            
        NODE_LOOKUP(cn).left_parent = self;
    }
    
    return BH_SUCCESS;
}

/* Inserts a node into the graph
 *
 * @bhir The graph to update
 * @self The node to update
 * @newparent The node to append 
 * @return Error codes (BH_SUCCESS)
 */
bh_error bh_grap_node_add_parent(bh_ir* bhir, bh_node_index self, bh_node_index newparent)
{
    if (NODE_LOOKUP(self).left_parent == newparent || NODE_LOOKUP(self).right_parent == newparent || newparent == INVALID_NODE)
        return BH_SUCCESS;
    else if (NODE_LOOKUP(self).left_parent == INVALID_NODE)
        NODE_LOOKUP(self).left_parent = newparent;
    else if (NODE_LOOKUP(self).right_parent == INVALID_NODE)
        NODE_LOOKUP(self).right_parent = newparent;
    else
    {
        bh_node_index cn = bh_graph_new_node(bhir, BH_COLLECTION, INVALID_INSTRUCTION);
        if (cn == INVALID_NODE)
            return BH_ERROR;
            
        NODE_LOOKUP(cn).left_parent = NODE_LOOKUP(self).left_parent;
        NODE_LOOKUP(cn).right_parent = NODE_LOOKUP(self).right_parent;
        
        if (NODE_LOOKUP(NODE_LOOKUP(self).left_parent).left_child == self)
            NODE_LOOKUP(NODE_LOOKUP(cn).left_parent).left_child = cn;
        else if (NODE_LOOKUP(NODE_LOOKUP(self).left_parent).right_child == self)
            NODE_LOOKUP(NODE_LOOKUP(cn).left_parent).right_child = cn;
         
        if (NODE_LOOKUP(NODE_LOOKUP(self).right_parent).left_child == self)
            NODE_LOOKUP(NODE_LOOKUP(cn).right_parent).left_child = cn;
        else if (NODE_LOOKUP(NODE_LOOKUP(self).right_parent).right_child == self)
            NODE_LOOKUP(NODE_LOOKUP(cn).right_parent).right_child = cn;

        NODE_LOOKUP(self).left_parent = cn;
        NODE_LOOKUP(self).right_parent = newparent;
        NODE_LOOKUP(cn).left_child = self;
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
    hashmap<bh_array*, bh_intp, hash_bh_array_p> nameDict;
    hashmap<bh_array*, bh_intp, hash_bh_array_p>::iterator it;
        
    bh_intp lastName = 0;
    bh_intp constName = 0;
    
    std::ofstream fs(filename);
  
    fs << "digraph {" << std::endl;
    
    for(bh_intp i = 0; i < bhir->instructions->count; i++)
    {
        bh_instruction* n = &INSTRUCTION_LOOKUP(i);
        
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

/* Prints a graph representation of the node in DOT format.
 *
 * @bhir The graph to print
 * @return Error codes (BH_SUCCESS)
 */
bh_error bh_graph_print_graph(bh_ir* bhir, const char* filename)
{
    hashmap<bh_node_index, bh_intp, hash_bh_intp> nameTable;
    hashmap<bh_node_index, bh_intp, hash_bh_intp>::iterator it;
    bh_graph_iterator* graph_it;

    if (bh_graph_iterator_create(bhir, &graph_it) != BH_SUCCESS)
        return BH_ERROR;        
    
    std::ofstream fs(filename);
  
    fs << "digraph {" << std::endl;

    bh_intp lastName = 0L;

    bh_node_index node;
    while(bh_graph_iterator_next_node(graph_it, &node) == BH_SUCCESS)
    {
        if (node == INVALID_NODE)
            continue;
            
        const char T = NODE_LOOKUP(node).type == BH_INSTRUCTION ? 'I' : 'C';
        bh_intp nodeName;
    
        it = nameTable.find(node);
        if (it == nameTable.end())
        {
            nodeName = lastName++;
            nameTable[node] = nodeName;
        } 
        else
            nodeName = it->second;
    
        if (NODE_LOOKUP(node).type == BH_INSTRUCTION)
        {
            const char* color = "#CBD5E8"; // = roots.Contains(node) ? "#CBffff" : "#CBD5E8";
            const char* style = INSTRUCTION_LOOKUP(NODE_LOOKUP(node).instruction).opcode == BH_DISCARD ? "dashed,rounded" : "filled,rounded";
            fs << T << "_" << nodeName << " [shape=box style=" << style << " fillcolor=\"" << color << "\" label=\"" << T << "_" << nodeName << " - " << bh_opcode_text(INSTRUCTION_LOOKUP(NODE_LOOKUP(node).instruction).opcode) << "\"];" << std::endl;
        }
        else if (NODE_LOOKUP(node).type == BH_COLLECTION)
        {
            fs << T << "_" << nodeName << " [shape=box, style=filled, fillcolor=""#ffffE8"", label=\"" << T << nodeName << " - COLLECTION\"];" << std::endl;
        }
    
        if (NODE_LOOKUP(node).left_child != INVALID_NODE)
        {
            const char T2 = NODE_LOOKUP(NODE_LOOKUP(node).left_child).type == BH_INSTRUCTION ? 'I' : 'C';
            bh_intp childName;
            it = nameTable.find(NODE_LOOKUP(node).left_child);
            if (it == nameTable.end())
            {
                childName = lastName++;
                nameTable[NODE_LOOKUP(node).left_child] = childName;
            } 
            else
                childName = it->second;
 
            fs << T << "_" << nodeName << " -> " << T2 << "_" << childName << ";" << std::endl;
        }
        if (NODE_LOOKUP(node).right_child != INVALID_NODE)
        {
            const char T2 = NODE_LOOKUP(NODE_LOOKUP(node).right_child).type == BH_INSTRUCTION ? 'I' : 'C';
            bh_intp childName;
            it = nameTable.find(NODE_LOOKUP(node).right_child);
            if (it == nameTable.end())
            {
                childName = lastName++;
                nameTable[NODE_LOOKUP(node).right_child] = childName;
            } 
            else
                childName = it->second;
 
            fs << T << "_" << nodeName << " -> " << T2 << "_" << childName << ";" << std::endl;
        }
    }

    fs << "}" << std::endl;
    fs.close();
    
    bh_graph_iterator_destroy(graph_it);
    
    return BH_SUCCESS;
}