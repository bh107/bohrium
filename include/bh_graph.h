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
#ifndef __BH_GRAPH_H
#define __BH_GRAPH_H

#include <bh.h>
#include <bh_dynamic_list.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef bh_intp bh_node_index;
typedef bh_intp bh_instruction_index;
#define INVALID_NODE (-1)
#define INVALID_INSTRUCTION (-1)

#define LEFT_C(x)      (((bh_graph_node*)bhir->nodes->data)[(x)].left_child)
#define RIGHT_C(x)     (((bh_graph_node*)bhir->nodes->data)[(x)].right_child)
#define LEFT_P(x)      (((bh_graph_node*)bhir->nodes->data)[(x)].left_parent)
#define RIGHT_P(x)     (((bh_graph_node*)bhir->nodes->data)[(x)].right_parent)
#define NODE_LOOKUP(x) (((bh_graph_node*)bhir->nodes->data)[(x)])
#define INSTRUCTION_LOOKUP(x) (((bh_instruction*)bhir->instructions->data)[(x)])

//Basic entry in a parsed graph
struct bh_graph_node {
    // The node type
	bh_intp type;
	
	// The index into the instruction list
	bh_intp instruction;
	
	//The left parent node for this element, or INVALID_NODE if this is a root node
	bh_node_index left_parent;

	//The parent node for this element or INVALID_NODE
	bh_node_index right_parent;
	
	//A pointer to the left node, or INVALID_NODE if this is a leaf node
	bh_node_index left_child;

	//A pointer to the right node, or INVALID_NODE if there is only a single child node
	bh_node_index right_child;
};

// "Secret" implementation, it uses C++ STL so we hide it from the C API
typedef struct bh_graph_iterator bh_graph_iterator;

// A parsed graph, representing an execution batch
typedef struct {
    // The graph root node
    bh_intp root;
        
    // The allocated instruction storage
    bh_dynamic_list* instructions;

    // The allocated node storage
    bh_dynamic_list* nodes;
    
} bh_ir;


/* Node types codes */
enum /* bh_node_types */
{
    BH_INSTRUCTION,     // The node contains an actual instructions
    BH_COLLECTION       // The node is a collection node
};

/* Creates a new graph storage element
 *
 * @bhir A pointer to the result
 * @instructions The initial instruction list (can be NULL if @instruction_count is 0)
 * @instruction_count The number of instructions in the initial list
 * @return BH_ERROR on allocation failure, otherwise BH_SUCCESS
 */
DLLEXPORT bh_error bh_graph_create(bh_ir** bhir, bh_instruction* instructions, bh_intp instruction_count);

/* Cleans up all memory used by the graph
 *
 * @bhir The graph to destroy
 * @return Error codes (BH_SUCCESS) 
 */
DLLEXPORT bh_error bh_graph_destroy(bh_ir* bhir);

/* Appends a new instruction to the current graph
 *
 * @bhir The graph to update
 * @instructions The instructions to append
 * @instruction_count The number of instructions in the list
 * @return Error codes (BH_SUCCESS) 
 */
DLLEXPORT bh_error bh_graph_append(bh_ir* bhir, bh_instruction* instruction, bh_intp instruction_count);

/* Creates a new graph node.
 *
 * @bhir The bh_ir structure
 * @type The node type 
 * @instruction The instruction attached to the node, or NULL
 * @return Error codes (BH_SUCCESS)
 */
DLLEXPORT bh_node_index bh_graph_new_node(bh_ir* bhir, bh_intp type, bh_instruction_index instruction);

/* Destroys a new graph node.
 *
 * @bhir The bh_ir structure
 * @node The node to free
 */
DLLEXPORT void bh_graph_free_node(bh_ir* bhir, bh_node_index node);

/* Inserts a node into the graph
 *
 * @bhir The graph to update
 * @self The node to insert before
 * @other The node to insert 
 * @return Error codes (BH_SUCCESS)
 */
DLLEXPORT bh_error bh_grap_node_insert_before(bh_ir* bhir, bh_node_index self, bh_node_index other);

/* Appends a node onto another node in the graph
 *
 * @bhir The graph to update
 * @self The node to append to
 * @newchild The node to append 
 * @return Error codes (BH_SUCCESS)
 */
DLLEXPORT bh_error bh_grap_node_add_child(bh_ir* bhir, bh_node_index self, bh_node_index newchild);

/* Inserts a node into the graph
 *
 * @bhir The graph to update
 * @self The node to update
 * @newparent The node to append 
 * @return Error codes (BH_SUCCESS)
 */
DLLEXPORT bh_error bh_grap_node_add_parent(bh_ir* bhir, bh_node_index self, bh_node_index newparent);


/* Uses the instruction list to calculate dependencies and print a graph in DOT format.
 *
 * @bhir The graph to print from
 * @bhir The input instructions
 * @return Error codes (BH_SUCCESS)
 */
DLLEXPORT bh_error bh_graph_print_from_instructions(bh_ir* bhir, const char* filename);

/* Prints a graph representation of the node in DOT format.
 *
 * @bhir The graph to print
 * @return Error codes (BH_SUCCESS)
 */
DLLEXPORT bh_error bh_graph_print_graph(bh_ir* bhir, const char* filename);

/* Creates a list of instructions from a graph representation.
 *
 * @bhir The graph to serialize
 * @instructions Storage for retrieving the instructions
 * @instruction_count Input the size of the list, outputs the number of instructions
 * @return BH_SUCCESS if all nodes are added to the list, BH_ERROR if the storage was insufficient
 */
DLLEXPORT bh_error bh_graph_serialize(bh_ir* bhir, bh_instruction* instructions, bh_intp* instruction_count);

/* Creates a new iterator for visiting nodes in the graph
 *
 * @bhir The graph to iterate
 * @iterator The new iterator
 * @return BH_SUCCESS if the iterator is create, BH_ERROR otherwise
 */
DLLEXPORT bh_error bh_graph_iterator_create(bh_ir* bhir, bh_graph_iterator** iterator);

/* Resets a graph iterator 
 *
 * @iterator The iterator to reset
 * @return BH_SUCCESS if the iterator is reset, BH_ERROR otherwise
 */
DLLEXPORT bh_error bh_graph_iterator_reset(bh_graph_iterator* iterator);

/* Moves a graph iterator to next instruction
 *
 * @iterator The iterator to move
 * @instruction The next instruction
 * @return BH_SUCCESS if the iterator moved, BH_ERROR otherwise
 */
DLLEXPORT bh_error bh_graph_iterator_next_instruction(bh_graph_iterator* iterator, bh_instruction** instruction);

/* Moves a graph iterator to next node
 *
 * @iterator The iterator to move
 * @node The next node index
 * @return BH_SUCCESS if the iterator moved, BH_ERROR otherwise
 */
DLLEXPORT bh_error bh_graph_iterator_next_node(bh_graph_iterator* iterator, bh_node_index* node);

/* Destroys a graph iterator 
 *
 * @iterator The iterator to destroy
 * @return BH_SUCCESS if the iterator is destroyed, BH_ERROR otherwise
 */
DLLEXPORT bh_error bh_graph_iterator_destroy(bh_graph_iterator* iterator);


#ifdef __cplusplus
}
#endif

#endif

