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
#ifndef __BH_VE_GRAPH_H
#define __BH_VE_GRAPH_H

#include <bh.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct __bh_graph_node bh_graph_node;

//Basic entry in a parsed graph
struct __bh_graph_node {
    // The node type
	bh_intp type;
	
	// The index into the instruction list
	bh_instruction* instruction;
	
	//The left parent node for this element, or NULL if this is a root node
	bh_graph_node* left_parent;

	//The parent node for this element or NULL
	bh_graph_node* right_parent;
	
	//A pointer to the left node, or NULL if this is a leaf node
	bh_graph_node* left_child;

	//A pointer to the right node, or NULL if there is only a single child node
	bh_graph_node* right_child;
};

// A parsed graph, representing an execution batch
typedef struct {
    // The graph root node
    bh_graph_node* node;
    
    // The number of instructions in the batch
    bh_intp instruction_count;
    
    // The instruction batch
    bh_instruction* instructions;
    
#ifdef DEBUG
    char[1000] tag;
#endif
} bh_ir;


/* Node types codes */
enum /* bh_node_types */
{
    BH_INSTRUCTION,    // The node contains an actual instructions
    BH_COLLECTION,     // The node is a collection node
};

/* Creates a new graph node.
 *
 * @type The node type
 * @instruction The instruction attached to the node, or NULL
 * @return Error codes (BH_SUCCESS)
 */
bh_graph_node* bh_graph_new_node(bh_intp type, bh_instruction* instruction);

/* Destroys a new graph node.
 *
 * @node The node to free
 */
void bh_graph_free_node(bh_graph_node* node);

/* Parses a list of instructions into a graph representation.
 *
 * @bhir Contains the input instructions, 
 * and will be updated with the root node extracted 
 * from the parsing
 * @return Error codes (BH_SUCCESS)
 */
DLLEXPORT bh_error bh_graph_from_list(bh_ir* bhir);

/* Cleans up all memory used by the graph
 *
 * @bhir The entry to remove
 * @return Error codes (BH_SUCCESS) 
 */
DLLEXPORT bh_error bh_graph_destroy(bh_ir* bhir);

/* Inserts a node into the graph
 *
 * @self The node to insert before
 * @other The node to insert 
 * @return Error codes (BH_SUCCESS)
 */
DLLEXPORT bh_error bh_grap_node_insert_before(bh_graph_node* self, bh_graph_node* other);

/* Appends a node onto another node in the graph
 *
 * @self The node to append to
 * @newchild The node to append 
 * @return Error codes (BH_SUCCESS)
 */
DLLEXPORT bh_error bh_grap_node_add_child(bh_graph_node* self, bh_graph_node* newchild);

/* Inserts a node into the graph
 *
 * @self The node to update
 * @newparent The node to append 
 * @return Error codes (BH_SUCCESS)
 */
DLLEXPORT bh_error bh_grap_node_add_parent(bh_graph_node* self, bh_graph_node* newparent);


/* Uses the instruction list to calculate dependencies and print a graph in DOT format.
 *
 * @bhir The input instructions
 * @return Error codes (BH_SUCCESS)
 */
DLLEXPORT bh_error bh_graph_print_from_instructions(bh_ir* bhir, const char* filename);

/* Prints a graph representation of the node in DOT format.
 *
 * @root The root node to draw
 * @return Error codes (BH_SUCCESS)
 */
DLLEXPORT bh_error bh_graph_print_graph(bh_graph_node* root, const char* filename);

/* Creates a list of instructions from a graph representation.
 *
 * @root The root node
 * instructions Storage for retrieving the instructions
 * instruction_count Input the size of the list, outputs the number of instructions
 * @return BH_SUCCESS if all nodes are added to the list, BH_ERROR if the storage was insufficient
 */
DLLEXPORT bh_error bh_graph_serialize(bh_graph_node* root, bh_instruction* instructions, bh_intp* instruction_count);

#ifdef __cplusplus
}
#endif

#endif

