/*
 * ClientExec.cpp
 *
 *  Created on: Feb 3, 2014
 *      Author: d
 */

#include <iostream>
//#include <bh.h>

#include "client_comp.h"

//Our self
static bh_component myself;

//Function pointers to our child.
bh_component_iface *child;


int initcount = 0;

/* Component interface: init (see bh_component.h) */
bh_error client_comp_init(const char *component_name)
{
	bh_error err;

	if((err = bh_component_init(&myself, component_name)) != BH_SUCCESS)
		return err;

	// We are assuming that the component is connected to exactly 1 child
	if(myself.nchildren != 1)
	{
		std::cerr << "[client-VEM] Unexpected number of children, must be 1, is " << myself.nchildren << std::endl;
		return BH_ERROR;
	}

	//Let us initiate the child.
	
	child = &myself.children[0];
	if(initcount == 0 && (err = child->init(child->name)) != 0)
		return err;

	initcount++;
	return BH_SUCCESS;

}

/* Component interface: shutdown (see bh_component.h) */
bh_error client_comp_shutdown(void)
{

	bh_error err = BH_SUCCESS;
    initcount--;
    if(initcount==0)
    {
    	printf("destr");
    	err = child->shutdown();
    	bh_component_destroy(&myself);
    }

    return err;
}

/* Component interface: extmethod (see bh_component.h) */
bh_error client_comp_extmethod(const char *name, bh_opcode opcode)
{
	return child->extmethod(name, opcode);
}




/* Execute a BhIR where all operands are global arrays
 *
 * @bhir   The BhIR in question
 * @return Error codes
 */
bh_error client_comp_execute(bh_ir *bhir)
{

    bh_error ret = child->execute(bhir);

    return ret;
}
