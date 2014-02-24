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

#include <cassert>
#include <cstring>
#include <iostream>
#include <bh.h>
#include <set>

#include "bh_proxy_server.h"
#include "ProxyNetworking.h"

//Function pointers to our child.
//static bh_component_iface *child;

//Our self
static bh_component vem_proxy_server_myself;

//Allocated base arrays
static std::set<bh_base*> allocated_bases;


//Network status
//static bool netstat = false;

#ifdef BH_TIMING
    //Number of elements executed
    static bh_intp total_execution_size = 0;
#endif

bool bh_string_option(char *&option, const char *env_name, const char *conf_name)
{
	option = getenv(env_name);           // For the compiler
	if (NULL==option) {
		option = bh_component_config_lookup(&vem_proxy_server_myself, conf_name);
	}

	if (!option) {
		return false;
	}
	return true;
}

/* Component interface: init (see bh_component.h) */
bh_error bh_vem_proxy_init(const char* name)
{
    bh_error err;

    //std::cout << "Initializing client" << std::endl;

    if((err = bh_component_init(&vem_proxy_server_myself, name)) != BH_SUCCESS)
        return err;

    // set configurations

    char * port;

    bh_string_option( port,   "BH_VE_PROXY_PORT", "port");

    // set up network

    if(Init_Networking(port) != BH_SUCCESS)
        return BH_ERROR;



    //Let us initiate the child.
    err = nw_init(name);
    if(err != 0)
        return err;


    return BH_SUCCESS;
}

/* Component interface: shutdown (see bh_component.h) */
bh_error bh_vem_proxy_shutdown(void)
{
    bh_error err = nw_shutdown(); //= child->shutdown();
    bh_component_destroy(&vem_proxy_server_myself);
    Shutdown_Networking();

    #ifdef BH_TIMING
        printf("Number of elements executed: %ld\n", total_execution_size);
    #endif

    // shut down network
    return err;
}

/* Component interface: extmethod (see bh_component.h) */
bh_error bh_vem_proxy_extmethod(const char *name, bh_opcode opcode)
{
    return nw_extmethod(name, opcode);
}

/* Component interface: execute (see bh_component.h) */
bh_error bh_vem_proxy_execute(bh_ir* bhir)
{

    bh_error ret = nw_execute(bhir);

    return ret;
}
