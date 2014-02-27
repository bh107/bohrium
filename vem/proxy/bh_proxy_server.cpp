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
    bh_error res = BH_SUCCESS;

    res = bh_component_init(&vem_proxy_server_myself, name);
    if (BH_SUCCESS != res) {
        return res;
    }

    char* port;
    bh_string_option(port, "BH_VEM_PROXY_PORT", "port");
    if(port == NULL)
    {
        fprintf(stderr, "[PROXY-VEM] The server port must be specified "
                "through the config file or the env BH_VEM_PROXY_PORT \n");
        return BH_ERROR;
    }

    // set up network
    res = Init_Networking(atoi(port));
    if (BH_SUCCESS != res) {
        return BH_ERROR;
    }

    // Let us initiate the child.
    res = nw_init(name);
    if (BH_SUCCESS != res) {
        return res;
    }

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

