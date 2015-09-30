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
#include "exec.h"
#include "bh_serialize.h"
#include "bh_vem_proxy.h"
#include "comm.h"

#include <vector>


using namespace std;
using namespace boost;
using namespace bohrium::proxy;
using namespace bohrium::serialize;

static CommFrontend comm_front;

bh_error bh_vem_proxy_init(const char* name)
{
    cout << "init backend" << endl;
    bh_error e;

    //Execute our self
    if((e = exec_init(name)) != BH_SUCCESS)
        return e;

    int port = bh_component_config_lookup_int(exec_get_self_component(), "port", 4200);
    comm_front = CommFrontend(name, "localhost", port);

    return BH_SUCCESS;
}

bh_error bh_vem_proxy_shutdown(void)
{
    //Execute our self
    bh_error err = exec_shutdown();
    comm_front.shutdown();
    cout << "shutdown backend" << endl;
    return err;
}

bh_error bh_vem_proxy_extmethod(const char *name, bh_opcode opcode)
{
    bh_error e;
    assert(name != NULL);

    //The Master will find the new 'id' if required.
    if((e = exec_extmethod(name, opcode)) != BH_SUCCESS)
        return e;

    return BH_SUCCESS;
}

bh_error bh_vem_proxy_execute(bh_ir* bhir)
{
    vector<char> buffer;
    comm_front.execute(*bhir);

    return BH_SUCCESS;
}
