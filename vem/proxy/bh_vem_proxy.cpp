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

#include <vector>


using namespace std;
using namespace boost;
using namespace bohrium::proxy;
using namespace bohrium::serialize;

bh_error bh_vem_proxy_init(const char* name)
{
    bh_error e;

    //Execute our self
    if((e = exec_init(name)) != BH_SUCCESS)
        return e;

    return BH_SUCCESS;
}

bh_error bh_vem_proxy_shutdown(void)
{
    //Execute our self
    bh_error err = exec_shutdown();
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

ExecuteFrontend front;
ExecuteBackend back;

bh_error bh_vem_proxy_execute(bh_ir* bhir)
{
    vector<char> buffer;

    vector<bh_base*> front_data_send;
    vector<bh_base*> front_data_recv;
    front.serialize(*bhir, buffer, front_data_send, front_data_recv);

    bh_ir new_bhir;
    vector<bh_base*> back_data_send;
    vector<bh_base*> back_data_recv;
    back.deserialize(new_bhir, buffer, back_data_send, back_data_recv);

    //bh_pprint_instr_list(&bhir->instr_list[0],  bhir->instr_list.size(), "frontend");

    assert(front_data_send.size() == back_data_recv.size());
    for(size_t i=0; i<front_data_send.size(); ++i)
    {
        bh_base *front = front_data_send[i];
        bh_base *back = back_data_recv[i];
        assert(back->data != NULL);
        assert(front->data != NULL);
        back->data = NULL;
        bh_data_malloc(back);

        memcpy(back->data, front->data, bh_base_size(back));
    }

    front.cleanup(*bhir);

    //bh_pprint_instr_list(&new_bhir.instr_list[0],  new_bhir.instr_list.size(), "backend preexec");

    bh_error e = exec_execute(&new_bhir);
    if(e != BH_SUCCESS)
        return e;

    //bh_pprint_instr_list(&new_bhir.instr_list[0],  new_bhir.instr_list.size(), "backend postexec");

    assert(back_data_send.size() == front_data_recv.size());
    for(size_t i=0; i<back_data_send.size(); ++i)
    {
        bh_base *front = front_data_recv[i];
        bh_base *back = back_data_send[i];
        bh_data_malloc(front);
        bh_data_malloc(back);
        memcpy(front->data, back->data, bh_base_size(back));
    }
    back.cleanup(*bhir);

    return BH_SUCCESS;
}
