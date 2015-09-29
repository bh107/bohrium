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
#include "comm.h"
#include "exec.h"

using namespace std;
using namespace bohrium;
using namespace bohrium::proxy;


int main()
{
    bh_error e;
    CommBackend comm_backend = CommBackend("localhost", 4200);
    serialize::ExecuteBackend exec;

    while(1)
    {
        serialize::Header head = comm_backend.next_message_head();
        cout << "backend received message type: ";
        switch(head.type)
        {
            case serialize::TYPE_INIT:
            {
                cout << "INIT" << endl;
                std::vector<char> buffer(head.body_size);
                comm_backend.next_message_body(buffer);
                serialize::Init body(buffer);

                if((e = exec_init(body.component_name.c_str())) != BH_SUCCESS)
                    return e;
                break;
            }
            case serialize::TYPE_SHUTDOWN:
            {
                cout << "SHUTDOWN" << endl;
                e = exec_shutdown();
                comm_backend.shutdown();
                return e;
            }
            case serialize::TYPE_EXEC:
            {
                cout << "EXEC" << endl;
                cout << "EXEC read next body" << endl;
                std::vector<char> buffer(head.body_size);
                comm_backend.next_message_body(buffer);

                vector<bh_base*> data_send;
                vector<bh_base*> data_recv;
                bh_ir bhir = exec.deserialize(buffer, data_send, data_recv);

                cout << "EXEC recv new base data: ";
                //Receive new base array data
                for(size_t i=0; i<data_recv.size(); ++i)
                {
                    bh_base *base = data_recv[i];
                    base->data = NULL;
                    bh_data_malloc(base);
                    printf("%p ", base);
                    comm_backend.recv_array_data(base);
                }
                cout << endl;

                cout << "EXEC execute bytecode" << endl;
                bh_error e = exec_execute(&bhir);
                if(e != BH_SUCCESS)
                    return e;

                //Send sync'ed array data
                cout << "EXEC send sync'ed data: ";
                for(size_t i=0; i<data_send.size(); ++i)
                {
                    bh_base *base = data_send[i];
                    bh_data_malloc(base);
                    printf("%p ", base);
                    comm_backend.send_array_data(base);
                }
                cout << endl;
                exec.cleanup(bhir);
                break;
            }
            default:
            {
                cerr << "[VEM-PROXY] the backend received a unknown message type" << endl;
                return -1;
            }
        }
    }

    return 0;
}
