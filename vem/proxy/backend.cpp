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

#include <bh.hpp>
#include "comm.h"
#include "exec.h"

using namespace std;
using namespace bohrium;
using namespace bohrium::proxy;



bh_error service(const std::string &address, int port)
{
    bh_error e;
    CommBackend comm_backend = CommBackend(address, port);
    serialize::ExecuteBackend exec;

    while(1)
    {
        serialize::Header head = comm_backend.next_message_head();
        switch(head.type)
        {
            case serialize::TYPE_INIT:
            {
                std::vector<char> buffer(head.body_size);
                comm_backend.next_message_body(buffer);
                serialize::Init body(buffer);

                if((e = exec_init(body.component_name.c_str())) != BH_SUCCESS)
                    return e;
                break;
            }
            case serialize::TYPE_SHUTDOWN:
            {
                e = exec_shutdown();
                comm_backend.shutdown();
                return e;
            }
            case serialize::TYPE_EXEC:
            {
                std::vector<char> buffer(head.body_size);
                comm_backend.next_message_body(buffer);

                vector<bh_base*> data_send;
                vector<bh_base*> data_recv;
                bh_ir bhir = exec.deserialize(buffer, data_send, data_recv);

                //Receive new base array data
                for(size_t i=0; i<data_recv.size(); ++i)
                {
                    bh_base *base = data_recv[i];
                    base->data = NULL;
                    bh_data_malloc(base);
                    comm_backend.recv_array_data(base);
                }

                bh_error e = exec_execute(&bhir);
                if(e != BH_SUCCESS)
                    return e;

                //Send sync'ed array data
                for(size_t i=0; i<data_send.size(); ++i)
                {
                    bh_base *base = data_send[i];
                    bh_data_malloc(base);
                    comm_backend.send_array_data(base);
                }
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

int main(int argc, char * argv[])
{
    char *address = NULL;
    int port = 0;

    if (argc == 5 && \
        (strncmp(argv[1], "-a\0", 3) == 0) && \
        (strncmp(argv[3], "-p\0", 3) == 0)) {
        address = argv[2];
        port = atoi(argv[4]);
    } else {
        printf("Usage: %s -a ipaddress -p port\n", argv[0]);
        return 0;
    }
    if (!address) {
        fprintf(stderr, "Please supply address.\n");
        return 0;
    }
    service(address, port);
}