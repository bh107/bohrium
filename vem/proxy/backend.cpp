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

#include <bh_component.hpp>
#include <bh_util.hpp>

#include "comm.hpp"

using namespace std;
using namespace bohrium;
using namespace component;

static void service(const std::string &address, int port)
{
    CommBackend comm_backend(address, port);
    unique_ptr<ConfigParser> config;
    unique_ptr<ComponentFace> child;
    std::map<const bh_base*, bh_base> remote2local;

    while(true) {
        // Let's read the head of the message
        vector<char> buf_head(msg::HeaderSize);
        comm_backend.read(buf_head);
        msg::Header head(buf_head);

        switch(head.type) {
            case msg::Type::INIT:
            {
                std::vector<char> buffer(head.body_size);
                comm_backend.read(buffer);
                msg::Init body(buffer);
                if (child.get() != nullptr) {
                    throw runtime_error("[VEM-PROXY] Received INIT messages multiple times!");
                }
                config.reset(new ConfigParser(body.stack_level));
                child.reset(new ComponentFace(config->getChildLibraryPath(), config->stack_level+1));
                break;
            }
            case msg::Type::SHUTDOWN:
            {
                return;
            }
            case msg::Type::EXEC:
            {
                std::vector<char> buffer(head.body_size);
                comm_backend.read(buffer);
                vector<bh_base*> data_recv;
                set<bh_base*> freed;
                BhIR bhir(buffer, remote2local, data_recv, freed);

                // Receive new base array data
                for (bh_base *base: data_recv) {
                    base->data = nullptr;
                    comm_backend.recv_array_data(base);
                }

                // Send the bhir down to the child
                child->execute(&bhir);

                // Let's remove the freed base arrays
                for (const bh_base *base: freed) {
                    bh_data_free(&remote2local[base]);
                    remote2local.erase(base);
                }
                break;
            }
            case msg::Type::GET_DATA:
            {
                std::vector<char> buffer(head.body_size);
                comm_backend.read(buffer);
                msg::GetData body(buffer);

                if (util::exist(remote2local, body.base)) {
                    bh_base &local_base = remote2local.at(body.base);
                    void *data = child->get_mem_ptr(local_base, true, false, body.nullify);
                    comm_backend.send_array_data(data, bh_base_size(&local_base));
                } else {
                    comm_backend.send_array_data(nullptr, 0);
                }
                break;
            }
            case msg::Type::MSG:
            {
                break;
            }
            default:
            {
                throw runtime_error("[VEM-PROXY] the backend received a unknown message type");
            }
        }
    }
}

int main(int argc, char * argv[])
{
    char *address = nullptr;
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