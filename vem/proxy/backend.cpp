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
#include <bh_main_memory.hpp>

#include "comm.hpp"
#include "compression.hpp"

using namespace std;
using namespace bohrium;
using namespace component;

static void service(const std::string &address, int port) {
    CommBackend comm_backend(address, port);
    unique_ptr<ConfigParser> config;
    unique_ptr<ComponentFace> child;
    Compression compression;
    string compress_param;
    std::map<const bh_base *, bh_base> remote2local;

    // Some statistics
    std::chrono::duration<double> time_mem_copy_total{0};
    std::chrono::duration<double> time_mem_copy_zip{0};
    uint64_t nbytes_send{0};

    while (true) {
        // Let's read the head of the message
        vector<char> buf_head(msg::HeaderSize);
        comm_backend.read(buf_head);
        msg::Header head(buf_head);

        switch (head.type) {
            case msg::Type::INIT: {
                std::vector<char> buffer(head.body_size);
                comm_backend.read(buffer);
                msg::Init body(buffer);
                if (child.get() != nullptr) {
                    throw runtime_error("[VEM-PROXY] Received INIT messages multiple times!");
                }
                config.reset(new ConfigParser(body.stack_level));
                child.reset(new ComponentFace(config->getChildLibraryPath(), config->stack_level + 1));
                compress_param = config->defaultGet<string>("compress_param", "zlib");
                break;
            }
            case msg::Type::SHUTDOWN: {
                if (config->defaultGet("prof", false)) {
                    cout << "Backend:\n";
                    cout << "  MemCopy: " << time_mem_copy_total.count() << "s" << endl;
                    cout << "    Zip:   " << time_mem_copy_zip.count() << "s" << endl;
                    cout << "    Send:  " << nbytes_send / 1024.0 / 1024.0 << "MB" << endl;
                }
                return;
            }
            case msg::Type::EXEC: {
                std::vector<char> buffer(head.body_size);
                comm_backend.read(buffer);
                vector<bh_base *> data_recv;
                set<bh_base *> freed;
                BhIR bhir(buffer, remote2local, data_recv, freed);

                // Receive new base array data
                for (bh_base *base: data_recv) {
                    base->data = nullptr;
                    auto data = comm_backend.recv_data();
                    if (not data.empty()) {
                        bh_data_malloc(base);
                        compression.uncompress(data, *base, compress_param);
                    }
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
            case msg::Type::GET_DATA: {
                std::vector<char> buffer(head.body_size);
                comm_backend.read(buffer);
                msg::GetData body(buffer);

                if (util::exist(remote2local, body.base)) {
                    bh_base &local_base = remote2local.at(body.base);
                    child->getMemoryPointer(local_base, true, false, false); // Note, we delay nullify to after comm.
                    if (local_base.data != nullptr) {
                        auto data = compression.compress(local_base, compress_param);
                        comm_backend.send_data(data);
                    } else {
                        comm_backend.send_data({});
                    }
                    if (body.nullify) {
                        bh_data_free(&local_base);
                        local_base.data = nullptr;
                    }
                } else {
                    comm_backend.send_data({});
                }
                if (body.nullify) {
                    remote2local.erase(body.base);
                }
                break;
            }
            case msg::Type::MEM_COPY: {
                auto t1 = chrono::steady_clock::now();
                std::vector<char> buffer(head.body_size);
                comm_backend.read(buffer);
                msg::MemCopy body(buffer);
                if (util::exist(remote2local, body.src.base)) {
                    bh_view src = body.src;
                    src.base = &remote2local.at(body.src.base);
                    child->getMemoryPointer(*src.base, true, false, false);
                    if (src.base->data != nullptr) {
                        auto t2 = chrono::steady_clock::now();
                        auto data = compression.compress(src, body.param);
                        time_mem_copy_zip += chrono::steady_clock::now() - t2;
                        nbytes_send += data.size();
                        comm_backend.send_data(data);
                    } else {
                        comm_backend.send_data({});
                    }
                } else {
                    comm_backend.send_data({});
                }
                time_mem_copy_total += chrono::steady_clock::now() - t1;
                break;
            }
            case msg::Type::MSG: {
                std::vector<char> buffer(head.body_size);
                comm_backend.read(buffer);
                msg::Message body(buffer);
                stringstream ss;
                if (body.msg == "info") {
                    ss << "  Backend: " << "\n";
                    ss << "    Hostname: " << comm_backend.hostname() << "\n";
                    ss << "    IP: "       << comm_backend.ip() << "\n";
                }
                ss << child->message(body.msg);
                comm_backend.write(ss.str());
                break;
            }
            default: {
                throw runtime_error("[VEM-PROXY] the backend received a unknown message type");
            }
        }
    }
}

int main(int argc, char *argv[]) {
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
