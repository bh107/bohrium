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

#include <iostream>
#include <bh_component.hpp>
#include "serialize.hpp"
#include <bh_util.hpp>

#include "comm.hpp"

using namespace bohrium;
using namespace component;
using namespace std;

namespace {
class Impl : public ComponentImpl {
private:
    CommFrontend comm_front;
    std::set<bh_base *> known_base_arrays;

public:
    Impl(int stack_level) : ComponentImpl(stack_level),
                            comm_front(stack_level,
                                       config.defaultGet<string>("address", "127.0.0.1"),
                                       config.defaultGet<int>("port", 4200)) {}
    ~Impl() {}

    void execute(BhIR *bhir);

    void extmethod(const std::string &name, bh_opcode opcode) {
        throw runtime_error("[PROXY-VEM] extmethod() not implemented!");
    };

    // Handle messages from parent
    string message(const string &msg) {
        throw runtime_error("[PROXY-VEM] message() not implemented!");
    }

    // Handle memory pointer retrieval
    void* get_mem_ptr(bh_base &base, bool copy2host, bool force_alloc, bool nullify) {
        if (not copy2host) {
            throw runtime_error("PROXY - get_mem_ptr(): `copy2host` is not True");
        }

        // Serialize message body
        vector<char> buf_body;
        msg::GetData body(&base, nullify);
        body.serialize(buf_body);

        // Serialize message head
        vector<char> buf_head;
        msg::Header head(msg::Type::GET_DATA, buf_body.size());
        head.serialize(buf_head);

        // Send serialized message
        comm_front.write(buf_head);
        comm_front.write(buf_body);

        // Receive the array data
        comm_front.recv_array_data(&base);

        if (force_alloc) {
            bh_data_malloc(&base);
        }

        // Nullify the data pointer
        void *ret = base.data;
        if (nullify) {
            base.data = nullptr;
        }
        return ret;
    }

    // Handle memory pointer obtainment
    void set_mem_ptr(bh_base *base, bool host_ptr, void *mem) {
        if (not host_ptr) {
            throw runtime_error("PROXY - set_mem_ptr(): `host_ptr` is not True");
        }
        throw runtime_error("PROXY - set_mem_ptr(): not implemented");
    }

    // We have no context so returning NULL
    void* get_device_context() {
        return nullptr;
    };

    // We have no context so doing nothing
    void set_device_context(void* device_context) {};
};
} //Unnamed namespace


extern "C" ComponentImpl* create(int stack_level) {
    return new Impl(stack_level);
}
extern "C" void destroy(ComponentImpl* self) {
    delete self;
}


void Impl::execute(BhIR *bhir) {

    // Serialize the BhIR, which becomes the message body
    vector<bh_base *> new_data; // New data in the order they appear in the instruction list
    vector<char> buf_body = bhir->write_serialized_archive(known_base_arrays, new_data);

    // Serialize message head
    vector<char> buf_head;
    msg::Header head(msg::Type::EXEC, buf_body.size());
    head.serialize(buf_head);

    // Send serialized message (head and body)
    comm_front.write(buf_head);
    comm_front.write(buf_body);

    // Send array data
    for (bh_base *base: new_data) {
        assert(base->data != nullptr);
        comm_front.send_array_data(base);
    }

    // Cleanup freed base array and make them unknown.
    for (const bh_instruction &instr: bhir->instr_list) {
        if (instr.opcode == BH_FREE) {
            bh_data_free(instr.operand[0].base);
            known_base_arrays.erase(instr.operand[0].base);
        }
    }
}