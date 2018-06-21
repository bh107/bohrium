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
#include <bh_main_memory.hpp>
#include <bh_util.hpp>

#include "serialize.hpp"
#include "comm.hpp"
#include "compression.hpp"

using namespace bohrium;
using namespace component;
using namespace std;

namespace {
class Impl : public ComponentVE {
private:
    Compression compressor;
    CommFrontend comm_front;
    std::set<bh_base *> known_base_arrays;
    string compress_param;

    bool stat_print_on_exit;
    std::chrono::duration<double> time_mem_copy_total{0};
    std::chrono::duration<double> time_mem_copy_unzip{0};
    uint64_t nbytes_recv{0};

public:
    Impl(int stack_level) : ComponentVE(stack_level, false),
                            comm_front(stack_level,
                                       config.defaultGet<string>("address", "127.0.0.1"),
                                       config.defaultGet<int>("port", 4200),
                                       config.defaultGet<uint64_t>("delay", 0)),
                            compress_param(config.defaultGet<string>("compress_param", "zlib")),
                            stat_print_on_exit(config.defaultGet("prof", false)) {}
    ~Impl() override {
        if (stat_print_on_exit) {
            cout << compressor.pprintStats();
            cout << "Frontend:\n";
            cout << "  MemCopy: " << time_mem_copy_total.count() << "s" << endl;
            cout << "    UnZip: " << time_mem_copy_unzip.count() << "s" << endl;
            cout << "    Recv:  " << nbytes_recv / 1024.0 / 1024.0 << "MB" << endl;
        }
    }

    void execute(BhIR *bhir) override;

    void extmethod(const string &name, bh_opcode opcode) override {
        // ExtmethodFace does not have a default or copy constructor thus
        // we have to use its move constructor.
        extmethods.insert(make_pair(opcode, extmethod::ExtmethodFace(config, name)));
    }

    // Handle messages from parent
    string message(const string &msg) override {
        // Serialize message body
        vector<char> buf_body;
        msg::Message body(msg);
        body.serialize(buf_body);

        // Serialize message head
        vector<char> buf_head;
        msg::Header head(msg::Type::MSG, buf_body.size());
        head.serialize(buf_head);

        // Send serialized message
        comm_front.write(buf_head);
        comm_front.write(buf_body);

        stringstream ss;
        if (msg == "info") {
            ss << "----" << "\n";
            ss << "Proxy:" << "\n";
            ss << "  Frontend: " << "\n";
            ss << "    Hostname: " << comm_front.hostname() << "\n";
            ss << "    IP: "       << comm_front.ip();
        } else if (msg == "statistics-detail") {
            ss << "----" << "\n";
            ss << "Proxy:" << "\n";
            ss << compressor.pprintStatsDetail();
        }
        ss << comm_front.read(); // Read the message from the backend
        return ss.str();
    }

    // Handle memory pointer retrieval
    void* getMemoryPointer(bh_base &base, bool copy2host, bool force_alloc, bool nullify) override {
        if (not copy2host) {
            throw runtime_error("PROXY - getMemoryPointer(): `copy2host` is not True");
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
        vector<unsigned char> data = comm_front.recv_data();
        if (not data.empty()) {
            bh_data_malloc(&base);
            compressor.uncompress(data, base, compress_param);
        }

        if (force_alloc) {
            bh_data_malloc(&base);
        }

        // Nullify the data pointer
        void *ret = base.data;
        if (nullify) {
            base.data = nullptr;
            known_base_arrays.erase(&base);
        }
        return ret;
    }

    // Handle memory pointer obtainment
    void setMemoryPointer(bh_base *base, bool host_ptr, void *mem) override {
        if (not host_ptr) {
            throw runtime_error("PROXY - setMemoryPointer(): `host_ptr` is not True");
        }
        throw runtime_error("PROXY - setMemoryPointer(): not implemented");
    }

    // Handle memory copy
    void memCopy(bh_view &src, bh_view &dst, const std::string &param) override {
        if (bh_is_constant(&src) or bh_is_constant(&dst)) {
            throw runtime_error("PROXY - memCopy(): `src` and `dst` cannot be constants");
        }
        if (bh_nelements(src) != bh_nelements(dst)) {
            throw runtime_error("PROXY - memCopy(): `src` and `dst` must have same size");
        }
        if (util::exist(known_base_arrays, dst.base) or dst.base->data != nullptr) {
            throw runtime_error("PROXY - memCopy(): `dst` must be un-initiated");
        }

        auto t1 = chrono::steady_clock::now();

        // Serialize message body
        vector<char> buf_body;
        msg::MemCopy body(src, param);
        body.serialize(buf_body);

        // Serialize message head
        vector<char> buf_head;
        msg::Header head(msg::Type::MEM_COPY, buf_body.size());
        head.serialize(buf_head);

        // Send serialized message
        comm_front.write(buf_head);
        comm_front.write(buf_body);

        // Receive the array data
        vector<unsigned char> data = comm_front.recv_data();
        if (not data.empty()) {
            bh_data_malloc(dst.base);
            auto t2 = chrono::steady_clock::now();
            compressor.uncompress(data, dst, param);
            time_mem_copy_unzip += chrono::steady_clock::now() - t2;
            nbytes_recv += data.size();
        }
        time_mem_copy_total += chrono::steady_clock::now() - t1;
    }

    // We have no context so returning NULL
    void* getDeviceContext() override {
        return nullptr;
    };

    // We have no context so doing nothing
    void setDeviceContext(void* device_context) override {} ;

    // Handle extension methods in `bhir`
    void handleExtmethod(BhIR *bhir){
        std::vector<bh_instruction> instr_list;
        for (bh_instruction &instr: bhir->instr_list) {
            auto ext = extmethods.find(instr.opcode);

            if (ext != extmethods.end()) { // Execute the instructions up until now
                BhIR b(std::move(instr_list), bhir->getSyncs());
                execute(&b);
                instr_list.clear(); // Notice, it is legal to clear a moved vector.
                for (bh_view &op: instr.operand) {
                    getMemoryPointer(*op.base, true, true, false);
                }
                ext->second.execute(&instr, nullptr); // Execute the extension method
            } else {
                instr_list.push_back(instr);
            }
        }
        bhir->instr_list = instr_list;
    }
};
} //Unnamed namespace


extern "C" ComponentImpl* create(int stack_level) {
    return new Impl(stack_level);
}
extern "C" void destroy(ComponentImpl* self) {
    delete self;
}


void Impl::execute(BhIR *bhir) {

    handleExtmethod(bhir);

    // Serialize the BhIR, which becomes the message body
    vector<bh_base *> new_data; // New data in the order they appear in the instruction list
    vector<char> buf_body = bhir->writeSerializedArchive(known_base_arrays, new_data);

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
        auto data = compressor.compress(*base, compress_param);
        comm_front.send_data(data);
    }

    // Cleanup freed base array and make them unknown.
    for (const bh_instruction &instr: bhir->instr_list) {
        if (instr.opcode == BH_FREE) {
            bh_data_free(instr.operand[0].base);
            known_base_arrays.erase(instr.operand[0].base);
        }
    }
}
