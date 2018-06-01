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
#include <boost/asio.hpp>
#include <thread>         // std::this_thread::sleep_for
#include <chrono>         // std::chrono::seconds
#include <zlib.h>
#include <bh_main_memory.hpp>

#include "serialize.hpp"
#include "comm.hpp"


using boost::asio::ip::tcp;
using namespace std;

namespace {
void comm_send_data(boost::asio::ip::tcp::socket &socket, const std::vector<unsigned char> &data) {
    if (data.empty()) {
        const size_t size[] = {0};
        boost::asio::write(socket, boost::asio::buffer(size));
    } else {
        const size_t size[] = {data.size()};
        boost::asio::write(socket, boost::asio::buffer(size));
        boost::asio::write(socket, boost::asio::buffer(data, data.size()));
    }
}

std::vector<unsigned char> comm_recv_data(boost::asio::ip::tcp::socket &socket) {
    size_t size[1];
    boost::asio::read(socket, boost::asio::buffer(size));
    std::vector<unsigned char> ret(size[0]);
    if (not ret.empty()) {
        boost::asio::read(socket, boost::asio::buffer(ret));
    }
    return ret;
}
}

CommFrontend::CommFrontend(int stack_level,
                           const std::string &address,
                           int port,
                           uint64_t sim_bandwidth) : sim_bandwidth(sim_bandwidth), socket(io_service) {
    constexpr unsigned int retries = 100;
    for (unsigned int i = 1; i <= retries; ++i) {
        try {
            cout << "[PROXY-VEM] Connecting to " << address << ":" << port << endl;
            // Get a list of endpoints corresponding to the server name.
            tcp::resolver resolver(io_service);
            tcp::resolver::query query(address, to_string(port));
            tcp::resolver::iterator endpoint_iterator = resolver.resolve(query);
            tcp::resolver::iterator end;

            // Try each endpoint until we successfully establish a connection.
            boost::system::error_code error = boost::asio::error::host_not_found;
            while (error && endpoint_iterator != end) {
                socket.close();
                socket.connect(*endpoint_iterator++, error);
            }
            if (error)
                throw boost::system::system_error(error);
            socket.set_option(boost::asio::ip::tcp::no_delay(true));
            goto connected;
        }
        catch (const boost::system::system_error &e) {
            this_thread::sleep_for(chrono::seconds(1));
            cerr << e.what() << endl;
            cout << "Retrying - attempt number " << i << " of " << retries << endl;
        }
    }
    throw runtime_error("[PROXY-VEM] No connection!");

    connected:
    // Serialize message body
    vector<char> buf_body;
    msg::Init body(stack_level);
    body.serialize(buf_body);

    //Serialize message head
    vector<char> buf_head;
    msg::Header head(msg::Type::INIT, buf_body.size());
    head.serialize(buf_head);

    //Send serialized message
    write(buf_head);
    write(buf_body);
}

CommFrontend::~CommFrontend() {
    //Serialize message head
    vector<char> buf_head;
    msg::Header head(msg::Type::SHUTDOWN, 0);
    head.serialize(buf_head);

    //Send serialized message
    boost::asio::write(socket, boost::asio::buffer(buf_head));
    socket.shutdown(boost::asio::ip::tcp::socket::shutdown_both);
    socket.close();
}

void CommFrontend::send_data(const std::vector<unsigned char> &data) {
    auto t = chrono::steady_clock::now();
    comm_send_data(socket, data);
    std::chrono::duration<double> comm_time = chrono::steady_clock::now() - t;
    std::chrono::duration<double> sim_time = std::chrono::duration<double>{data.size() / (double) sim_bandwidth};
    if (comm_time < sim_time) {
        sim_time -= comm_time;
    }
    if (sim_time.count() > 0) {
        std::this_thread::sleep_for(sim_time);
    }
}

std::vector<unsigned char> CommFrontend::recv_data() {
    auto t = chrono::steady_clock::now();
    std::vector<unsigned char> ret = comm_recv_data(socket);
    std::chrono::duration<double> comm_time = chrono::steady_clock::now() - t;
    std::chrono::duration<double> sim_time = std::chrono::duration<double>{ret.size() / (double) sim_bandwidth};
    if (comm_time < sim_time) {
        sim_time -= comm_time;
    }
    if (sim_time.count() > 0) {
        std::this_thread::sleep_for(sim_time);
    }
    return ret;
}

std::string CommFrontend::read() {
    vector<char> str_vec;
    while(1) {
        char buf;
        size_t bytes = boost::asio::read(socket, boost::asio::buffer(&buf, 1));
        if (bytes != 1 or buf == '\0') {
            break;
        }
        str_vec.push_back(buf);
    }
    return std::string(str_vec.begin(), str_vec.end());
}

CommBackend::CommBackend(const std::string &address, int port) : socket(io_service) {
    cout << "[PROXY-VEM] Server listen on port " << port << endl;
    tcp::acceptor acceptor(io_service, tcp::endpoint(tcp::v4(), port));
    acceptor.accept(socket);
    socket.set_option(boost::asio::ip::tcp::no_delay(true));
}

CommBackend::~CommBackend() {
    socket.shutdown(boost::asio::ip::tcp::socket::shutdown_both);
    socket.close();
}

void CommBackend::send_data(const std::vector<unsigned char> &data) {
    comm_send_data(socket, data);
}

std::vector<unsigned char> CommBackend::recv_data() {
    return comm_recv_data(socket);
}
