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

#include "serialize.hpp"
#include "comm.hpp"

#include "zlib.h"


using boost::asio::ip::tcp;
using namespace std;

namespace {
void comm_send_array_data(boost::asio::ip::tcp::socket &socket, const void *data, size_t nbytes) {
    if (nbytes == 0 or data == nullptr) {
        const size_t size[] = {0};
        boost::asio::write(socket, boost::asio::buffer(size));
    } else {
        const size_t org_size = nbytes;
        size_t new_size = compressBound(org_size);
        vector<Bytef> buffer(new_size);
        compress(&buffer[0], &new_size, (Bytef *) data, org_size);

        const size_t size[] = {new_size};
        boost::asio::write(socket, boost::asio::buffer(size));
        boost::asio::write(socket, boost::asio::buffer(&buffer[0], new_size));
    }
}

void comm_recv_array_data(boost::asio::ip::tcp::socket &socket, bh_base *base) {
    size_t size[1];
    boost::asio::read(socket, boost::asio::buffer(size));

    if (size[0] > 0) {
        bh_data_malloc(base);
        const size_t org_size = bh_base_size(base);
        size_t new_size = org_size;

        vector<char> compressed(size[0]);
        boost::asio::read(socket, boost::asio::buffer(compressed));

        uncompress((Bytef *) base->data, &new_size, (Bytef *) (&compressed[0]), compressed.size());
        assert(new_size == org_size);
    }
}
}

CommFrontend::CommFrontend(int stack_level, const std::string &address, int port) : socket(io_service) {
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

void CommFrontend::send_array_data(const bh_base *base) {
    comm_send_array_data(socket, base->data, bh_base_size(base));
}

void CommFrontend::recv_array_data(bh_base *base) {
    comm_recv_array_data(socket, base);
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

void CommBackend::send_array_data(const void *data, size_t nbytes) {
    comm_send_array_data(socket, data, nbytes);
}

void CommBackend::recv_array_data(bh_base *base) {
    comm_recv_array_data(socket, base);
}
